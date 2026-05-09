"""Deployment ensemble: rank-normalized mean of three model families.

The deployment score is the **rank-normalized mean probability** across:

    1. ``DeploymentXGBModel``: 6-booster XGBoost on MACCS+pocket+physchem
       (183 features). The original deployment model from try4.
    2. ``MorganXGBModel``: 6-booster XGBoost on Morgan ECFP4 (2048) + physchem
       (8 cols); the try3 ``xgb_no_val`` recipe, retrained with 6-fold LOFO.
    3. ``ChemeleonModel``: 6-booster CheMeleon transfer-learning MPNN;
       the try3 ``chemeleon_no_val`` recipe, retrained with 6-fold LOFO.

Rationale: each model captures a different chemical-space neighborhood;
their probability scales are not comparable (scale_pos_weight inflates
the XGB scores, CheMeleon's softmax outputs stay near the base rate).
Rank-normalizing each model's scores to [0, 1] gives every model an
equal vote in the mean. See the analysis in
``/tmp/opencode/model-weakness/`` for the data-driven comparison.

Feature pipelines at inference time
-----------------------------------
- ``DeploymentXGBModel``: MACCS (167) + pocket (8) + physchem (8) = 183.
- ``MorganXGBModel``: Morgan ECFP4 2048 + physchem (8) = 2056.
- ``ChemeleonModel``: SMILES go straight into the chemprop MPNN.

Example:
    >>> from tbxt_hackathon.deployment import EnsembleModel
    >>> model = EnsembleModel.load()
    >>> out = model.predict(["CCOC(=O)c1ccc(O)cc1", "Nc1ccc(Cl)cc1"])
    >>> out["ensemble_rank_score"]
    array([0.82, 0.65])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import xgboost as xgb
from loguru import logger
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from scipy.stats import rankdata
from typeguard import typechecked

from .exceptions import DataError
from .fingerprints import maccs_ndarray, morgan_ndarray
from .pocket_assigner import PocketAssigner
from .xgb_baseline import PHYSCHEM_COLUMNS, load_xgb_model, predict_proba_xgb

# Relative paths: we resolve against the repo root (two levels up from this file)
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = _REPO_ROOT / "data" / "deployment-model"
DEFAULT_FRAGMENTS_CSV = _REPO_ROOT / "data" / "structures" / "sgc_fragments.csv"

POCKETS: tuple[str, ...] = ("A", "B", "C", "D")
POCKET_COLUMNS: tuple[str, ...] = tuple(
    f"pocket_{p}_{kind}" for p in POCKETS for kind in ("tanimoto", "substruct")
)

# ---- feature spec(s): source of truth for each model's pipeline -----------

DEPLOYMENT_FEATURE_SPEC: dict[str, Any] = {
    "fingerprint": {"type": "maccs", "n_bits": 167},
    "pocket": {
        "pockets": list(POCKETS),
        "per_pocket_columns": ["tanimoto", "substruct"],
        "n_cols": len(POCKET_COLUMNS),
        "fragments_csv": "data/structures/sgc_fragments.csv",
        "radius": 2,
        "threshold": 0.35,
    },
    "physchem": {
        "columns": list(PHYSCHEM_COLUMNS),
        "n_cols": len(PHYSCHEM_COLUMNS),
    },
    "column_order": ["maccs", "pocket", "physchem"],
    "total_cols": 167 + len(POCKET_COLUMNS) + len(PHYSCHEM_COLUMNS),
}

MORGAN_FEATURE_SPEC: dict[str, Any] = {
    "fingerprint": {"type": "morgan", "n_bits": 2048, "radius": 2},
    "physchem": {
        "columns": list(PHYSCHEM_COLUMNS),
        "n_cols": len(PHYSCHEM_COLUMNS),
    },
    "column_order": ["morgan", "physchem"],
    "total_cols": 2048 + len(PHYSCHEM_COLUMNS),
}

CHEMELEON_SPEC: dict[str, Any] = {
    "input": "canonical_smiles",
    "encoder": "chemprop BondMessagePassing with CheMeleon pretrained weights",
    "head": "BinaryClassificationFFN (2 hidden layers, hidden_dim=256, dropout=0.2)",
    "training": "no-val, 15 epochs, batch_size=32, lr=1e-3",
}

ENSEMBLE_SPEC: dict[str, Any] = {
    "models": ["deployment_xgb_maccs", "morgan_xgb", "chemeleon"],
    "aggregation": "mean of per-model rank-normalized scores",
    "rank_norm": "scipy.stats.rankdata(-p) / N  -> lower is rank 1, higher is rank N",
    "score_direction": "higher rank_score == more likely binder",
}


# ---- physchem computed from SMILES ---------------------------------------


@typechecked
def _physchem_from_mol(mol: Chem.Mol) -> dict[str, float]:
    """Compute the 8 physchem descriptors we use at training time.

    Order matches ``PHYSCHEM_COLUMNS`` exactly.
    """
    return {
        "mw": float(Descriptors.MolWt(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "hbd": float(Lipinski.NumHDonors(mol)),
        "hba": float(Lipinski.NumHAcceptors(mol)),
        "heavy_atoms": float(mol.GetNumHeavyAtoms()),
        "num_rings": float(rdMolDescriptors.CalcNumRings(mol)),
        "tpsa": float(Descriptors.TPSA(mol)),
        "rotatable_bonds": float(Lipinski.NumRotatableBonds(mol)),
    }


@typechecked
def _physchem_matrix(mols: list[Chem.Mol]) -> np.ndarray:
    """Build a (n, 8) float32 matrix of physchem features."""
    rows = [_physchem_from_mol(m) for m in mols]
    return np.asarray(
        [[r[col] for col in PHYSCHEM_COLUMNS] for r in rows],
        dtype=np.float32,
    )


# ---- deployment (MACCS+pocket+physchem) featurizer ------------------------


@dataclass
class DeploymentFeatureMatrix:
    """Output of :func:`build_deployment_features`.

    Attributes:
        X: (n, 183) float32 feature matrix.
        maccs_end: end of the MACCS slice (``X[:, :maccs_end]``).
        pocket_end: end of the pocket slice (``X[:, maccs_end:pocket_end]``).
        phys_end: end of the physchem slice (``X[:, pocket_end:phys_end]``).
    """

    X: np.ndarray
    maccs_end: int
    pocket_end: int
    phys_end: int


@typechecked
def build_deployment_features(
    frame: pl.DataFrame,
    smiles_col: str = "canonical_smiles",
) -> DeploymentFeatureMatrix:
    """Build the 183-column MACCS+pocket+physchem matrix from a dataframe.

    Training-time entry point. The frame must already carry the pocket
    and physchem columns; use :func:`featurize_smiles_deployment` for
    inference from raw SMILES.

    Raises:
        DataError: if required columns are missing or contain NaN.

    Example:
        >>> feats = build_deployment_features(df)
        >>> feats.X.shape
        (708, 183)
    """
    missing_pocket = [c for c in POCKET_COLUMNS if c not in frame.columns]
    if missing_pocket:
        raise DataError(
            f"frame missing pocket columns: {missing_pocket}. "
            "Run scripts/classification-models-try4-rjg/02_make_pocket_features.py "
            "then join onto your frame, or use DeploymentXGBModel.predict(smiles_list) "
            "which handles pocket featurization internally.",
        )
    missing_phys = [c for c in PHYSCHEM_COLUMNS if c not in frame.columns]
    if missing_phys:
        raise DataError(f"frame missing physchem columns: {missing_phys}")

    smiles = frame[smiles_col].to_list()
    maccs, _ = maccs_ndarray(smiles)
    maccs = maccs.astype(np.float32)
    pocket = frame.select(list(POCKET_COLUMNS)).to_numpy().astype(np.float32)
    phys = frame.select(list(PHYSCHEM_COLUMNS)).to_numpy().astype(np.float32)

    if np.isnan(pocket).any():
        raise DataError("pocket features contain NaN")
    if np.isnan(phys).any():
        raise DataError("physchem features contain NaN")

    X = np.concatenate([maccs, pocket, phys], axis=1)
    expected_cols = DEPLOYMENT_FEATURE_SPEC["total_cols"]
    assert X.shape[1] == expected_cols, (
        f"built {X.shape[1]} cols, expected {expected_cols}"
    )
    return DeploymentFeatureMatrix(
        X=X,
        maccs_end=maccs.shape[1],
        pocket_end=maccs.shape[1] + pocket.shape[1],
        phys_end=maccs.shape[1] + pocket.shape[1] + phys.shape[1],
    )


@typechecked
def featurize_smiles_deployment(
    smiles_list: list[str],
    fragments_csv: Path | str | None = None,
) -> np.ndarray:
    """End-to-end: SMILES -> (n, 183) MACCS+pocket+physchem feature matrix.

    Args:
        smiles_list: list of canonical SMILES.
        fragments_csv: path to ``sgc_fragments.csv``; defaults to repo-standard.

    Returns:
        (len(smiles_list), 183) float32 matrix, training-time column order.

    Raises:
        DataError: if any SMILES fails to parse.
    """
    fragments_csv = Path(fragments_csv) if fragments_csv else DEFAULT_FRAGMENTS_CSV

    mols: list[Chem.Mol] = []
    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise DataError(f"invalid SMILES at index {idx}: {smi!r}")
        mols.append(mol)

    maccs, _ = maccs_ndarray(smiles_list)
    maccs = maccs.astype(np.float32)

    assigner = PocketAssigner.from_csv(fragments_csv)
    pocket_rows = []
    for scores in assigner.score_batch(smiles_list):
        row = []
        for p in POCKETS:
            if p in scores:
                row.extend([
                    float(scores[p].tanimoto),
                    1.0 if scores[p].substruct else 0.0,
                ])
            else:
                row.extend([0.0, 0.0])
        pocket_rows.append(row)
    pocket = np.asarray(pocket_rows, dtype=np.float32)
    assert pocket.shape == (len(smiles_list), len(POCKET_COLUMNS))

    phys = _physchem_matrix(mols)

    X = np.concatenate([maccs, pocket, phys], axis=1)
    expected = DEPLOYMENT_FEATURE_SPEC["total_cols"]
    assert X.shape[1] == expected, f"built {X.shape[1]} cols, expected {expected}"
    return X


# ---- Morgan (ECFP4) + physchem featurizer --------------------------------


@typechecked
def featurize_smiles_morgan(
    smiles_list: list[str],
    n_bits: int = 2048,
    radius: int = 2,
) -> np.ndarray:
    """End-to-end: SMILES -> (n, 2056) Morgan FP + physchem feature matrix.

    Matches the try3 ``xgb_no_val`` training pipeline exactly.

    Raises:
        DataError: if any SMILES fails to parse.
    """
    mols: list[Chem.Mol] = []
    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise DataError(f"invalid SMILES at index {idx}: {smi!r}")
        mols.append(mol)

    fp_arr, _ = morgan_ndarray(smiles_list, n_bits=n_bits, radius=radius)
    fp_arr = fp_arr.astype(np.float32)
    phys = _physchem_matrix(mols)
    X = np.concatenate([fp_arr, phys], axis=1)
    expected = MORGAN_FEATURE_SPEC["total_cols"]
    assert X.shape[1] == expected, f"built {X.shape[1]} cols, expected {expected}"
    return X


# Back-compat alias used by the prior 01_train_ensemble.py
featurize_smiles = featurize_smiles_deployment


# ---- rank normalization helper -------------------------------------------


@typechecked
def rank_norm(scores: np.ndarray) -> np.ndarray:
    """Map a score vector to [0, 1] by rank / N.

    Higher ``scores[i]`` maps to higher output. NaN inputs remain NaN
    on output and are excluded from the ranking denominator.

    Example:
        >>> import numpy as np
        >>> rank_norm(np.array([0.1, 0.9, 0.5])).tolist()
        [0.333..., 1.0, 0.666...]
    """
    out = np.full_like(scores, np.nan, dtype=np.float64)
    mask = ~np.isnan(scores)
    n = int(mask.sum())
    if n == 0:
        return out
    # rankdata assigns rank 1 to smallest. We want larger score -> larger output,
    # so rank of x in ascending order / N gives us [1/N, 2/N, ..., 1].
    out[mask] = rankdata(scores[mask], method="average") / n
    return out


# ---- XGB-based model wrappers --------------------------------------------


@dataclass
class _XGBEnsembleBase:
    """Common behaviour for the two XGB-based per-model ensembles.

    Subclasses define ``featurize(smiles_list) -> np.ndarray`` and a
    ``MODEL_GLOB`` class attribute. ``predict`` averages per-booster
    probabilities.
    """

    boosters: list[xgb.XGBClassifier]
    model_dir: Path

    MODEL_GLOB: str = ""

    @typechecked
    def predict(
        self,
        smiles_list: list[str],
        return_per_booster: bool = False,
    ) -> np.ndarray:
        """Return mean P(binder) or the per-booster matrix."""
        X = self.featurize(smiles_list)
        per_booster = np.column_stack([predict_proba_xgb(b, X) for b in self.boosters])
        if return_per_booster:
            return per_booster
        return per_booster.mean(axis=1)

    def featurize(self, smiles_list: list[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


@dataclass
class DeploymentXGBModel(_XGBEnsembleBase):
    """6-booster XGBoost on MACCS (167) + pocket (8) + physchem (8) = 183 features.

    The original try4 deployment model, trained by
    ``scripts/deployment-model/01_train_ensemble.py``.
    """

    fragments_csv: Path = field(default_factory=lambda: DEFAULT_FRAGMENTS_CSV)
    MODEL_GLOB: str = "xgb_deploy_fold_*.ubj"

    @classmethod
    @typechecked
    def load(
        cls,
        model_dir: Path | str | None = None,
        fragments_csv: Path | str | None = None,
    ) -> "DeploymentXGBModel":
        model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        fragments_csv = Path(fragments_csv) if fragments_csv else DEFAULT_FRAGMENTS_CSV

        booster_paths = sorted(model_dir.glob(cls.MODEL_GLOB))
        if not booster_paths:
            raise FileNotFoundError(
                f"no boosters matching {cls.MODEL_GLOB} in {model_dir}. "
                "Run scripts/deployment-model/01_train_ensemble.py first.",
            )
        boosters = [load_xgb_model(p) for p in booster_paths]
        logger.info(f"loaded {len(boosters)} deployment XGB boosters from {model_dir}")
        return cls(boosters=boosters, model_dir=model_dir, fragments_csv=fragments_csv)

    def featurize(self, smiles_list: list[str]) -> np.ndarray:
        return featurize_smiles_deployment(smiles_list, fragments_csv=self.fragments_csv)


@dataclass
class MorganXGBModel(_XGBEnsembleBase):
    """6-booster XGBoost on Morgan ECFP4 (2048) + physchem (8) = 2056 features.

    try3 ``xgb_no_val`` recipe retrained with 6-fold leave-one-out for the
    deployment ensemble.
    """

    MODEL_GLOB: str = "xgb_morgan_fold_*.ubj"

    @classmethod
    @typechecked
    def load(cls, model_dir: Path | str | None = None) -> "MorganXGBModel":
        model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        booster_paths = sorted(model_dir.glob(cls.MODEL_GLOB))
        if not booster_paths:
            raise FileNotFoundError(
                f"no boosters matching {cls.MODEL_GLOB} in {model_dir}. "
                "Run scripts/deployment-model/01_train_ensemble.py first.",
            )
        boosters = [load_xgb_model(p) for p in booster_paths]
        logger.info(f"loaded {len(boosters)} Morgan XGB boosters from {model_dir}")
        return cls(boosters=boosters, model_dir=model_dir)

    def featurize(self, smiles_list: list[str]) -> np.ndarray:
        return featurize_smiles_morgan(smiles_list)


# Back-compat alias for the old single-model deployment API
DeploymentModel = DeploymentXGBModel


# ---- CheMeleon wrapper ---------------------------------------------------


@dataclass
class ChemeleonModel:
    """6-checkpoint CheMeleon MPNN ensemble.

    Loading the checkpoints triggers a torch import, which is slow and
    pulls in a large dependency graph. Only import the chemeleon_transfer
    submodule when this class is instantiated, so the XGB-only code path
    stays light.

    Attributes:
        checkpoint_paths: paths to the 6 lightning checkpoints.
        model_dir: directory from which checkpoints were loaded.
        accelerator: passed to chemprop's ``predict_proba``.
    """

    checkpoint_paths: list[Path]
    model_dir: Path
    accelerator: str = "auto"

    MODEL_GLOB: str = "chemeleon_fold_*.ckpt"

    @classmethod
    @typechecked
    def load(
        cls,
        model_dir: Path | str | None = None,
        accelerator: str = "auto",
    ) -> "ChemeleonModel":
        """Discover the checkpoints but defer loading to predict()."""
        model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        ckpt_paths = sorted(model_dir.glob(cls.MODEL_GLOB))
        if not ckpt_paths:
            raise FileNotFoundError(
                f"no checkpoints matching {cls.MODEL_GLOB} in {model_dir}. "
                "Run scripts/deployment-model/01_train_ensemble.py first.",
            )
        logger.info(f"discovered {len(ckpt_paths)} CheMeleon checkpoints in {model_dir}")
        return cls(
            checkpoint_paths=ckpt_paths, model_dir=model_dir, accelerator=accelerator,
        )

    @typechecked
    def predict(
        self,
        smiles_list: list[str],
        return_per_booster: bool = False,
        batch_size: int = 128,
    ) -> np.ndarray:
        """Score SMILES with all 6 checkpoints; return mean or per-model matrix."""
        # Defer heavy imports until we actually predict.
        from chemprop.models import MPNN

        from .chemeleon_transfer import predict_proba

        per_model = np.zeros((len(smiles_list), len(self.checkpoint_paths)), dtype=np.float32)
        for i, ckpt in enumerate(self.checkpoint_paths):
            logger.info(f"chemeleon predict: {ckpt.name}")
            model = MPNN.load_from_checkpoint(str(ckpt))
            per_model[:, i] = predict_proba(
                model,
                smiles_list,
                batch_size=batch_size,
                accelerator=self.accelerator,
            )
            # Release as soon as possible; 6 of these times 2048-dim MPNN
            # weights will add up quickly.
            del model
        if return_per_booster:
            return per_model
        return per_model.mean(axis=1)


# ---- combined 3-model rank-ensemble --------------------------------------


@dataclass
class EnsemblePrediction:
    """Per-compound scores from each constituent model + the combined score.

    All arrays have shape ``(n,)`` unless otherwise noted. Rank columns
    are in [1, n]; rank-norm columns are in [1/n, 1].

    Attributes:
        p_deploy: P(binder) from DeploymentXGBModel (MACCS+pocket+physchem).
        p_morgan: P(binder) from MorganXGBModel.
        p_chemeleon: P(binder) from ChemeleonModel.
        rank_deploy: 1-based rank of p_deploy across the scored batch, 1 = best.
        rank_morgan: 1-based rank of p_morgan, 1 = best.
        rank_chemeleon: 1-based rank of p_chemeleon, 1 = best.
        score_deploy_rn: rank-normalized p_deploy, 1.0 = top compound in batch.
        score_morgan_rn: rank-normalized p_morgan.
        score_chemeleon_rn: rank-normalized p_chemeleon.
        ensemble_rank_score: mean of the three rank-normalized scores, in [0, 1].
    """

    p_deploy: np.ndarray
    p_morgan: np.ndarray
    p_chemeleon: np.ndarray
    rank_deploy: np.ndarray
    rank_morgan: np.ndarray
    rank_chemeleon: np.ndarray
    score_deploy_rn: np.ndarray
    score_morgan_rn: np.ndarray
    score_chemeleon_rn: np.ndarray
    ensemble_rank_score: np.ndarray


@dataclass
class EnsembleModel:
    """Three-model rank-normalized-mean ensemble: deploy-XGB + Morgan-XGB + CheMeleon.

    Scores are rank-normalized WITHIN EACH CALL to ``predict``. That means
    ``ensemble_rank_score`` for a compound depends on what else is in the
    input batch. For onepot screening the intended use is: score all
    100k+ candidates in one call, then take the top-K.

    Example:
        >>> model = EnsembleModel.load()
        >>> out = model.predict(["CCOC(=O)c1ccc(O)cc1", "Nc1ccc(Cl)cc1"])
        >>> out.ensemble_rank_score.shape
        (2,)
    """

    deploy: DeploymentXGBModel
    morgan: MorganXGBModel
    chemeleon: ChemeleonModel

    @classmethod
    @typechecked
    def load(
        cls,
        model_dir: Path | str | None = None,
        fragments_csv: Path | str | None = None,
        accelerator: str = "auto",
    ) -> "EnsembleModel":
        """Load all three constituent models from a shared directory."""
        model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        return cls(
            deploy=DeploymentXGBModel.load(
                model_dir=model_dir, fragments_csv=fragments_csv,
            ),
            morgan=MorganXGBModel.load(model_dir=model_dir),
            chemeleon=ChemeleonModel.load(model_dir=model_dir, accelerator=accelerator),
        )

    @typechecked
    def predict(
        self,
        smiles_list: list[str],
        chemeleon_batch_size: int = 128,
    ) -> EnsemblePrediction:
        """Score ``smiles_list`` with all three models and return the combined result.

        The ranking used for rank-normalization is performed across the
        input batch. Pass the full set of compounds you want to compare
        in a single call.
        """
        logger.info(
            f"ensemble predict: {len(smiles_list)} compounds -> "
            "deploy_xgb, morgan_xgb, chemeleon",
        )
        p_deploy = self.deploy.predict(smiles_list)
        p_morgan = self.morgan.predict(smiles_list)
        p_chemeleon = self.chemeleon.predict(
            smiles_list, batch_size=chemeleon_batch_size,
        )

        # 1 = best (highest prob), so we rank by descending prob.
        rank_deploy = rankdata(-p_deploy, method="average")
        rank_morgan = rankdata(-p_morgan, method="average")
        rank_chemeleon = rankdata(-p_chemeleon, method="average")

        # rank_norm: higher prob -> higher rn score (closer to 1.0)
        score_deploy_rn = rank_norm(p_deploy)
        score_morgan_rn = rank_norm(p_morgan)
        score_chemeleon_rn = rank_norm(p_chemeleon)

        ensemble_rank_score = (
            score_deploy_rn + score_morgan_rn + score_chemeleon_rn
        ) / 3.0

        return EnsemblePrediction(
            p_deploy=p_deploy,
            p_morgan=p_morgan,
            p_chemeleon=p_chemeleon,
            rank_deploy=rank_deploy,
            rank_morgan=rank_morgan,
            rank_chemeleon=rank_chemeleon,
            score_deploy_rn=score_deploy_rn,
            score_morgan_rn=score_morgan_rn,
            score_chemeleon_rn=score_chemeleon_rn,
            ensemble_rank_score=ensemble_rank_score,
        )
