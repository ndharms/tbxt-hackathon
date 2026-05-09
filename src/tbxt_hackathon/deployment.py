"""Deployment model: inference helpers for the MACCS+pocket+physchem XGB ensemble.

This module consolidates the feature pipeline and prediction ensemble used
by the deployment model produced by
``scripts/deployment-model/01_train_ensemble.py``. Use it from notebooks or
other scripts when you want to score new SMILES.

Feature recipe
--------------
    MACCS keys (167 bits)
  + pocket scores (4 pockets x [tanimoto, substruct] = 8 columns)
  + physchem descriptors (8 columns: mw, logp, hbd, hba, heavy_atoms,
    num_rings, tpsa, rotatable_bonds)
  -----------------------------------------------------------------
    183 float32 columns, same ordering as training time.

Example:
    >>> from tbxt_hackathon.deployment import DeploymentModel
    >>> model = DeploymentModel.load()
    >>> model.predict(["CCOC(=O)c1ccc(O)cc1", "Nc1ccc(Cl)cc1"])
    array([0.17, 0.42])
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import xgboost as xgb
from loguru import logger
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
from typeguard import typechecked

from .exceptions import DataError
from .fingerprints import maccs_ndarray
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

# Single source of truth for the feature pipeline. The training script
# writes this dict to feature_spec.json so downstream users can confirm
# they're running compatible featurization.
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


@dataclass
class DeploymentFeatureMatrix:
    """Output of :func:`build_deployment_features`.

    Attributes:
        X: (n, 183) float32 feature matrix.
        maccs_end: end of the MACCS slice (``X[:, :maccs_end]``).
        pocket_end: end of the pocket slice (``X[:, maccs_end:pocket_end]``).
        phys_end: end of the physchem slice (``X[:, pocket_end:phys_end]``); equals X.shape[1].
    """

    X: np.ndarray
    maccs_end: int
    pocket_end: int
    phys_end: int


# -- featurization helpers used at training time ----------------------------


@typechecked
def build_deployment_features(
    frame: pl.DataFrame,
    smiles_col: str = "canonical_smiles",
) -> DeploymentFeatureMatrix:
    """Build the full 183-column feature matrix from a dataframe.

    The frame must already carry the pocket and physchem columns
    (``pocket_A_tanimoto`` ... and ``mw, logp, ...``). This is the
    training-time entry point; ``DeploymentModel.predict`` handles
    featurization from raw SMILES.

    Args:
        frame: polars frame with SMILES + pocket + physchem columns.
        smiles_col: name of the SMILES column.

    Returns:
        DeploymentFeatureMatrix.

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
            "then join onto your frame, or use DeploymentModel.predict(smiles_list) "
            "which handles pocket featurization internally."
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
    maccs_end = maccs.shape[1]
    pocket_end = maccs_end + pocket.shape[1]
    phys_end = pocket_end + phys.shape[1]
    return DeploymentFeatureMatrix(
        X=X, maccs_end=maccs_end, pocket_end=pocket_end, phys_end=phys_end
    )


# -- physchem computed from SMILES (for inference) --------------------------


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
def featurize_smiles(
    smiles_list: list[str],
    fragments_csv: Path | str | None = None,
) -> np.ndarray:
    """End-to-end: SMILES -> (n, 183) feature matrix for the deployment model.

    Unlike :func:`build_deployment_features`, this function does NOT require
    a prepared dataframe -- it computes pocket scores and physchem on the
    fly from the SMILES strings alone.

    Args:
        smiles_list: list of canonical SMILES.
        fragments_csv: path to ``sgc_fragments.csv``; defaults to the
            repo-standard location.

    Returns:
        (len(smiles_list), 183) float32 feature matrix with the training-
        time column ordering.

    Raises:
        DataError: if any SMILES fails to parse.
    """
    fragments_csv = Path(fragments_csv) if fragments_csv else DEFAULT_FRAGMENTS_CSV

    # Validate SMILES up front so we can point the user at the bad one
    mols: list[Chem.Mol] = []
    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise DataError(f"invalid SMILES at index {idx}: {smi!r}")
        mols.append(mol)

    # MACCS
    maccs, _ = maccs_ndarray(smiles_list)
    maccs = maccs.astype(np.float32)

    # Pocket features
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

    # Physchem
    phys_rows = [_physchem_from_mol(mol) for mol in mols]
    phys = np.asarray(
        [[row[col] for col in PHYSCHEM_COLUMNS] for row in phys_rows],
        dtype=np.float32,
    )

    X = np.concatenate([maccs, pocket, phys], axis=1)
    expected = DEPLOYMENT_FEATURE_SPEC["total_cols"]
    assert X.shape[1] == expected, f"built {X.shape[1]} cols, expected {expected}"
    return X


# -- ensemble wrapper ------------------------------------------------------


@dataclass
class DeploymentModel:
    """Leave-one-fold-out ensemble wrapper around 6 XGBoost classifiers.

    Attributes:
        boosters: list of trained XGBoost classifiers (one per held-out fold).
        fragments_csv: path to SGC fragments CSV used for pocket featurization.

    Example:
        >>> model = DeploymentModel.load()
        >>> p = model.predict(["CCOC(=O)c1ccc(O)cc1", "Nc1ccc(Cl)cc1"])
        >>> p.shape
        (2,)
        >>> all(0.0 <= pi <= 1.0 for pi in p)
        True
    """

    boosters: list[xgb.XGBClassifier]
    fragments_csv: Path
    model_dir: Path

    @classmethod
    @typechecked
    def load(
        cls,
        model_dir: Path | str | None = None,
        fragments_csv: Path | str | None = None,
    ) -> "DeploymentModel":
        """Load the 6 saved boosters from disk.

        Args:
            model_dir: directory containing the ``xgb_deploy_fold_*.ubj``
                boosters. Defaults to ``data/deployment-model/``.
            fragments_csv: path to ``sgc_fragments.csv``. Defaults to
                ``data/structures/sgc_fragments.csv``.

        Returns:
            A ready-to-predict DeploymentModel.
        """
        model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        fragments_csv = Path(fragments_csv) if fragments_csv else DEFAULT_FRAGMENTS_CSV

        booster_paths = sorted(model_dir.glob("xgb_deploy_fold_*.ubj"))
        if not booster_paths:
            raise FileNotFoundError(
                f"no boosters found in {model_dir}. "
                "Run scripts/deployment-model/01_train_ensemble.py first."
            )
        boosters = [load_xgb_model(p) for p in booster_paths]
        logger.info(f"loaded {len(boosters)} deployment boosters from {model_dir}")
        return cls(boosters=boosters, fragments_csv=fragments_csv, model_dir=model_dir)

    @typechecked
    def predict(
        self,
        smiles_list: list[str],
        return_per_model: bool = False,
    ) -> np.ndarray:
        """Score a list of SMILES with the ensemble.

        Args:
            smiles_list: list of canonical SMILES.
            return_per_model: if True, return a (n_compounds, n_boosters)
                matrix of per-booster probabilities instead of the mean.

        Returns:
            If ``return_per_model`` is False (default): a (len(smiles_list),)
            array of ensemble-mean P(binder).
            Otherwise: a (len(smiles_list), len(self.boosters)) matrix.

        Example:
            >>> model = DeploymentModel.load()
            >>> model.predict(["CCO"]).shape
            (1,)
            >>> model.predict(["CCO"], return_per_model=True).shape
            (1, 6)
        """
        X = featurize_smiles(smiles_list, fragments_csv=self.fragments_csv)
        per_model = np.column_stack([predict_proba_xgb(b, X) for b in self.boosters])
        if return_per_model:
            return per_model
        return per_model.mean(axis=1)
