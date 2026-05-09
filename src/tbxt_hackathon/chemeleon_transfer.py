"""Transfer-learning from CheMeleon: encoder + 2-layer NN classifier head.

The CheMeleon pretrained weights load into a chemprop ``BondMessagePassing``
encoder (2048-dim output). We attach a ``BinaryClassificationFFN`` with two
hidden layers instead of the chemprop default single-hidden-layer FFN.

Training uses chemprop's Lightning infrastructure with an early-stop on
validation AUPRC, as is standard for imbalanced binder prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from chemprop import data as cp_data
from chemprop import featurizers, nn
from chemprop.models import MPNN
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from loguru import logger
from rdkit.Chem import Mol
from typeguard import typechecked

from .exceptions import ModelError

CHEMELEON_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
DEFAULT_CHEMELEON_PATH = Path.home() / ".chemprop" / "chemeleon_mp.pt"


@typechecked
def download_chemeleon(path: Path = DEFAULT_CHEMELEON_PATH) -> Path:
    """Download the pretrained CheMeleon message-passing checkpoint if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        logger.info(f"downloading CheMeleon weights to {path}")
        urlretrieve(CHEMELEON_URL, path)
    return path


@typechecked
def build_chemeleon_encoder(
    checkpoint_path: Path = DEFAULT_CHEMELEON_PATH,
) -> nn.BondMessagePassing:
    """Load the CheMeleon pretrained BondMessagePassing stack."""
    if not checkpoint_path.exists():
        download_chemeleon(checkpoint_path)
    ckpt = torch.load(checkpoint_path, weights_only=True)
    mp = nn.BondMessagePassing(**ckpt["hyper_parameters"])
    mp.load_state_dict(ckpt["state_dict"])
    logger.info(f"loaded CheMeleon encoder (output_dim={mp.output_dim})")
    return mp


@dataclass(frozen=True)
class ClassifierConfig:
    """Hyperparameters for the transfer-learning classifier.

    Attributes:
        hidden_dim: size of each of the two hidden layers in the FFN head.
        dropout: dropout applied between FFN layers.
        max_epochs: hard cap on training epochs (early-stop usually hits first).
        patience: early-stop patience (epochs without val improvement).
        batch_size: training mini-batch size.
        learning_rate: initial LR for AdamW.
        weight_decay: L2 coefficient.
        init_lr_ratio: chemprop's NoamLR ramps between init and max LR.
        final_lr_ratio: NoamLR final LR ratio.
        random_state: torch manual seed.
    """

    hidden_dim: int = 256
    dropout: float = 0.2
    max_epochs: int = 60
    patience: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    init_lr_ratio: float = 1e-4
    final_lr_ratio: float = 1e-4
    random_state: int = 0


@typechecked
def build_transfer_model(
    cfg: ClassifierConfig,
    checkpoint_path: Path = DEFAULT_CHEMELEON_PATH,
) -> MPNN:
    """Build a chemprop ``MPNN`` with CheMeleon encoder + 2-layer classifier FFN."""
    mp = build_chemeleon_encoder(checkpoint_path)
    agg = nn.MeanAggregation()
    # chemprop's BinaryClassificationFFN: ``n_layers`` counts hidden layers
    # between input and output; n_layers=2 => two hidden linear layers
    # (the demo used the default n_layers=1 single hidden; we override).
    ffn = nn.BinaryClassificationFFN(
        input_dim=mp.output_dim,
        hidden_dim=cfg.hidden_dim,
        n_layers=2,
        dropout=cfg.dropout,
    )
    metrics = [nn.metrics.BinaryAUROC(), nn.metrics.BinaryAUPRC()]
    model = MPNN(
        message_passing=mp,
        agg=agg,
        predictor=ffn,
        batch_norm=False,
        metrics=metrics,
        init_lr=cfg.learning_rate * cfg.init_lr_ratio,
        max_lr=cfg.learning_rate,
        final_lr=cfg.learning_rate * cfg.final_lr_ratio,
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"built transfer model; trainable params: {n_trainable:,}")
    return model


@typechecked
def _build_dataset(
    smiles: list[str],
    labels: np.ndarray,
    featurizer: featurizers.SimpleMoleculeMolGraphFeaturizer,
) -> cp_data.MoleculeDataset:
    """Chemprop MoleculeDataset from SMILES + (n,) binary labels."""
    if labels.ndim != 1:
        raise ModelError(f"labels must be 1D; got shape {labels.shape}")
    y = labels.astype(np.float32).reshape(-1, 1)
    dps = [cp_data.MoleculeDatapoint.from_smi(s, y[i]) for i, s in enumerate(smiles)]
    return cp_data.MoleculeDataset(dps, featurizer)


@dataclass
class TrainResult:
    """Outputs from one training run.

    Attributes:
        model: trained chemprop MPNN (restored to best val checkpoint).
        best_val_loss: lowest validation loss observed (BCE).
        best_val_auprc: val AUPRC at the best-loss epoch.
        best_val_auroc: val AUROC at the best-loss epoch.
        best_epoch: epoch index (0-based) of the best checkpoint.
        ckpt_path: filesystem path of the best checkpoint.
    """

    model: MPNN
    best_val_loss: float
    best_val_auprc: float | None
    best_val_auroc: float | None
    best_epoch: int
    ckpt_path: Path


@typechecked
def train_one(
    train_smiles: list[str],
    train_labels: np.ndarray,
    val_smiles: list[str],
    val_labels: np.ndarray,
    cfg: ClassifierConfig,
    checkpoint_dir: Path,
    accelerator: str = "auto",
    num_workers: int = 0,
) -> TrainResult:
    """Train one CheMeleon-transfer classifier with early stopping on val loss.

    Args:
        train_smiles/train_labels: training data. Labels must be 0/1.
        val_smiles/val_labels: validation data for early stopping.
        cfg: hyperparameters.
        checkpoint_dir: where to write checkpoints for this run.
        accelerator: passed to Lightning; ``"mps"`` for Apple Silicon,
            ``"cpu"``, ``"gpu"``, or ``"auto"``.
        num_workers: DataLoader workers; 0 on MPS to avoid fork issues.
    """
    torch.manual_seed(cfg.random_state)

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_dset = _build_dataset(train_smiles, train_labels, featurizer)
    val_dset = _build_dataset(val_smiles, val_labels, featurizer)

    train_loader = cp_data.build_dataloader(
        train_dset,
        num_workers=num_workers,
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    val_loader = cp_data.build_dataloader(
        val_dset,
        num_workers=num_workers,
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    model = build_transfer_model(cfg)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-{epoch}-{val_loss:.3f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    early_cb = EarlyStopping(monitor="val_loss", mode="min", patience=cfg.patience)

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=False,
        accelerator=accelerator,
        devices=1,
        max_epochs=cfg.max_epochs,
        callbacks=[ckpt_cb, early_cb],
        deterministic=False,
    )
    trainer.fit(model, train_loader, val_loader)

    best_path = Path(ckpt_cb.best_model_path)
    best_val = float(ckpt_cb.best_model_score) if ckpt_cb.best_model_score is not None else float("nan")
    logger.info(f"best val_loss={best_val:.4f} at {best_path.name}")

    # Restore best weights for inference
    best_model = MPNN.load_from_checkpoint(best_path)

    # Evaluate best model on val to get AUPRC/AUROC at that checkpoint.
    # chemprop logs val metrics under the metric's ``alias`` attribute, e.g.
    # ``val/roc``, ``val/prc`` for BinaryAUROC / BinaryAUPRC.
    val_metrics = trainer.validate(best_model, val_loader, verbose=False)[0]
    auprc_val = val_metrics.get("val/prc")
    auroc_val = val_metrics.get("val/roc")
    if auprc_val is not None:
        auprc_val = float(auprc_val)
    if auroc_val is not None:
        auroc_val = float(auroc_val)

    return TrainResult(
        model=best_model,
        best_val_loss=best_val,
        best_val_auprc=auprc_val,
        best_val_auroc=auroc_val,
        best_epoch=int(trainer.current_epoch),
        ckpt_path=best_path,
    )


@typechecked
def train_one_novalid(
    train_smiles: list[str],
    train_labels: np.ndarray,
    cfg: ClassifierConfig,
    save_path: Path | None = None,
    accelerator: str = "auto",
    num_workers: int = 0,
) -> MPNN:
    """Train a classifier for a fixed number of epochs with no validation set.

    Trades the early-stopping signal for ~33% more training data. Motivated by
    the observation that the validated-variant typically converges to its best
    val-loss within 12-14 epochs after a 2-epoch warmup. ``cfg.max_epochs``
    controls the training length; when ``save_path`` is provided the final
    Lightning checkpoint is written there (compatible with
    ``MPNN.load_from_checkpoint``).

    ``num_workers`` defaults to 0 because MPS + forked DataLoader workers
    deadlocks silently on macOS (especially once heavy C extensions like
    xgboost have been imported in the parent). Override to a positive int
    only on Linux/CUDA.
    """
    torch.manual_seed(cfg.random_state)

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_dset = _build_dataset(train_smiles, train_labels, featurizer)
    train_loader = cp_data.build_dataloader(
        train_dset,
        num_workers=num_workers,
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    model = build_transfer_model(cfg)
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accelerator=accelerator,
        devices=1,
        max_epochs=cfg.max_epochs,
        deterministic=False,
    )
    trainer.fit(model, train_loader)
    logger.info(f"no-val training complete at epoch {trainer.current_epoch} / {cfg.max_epochs}")
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(save_path))
        logger.info(f"saved final no-val checkpoint -> {save_path}")
    return model


@typechecked
def build_transfer_regression_model(
    cfg: ClassifierConfig,
    checkpoint_path: Path = DEFAULT_CHEMELEON_PATH,
) -> MPNN:
    """Build a chemprop ``MPNN`` with CheMeleon encoder + 2-layer regression FFN.

    Mirrors :func:`build_transfer_model` but targets continuous outputs
    (e.g. pKD). Uses MSE as the criterion and reports RMSE/MAE/R2 as val
    metrics for parity with the XGBoost regression baseline.
    """
    mp = build_chemeleon_encoder(checkpoint_path)
    agg = nn.MeanAggregation()
    ffn = nn.RegressionFFN(
        input_dim=mp.output_dim,
        hidden_dim=cfg.hidden_dim,
        n_layers=2,
        dropout=cfg.dropout,
        criterion=nn.metrics.MSE(),
    )
    metrics = [nn.metrics.RMSE(), nn.metrics.MAE(), nn.metrics.R2Score()]
    model = MPNN(
        message_passing=mp,
        agg=agg,
        predictor=ffn,
        batch_norm=False,
        metrics=metrics,
        init_lr=cfg.learning_rate * cfg.init_lr_ratio,
        max_lr=cfg.learning_rate,
        final_lr=cfg.learning_rate * cfg.final_lr_ratio,
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"built regression transfer model; trainable params: {n_trainable:,}")
    return model


@dataclass
class RegressionTrainResult:
    """Outputs from one regression training run.

    Attributes:
        model: trained chemprop MPNN (restored to best val checkpoint).
        best_val_loss: lowest validation loss observed (MSE).
        best_val_rmse: val RMSE at the best-loss epoch.
        best_val_mae: val MAE at the best-loss epoch.
        best_val_r2: val R2 at the best-loss epoch.
        best_epoch: epoch index (0-based) of the best checkpoint.
        ckpt_path: filesystem path of the best checkpoint.
    """

    model: MPNN
    best_val_loss: float
    best_val_rmse: float | None
    best_val_mae: float | None
    best_val_r2: float | None
    best_epoch: int
    ckpt_path: Path


@typechecked
def train_one_regression(
    train_smiles: list[str],
    train_targets: np.ndarray,
    val_smiles: list[str],
    val_targets: np.ndarray,
    cfg: ClassifierConfig,
    checkpoint_dir: Path,
    accelerator: str = "auto",
    num_workers: int = 0,
) -> RegressionTrainResult:
    """Train one CheMeleon-transfer regressor with early stopping on val loss.

    Regression analog of :func:`train_one`. Targets are continuous floats
    (e.g. pKD). No standardization is applied; if the caller wants z-scored
    targets it must transform inputs and invert predictions itself.
    """
    torch.manual_seed(cfg.random_state)

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_dset = _build_dataset(train_smiles, train_targets, featurizer)
    val_dset = _build_dataset(val_smiles, val_targets, featurizer)

    train_loader = cp_data.build_dataloader(
        train_dset,
        num_workers=num_workers,
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    val_loader = cp_data.build_dataloader(
        val_dset,
        num_workers=num_workers,
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    model = build_transfer_regression_model(cfg)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-{epoch}-{val_loss:.3f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    early_cb = EarlyStopping(monitor="val_loss", mode="min", patience=cfg.patience)

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=False,
        accelerator=accelerator,
        devices=1,
        max_epochs=cfg.max_epochs,
        callbacks=[ckpt_cb, early_cb],
        deterministic=False,
    )
    trainer.fit(model, train_loader, val_loader)

    best_path = Path(ckpt_cb.best_model_path)
    best_val = float(ckpt_cb.best_model_score) if ckpt_cb.best_model_score is not None else float("nan")
    logger.info(f"best val_loss (MSE)={best_val:.4f} at {best_path.name}")

    best_model = MPNN.load_from_checkpoint(best_path)

    # chemprop logs regression metrics under the metric's alias; RMSE/MAE/R2
    # keys are e.g. "val/rmse", "val/mae", "val/r2". Fall back to None if
    # chemprop's alias table changes between versions.
    val_metrics = trainer.validate(best_model, val_loader, verbose=False)[0]
    rmse_val = val_metrics.get("val/rmse")
    mae_val = val_metrics.get("val/mae")
    r2_val = val_metrics.get("val/r2")
    rmse_val = float(rmse_val) if rmse_val is not None else None
    mae_val = float(mae_val) if mae_val is not None else None
    r2_val = float(r2_val) if r2_val is not None else None

    return RegressionTrainResult(
        model=best_model,
        best_val_loss=best_val,
        best_val_rmse=rmse_val,
        best_val_mae=mae_val,
        best_val_r2=r2_val,
        best_epoch=int(trainer.current_epoch),
        ckpt_path=best_path,
    )


@typechecked
def train_one_novalid_regression(
    train_smiles: list[str],
    train_targets: np.ndarray,
    cfg: ClassifierConfig,
    save_path: Path | None = None,
    accelerator: str = "auto",
    num_workers: int = 0,
) -> MPNN:
    """Regression counterpart to :func:`train_one_novalid`.

    Fixed-epoch training on the full train pool with no validation set.
    """
    torch.manual_seed(cfg.random_state)

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_dset = _build_dataset(train_smiles, train_targets, featurizer)
    train_loader = cp_data.build_dataloader(
        train_dset,
        num_workers=num_workers,
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    model = build_transfer_regression_model(cfg)
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accelerator=accelerator,
        devices=1,
        max_epochs=cfg.max_epochs,
        deterministic=False,
    )
    trainer.fit(model, train_loader)
    logger.info(f"no-val regression training complete at epoch {trainer.current_epoch} / {cfg.max_epochs}")
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(save_path))
        logger.info(f"saved final no-val regression checkpoint -> {save_path}")
    return model


@typechecked
def predict_regression(
    model: MPNN,
    smiles: list[str | Mol],
    batch_size: int = 128,
    accelerator: str = "auto",
    num_workers: int = 0,
) -> np.ndarray:
    """Return continuous predictions (e.g. pKD) for each SMILES. Shape (n,).

    Same plumbing as :func:`predict_proba` but with no probability clamp;
    the model's raw head output is returned.
    """
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    placeholder = np.zeros(len(smiles), dtype=np.float32)
    dps = []
    for i, item in enumerate(smiles):
        y = np.array([placeholder[i]], dtype=np.float32)
        if isinstance(item, Mol):
            dps.append(cp_data.MoleculeDatapoint(mol=item, y=y))
        else:
            dps.append(cp_data.MoleculeDatapoint.from_smi(item, y))
    dset = cp_data.MoleculeDataset(dps, featurizer)
    loader = cp_data.build_dataloader(
        dset, num_workers=num_workers, batch_size=batch_size, shuffle=False,
    )
    trainer = pl.Trainer(
        logger=False, enable_progress_bar=False, accelerator=accelerator, devices=1,
    )
    preds = trainer.predict(model, loader)
    arr = torch.cat([p for p in preds], dim=0).cpu().numpy().reshape(-1)
    return arr


@typechecked
def predict_proba(
    model: MPNN,
    smiles: list[str | Mol],
    batch_size: int = 128,
    accelerator: str = "auto",
    num_workers: int = 0,
) -> np.ndarray:
    """Return P(binder) for each SMILES. Shape (n,).

    ``num_workers=0`` by default for the same MPS-fork-hang reason as
    :func:`train_one_novalid`.
    """
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    # labels are ignored at inference; pass zeros of right shape
    placeholder = np.zeros(len(smiles), dtype=np.float32)
    # from_smi only accepts strings; handle Mol -> SMILES for dataset build via datapoints
    dps = []
    for i, item in enumerate(smiles):
        y = np.array([placeholder[i]], dtype=np.float32)
        if isinstance(item, Mol):
            dps.append(cp_data.MoleculeDatapoint(mol=item, y=y))
        else:
            dps.append(cp_data.MoleculeDatapoint.from_smi(item, y))
    dset = cp_data.MoleculeDataset(dps, featurizer)
    loader = cp_data.build_dataloader(
        dset, num_workers=num_workers, batch_size=batch_size, shuffle=False,
    )
    trainer = pl.Trainer(
        logger=False, enable_progress_bar=False, accelerator=accelerator, devices=1,
    )
    preds = trainer.predict(model, loader)
    # chemprop returns list[Tensor(batch, 1)]; concatenate
    arr = torch.cat([p for p in preds], dim=0).cpu().numpy().reshape(-1)
    return arr
