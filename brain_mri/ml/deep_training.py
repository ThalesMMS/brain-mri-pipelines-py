import hashlib
import logging
import os
import random
import time
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from .datasets import MultiOrientMRIDataset
    from .multistream_models import MultiOrientTabularFusionNet
    from .training_utils import ExponentialMovingAverage, build_transforms, focal_loss, select_device
    try:
        from .debug_tools import debug_batch, debug_one_step
    except ImportError:
        debug_batch = debug_one_step = None
    TORCH_AVAILABLE = True
except ImportError:
    torch = nn = optim = DataLoader = None
    ExponentialMovingAverage = build_transforms = focal_loss = select_device = None
    debug_batch = debug_one_step = None
    TORCH_AVAILABLE = False

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        r2_score,
        recall_score,
    )
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    accuracy_score = balanced_accuracy_score = confusion_matrix = f1_score = None
    mean_absolute_error = mean_squared_error = precision_score = r2_score = recall_score = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False

try:
    from matplotlib.figure import Figure
except ImportError:
    Figure = None

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PILImage = None
    PIL_AVAILABLE = False

from .dataset_builder import populate_orientation_paths
from .embedding_export import _export_embeddings
from .training_utils import load_split_dataframe


CLASS_LABEL_MAP = {"Nondemented": 0, "Demented": 1}


@dataclass(frozen=True)
class DeepTrainingConfig:
    split_csv_path: Path
    output_dir: Path
    mode: str
    backbone: str
    hyperparameters: dict[str, Any] | None = None
    save_experiment_fn: Any = None
    headless: bool = False
    dataset_dir: Path | None = None
    show_plot_window_fn: Any = None
    plot_confusion_matrix_fn: Any = None


@dataclass(frozen=True)
class DeepTrainingResult:
    backbone: str
    mode: str
    best_checkpoint_path: Path
    legacy_checkpoint_path: Path
    learning_curves: dict[str, Any]
    metrics: dict[str, Any]
    experiment_payload: dict[str, Any]
    summary_message: str
    artifact_paths: dict[str, Path]


def _parse_bool(value, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
        raise ValueError(f"Cannot parse boolean value {value!r}.")
    return bool(value)


def _format_mri_id_for_error(mri_id) -> str:
    if _parse_bool(os.getenv("BRAIN_MRI_DEBUG_IDENTIFIERS", "0")):
        return repr(mri_id)
    if mri_id is None:
        return "<missing>"
    digest = hashlib.sha256(str(mri_id).encode("utf-8")).hexdigest()[:12]
    return f"<redacted:{digest}>"


def _validate_final_group_labels(df, split_name: str) -> None:
    if "Final_Group" not in df.columns:
        raise ValueError(f"Split {split_name} is missing required Final_Group labels.")
    invalid_mask = df["Final_Group"].isna() | ~df["Final_Group"].isin(CLASS_LABEL_MAP)
    if not invalid_mask.any():
        return

    invalid_rows = df.loc[invalid_mask]
    invalid_values = ", ".join(sorted({repr(value) for value in invalid_rows["Final_Group"].tolist()}))
    details = []
    for row_index, row in invalid_rows.head(10).iterrows():
        safe_mri_id = _format_mri_id_for_error(row.get("MRI_ID"))
        details.append(
            f"index={row_index}, MRI_ID={safe_mri_id}, Final_Group={row.get('Final_Group')!r}"
        )
    if len(invalid_rows) > 10:
        details.append(f"... {len(invalid_rows) - 10} more")
    expected = ", ".join(sorted(CLASS_LABEL_MAP))
    raise ValueError(
        f"Invalid Final_Group labels in split {split_name}: values [{invalid_values}] at {'; '.join(details)}. "
        f"Expected one of: {expected}."
    )


def _count_trainable_parameters(model, context: str) -> int | None:
    try:
        return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    except (AttributeError, TypeError) as exc:
        logger.exception("Failed counting trainable params for model during %s: %s", context, exc)
        return None


def _coerce_config(config) -> DeepTrainingConfig:
    """
    Normalize input into a DeepTrainingConfig instance.
    
    Parameters:
        config (DeepTrainingConfig | Mapping): Either an existing DeepTrainingConfig to be canonicalized
            (converts paths/strings/booleans to their canonical types) or a mapping with keys:
            "split_csv_path", "output_dir", "mode", "backbone". Optional mapping keys: "hyperparameters",
            "save_experiment_fn", "headless", "dataset_dir", "show_plot_window_fn", "plot_confusion_matrix_fn".
    
    Returns:
        DeepTrainingConfig: A canonicalized DeepTrainingConfig with Path objects for paths, strings for
        mode/backbone, a dict for hyperparameters, and normalized optional fields.
    
    Raises:
        TypeError: If `config` is neither a DeepTrainingConfig nor a Mapping.
    """
    if isinstance(config, DeepTrainingConfig):
        dataset_dir = Path(config.dataset_dir) if config.dataset_dir is not None else None
        return DeepTrainingConfig(
            split_csv_path=Path(config.split_csv_path),
            output_dir=Path(config.output_dir),
            mode=str(config.mode),
            backbone=str(config.backbone),
            hyperparameters=dict(config.hyperparameters or {}),
            save_experiment_fn=config.save_experiment_fn,
            headless=_parse_bool(config.headless),
            dataset_dir=dataset_dir,
            show_plot_window_fn=config.show_plot_window_fn,
            plot_confusion_matrix_fn=config.plot_confusion_matrix_fn,
        )
    if isinstance(config, Mapping):
        dataset_dir = config.get("dataset_dir")
        return DeepTrainingConfig(
            split_csv_path=Path(config["split_csv_path"]),
            output_dir=Path(config["output_dir"]),
            mode=str(config["mode"]),
            backbone=str(config["backbone"]),
            hyperparameters=dict(config.get("hyperparameters") or {}),
            save_experiment_fn=config.get("save_experiment_fn"),
            headless=_parse_bool(config.get("headless", False)),
            dataset_dir=Path(dataset_dir) if dataset_dir is not None else None,
            show_plot_window_fn=config.get("show_plot_window_fn"),
            plot_confusion_matrix_fn=config.get("plot_confusion_matrix_fn"),
        )
    raise TypeError("config must be a DeepTrainingConfig or mapping.")


def _require_dependencies(raise_on_missing: bool = True) -> bool:
    """
    Validate that required third-party libraries are available.

    When raise_on_missing is False, return False instead of raising for missing dependencies.
    
    Raises:
        ImportError: If `scikit-learn`, `torch`/`torchvision`, `pandas`, or Pillow is not importable. The raised message indicates which dependency is missing and suggests a pip install command.

    Returns:
        bool: True when all required dependencies are available; False only when raise_on_missing is False and a dependency is missing.
    """
    if not SKLEARN_AVAILABLE:
        if not raise_on_missing:
            return False
        raise ImportError(
            "O módulo 'scikit-learn' é necessário para normalização e métricas.\n"
            "Instale com 'pip install scikit-learn'."
        )
    if not TORCH_AVAILABLE:
        if not raise_on_missing:
            return False
        raise ImportError(
            "PyTorch/torchvision são necessários para este treino.\n"
            "Instale com 'pip install torch torchvision'."
        )
    if not PANDAS_AVAILABLE:
        if not raise_on_missing:
            return False
        raise ImportError(
            "O módulo 'pandas' é necessário para preparar os datasets de treino.\n"
            "Instale com 'pip install pandas'."
        )
    if not PIL_AVAILABLE:
        if not raise_on_missing:
            return False
        raise ImportError(
            "O módulo 'Pillow' é necessário para carregar imagens de treino.\n"
            "Instale com 'pip install pillow'."
        )
    return True


def _resolve_dataset_dir(cfg: DeepTrainingConfig) -> Path:
    """
    Determine the dataset root used to resolve orientation file paths.
    
    If `cfg.dataset_dir` points to the axl directory or a path inside it, the
    parent of that axl directory is returned. Otherwise the provided path is
    treated as the dataset root. When no dataset_dir is configured, the existing
    fallback axl sibling of output_dir is normalized the same way.
    
    Parameters:
        cfg (DeepTrainingConfig): Training configuration containing `dataset_dir` and `output_dir`.
    
    Returns:
        Path: Resolved dataset root directory.
    """
    dataset_path = Path(cfg.dataset_dir) if cfg.dataset_dir is not None else Path(cfg.output_dir).parent / "axl"
    if dataset_path.name == "axl":
        return dataset_path.parent
    for ancestor in dataset_path.parents:
        if ancestor.name == "axl":
            return ancestor.parent
    return dataset_path


def _maybe_show_plot(cfg: DeepTrainingConfig, title: str, figure) -> None:
    """
    Display a figure using the configured display callback when available.
    
    If cfg.headless is True or cfg.show_plot_window_fn is not callable, this function does nothing.
    Any exception raised by the callback is logged and suppressed.
    
    Parameters:
        cfg (DeepTrainingConfig): Configuration containing `headless` and optional `show_plot_window_fn`.
        title (str): Title to pass to the display callback.
        figure: Figure object to pass to the display callback (typically a matplotlib.figure.Figure).
    """
    if cfg.headless or not callable(cfg.show_plot_window_fn):
        return
    try:
        cfg.show_plot_window_fn(title, figure)
    except Exception as exc:
        callback_name = getattr(cfg.show_plot_window_fn, "__name__", repr(cfg.show_plot_window_fn))
        logger.exception(
            "Error in show_plot_window callback %s for title=%r figure_type=%s: %s",
            callback_name,
            title,
            type(figure).__name__,
            exc,
        )


def train_pytorch_model(config):
    """
    Train a multi-orientation MRI model (optionally with clinical/tabular features) for classification or regression and return the resulting artifacts and metrics.
    
    Parameters:
        config (DeepTrainingConfig | Mapping): Training configuration or a mapping that can be coerced to DeepTrainingConfig. Contains dataset split CSV path, output directory, mode ("classification" or "regression"), backbone id, optional hyperparameters, and optional callbacks for persisting or plotting results.
    
    Returns:
        DeepTrainingResult: Object containing backbone/mode identifiers, paths to saved checkpoints, learning-curve history, computed metrics, the persisted experiment payload, a human-readable summary message, and collected artifact paths.
    
    Raises:
        ImportError: If required runtime dependencies (e.g., torch, pandas, scikit-learn) are missing.
        TypeError: If `config` cannot be coerced into the expected DeepTrainingConfig.
        ValueError: If any dataset split (train/validation/test) is empty either before or after resolving orientation paths.
    """
    _require_dependencies()
    cfg = _coerce_config(config)
    dataset_root = _resolve_dataset_dir(cfg)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    mode = str(cfg.mode).lower()
    if mode not in {"classification", "regression"}:
        raise ValueError(f"mode must be 'classification' or 'regression', got {cfg.mode!r}")
    backbone = cfg.backbone
    hparams = dict(cfg.hyperparameters or {})
    clinical_features = hparams.get("clinical_features")
    if clinical_features is None:
        clinical_features = ["age", "education", "nwbv", "etiv", "asf"] if os.getenv("USE_MULTIMODAL", "1") == "1" else None
    label_column = "age_normalized" if mode == "regression" else "Final_Group"
    if clinical_features is not None:
        excluded_clinical_features = {label_column}
        if mode == "regression":
            excluded_clinical_features.add("age")
        clinical_features = [feature for feature in clinical_features if feature not in excluded_clinical_features]

    checkpoint_suffix = "classifier" if mode == "classification" else "regressor"
    best_checkpoint_path = output_dir / f"best_{backbone}_{checkpoint_suffix}.pth"
    legacy_checkpoint_path = output_dir / f"{backbone}_{mode}.pth"

    df_path = Path(cfg.split_csv_path)
    required_columns = ["split", "Final_Group", "MRI_ID"]
    if mode == "regression" or (clinical_features is not None and "age" in clinical_features):
        required_columns.append("age")
    if clinical_features is not None:
        for feature in clinical_features:
            if feature not in required_columns:
                required_columns.append(feature)
    df = load_split_dataframe(df_path, required_columns=required_columns)
    logger.info("[DATA] USING SPLIT CSV: %s | shape=%s", df_path, df.shape)
    for split_name in ["train", "validation", "test"]:
        split_df = df[df["split"] == split_name]
        if len(split_df) == 0:
            raise ValueError(f"Split {split_name} está vazio (treino inválido).")
        _validate_final_group_labels(split_df, split_name)
        counts = split_df["Final_Group"].value_counts(dropna=False).to_dict()
        logger.info("[DATA] %s: n=%s | by_class=%s", split_name, len(split_df), counts)

    device = select_device()
    logger.info("Dispositivo selecionado: %s | Torch threads: %s | Backbone: %s", device, torch.get_num_threads(), backbone)

    defaults = {
        "weight_decay": float(os.getenv("RESNET_WEIGHT_DECAY", 1e-4 if mode == "classification" else 0.0)),
        "dropout": float(os.getenv("RESNET_DROPOUT", 0.25)),
        "label_smoothing": float(os.getenv("RESNET_LABEL_SMOOTH", 0.05 if mode == "classification" else 0.0)),
        "mixup_alpha": float(os.getenv("RESNET_MIXUP", 0.0)),
        "freeze_backbone": _parse_bool(os.getenv("RESNET_FREEZE", "1")),
        "freeze_warmup_epochs": int(os.getenv("RESNET_WARMUP_EPOCHS", "0")),
        "seed": int(os.getenv("RESNET_SEED", "42")),
        "medicalnet_depth": 18,
        "batch_size": 16,
        "grad_clip": 1.0,
        "pretrained": True,
    }
    defaults.update(hparams)
    defaults["freeze_backbone"] = _parse_bool(defaults.get("freeze_backbone", True))
    defaults["pretrained"] = _parse_bool(defaults.get("pretrained", True), default=True)
    if "lr" not in hparams:
        effective_default_lr = 1e-3 if defaults["freeze_backbone"] else (5e-5 if mode == "classification" else 1e-3)
        defaults["lr"] = float(os.getenv("RESNET_LR", str(effective_default_lr)))
    else:
        defaults["lr"] = float(defaults["lr"])

    seed = int(defaults.get("seed", 42))
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        split_csv_sha256 = hashlib.sha256(df_path.read_bytes()).hexdigest()
    except OSError:
        split_csv_sha256 = None

    deep_scenario_base = (os.getenv("DEEP_SCENARIO", "deep_current_split") or "deep_current_split").strip()
    deep_scenario_label = f"{deep_scenario_base}_{backbone}_{mode}_seed{seed}"

    lr = float(defaults["lr"])
    weight_decay = float(defaults["weight_decay"])
    dropout_rate = float(defaults["dropout"])
    label_smoothing = float(defaults["label_smoothing"])
    mixup_alpha = float(defaults["mixup_alpha"])
    freeze_backbone = _parse_bool(defaults["freeze_backbone"])
    freeze_warmup_epochs = int(defaults.get("freeze_warmup_epochs", 0) or 0)
    pretrained = _parse_bool(defaults.get("pretrained", True), default=True)

    age_scaler = None
    if mode == "regression":
        age_scaler = StandardScaler()
        df_train = df[df["split"] == "train"].copy()
        df_val = df[df["split"] == "validation"].copy()
        df_test = df[df["split"] == "test"].copy()
        df_train["age_normalized"] = age_scaler.fit_transform(df_train[["age"]])
        df_val["age_normalized"] = age_scaler.transform(df_val[["age"]])
        df_test["age_normalized"] = age_scaler.transform(df_test[["age"]])
        df.loc[df["split"] == "train", "age_normalized"] = df_train["age_normalized"]
        df.loc[df["split"] == "validation", "age_normalized"] = df_val["age_normalized"]
        df.loc[df["split"] == "test", "age_normalized"] = df_test["age_normalized"]

    train_tf, val_tf = build_transforms()
    if clinical_features:
        logger.info("[Multimodal] Integrando dados clínicos: %s", clinical_features)

    train_df = populate_orientation_paths(df[df["split"] == "train"], dataset_root)
    val_df = populate_orientation_paths(df[df["split"] == "validation"], dataset_root)
    test_df = populate_orientation_paths(df[df["split"] == "test"], dataset_root)
    for split_name, split_df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        if len(split_df) == 0:
            raise ValueError(
                f"Split {split_name} became empty after resolving orientation paths; check missing or invalid MRI_ID values."
            )
        _validate_final_group_labels(split_df, split_name)

    train_ds = MultiOrientMRIDataset(train_df, train_tf, dataset_root, "original_path", label_column, clinical_features=clinical_features)
    val_ds = MultiOrientMRIDataset(val_df, val_tf, dataset_root, "original_path", label_column, clinical_features=clinical_features)
    test_ds = MultiOrientMRIDataset(test_df, val_tf, dataset_root, "original_path", label_column, clinical_features=clinical_features)
    train_ds._split_name = "train"
    val_ds._split_name = "val"
    test_ds._split_name = "test"

    if len(val_ds) == 0:
        raise ValueError("Split de validação vazio.")

    epochs = int(hparams.get("max_epochs", 40 if mode == "classification" else 20))
    batch_size = int(defaults.get("batch_size", 16))
    early_stop_patience = int(os.getenv("RESNET_PATIENCE", "7")) if mode == "classification" else None
    use_mixup = False if clinical_features else (mode == "classification" and mixup_alpha > 0)
    use_focal = mode == "classification" and not use_mixup
    focal_gamma = float(os.getenv("RESNET_FOCAL_GAMMA", 2.0))

    train_sampler = None
    shuffle_train = True
    if mode == "classification" and "Final_Group" in train_df.columns:
        train_labels = train_df["Final_Group"].map(CLASS_LABEL_MAP).astype(int).values
        class_counts = np.maximum(np.bincount(train_labels), 1)
        sample_weights = (1.0 / class_counts)[train_labels]
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle_train = False

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train, sampler=train_sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = MultiOrientTabularFusionNet(
        backbone=backbone,
        mode=mode,
        num_tabular_features=len(clinical_features) if clinical_features else 0,
        medicalnet_depth=int(defaults.get("medicalnet_depth", 18)),
        pretrained=pretrained,
        share_encoder=True,
        dropout=dropout_rate,
    ).to(device)

    trainable_params_initial = None
    trainable_params_after_unfreeze = None
    did_unfreeze_backbone = False
    unfreeze_epoch_1based = None
    use_freeze_warmup = freeze_backbone and freeze_warmup_epochs > 0
    if use_freeze_warmup:
        for encoder in [model.enc_axl, model.enc_cor, model.enc_sag]:
            for parameter in encoder.parameters():
                parameter.requires_grad = False
    trainable_params_initial = _count_trainable_parameters(model, "initial setup")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=10000.0,
        anneal_strategy="cos",
    )

    loss_weights = None
    if mode == "classification" and train_sampler is None and "Final_Group" in train_df.columns:
        counts = train_df["Final_Group"].map(CLASS_LABEL_MAP).value_counts()
        n_nondemented = max(counts.get(CLASS_LABEL_MAP["Nondemented"], 0), 1)
        n_demented = max(counts.get(CLASS_LABEL_MAP["Demented"], 0), 1)
        total = n_nondemented + n_demented
        loss_weights = torch.tensor(
            [total / (2.0 * n_nondemented), total / (2.0 * n_demented)],
            dtype=torch.float32,
            device=device,
        )

    if mode == "regression":
        criterion = nn.MSELoss()
    else:
        try:
            criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=loss_weights)
        except TypeError:
            criterion = nn.CrossEntropyLoss(weight=loss_weights)
    ema = ExponentialMovingAverage(model, decay=0.999) if mode == "classification" else None

    history_train_loss, history_val_loss = [], []
    history_train_acc, history_val_acc, history_val_acc_raw = [], [], []
    val_metric_value = None
    best_state, best_epoch = None, 0
    best_bal_acc_raw = -float("inf") if mode == "classification" else None
    best_bal_acc_adj = -float("inf") if mode == "classification" else None
    best_val_acc_raw = -float("inf") if mode == "classification" else None
    best_val_metric = float("inf") if mode == "regression" else None
    no_improve = 0

    amp_available = hasattr(torch, "amp")
    scaler = None
    if amp_available and device.type != "cpu":
        try:
            scaler = torch.amp.GradScaler(device_type=device.type)
        except TypeError:
            if device.type == "cuda" and hasattr(torch, "cuda") and hasattr(torch.cuda, "amp"):
                scaler = torch.cuda.amp.GradScaler()
    use_amp = scaler is not None

    artifact_paths: dict[str, Path] = {}
    artifact_errors: dict[str, str] = {}
    history_train_mae_denorm, history_val_mae_denorm = [], []

    for epoch in range(epochs):
        if use_freeze_warmup and not did_unfreeze_backbone and epoch >= freeze_warmup_epochs:
            for encoder in [model.enc_axl, model.enc_cor, model.enc_sag]:
                for parameter in encoder.parameters():
                    parameter.requires_grad = True
            did_unfreeze_backbone = True
            unfreeze_epoch_1based = epoch + 1
            trainable_params_after_unfreeze = _count_trainable_parameters(model, "backbone unfreeze")

        model.train()
        epoch_train_losses, epoch_train_true, epoch_train_preds = [], [], []
        epoch_train_true_denorm, epoch_train_preds_denorm = [], []
        for batch_x, lbls in train_loader:
            axl = batch_x["axl"].to(device)
            cor = batch_x["cor"].to(device)
            sag = batch_x["sag"].to(device)
            clin = batch_x.get("clin")
            if clin is not None:
                clin = clin.to(device)
            lbls = lbls.to(device)

            if debug_batch is not None:
                try:
                    debug_batch(axl, cor, sag, clin, lbls)
                except Exception as exc:
                    logger.exception(
                        "Error in debug_batch hook for batch shapes axl=%s cor=%s sag=%s clin=%s labels=%s: %s",
                        tuple(axl.shape),
                        tuple(cor.shape),
                        tuple(sag.shape),
                        tuple(clin.shape) if clin is not None else None,
                        tuple(lbls.shape),
                        exc,
                    )

            optimizer.zero_grad(set_to_none=True)
            mixup_indices = None
            mixup_lambda = None
            if use_mixup:
                mixup_lambda = float(np.random.beta(mixup_alpha, mixup_alpha))
                mixup_indices = torch.randperm(axl.size(0), device=device)
                axl = mixup_lambda * axl + (1.0 - mixup_lambda) * axl[mixup_indices]
                cor = mixup_lambda * cor + (1.0 - mixup_lambda) * cor[mixup_indices]
                sag = mixup_lambda * sag + (1.0 - mixup_lambda) * sag[mixup_indices]

            amp_context = torch.autocast(device_type=device.type, enabled=use_amp) if hasattr(torch, "autocast") else nullcontext()
            with amp_context:
                outputs = model(axl, cor, sag, clin)
                if mode == "classification" and use_mixup:
                    loss = mixup_lambda * criterion(outputs, lbls.long()) + (1.0 - mixup_lambda) * criterion(outputs, lbls[mixup_indices].long())
                elif mode == "classification" and use_focal:
                    loss = focal_loss(outputs, lbls.long(), gamma=focal_gamma, weight=loss_weights)
                elif mode == "regression":
                    loss = criterion(outputs.squeeze(-1), lbls.float())
                else:
                    loss = criterion(outputs, lbls.long())

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(defaults.get("grad_clip", 1.0)))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(defaults.get("grad_clip", 1.0)))
                optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(model)

            epoch_train_losses.append(float(loss.detach().cpu().item()))
            if mode == "classification" and not use_mixup:
                epoch_train_true.extend(lbls.detach().cpu().numpy().tolist())
                epoch_train_preds.extend(outputs.argmax(dim=1).detach().cpu().numpy().tolist())
            elif mode == "regression":
                preds = outputs.squeeze(-1).detach().cpu().numpy().reshape(-1, 1)
                true = lbls.detach().cpu().numpy().reshape(-1, 1)
                preds_denorm = age_scaler.inverse_transform(preds).reshape(-1)
                true_denorm = age_scaler.inverse_transform(true).reshape(-1)
                epoch_train_preds_denorm.extend(preds_denorm.tolist())
                epoch_train_true_denorm.extend(true_denorm.tolist())

        history_train_loss.append(float(np.mean(epoch_train_losses)) if epoch_train_losses else 0.0)
        if mode == "classification":
            if not use_mixup:
                train_acc = accuracy_score(epoch_train_true, epoch_train_preds) if epoch_train_true else 0.0
                history_train_acc.append(float(train_acc))
        else:
            train_mae_epoch = mean_absolute_error(epoch_train_true_denorm, epoch_train_preds_denorm) if epoch_train_true_denorm else 0.0
            history_train_mae_denorm.append(float(train_mae_epoch))

        if ema is not None:
            ema.apply_shadow(model)

        model.eval()
        val_losses, val_true, val_preds = [], [], []
        val_true_denorm, val_preds_denorm = [], []
        with torch.no_grad():
            for batch_x, lbls in val_loader:
                axl = batch_x["axl"].to(device)
                cor = batch_x["cor"].to(device)
                sag = batch_x["sag"].to(device)
                clin = batch_x.get("clin")
                if clin is not None:
                    clin = clin.to(device)
                lbls = lbls.to(device)

                outputs = model(axl, cor, sag, clin)
                if mode == "classification" and use_focal:
                    loss = focal_loss(outputs, lbls.long(), gamma=focal_gamma, weight=loss_weights)
                elif mode == "regression":
                    loss = criterion(outputs.squeeze(-1), lbls.float())
                else:
                    loss = criterion(outputs, lbls.long())

                val_losses.append(float(loss.detach().cpu().item()))
                if mode == "classification":
                    val_true.extend(lbls.detach().cpu().numpy().tolist())
                    val_preds.extend(outputs.argmax(dim=1).detach().cpu().numpy().tolist())
                else:
                    preds = outputs.squeeze(-1).detach().cpu().numpy().reshape(-1, 1)
                    true = lbls.detach().cpu().numpy().reshape(-1, 1)
                    preds_denorm = age_scaler.inverse_transform(preds).reshape(-1)
                    true_denorm = age_scaler.inverse_transform(true).reshape(-1)
                    val_preds_denorm.extend(preds_denorm.tolist())
                    val_true_denorm.extend(true_denorm.tolist())

        val_loss_epoch = float(np.mean(val_losses)) if val_losses else 0.0
        history_val_loss.append(val_loss_epoch)
        if mode == "classification":
            val_acc_raw = accuracy_score(val_true, val_preds) if val_true else 0.0
            val_bal_acc = balanced_accuracy_score(val_true, val_preds) if val_true else 0.0
            history_val_acc_raw.append(float(val_acc_raw))
            history_val_acc.append(float(val_bal_acc))
            val_metric_value = float(val_bal_acc)
            improved = val_metric_value > float(best_bal_acc_adj)
            if improved:
                best_bal_acc_adj = float(val_bal_acc)
                best_bal_acc_raw = float(val_bal_acc)
                best_val_acc_raw = float(val_acc_raw)
        else:
            val_mae_epoch = mean_absolute_error(val_true_denorm, val_preds_denorm) if val_true_denorm else 0.0
            history_val_mae_denorm.append(float(val_mae_epoch))
            val_metric_value = float(val_mae_epoch)
            improved = val_metric_value < float(best_val_metric)
            if improved:
                best_val_metric = float(val_mae_epoch)

        if improved:
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_epoch = epoch + 1
            torch.save(best_state, best_checkpoint_path)
            torch.save(best_state, legacy_checkpoint_path)
            no_improve = 0
        else:
            no_improve += 1

        if ema is not None:
            ema.restore(model)

        if early_stop_patience is not None and no_improve >= early_stop_patience:
            logger.info("[EarlyStop] epoch=%s patience=%s", epoch + 1, early_stop_patience)
            break

    if best_state is None:
        best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        best_epoch = max(best_epoch, 1)
        torch.save(best_state, best_checkpoint_path)
        torch.save(best_state, legacy_checkpoint_path)

    model.load_state_dict(best_state, strict=False)

    def _evaluate(loader):
        """
        Evaluate the current model over a DataLoader and return the mean loss plus true and predicted targets.
        
        Parameters:
            loader: An iterable yielding tuples (batch_x, labels) where `batch_x` is a dict containing keys
                "axl", "cor", "sag" (and optionally "clin") mapped to tensors, and `labels` is a tensor of targets.
        
        Returns:
            tuple: (mean_loss, y_true, y_pred)
                - mean_loss (float): Mean batch loss across the loader; returns 0.0 if the loader is empty.
                - y_true (np.ndarray): 1-D array of true targets. For regression, values are inverse-transformed
                  via the module's `age_scaler`; for classification, these are class indices.
                - y_pred (np.ndarray): 1-D array of predicted targets. For regression, predictions are
                  inverse-transformed via `age_scaler`; for classification, these are predicted class indices.
        """
        model.eval()
        losses, y_true, y_pred = [], [], []
        with torch.no_grad():
            for batch_x, lbls in loader:
                axl = batch_x["axl"].to(device)
                cor = batch_x["cor"].to(device)
                sag = batch_x["sag"].to(device)
                clin = batch_x.get("clin")
                if clin is not None:
                    clin = clin.to(device)
                lbls = lbls.to(device)
                outputs = model(axl, cor, sag, clin)

                if mode == "classification":
                    if use_focal:
                        loss = focal_loss(outputs, lbls.long(), gamma=focal_gamma, weight=loss_weights)
                    else:
                        loss = criterion(outputs, lbls.long())
                    preds = outputs.argmax(dim=1).detach().cpu().numpy().reshape(-1)
                    true = lbls.detach().cpu().numpy().reshape(-1)
                else:
                    loss = criterion(outputs.squeeze(-1), lbls.float())
                    preds = age_scaler.inverse_transform(outputs.squeeze(-1).detach().cpu().numpy().reshape(-1, 1)).reshape(-1)
                    true = age_scaler.inverse_transform(lbls.detach().cpu().numpy().reshape(-1, 1)).reshape(-1)

                losses.append(float(loss.detach().cpu().item()))
                y_true.extend(true.tolist())
                y_pred.extend(preds.tolist())
        return float(np.mean(losses)) if losses else 0.0, np.asarray(y_true), np.asarray(y_pred)

    train_loss_final, train_true_final, train_pred_final = _evaluate(train_loader)
    val_loss_final, val_true_final, val_pred_final = _evaluate(val_loader)
    test_loss_final, test_true_final, test_pred_final = _evaluate(test_loader)

    metrics = {
        "best_epoch": int(best_epoch),
        "training_time_seconds": float(time.time() - start_time),
        "trainable_params_initial": trainable_params_initial,
        "trainable_params_after_unfreeze": trainable_params_after_unfreeze,
        "did_unfreeze_backbone": bool(did_unfreeze_backbone),
        "unfreeze_epoch_1based": unfreeze_epoch_1based,
    }

    if mode == "classification":
        train_acc_final = accuracy_score(train_true_final, train_pred_final) if train_true_final.size else 0.0
        val_acc_final = accuracy_score(val_true_final, val_pred_final) if val_true_final.size else 0.0
        test_acc_final = accuracy_score(test_true_final, test_pred_final) if test_true_final.size else 0.0
        train_bal_final = balanced_accuracy_score(train_true_final, train_pred_final) if train_true_final.size else 0.0
        val_bal_final = balanced_accuracy_score(val_true_final, val_pred_final) if val_true_final.size else 0.0
        test_bal_final = balanced_accuracy_score(test_true_final, test_pred_final) if test_true_final.size else 0.0
        metrics.update(
            {
                "train_loss": float(train_loss_final),
                "val_loss": float(val_loss_final),
                "test_loss": float(test_loss_final),
                "train_accuracy": float(train_acc_final),
                "val_accuracy": float(val_acc_final),
                "test_accuracy": float(test_acc_final),
                "train_balanced_accuracy": float(train_bal_final),
                "val_balanced_accuracy": float(val_bal_final),
                "test_balanced_accuracy": float(test_bal_final),
                "best_val_accuracy_raw": float(best_val_acc_raw),
                "best_val_balanced_accuracy": float(best_bal_acc_raw),
                "train_confusion_matrix": confusion_matrix(train_true_final, train_pred_final).tolist() if train_true_final.size else None,
                "val_confusion_matrix": confusion_matrix(val_true_final, val_pred_final).tolist() if val_true_final.size else None,
                "test_confusion_matrix": confusion_matrix(test_true_final, test_pred_final).tolist() if test_true_final.size else None,
            }
        )
        summary_message = (
            f"Treino concluído ({backbone}/{mode}).\n"
            f"Val balanced acc: {val_bal_final:.2%}\n"
            f"Teste balanced acc: {test_bal_final:.2%}"
        )
    else:
        train_mae_orig = mean_absolute_error(train_true_final, train_pred_final) if train_true_final.size else 0.0
        val_mae_orig = mean_absolute_error(val_true_final, val_pred_final) if val_true_final.size else 0.0
        test_mae_orig = mean_absolute_error(test_true_final, test_pred_final) if test_true_final.size else 0.0
        train_rmse = float(np.sqrt(mean_squared_error(train_true_final, train_pred_final))) if train_true_final.size else 0.0
        val_rmse = float(np.sqrt(mean_squared_error(val_true_final, val_pred_final))) if val_true_final.size else 0.0
        test_rmse = float(np.sqrt(mean_squared_error(test_true_final, test_pred_final))) if test_true_final.size else 0.0
        train_r2 = r2_score(train_true_final, train_pred_final) if train_true_final.size else 0.0
        val_r2 = r2_score(val_true_final, val_pred_final) if val_true_final.size else 0.0
        test_r2 = r2_score(test_true_final, test_pred_final) if test_true_final.size else 0.0
        metrics.update(
            {
                "train_loss": float(train_loss_final),
                "val_loss": float(val_loss_final),
                "test_loss": float(test_loss_final),
                "train_mae": float(train_mae_orig),
                "val_mae": float(val_mae_orig),
                "test_mae": float(test_mae_orig),
                "train_rmse": float(train_rmse),
                "val_rmse": float(val_rmse),
                "test_rmse": float(test_rmse),
                "train_r2": float(train_r2),
                "val_r2": float(val_r2),
                "test_r2": float(test_r2),
            }
        )
        summary_message = (
            f"Treino concluído ({backbone}/{mode}).\n"
            f"Val MAE: {val_mae_orig:.2f}\n"
            f"Teste MAE: {test_mae_orig:.2f}"
        )

    learning_curves = {
        "train_loss": history_train_loss,
        "val_loss": history_val_loss,
    }
    if mode == "classification":
        learning_curves["train_accuracy"] = history_train_acc
        learning_curves["val_balanced_accuracy"] = history_val_acc
        learning_curves["val_accuracy_raw"] = history_val_acc_raw
    else:
        learning_curves["train_mae"] = history_train_mae_denorm
        learning_curves["val_mae"] = history_val_mae_denorm

    if Figure is not None:
        fig = Figure(figsize=(8, 4))
        ax1, ax2 = fig.subplots(1, 2)
        ax1.plot(history_train_loss, label="train")
        ax1.plot(history_val_loss, label="val")
        ax1.set_title("Loss")
        ax1.legend()
        if mode == "classification":
            ax2.plot(history_train_acc, label="train_acc")
            ax2.plot(history_val_acc, label="val_bal_acc")
            ax2.set_title("Accuracy")
        else:
            ax2.plot(history_train_mae_denorm, label="train_mae")
            ax2.plot(history_val_mae_denorm, label="val_mae")
            ax2.set_title("MAE")
        ax2.legend()
        curves_path = output_dir / f"{backbone}_{mode}_learning_curves.png"
        fig.savefig(curves_path, dpi=150, bbox_inches="tight")
        artifact_paths["learning_curves"] = curves_path
        _maybe_show_plot(cfg, f"Curvas {backbone}/{mode}", fig)

    if mode == "classification" and callable(cfg.plot_confusion_matrix_fn):
        try:
            cm_fig = cfg.plot_confusion_matrix_fn(metrics["test_confusion_matrix"], ["Nondemented", "Demented"])
            if cm_fig is not None:
                cm_path = output_dir / f"{backbone}_{mode}_confusion_matrix.png"
                cm_fig.savefig(cm_path, dpi=150, bbox_inches="tight")
                artifact_paths["confusion_matrix"] = cm_path
                _maybe_show_plot(cfg, f"Matriz {backbone}/{mode}", cm_fig)
        except Exception as exc:
            callback_name = getattr(cfg.plot_confusion_matrix_fn, "__name__", repr(cfg.plot_confusion_matrix_fn))
            logger.exception("Error in plot_confusion_matrix callback %s for mode=%s backbone=%s: %s", callback_name, mode, backbone, exc)

    artifact_paths["best_checkpoint"] = best_checkpoint_path
    artifact_paths["legacy_checkpoint"] = legacy_checkpoint_path

    try:
        train_emb_path = _export_embeddings(
            model=model,
            dataset_obj=train_ds,
            device=device,
            batch_size=batch_size,
            mode=mode,
            backbone=backbone,
            output_dir=output_dir,
        )
        if train_emb_path is not None:
            artifact_paths["train_embeddings"] = train_emb_path
    except Exception as exc:
        logger.exception("Error exporting train embeddings")
        artifact_errors["train_embeddings"] = str(exc)

    experiment_payload = {
        "model": backbone,
        "mode": mode,
        "scenario": deep_scenario_label,
        "hparams": {
            "lr": lr,
            "weight_decay": weight_decay,
            "dropout": dropout_rate,
            "label_smoothing": label_smoothing,
            "mixup_alpha": mixup_alpha,
            "freeze_backbone": freeze_backbone,
            "freeze_warmup_epochs": freeze_warmup_epochs,
            "pretrained": pretrained,
            "batch_size": batch_size,
            "epochs": epochs,
            "seed": seed,
            "clinical_features": clinical_features,
        },
        "metrics": metrics,
        "split_csv_path": str(df_path),
        "split_csv_sha256": split_csv_sha256,
        "artifacts": {key: str(path) for key, path in artifact_paths.items()},
    }
    if artifact_errors:
        experiment_payload["artifact_errors"] = artifact_errors
    if callable(cfg.save_experiment_fn):
        cfg.save_experiment_fn(experiment_payload)

    return DeepTrainingResult(
        backbone=backbone,
        mode=mode,
        best_checkpoint_path=best_checkpoint_path,
        legacy_checkpoint_path=legacy_checkpoint_path,
        learning_curves=learning_curves,
        metrics=metrics,
        experiment_payload=experiment_payload,
        summary_message=summary_message,
        artifact_paths=artifact_paths,
    )
