from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)
_INTEROP_THREADS_CONFIGURED = False

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms

    from .medicalnet_models import resnet10_2d, resnet18_2d, resnet34_2d, resnet50_2d

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    models = transforms = None
    resnet10_2d = resnet18_2d = resnet34_2d = resnet50_2d = None
    TORCH_AVAILABLE = False


def _require_torch() -> None:
    """
    Ensure PyTorch and torchvision are available and raise an informative ImportError if they are not.

    Raises:
        ImportError: If PyTorch/torchvision are not installed; the exception message includes an installation hint.
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch/torchvision são necessários para este utilitário.\n"
            "Instale com 'pip install torch torchvision'."
        )


class ExponentialMovingAverage:
    """Simple EMA wrapper to smooth weights during training."""

    def __init__(self, model, decay: float = 0.999):
        """
        Initialize an ExponentialMovingAverage that tracks floating-point parameters of `model`.

        Creates a shadow copy of every floating-point tensor from `model.state_dict()` and stores the EMA decay factor.

        Parameters:
            model: The torch.nn.Module whose parameters will be tracked.
            decay (float): Smoothing factor in [0,1) controlling the EMA update (higher means slower updates).

        Attributes:
            decay (float): The configured EMA decay.
            shadow (dict): Mapping from state_dict keys to cloned floating-point tensors used as EMA "shadow" values.
            backup (dict): Empty dict reserved for temporarily backing up model parameters when applying/restoring shadow.

        Raises:
            ImportError: If PyTorch/torchvision are not available (enforced by `_require_torch()`).
        """
        _require_torch()
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.is_floating_point()}
        self.backup = {}

    def update(self, model):
        with torch.no_grad():
            for name, param in model.state_dict().items():
                if name not in self.shadow or not param.is_floating_point():
                    continue
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        """
        Store the current model parameters and replace them with the EMA (shadow) parameters.

        Backs up tensors from the model that correspond to keys in the EMA shadow, then loads the shadow parameters into the model under no-grad mode using a non-strict state dict update.

        Parameters:
            model (torch.nn.Module): The model whose parameters will be backed up and replaced with EMA shadow values.
        """
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items() if k in self.shadow}
        with torch.no_grad():
            model.load_state_dict(self.shadow, strict=False)

    def restore(self, model):
        """
        Restore model parameters from the stored backup.

        If a backup of parameters exists, loads those tensors into `model` using
        `load_state_dict(..., strict=False)` under a no-grad context, then clears
        the internal backup so it won't be reused.

        Parameters:
            model (torch.nn.Module): The model whose parameters will be replaced from the backup.
        """
        if self.backup:
            with torch.no_grad():
                model.load_state_dict(self.backup, strict=False)
            self.backup = {}


def focal_loss(logits, targets, gamma: float = 2.0, alpha=None, weight=None):
    """
    Compute the focal loss for multi-class classification.

    Parameters:
        logits (torch.Tensor): Predicted unnormalized scores with shape (N, C).
        targets (torch.Tensor): Integer class labels with shape (N,) or (N,1).
        gamma (float): Focusing parameter that down-weights well-classified examples.
        alpha (torch.Tensor or None): Optional per-class weighting tensor of shape (C,); when provided, each target's loss is scaled by the corresponding class weight.
        weight (torch.Tensor or None): Alias for `alpha`; if `alpha` is None, `weight` will be used.

    Returns:
        torch.Tensor: Scalar tensor containing the mean focal loss over the batch.
    """
    _require_torch()
    if alpha is None:
        alpha = weight

    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    probs = torch.exp(log_probs)
    targets = targets.view(-1, 1)
    one_hot = torch.zeros_like(log_probs).scatter_(1, targets, 1.0)
    focal_weight = (1 - probs) ** gamma

    if alpha is not None:
        alpha_t = alpha[targets.squeeze()].unsqueeze(1)
        focal_weight = focal_weight * alpha_t

    loss = -(one_hot * focal_weight * log_probs).sum(dim=1)
    return loss.mean()


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Create torchvision transform pipelines for training and validation.

    The training pipeline applies strong augmentations suitable for low-data regimes (random resized crop to 224×224, horizontal flip, rotation, small affine transforms, color jitter, optional Gaussian blur, and random erasing), then converts images to tensors and normalizes using ImageNet mean/std. The validation pipeline resizes to 224×224, converts to tensor, and applies the same normalization.

    Returns:
        (train_tf, val_tf) (Tuple[transforms.Compose, transforms.Compose]): A tuple with the training transforms as the first element and the validation transforms as the second.
    """
    _require_torch()
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.08), ratio=(0.3, 3.3), value="random"),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def build_medicalnet(
    mode: str = "classification",
    depth: int = 18,
    dropout_rate: float = 0.3,
    pretrained: bool = True,
) -> nn.Module:
    """
    Constructs a MedicalNet ResNet and replaces its final fully connected head for the requested task.

    Parameters:
        mode (str): Task mode; use "regression" to produce a single-output head, any other value produces a 2-class classification head.
        depth (int): ResNet depth to use; supported values are 10, 18, 34, and 50.
        dropout_rate (float): Dropout probability applied before the final linear layer.
        pretrained (bool): If True, load pretrained weights for the backbone.

    Returns:
        nn.Module: The constructed model with its final `fc` replaced by a Dropout followed by a Linear layer sized for the chosen `mode`.

    Raises:
        ValueError: If `depth` is not one of the supported values (10, 18, 34, 50).
    """
    _require_torch()
    builders = {
        10: resnet10_2d,
        18: resnet18_2d,
        34: resnet34_2d,
        50: resnet50_2d,
    }

    if depth not in builders:
        raise ValueError(f"MedicalNet depth {depth} not supported (10, 18, 34, 50).")

    model = builders[depth](pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features, 1 if mode == "regression" else 2),
    )
    return model


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    """
    Apply MixUp augmentation to a batch of inputs and targets.

    When alpha > 0, samples a mixing coefficient lambda ~ Beta(alpha, alpha), permutes the batch,
    and returns inputs and paired targets mixed by lambda. If alpha <= 0, returns inputs and targets unchanged.

    Parameters:
        x: Batch of input tensors.
        y: Batch of target tensors or labels.
        alpha (float): Concentration parameter for the Beta distribution; larger values produce stronger mixing.
            If alpha <= 0, mixing is disabled.

    Returns:
        mixed_x, y_a, y_b, lam:
            mixed_x: Tensor of mixed inputs.
            y_a: Original targets (aligned with the first component of the mix).
            y_b: Permuted targets (aligned with the second component of the mix).
            lam: Mixing coefficient in [0, 1]; equals 1.0 when mixing is disabled.
    """
    _require_torch()
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def select_device():
    """
    Selects the best available torch device, preferring MPS, then CUDA, and falling back to CPU.

    When MPS is chosen, prints a warning if the installed PyTorch version is older than 2.1. When CPU is chosen, limits intra-op threads to at most 4 and sets inter-op threads to 1 when available.

    Returns:
        torch.device: The selected device (`mps`, `cuda`, or `cpu`).
    """
    _require_torch()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built()
    if has_mps:
        device = torch.device("mps")
        try:
            version = torch.__version__.split("+")[0]
            major, minor = map(int, version.split(".")[:2])
            if major < 2 or (major == 2 and minor < 1):
                print(f"[WARN] PyTorch {torch.__version__} detectado. Para melhor suporte MPS, use 2.1+.")
        except Exception as exc:
            logger.debug("Failed to parse torch.__version__=%s: %s", torch.__version__, exc)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        global _INTEROP_THREADS_CONFIGURED
        if hasattr(torch, "set_num_interop_threads") and not _INTEROP_THREADS_CONFIGURED:
            try:
                current_interop_threads = torch.get_num_interop_threads() if hasattr(torch, "get_num_interop_threads") else None
                if current_interop_threads != 1:
                    torch.set_num_interop_threads(1)
                _INTEROP_THREADS_CONFIGURED = True
            except RuntimeError as exc:
                logger.debug("Skipping torch.set_num_interop_threads(1): %s", exc)
                _INTEROP_THREADS_CONFIGURED = True
    return device


def ensure_required_columns(df, required_columns, context: str = "DataFrame"):
    """
    Validate that the given DataFrame contains all required columns.

    Parameters:
        df (pandas.DataFrame): DataFrame to validate.
        required_columns (Iterable[str]): Column names that must be present in `df`.
        context (str): Context label used in the error message when columns are missing.

    Returns:
        pandas.DataFrame: The same `df` if validation passes.

    Raises:
        ValueError: If any required columns are missing; the message lists the missing columns.
    """
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        missing_str = ", ".join(map(str, missing))
        raise ValueError(f"{context} is missing required columns: {missing_str}")
    return df


def load_split_dataframe(split_csv_path, required_columns=None):
    """
    Load a CSV split file into a pandas DataFrame and optionally validate required columns.

    Parameters:
        split_csv_path (str | Path): Path to the split CSV file to load.
        required_columns (Iterable[str] | None): If provided, list of column names that must exist in the CSV; validation uses the module's ensure_required_columns.

    Returns:
        pandas.DataFrame: The loaded DataFrame.

    Raises:
        ImportError: If pandas is not installed.
        FileNotFoundError: If the CSV file does not exist at the given path.
        ValueError: If `required_columns` is provided and one or more required columns are missing.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "O módulo 'pandas' é necessário para carregar o split do dataset."
        ) from exc

    split_path = Path(split_csv_path)
    if not split_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_path}")

    df = pd.read_csv(split_path)
    if required_columns:
        ensure_required_columns(df, required_columns, context=str(split_path))
    return df


def sanitize_artifact_label(label) -> str:
    """
    Sanitize a label for use in artifact filenames.

    Replaces characters outside ASCII letters, digits, hyphen, underscore,
    and dot with underscores, collapses repeated underscores, and trims
    leading/trailing dots and spaces.

    Parameters:
        label: The label to sanitize.

    Returns:
        sanitized_label (str): The sanitized label suitable for inclusion in filenames.
    """
    safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", str(label).strip())
    safe_label = re.sub(r"_+", "_", safe_label).strip(" .")
    if not safe_label or not re.search(r"[A-Za-z0-9]", safe_label):
        return "artifact"
    return safe_label


def build_artifact_path(output_dir, filename: str, label=None) -> Path:
    """
    Builds a filesystem path for an artifact, optionally inserting a sanitized label before the file extension.

    Parameters:
        output_dir: Path-like location where the artifact will be placed.
        filename (str): Base filename (may include an extension).
        label (optional): If provided and not empty, a sanitized version of this label is appended to the filename before its extension.

    Returns:
        Path: The resulting path combining `output_dir` and the (possibly labeled) filename.
    """
    output_path = Path(output_dir)
    if label in (None, ""):
        return output_path / filename

    safe_label = sanitize_artifact_label(label)
    stem, suffix = os.path.splitext(filename)
    return output_path / f"{stem}_{safe_label}{suffix}"
