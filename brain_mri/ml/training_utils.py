import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms


class ExponentialMovingAverage:
    """Simple EMA wrapper to smooth weights during training."""

    def __init__(self, model, decay: float = 0.999):
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
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items() if k in self.shadow}
        model.state_dict().update(self.shadow)

    def restore(self, model):
        if self.backup:
            model.state_dict().update(self.backup)
            self.backup = {}


def focal_loss(logits, targets, gamma: float = 2.0, weight=None):
    """Focal loss para classificação binária/multiclasse."""
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    probs = torch.exp(log_probs)
    targets = targets.view(-1, 1)
    one_hot = torch.zeros_like(log_probs).scatter_(1, targets, 1.0)
    focal_weight = (1 - probs) ** gamma
    if weight is not None:
        cls_w = weight[targets.squeeze()]
        focal_weight = focal_weight * cls_w.unsqueeze(1)
    loss = -(one_hot * focal_weight * log_probs).sum(dim=1)
    return loss.mean()


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Retorna transforms de treino/validação com forte augmentação para poucos dados."""
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.08), ratio=(0.3, 3.3), value='random')
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_tf, val_tf


def build_densenet(mode: str = 'classification', dropout_rate: float = 0.3) -> nn.Module:
    """Cria DenseNet-121 com cabeça ajustada e dropout extra."""
    weights = models.DenseNet121_Weights.IMAGENET1K_V1
    model = models.densenet121(weights=weights)
    n_feat = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(n_feat, 1 if mode == 'regression' else 2)
    )
    return model


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    """Aplica mixup ao batch."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def select_device():
    """Seleciona dispositivo com fallback para CPU e reduz threads em CPU."""
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built()
    if has_mps:
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
    return device
