import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from .medicalnet_models import resnet10_2d, resnet18_2d, resnet34_2d, resnet50_2d


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


def focal_loss(logits, targets, gamma: float = 2.0, alpha=None, weight=None):
    """Focal loss para classificação binária/multiclasse.
    
    Args:
        logits: Saída do modelo (N, C)
        targets: Rótulos verdadeiros (N,)
        gamma: Fator de foco - valores maiores penalizam mais exemplos fáceis
        alpha: Peso por classe (tensor de tamanho C) - balanceia classes desiguais
        weight: Alias para alpha (retrocompatibilidade)
    """
    if alpha is None:
        alpha = weight  # Retrocompatibilidade
    
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    probs = torch.exp(log_probs)
    targets = targets.view(-1, 1)
    one_hot = torch.zeros_like(log_probs).scatter_(1, targets, 1.0)
    
    # Peso focal: reduz contribuição de exemplos bem classificados
    focal_weight = (1 - probs) ** gamma
    
    # Aplica pesos de classe se fornecidos
    if alpha is not None:
        # alpha deve ser tensor de tamanho (num_classes,)
        alpha_t = alpha[targets.squeeze()].unsqueeze(1)  # (N, 1)
        focal_weight = focal_weight * alpha_t
    
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





def build_medicalnet(
    mode: str = 'classification',
    depth: int = 18,
    dropout_rate: float = 0.3,
    pretrained: bool = True
) -> nn.Module:
    """Cria MedicalNet ResNet com cabeça ajustada."""
    builders = {
        10: resnet10_2d,
        18: resnet18_2d,
        34: resnet34_2d,
        50: resnet50_2d
    }
    
    if depth not in builders:
        raise ValueError(f"MedicalNet depth {depth} not supported (10, 18, 34, 50).")
        
    model = builders[depth](pretrained=pretrained)
    
    # Substitui a fc final
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features, 1 if mode == 'regression' else 2)
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
        try:
            version = torch.__version__.split('+')[0]
            major, minor = map(int, version.split('.')[:2])
            if major < 2 or (major == 2 and minor < 1):
                print(f"[WARN] PyTorch {torch.__version__} detectado. Para melhor suporte MPS, use 2.1+.")
        except Exception:
            pass
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
    return device
