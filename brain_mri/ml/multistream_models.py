from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

from .medicalnet_models import resnet10_2d, resnet18_2d, resnet34_2d, resnet50_2d


def build_feature_encoder(backbone: str, pretrained: bool = True, medicalnet_depth: int = 18):
    """
    Retorna (encoder, feat_dim), onde encoder.forward(x) -> embedding (N, feat_dim).
    """
    backbone = backbone.lower()



    if backbone == "medicalnet":
        builders = {10: resnet10_2d, 18: resnet18_2d, 34: resnet34_2d, 50: resnet50_2d}
        if medicalnet_depth not in builders:
            raise ValueError("medicalnet_depth deve ser 10/18/34/50.")
        enc = builders[medicalnet_depth](pretrained=pretrained)
        feat_dim = enc.fc.in_features
        enc.fc = nn.Identity()
        return enc, feat_dim

    if backbone == "densenet":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        enc = models.densenet121(weights=weights)
        feat_dim = enc.classifier.in_features
        enc.classifier = nn.Identity()
        return enc, feat_dim

    if backbone == "efficientnet":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        enc = models.efficientnet_b0(weights=weights)
        # classifier = Dropout + Linear
        feat_dim = enc.classifier[1].in_features
        enc.classifier = nn.Identity()
        return enc, feat_dim

    raise ValueError(f"Backbone desconhecido: {backbone}")


class MultiOrientTabularFusionNet(nn.Module):
    def __init__(
        self,
        backbone: str,
        mode: str,  # "classification" ou "regression"
        num_tabular_features: int = 0,
        medicalnet_depth: int = 18,
        pretrained: bool = True,
        share_encoder: bool = True,
        proj_dim: int = 256,
        tab_hidden: int = 128,
        tab_dim: int = 64,
        fusion_hidden: int = 256,
        dropout: float = 0.25,
    ):
        super().__init__()
        assert mode in ("classification", "regression")

        # encoder(s)
        base_enc, feat_dim = build_feature_encoder(
            backbone=backbone,
            pretrained=pretrained,
            medicalnet_depth=medicalnet_depth,
        )

        if share_encoder:
            # Compartilha o mesmo objeto (pesos amarrados)
            self.enc_axl = base_enc
            self.enc_cor = base_enc
            self.enc_sag = base_enc
        else:
            # Cria encoders independentes se não quiser compartilhar
            # Nota: base_enc já foi criado, usamos ele como axl, e criamos outros 2
            self.enc_axl = base_enc
            self.enc_cor, _ = build_feature_encoder(backbone, pretrained, medicalnet_depth)
            self.enc_sag, _ = build_feature_encoder(backbone, pretrained, medicalnet_depth)
            
        # Para evitar problemas com .to(device) em módulos compartilhados,
        # PyTorch lida bem se eles são atributos do mesmo Module.
        # Mas se share_encoder=True, self.enc_cor e self.enc_sag são apenas referências.
        # Não precisamos registrar como submodule separado se for o mesmo objeto, mas...
        # Ao atribuir a self.x, o nn.Module registra. Se for o mesmo obj, ele sabe.
        
        # projeções por orientação (mantém simetria e controla dimensão da feature visual)
        def proj_block():
            return nn.Sequential(
                nn.Linear(feat_dim, proj_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )

        self.proj_axl = proj_block()
        self.proj_cor = proj_block()
        self.proj_sag = proj_block()

        # tabular MLP
        self.has_tab = num_tabular_features > 0
        if self.has_tab:
            self.tab_mlp = nn.Sequential(
                nn.Linear(num_tabular_features, tab_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(tab_hidden, tab_dim),
                nn.ReLU(inplace=True),
            )
            fusion_in = 3 * proj_dim + tab_dim
        else:
            self.tab_mlp = None
            fusion_in = 3 * proj_dim

        out_dim = 2 if mode == "classification" else 1

        # fusion head (1 hidden layer na junção)
        self.head = nn.Sequential(
            nn.BatchNorm1d(fusion_in),
            nn.Linear(fusion_in, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, out_dim),
        )

        self.mode = mode
        
        self._init_weights()

    def _encode_one(self, enc: nn.Module, proj: nn.Module, x: torch.Tensor):
        # x: [B, 3, H, W]
        f = enc(x)
        # Se saída for [B, C, 1, 1] ou [B, C], garante flatten
        if f.dim() > 2:
            f = torch.flatten(f, 1)
        return proj(f)

    def forward(self, axl: torch.Tensor, cor: torch.Tensor, sag: torch.Tensor, clin: Optional[torch.Tensor] = None):
        f_axl = self._encode_one(self.enc_axl, self.proj_axl, axl)
        f_cor = self._encode_one(self.enc_cor, self.proj_cor, cor)
        f_sag = self._encode_one(self.enc_sag, self.proj_sag, sag)

        feats = [f_axl, f_cor, f_sag]

        if self.has_tab:
            if clin is None:
                # Se foi instanciado com tabular, espera receber o tensor
                # Se não tiver dados para algum batch, idealmente o dataset retorna zeros.
                raise RuntimeError("Modelo foi criado com tabular, mas 'clin' veio None.")
            feats.append(self.tab_mlp(clin))

        z = torch.cat(feats, dim=1)
        out = self.head(z)
        return out

    def _init_weights(self):
        # Inicialização explícita para proj_*, tab_mlp e head
        # Encoders geralmente já vêm com pesos do ImageNet/MedicalNet (se pretrained=True)
        # Se pretrained=False, eles são inicializados pelo construtor do torchvision.

        modules_to_init = [self.proj_axl, self.proj_cor, self.proj_sag, self.head]
        if self.has_tab:
            modules_to_init.append(self.tab_mlp)
        
        for module in modules_to_init:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def extract_features(self, axl, cor, sag, clin=None):
        """
        Extrai vetor de features (embeddings) antes da head de classificação/regressão.
        Retorna tensor [B, fusion_hidden] ou similar, dependendo do ponto de corte.
        Aqui, retornamos a concatenação das projecões + tabular (input do self.head).
        """
        f_axl = self._encode_one(self.enc_axl, self.proj_axl, axl)
        f_cor = self._encode_one(self.enc_cor, self.proj_cor, cor)
        f_sag = self._encode_one(self.enc_sag, self.proj_sag, sag)

        feats = [f_axl, f_cor, f_sag]

        if self.has_tab:
            if clin is None:
                raise RuntimeError("Modelo espera dados tabulares para extração de features.")
            feats.append(self.tab_mlp(clin))

        return torch.cat(feats, dim=1)
