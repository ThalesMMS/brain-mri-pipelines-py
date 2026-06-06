import os
import torch
import torch.nn as nn
from typing import Union, List, Optional, Dict, Any

try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Mapeamento de modelos para repositórios no HuggingFace
# Usando os repositórios oficiais ou espelhos confiáveis do MedicalNet
MEDICALNET_REPO_ID = "TencentMedicalNet"
MEDICALNET_MODELS = {
    10: "MedicalNet-Resnet10",
    18: "MedicalNet-Resnet18",
    34: "MedicalNet-Resnet34",
    50: "MedicalNet-Resnet50",
    101: "MedicalNet-Resnet101",
}
MEDICALNET_FILENAMES = {
    10: "resnet_10.pth",
    18: "resnet_18.pth",
    34: "resnet_34.pth",
    50: "resnet_50.pth",
    101: "resnet_101.pth",
}

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Any] = None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Any] = None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet2D(nn.Module):
    def __init__(
        self,
        block: Any,
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Any] = None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        self.groups = groups
        self.base_width = width_per_group
        
        # Input layer adaptada para MedicalNet weights (originalmente 3D ResNet começa com conv7x7)
        # ResNet2D padrão usa conv7x7, stride2, padding3
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Any,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extrai embedding antes da camada fully-connected."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

def convert_3d_to_2d_weights(state_dict_3d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Converte pesos de um modelo ResNet 3D (MedicalNet) para 2D.
    Estratégia: Média dos pesos ao longo da dimensão de profundidade (dim 2) para kernels de convolução
    para preservar a escala de ativação.
    """
    state_dict_2d = {}
    
    # Mapeamento de chaves pode ser necessário se os nomes diferirem
    # MedicalNet geralmente usa keys como: 'module.layer1.0.conv1.weight' (DataParallel)
    # Precisamos remover o prefixo 'module.' se existir
    
    for key, value in state_dict_3d.items():
        new_key = key
        if new_key.startswith('module.'):
            new_key = new_key[7:]
            
        if 'conv' in new_key and value.dim() == 5:
            # Conv3d weights: (out, in, d, h, w) -> (out, in, h, w)
            # Média ao longo da dimensão de profundidade (dim 2) para manter escala
            weight_2d = value.mean(dim=2)
            state_dict_2d[new_key] = weight_2d
            
        elif 'downsample.0.weight' in new_key and value.dim() == 5:
             # Downsample conv weights
            weight_2d = value.mean(dim=2)
            state_dict_2d[new_key] = weight_2d
            
        elif 'bn' in new_key or 'downsample.1' in new_key or 'bias' in new_key:
            # Batch norm params e bias não precisam de alteração de dimensão, apenas cópia
            state_dict_2d[new_key] = value
            
        elif 'fc' in new_key:
            # Camada FC final - geralmente descartamos no transfer learning
            # Mas convertemos para manter consistência caso seja carregada
            state_dict_2d[new_key] = value
            
    return state_dict_2d

def download_medicalnet_weights(depth: int, cache_dir: Optional[str] = None) -> str:
    """Modela o download de pesos do HuggingFace Hub."""
    if not HF_AVAILABLE:
        raise RuntimeError("A biblioteca 'huggingface_hub' é obrigatória para baixar pesos MedicalNet.")
        
    repo_id = f"{MEDICALNET_REPO_ID}/{MEDICALNET_MODELS.get(depth)}"
    filename = MEDICALNET_FILENAMES.get(depth)
    
    if not repo_id or not filename:
        raise ValueError(f"Pesos para profundidade {depth} não mapeados.")
        
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/medicalnet")
    
    try:
        print(f"Baixando pesos MedicalNet-{depth}...")
        path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
        return path
    except Exception as e:
        raise RuntimeError(f"Falha ao baixar pesos MedicalNet-{depth}: {e}")

def _build_resnet_medical(depth: int, **kwargs) -> ResNet2D:
    depths = {
        10: [1, 1, 1, 1],
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
    }
    
    if depth not in depths:
        raise ValueError(f"Profundidade {depth} não suportada.")
    
    block = BasicBlock if depth <= 34 else Bottleneck
    model = ResNet2D(block, depths[depth], **kwargs)
    
    return model

def resnet10_2d(pretrained: bool = False, **kwargs) -> ResNet2D:
    model = _build_resnet_medical(10, **kwargs)
    if pretrained: load_medical_weights(model, 10)
    return model

def resnet18_2d(pretrained: bool = False, **kwargs) -> ResNet2D:
    model = _build_resnet_medical(18, **kwargs)
    if pretrained: load_medical_weights(model, 18)
    return model

def resnet34_2d(pretrained: bool = False, **kwargs) -> ResNet2D:
    model = _build_resnet_medical(34, **kwargs)
    if pretrained: load_medical_weights(model, 34)
    return model

def resnet50_2d(pretrained: bool = False, **kwargs) -> ResNet2D:
    model = _build_resnet_medical(50, **kwargs)
    if pretrained: load_medical_weights(model, 50)
    return model

def load_medical_weights(model: ResNet2D, depth: int):
    """Carrega pesos MedicalNet convertidos para 2D; falha se indisponíveis."""
    path = download_medicalnet_weights(depth)
    if not path or not os.path.exists(path):
        raise RuntimeError(f"Pesos MedicalNet não encontrados para ResNet{depth}. Baixe manualmente ou desative pretrained.")
        
    try:
        # MedicalNet salva o state_dict diretamente no arquivo .pth
        try:
            state_dict_3d = torch.load(path, map_location='cpu', weights_only=True)
        except TypeError:
            state_dict_3d = torch.load(path, map_location='cpu')
        if 'state_dict' in state_dict_3d:
            state_dict_3d = state_dict_3d['state_dict']
            
        state_dict_2d = convert_3d_to_2d_weights(state_dict_3d)
        
        # Filtra chaves que não existem no modelo atual (ex: fc layer dimensions diferentes)
        model_dict = model.state_dict()
        
        # Remove fc weights se dimensões não baterem (transfer learning normal)
        # O modelo 2D pode ter num_classes diferentes do modelo 3D pré-treinado
        # 3D weights fc: geralmente 2 classes ou N classes do dataset original
        
        filtered_dict = {}
        for k, v in state_dict_2d.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    filtered_dict[k] = v
                elif 'conv1.weight' in k and v.shape[1] == 1 and model_dict[k].shape[1] == 3:
                    # Adapta entrada de 1 canal (MedicalNet) para 3 canais (pipeline atual)
                    print(f"[MedicalNet] Adaptando {k} de 1 canal -> 3 canais (repetindo pesos).")
                    filtered_dict[k] = v.repeat(1, 3, 1, 1) / 3.0
                else:
                    print(f"[INFO] Pulando {k}: shapes {model_dict[k].shape} vs {v.shape}")
            else:
                pass 
        
        if not filtered_dict:
            raise RuntimeError(f"Sem pesos compatíveis encontrados no checkpoint MedicalNet-{depth}.")

        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        print(f"[INFO] Pesos MedicalNet-ResNet{depth} carregados com sucesso.")
        
    except Exception as e:
        raise RuntimeError(f"Falha ao carregar state_dict MedicalNet-{depth}: {e}")

class MultimodalMedicalNet(nn.Module):
    """
    Modelo multimodal que combina uma backbone (ex: ResNet, EfficientNet) para a imagem
    com um vetor de dados clínicos (ex: idade, gênero, etc.).
    """
    def __init__(self, backbone: nn.Module, num_clinical_features: int, num_classes: int = 2, dropout_rate: float = 0.2):
        super().__init__()
        self.backbone = backbone
        self.num_clinical_features = num_clinical_features
        
        # Descobre dimensão de saída da backbone
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            # Tenta chamar forward_features se existir (nossa ResNet e EfficientNet customizada podem ter)
            # Se a backbone for uma instância pronta da torchvision, precisaremos remover a FC
            if hasattr(self.backbone, 'forward_features'):
                features = self.backbone.forward_features(dummy)
                self.backbone_out_features = features.shape[1]
            elif hasattr(self.backbone, 'fc'):
                # Hack: assume ResNet-like e substitui fc por identidade temporariamente para descobrir dimensão
                # Ou melhor: inspeciona self.backbone.fc.in_features
                try:
                    self.backbone_out_features = self.backbone.fc.in_features
                except:
                     self.backbone_out_features = 512 # Fallback razoável
                self.backbone.fc = nn.Identity() # Remove head original
            elif hasattr(self.backbone, 'classifier'):
                # Inspeciona classifier[1].in_features para EfficientNet
                 try:
                     if isinstance(self.backbone.classifier, nn.Sequential):
                         self.backbone_out_features = self.backbone.classifier[1].in_features
                     else:
                         self.backbone_out_features = self.backbone.classifier.in_features
                 except:
                      self.backbone_out_features = 1280 # Fallback
                 self.backbone.classifier = nn.Identity()
            else:
                 # Último recurso: roda dummy
                 out = self.backbone(dummy)
                 self.backbone_out_features = out.shape[1]

        # Camada de fusão
        combined_features = self.backbone_out_features + num_clinical_features
        
        # Head de classificação final
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(combined_features),
            nn.Linear(combined_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, img: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        # Extrai features visuais
        if hasattr(self.backbone, 'forward_features'):
             x_img = self.backbone.forward_features(img)
        else:
             x_img = self.backbone(img)
             if isinstance(x_img, tuple): x_img = x_img[0] # Lidar com eventuais outputs estranhos
        
        # Achata se necessário
        if x_img.dim() > 2:
            x_img = torch.flatten(x_img, 1)

        # Concatena com dados clínicos
        # Garante que batch size bate
        if x_img.size(0) != clinical.size(0):
            raise RuntimeError(f"Mismatch batch size: Img {x_img.size(0)} vs Clin {clinical.size(0)}")

        x_combined = torch.cat((x_img, clinical), dim=1)
        
        # Classificação final
        out = self.classifier(x_combined)
        return out
