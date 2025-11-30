from pathlib import Path  # Manipulação de caminhos de arquivos de forma orientada a objetos

try:
    import torch  # Framework de deep learning PyTorch
    from torch.utils.data import Dataset  # Utilitários de datasets e carregadores no PyTorch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    Dataset = object  # Fallback para permitir import; métodos validarão disponibilidade
    TORCH_AVAILABLE = False
try:
    from PIL import Image  # Manipulação de imagens via Pillow
except ImportError:
    Image = None

from ..utils.image_utils import ImageUtils  # Utilitário de carregamento/normalização de imagens


class MRIDataset(Dataset):  # Dataset PyTorch específico para imagens de ressonância
    def __init__(self, df, transform=None, root_dir=None, path_col='original_path', label_col='Final_Group'):  # Inicializa dataset com DF e parâmetros
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch é necessário para usar MRIDataset. Instale com 'pip install torch'.")
        if Image is None:
            raise ImportError("Pillow é necessário para carregar imagens. Instale com 'pip install pillow'.")
        self.df = df.reset_index(drop=True)  # Reseta índices para acesso consistente
        self.transform = transform  # Guarda transformações de pré-processamento
        self.root = Path('.') if root_dir is None else Path(root_dir)  # Define diretório raiz das imagens
        self.path_col = path_col  # Coluna que contém caminhos das imagens
        self.label_col = label_col  # Coluna que contém rótulos/targets
        self.class_map = {'Nondemented': 0, 'Demented': 1}  # Mapeamento textual para rótulos inteiros

    def __len__(self): return len(self.df)  # Retorna quantidade de amostras no dataset

    def __getitem__(self, idx):  # Recupera item específico do dataset
        row = self.df.iloc[idx]  # Seleciona linha do DataFrame pelo índice
        # Se existir uma lista de caminhos de orientação, empilha em 3 canais (axl/cor/sag ou duplicando faltantes)
        orient_paths = row.get('orientation_paths') if isinstance(row, dict) else row.get('orientation_paths')
        if isinstance(orient_paths, str):
            try:
                import ast
                orient_paths = ast.literal_eval(orient_paths)
            except Exception:
                orient_paths = [orient_paths]
        if orient_paths and not isinstance(orient_paths, list):
            orient_paths = [orient_paths]

        if orient_paths:
            from torchvision.transforms.functional import to_tensor, normalize
            chans = []
            # garante axial em primeiro, depois cor, depois sag
            ordered = []
            for key in ['axl', 'cor', 'sag']:
                for p in orient_paths:
                    if f"_{key}" in p:
                        ordered.append(p)
                        break
            # preenche com axial se faltar
            if not ordered and orient_paths:
                ordered = orient_paths[:1]
            while len(ordered) < 3:
                ordered.append(ordered[-1])
            ordered = ordered[:3]
            for p in ordered:
                path = self.root / str(p)
                if not path.exists():
                    raise FileNotFoundError(f"Img não encontrada: {path}")
                img = ImageUtils.load_image_grayscale(path).resize((224, 224))
                t = to_tensor(img)  # 1xHxW
                chans.append(t)
            img_tensor = torch.cat(chans, dim=0)  # 3xHxW
            # normaliza com médias/std padrão (3 canais)
            img_tensor = normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            path = self.root / str(row.get(self.path_col, ''))  # Monta caminho completo da imagem
            if not path.exists(): raise FileNotFoundError(f"Img não encontrada: {path}")
            img = ImageUtils.load_image(path)  # Carrega imagem usando utilitário
            if self.transform: img = self.transform(img)  # Aplica transformações se definidas
            img_tensor = img

        y = row.get(self.label_col)  # Obtém rótulo da linha
        label = self.class_map[y] if isinstance(y, str) and y in self.class_map else float(y)  # Converte rótulo para numérico
        if not isinstance(label, int): label = torch.tensor(label, dtype=torch.float32)  # Transforma rótulo contínuo em tensor float

        return img_tensor, label  # Retorna tupla imagem e rótulo
