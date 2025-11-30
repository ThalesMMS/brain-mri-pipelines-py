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
        path = self.root / str(row.get(self.path_col, ''))  # Monta caminho completo da imagem

        if not path.exists(): raise FileNotFoundError(f"Img não encontrada: {path}")  # Lança erro se arquivo não existir

        img = ImageUtils.load_image(path)  # Carrega imagem usando utilitário
        if self.transform: img = self.transform(img)  # Aplica transformações se definidas

        y = row.get(self.label_col)  # Obtém rótulo da linha
        label = self.class_map[y] if isinstance(y, str) and y in self.class_map else float(y)  # Converte rótulo para numérico
        if not isinstance(label, int): label = torch.tensor(label, dtype=torch.float32)  # Transforma rótulo contínuo em tensor float

        return img, label  # Retorna tupla imagem e rótulo
