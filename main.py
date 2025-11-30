import tkinter as tk  # Interface gráfica com Tkinter base
from pathlib import Path  # Manipulação de caminhos de arquivos de forma orientada a objetos

from brain_mri.utils.dependencies import ensure_dependencies  # Bootstrap de dependências antes de carregar o app

# Garante dependências antes de importar módulos pesados (matplotlib, nibabel, etc.)
BASE_DIR = Path(__file__).resolve().parent
missing = ensure_dependencies(BASE_DIR / "requirements.txt")
if missing:
    print(f"Ainda faltam pacotes para instalar (tente manualmente se necessário): {', '.join(missing)}")
    raise SystemExit(1)

from brain_mri.experiments.history import ExperimentHistoryMixin  # Histórico de experimentos e plot de matrizes
from brain_mri.ml.ml_training import MLTrainingMixin  # Funções de criação de dataset e treino de modelos
from brain_mri.ui.navigation import NavigationMixin  # Navegação e exibição de imagens
from brain_mri.ui.segmentation import SegmentationMixin  # Segmentação e gerenciamento de viabilidade
from brain_mri.ui.ui_components import UIMixin  # Construção da UI
from brain_mri.ui.visualization import VisualizationMixin  # Visualizações auxiliares


class MedicalImagingApp(  # Controla UI Tkinter e lógica de processamento
    UIMixin,
    NavigationMixin,
    SegmentationMixin,
    VisualizationMixin,
    MLTrainingMixin,
    ExperimentHistoryMixin
):
    def __init__(self, root):  # Inicializa aplicativo com janela raiz Tkinter
        self.root = root  # Guarda referência da janela principal
        self.root.title("Trabalho de PAI")  # Define título da janela
        self.root.geometry("1400x900")  # Ajusta tamanho inicial da janela

        self.font_size = 10  # Tamanho de fonte padrão para botões/labels
        self.zoom_level = 1.0  # Nível de zoom inicial nas imagens
        self.show_roi = False  # Controle para exibir ou ocultar ROI
        self.models = {}  # Armazena modelos de ML treinados

        base_dir = Path(__file__).resolve().parent  # Diretório base do projeto
        self.dataset_dir = base_dir / "axl"  # Pasta de imagens de entrada
        self.output_dir = base_dir / "output"  # Pasta de saídas/resultados
        self.not_viable_dir = base_dir / "not_viable"  # Pasta para casos inviáveis
        self.csv_path = base_dir / "oasis_longitudinal_demographic.csv"  # Caminho do CSV original
        self.descriptors_csv = self.output_dir / "ventricle_descriptors.csv"  # CSV de descritores calculados
        self.experiment_history_path = self.output_dir / "training_experiments.json"  # Histórico de experimentos de treino

        for p in [self.output_dir, self.not_viable_dir]:  # Garante que diretórios de saída existam
            p.mkdir(exist_ok=True)  # Cria pastas caso não existam

        self.image_list = []  # Lista de caminhos de imagens carregadas
        self.current_image_index = -1  # Índice atual de imagem selecionada
        self.current_image = None  # Imagem processada atualmente exibida
        self.original_image = None  # Imagem original sem transformações
        self.current_segmented = None  # Máscara segmentada ativa na exibição
        self.original_segmented = None  # Máscara segmentada original
        self.segmented_images = {}  # Cache de máscaras por caminho de imagem
        self.roi_mask = None  # Máscara de ROI corrente
        self.image_path = None  # Caminho da imagem atual

        self._setup_ui()  # Monta todos os elementos de interface

        files = sorted(self.dataset_dir.glob("*.nii*"))  # Busca imagens NIfTI no diretório
        if files:  # Se encontrou arquivos
            self.image_list = [str(f) for f in files]  # Armazena caminhos ordenados

        self.load_existing_segmentations()  # Carrega segmentações já salvas, se houver

        if self.image_list:  # Se há imagens disponíveis
            self.current_image_index = 0  # Começa pela primeira imagem
            self.load_image_at_index(0)  # Carrega e exibe primeira imagem


if __name__ == "__main__":  # Ponto de entrada principal
    root = tk.Tk()
    app = MedicalImagingApp(root)
    root.mainloop()
