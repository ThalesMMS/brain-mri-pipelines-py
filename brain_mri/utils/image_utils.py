from pathlib import Path  # Manipulação de caminhos de arquivos de forma orientada a objetos

import numpy as np  # Operações numéricas e arrays
try:
    import nibabel as nib  # Leitura de arquivos NIfTI para imagens médicas
    NIB_AVAILABLE = True
except ImportError:
    nib = None
    NIB_AVAILABLE = False
try:
    from PIL import Image  # Manipulação de imagens via Pillow
except ImportError:
    Image = None
try:
    from scipy import ndimage  # Operações de processamento de imagens multidimensionais
    from scipy.ndimage import gaussian_filter  # Filtro gaussiano para suavização
    SCIPY_AVAILABLE = True
except ImportError:
    ndimage = gaussian_filter = None
    SCIPY_AVAILABLE = False
try:
    from skimage import filters, measure, morphology  # Filtros, medição de regiões e morfologia da scikit-image
    SKIMAGE_AVAILABLE = True
except ImportError:
    filters = measure = morphology = None
    SKIMAGE_AVAILABLE = False


class ImageUtils:  # Define utilitários estáticos para lidar com imagens
    @staticmethod  # Indica que o método não depende do estado da instância
    def load_image(path: Path) -> Image.Image:  # Carrega imagem de formatos comuns ou NIfTI e devolve PIL Image
        if Image is None:
            raise ImportError("O módulo Pillow não está instalado. Instale com 'pip install pillow'.")
        if path.suffix.lower() in {'.png', '.jpg', '.jpeg'}:  # Verifica se o arquivo é uma imagem convencional
            return Image.open(path).convert('RGB')  # Abre o arquivo e força conversão para RGB
        if path.suffix.lower() in {'.gz', '.nii'}:  # Trata arquivos NIfTI comprimidos ou não comprimidos
            if not NIB_AVAILABLE:
                raise ImportError("O módulo nibabel não está instalado. Instale com 'pip install nibabel'.")
            nii = nib.load(str(path))  # Carrega o volume NIfTI em memória
            data = np.squeeze(nii.get_fdata())  # Extrai array de dados e remove dimensões unitárias
            if data.ndim == 3:  # Se for volume 3D, seleciona um único corte para 2D
                data = data[:, :, 0]  # Mantém a primeira fatia axial para visualização
            denom = data.max() - data.min()  # Calcula intervalo de intensidades para normalização
            data = (data - data.min()) / (denom + 1e-8)  # Normaliza valores para 0-1 evitando divisão por zero
            return Image.fromarray((data * 255).astype(np.uint8)).convert('RGB')  # Converte array normalizado em imagem RGB de 8 bits
        raise ValueError(f"Formato não suportado: {path}")  # Erro explícito para extensões desconhecidas

    @staticmethod
    def load_image_grayscale(path: Path) -> Image.Image:
        """Carrega imagem e retorna em escala de cinza (1 canal)."""
        if Image is None:
            raise ImportError("O módulo Pillow não está instalado. Instale com 'pip install pillow'.")
        if path.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
            return Image.open(path).convert('L')
        if path.suffix.lower() in {'.gz', '.nii'}:
            if not NIB_AVAILABLE:
                raise ImportError("O módulo nibabel não está instalado. Instale com 'pip install nibabel'.")
            nii = nib.load(str(path))
            data = np.squeeze(nii.get_fdata())
            if data.ndim == 3:
                data = data[:, :, 0]
            denom = data.max() - data.min()
            data = (data - data.min()) / (denom + 1e-8)
            return Image.fromarray((data * 255).astype(np.uint8)).convert('L')
        raise ValueError(f"Formato não suportado: {path}")

    @staticmethod  # Indica método estático para normalizar arrays
    def normalize_array(img):  # Normaliza array de intensidades para faixa 0-1
        img = img.astype(np.float32)  # Converte para float32 para cálculos estáveis
        mn, mx = img.min(), img.max()  # Obtém valores mínimo e máximo do array
        return (img - mn) / (mx - mn) if mx > mn else img  # Normaliza quando há variação, senão retorna original

    @staticmethod  # Método estático para segmentar a imagem sem depender da instância
    def grow_region(image):  # Executa detecção de ROI e crescimento da região segmentada
        """Lógica de segmentação baseada em ROI central e crescimento de região."""  # Docstring descreve a estratégia usada
        if not (SCIPY_AVAILABLE and SKIMAGE_AVAILABLE):
            raise ImportError("SciPy e scikit-image são necessários para segmentação. Instale com 'pip install scipy scikit-image'.")
        img_u8 = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)  # Garante imagem em 8 bits para processamento morfológico
        h, w = img_u8.shape  # Captura altura e largura da imagem
        cy, cx = h // 2, w // 2  # Determina coordenadas do centro da imagem

        roi_w, roi_h = 49, 56  # Define dimensões do retângulo central usado como ROI
        roi_mask = np.zeros((h, w), dtype=bool)  # Cria máscara booleana vazia para marcar ROI
        roi_mask[cy - roi_h//2 : cy + roi_h//2, cx - roi_w//2 : cx + roi_w//2] = True  # Marca região central como ROI

        thresh = filters.threshold_otsu(img_u8)  # Calcula limiar Otsu para separar fundo e objeto
        binary = morphology.binary_opening(img_u8 < thresh, morphology.disk(1))  # Remove ruído fino aplicando abertura morfológica

        selem = morphology.disk(2)  # Estruturante em forma de disco para gradiente morfológico
        grad = morphology.dilation(img_u8, selem).astype(np.int16) - morphology.erosion(img_u8, selem).astype(np.int16)  # Calcula gradiente morfológico para detectar bordas

        labeled = measure.label(binary)  # Rotula componentes conectados na máscara binária
        seed_mask = np.zeros_like(binary, dtype=bool)  # Máscara para armazenar componentes que intersectam a ROI

        for region in measure.regionprops(labeled, intensity_image=img_u8):  # Itera sobre regiões rotuladas com intensidades
            if np.any((labeled == region.label) & roi_mask):  # Verifica se a região toca a ROI central
                seed_mask |= (labeled == region.label)  # Inclui região candidata como semente

        grown = seed_mask.copy()  # Inicializa máscara de crescimento com as sementes selecionadas

        for _ in range(3):  # Executa três iterações de crescimento controlado
            dilated = morphology.binary_dilation(grown, morphology.disk(1))  # Dilata máscara atual para expandir fronteira
            boundary = dilated & ~grown  # Calcula contorno recém-dilatado
            valid_growth = (boundary) & (img_u8 < thresh * 1.1) & (grad < np.percentile(grad, 80))  # Seleciona pixels de fronteira com intensidade baixa e gradiente moderado
            grown |= valid_growth  # Adiciona crescimento válido à máscara

        refined = grown  # Inicia refinamento a partir da máscara expandida
        refined &= (img_u8 < thresh * 1.15)  # Filtra pixels com intensidade um pouco abaixo do limiar ajustado
        refined = ndimage.binary_fill_holes(refined)  # Preenche buracos internos na máscara

        refined = gaussian_filter(refined.astype(np.float32), sigma=1.5) > 0.5  # Suaviza bordas e binariza novamente
        refined = ndimage.binary_fill_holes(refined)  # Repreenche buracos após suavização

        otsu_disp = (~(img_u8 < thresh)).astype(np.uint8)  # Cria máscara visual do limiar Otsu invertido
        return refined.astype(np.uint8), otsu_disp, thresh, roi_mask  # Retorna máscara segmentada, visual Otsu, limiar e ROI

    @staticmethod  # Método estático para extrair descritores morfológicos
    def calculate_descriptors(mask):  # Calcula métricas geométricas da máscara segmentada
        if not SKIMAGE_AVAILABLE:
            raise ImportError("scikit-image é necessário para calcular descritores. Instale com 'pip install scikit-image'.")
        if not np.any(mask):  # Se não houver pixels verdadeiros na máscara
            return {k: 0 for k in ['area', 'perimeter', 'circularity', 'eccentricity',
                                  'solidity', 'major_axis_length', 'minor_axis_length']}  # Retorna descritores zerados para casos vazios

        # Usa uma única máscara agregada (label 1) para considerar todas as regiões desconexas em conjunto
        labeled = measure.label(mask > 0)
        regions = measure.regionprops(labeled)

        # soma de propriedades simples
        area = sum(r.area for r in regions)
        perimeter = sum(r.perimeter for r in regions)

        # circularidade precisa ser recalculada com base nos valores somados
        circ = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        # para propriedades escalares onde não faz senso somar,
        # tiramos média ponderada pela área (mais correto anatomicamente)
        def weighted_avg(attr):
            num = sum(getattr(r, attr) * r.area for r in regions)
            den = sum(r.area for r in regions)
            return num / den if den > 0 else 0

        ecc = weighted_avg("eccentricity")
        sol = weighted_avg("solidity")
        maj = weighted_avg("major_axis_length")
        minr = weighted_avg("minor_axis_length")
        
        return {
            "area": float(area),
            "perimeter": float(perimeter),
            "circularity": float(circ),
            "eccentricity": float(ecc),
            "solidity": float(sol),
            "major_axis_length": float(maj),
            "minor_axis_length": float(minr),
        }
