import importlib  # Checagem de módulos instalados
import subprocess  # Execução de pip via subprocesso
import sys  # Caminho do executável Python
from pathlib import Path  # Localização do requirements.txt
from typing import List, Dict  # Tipagem leve

# Mapeia nome do pacote pip -> nome usado no import
PACKAGE_IMPORTS: Dict[str, str] = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "scikit-learn": "sklearn",
    "xgboost": "xgboost",
    "torch": "torch",
    "torchvision": "torchvision",
    "nibabel": "nibabel",
    "opencv-python": "cv2",
    "seaborn": "seaborn",
    "tqdm": "tqdm",
    "Pillow": "PIL",
    "scikit-image": "skimage",
    "scipy": "scipy",
}


def _is_installed(import_name: str) -> bool:
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False


def ensure_dependencies(requirements_path: Path) -> List[str]:
    """
    Garante que dependências básicas estejam instaladas.
    Retorna lista de pacotes que ainda faltarem após tentar instalar.
    """
    missing = [pkg for pkg, mod in PACKAGE_IMPORTS.items() if not _is_installed(mod)]
    if missing:
        # Se existir requirements.txt, prioriza instalar a lista completa para evitar versões conflitantes
        if requirements_path.exists():
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)]
        else:
            cmd = [sys.executable, "-m", "pip", "install", *missing]
        try:
            subprocess.run(cmd, check=False)
        except Exception:
            pass  # Segue para checar novamente

    remaining = [pkg for pkg, mod in PACKAGE_IMPORTS.items() if not _is_installed(mod)]
    return remaining
