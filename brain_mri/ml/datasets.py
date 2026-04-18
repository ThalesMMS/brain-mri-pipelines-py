import hashlib
import math
import os
from pathlib import Path

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    Dataset = object
    TORCH_AVAILABLE = False
try:
    from PIL import Image
except ImportError:
    Image = None

from ..utils.image_utils import ImageUtils


def _parse_bool(value, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
    return default


def _format_mri_id_for_error(mri_id) -> str:
    if mri_id is None or mri_id == "<missing>" or (isinstance(mri_id, float) and math.isnan(mri_id)):
        return "<missing>"
    if _parse_bool(os.getenv("BRAIN_MRI_DEBUG_IDENTIFIERS", "0")):
        return repr(mri_id)
    digest = hashlib.sha256(str(mri_id).encode("utf-8")).hexdigest()[:12]
    return f"<redacted:{digest}>"


class MRIDataset(Dataset):
    def __init__(self, df, transform=None, root_dir=None, path_col='original_path', label_col='Final_Group', features=None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch é necessário para usar MRIDataset. Instale com 'pip install torch'.")
        if Image is None:
            raise ImportError("Pillow é necessário para carregar imagens. Instale com 'pip install pillow'.")
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.root = Path('.') if root_dir is None else Path(root_dir)
        self.path_col = path_col
        self.label_col = label_col
        self.features = features
        self.class_map = {'Nondemented': 0, 'Demented': 1}

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
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
            ordered = []
            for key in ['axl', 'cor', 'sag']:
                for p in orient_paths:
                    if f"_{key}" in p:
                        ordered.append(p)
                        break
            if not ordered and orient_paths:
                ordered = orient_paths[:1]
            while len(ordered) < 3:
                ordered.append(ordered[-1])
            ordered = ordered[:3]

            imgs = []
            for p in ordered:
                path = self.root / str(p)
                if not path.exists():
                    raise FileNotFoundError(f"Img não encontrada: {path}")
                imgs.append(ImageUtils.load_image_grayscale(path).resize((224, 224)).convert("L"))

            img_rgb = Image.merge("RGB", imgs)
            if self.transform:
                img_tensor = self.transform(img_rgb)
            else:
                img_tensor = to_tensor(img_rgb)
                img_tensor = normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            path = self.root / str(row.get(self.path_col, ''))
            if not path.exists(): raise FileNotFoundError(f"Img não encontrada: {path}")
            img = ImageUtils.load_image(path)
            if self.transform: img = self.transform(img)
            img_tensor = img

        y = row.get(self.label_col)
        label = self.class_map[y] if isinstance(y, str) and y in self.class_map else float(y)
        if not isinstance(label, int): label = torch.tensor(label, dtype=torch.float32)

        if self.features:
            vals = row[self.features].fillna(0.0).values.astype(float)
            clin_tensor = torch.tensor(vals, dtype=torch.float32)
            return (img_tensor, clin_tensor), label

        return img_tensor, label


class MultiOrientMRIDataset(Dataset):
    """
    Retorna dict:
      x = {"axl": Tensor, "cor": Tensor, "sag": Tensor, "clin": Tensor(opcional)}
      y = label
    """
    def __init__(
        self,
        df,
        transform=None,
        root_dir=None,
        path_col="original_path",
        label_col="Final_Group",
        clinical_features=None,
        class_map=None,
        fallback_to_last=True,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch é necessário. Instale com 'pip install torch'.")
        
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.root = Path(".") if root_dir is None else Path(root_dir)
        self.path_col = path_col
        self.label_col = label_col
        self.clinical_features = clinical_features or None
        self.fallback_to_last = fallback_to_last

        self.class_map = class_map or {"Nondemented": 0, "Demented": 1}
        if self.label_col == "Final_Group":
            self._validate_final_group_labels()

    def __len__(self):
        return len(self.df)

    def _sample_context(self, idx, row):
        parts = [f"dataset_index={idx}"]
        split = row.get("split")
        if split is not None:
            parts.append(f"split={split!r}")
        sample_id = row.get("MRI_ID", row.get("Subject_ID", None))
        parts.append(f"MRI_ID={_format_mri_id_for_error(sample_id)}")
        return ", ".join(parts)

    def _validate_final_group_labels(self):
        if self.label_col not in self.df.columns:
            raise ValueError("MultiOrientMRIDataset requires a Final_Group column for classification labels.")
        invalid_mask = self.df[self.label_col].isna() | ~self.df[self.label_col].isin(self.class_map)
        if not invalid_mask.any():
            return
        invalid_rows = self.df.loc[invalid_mask]
        invalid_values = ", ".join(sorted({repr(value) for value in invalid_rows[self.label_col].tolist()}))
        details = []
        for idx, row in invalid_rows.head(10).iterrows():
            details.append(f"{self._sample_context(idx, row)}, Final_Group={row.get(self.label_col)!r}")
        if len(invalid_rows) > 10:
            details.append(f"... {len(invalid_rows) - 10} more")
        expected = ", ".join(sorted(self.class_map))
        raise ValueError(
            f"Invalid Final_Group labels in MultiOrientMRIDataset: values [{invalid_values}] at {'; '.join(details)}. "
            f"Expected one of: {expected}."
        )

    def _parse_orient_paths(self, row):
        # Tenta pegar 'orientation_paths' do row (series/dict)
        orient_paths = row.get("orientation_paths")
        
        if isinstance(orient_paths, str):
            try:
                import ast
                orient_paths = ast.literal_eval(orient_paths)
            except Exception:
                orient_paths = [orient_paths]
        
        if orient_paths and not isinstance(orient_paths, list):
            orient_paths = [orient_paths]
            
        return orient_paths or []

    def _pick_path(self, orient_paths, key):
        # key in {"axl","cor","sag"}
        for p in orient_paths:
            if f"_{key}" in str(p):
                return str(p)

        # fallback: usa original_path, ou o último disponível, ou nada
        if self.fallback_to_last and len(orient_paths) > 0:
            return str(orient_paths[-1])

        return str(orient_paths[0]) if orient_paths else ""

    def _load_as_rgb_tensor(self, rel_path: str):
        if not rel_path:
            # Tolerância a orientações ausentes: retorna tensor nulo padronizado.
            # Isso permite que exames com 1–2 planos ainda participem do treino.
            return torch.zeros((3, 224, 224), dtype=torch.float32)

        path = self.root / rel_path
        if not path.exists():
            # Tolerância a dados incompletos: em vez de abortar, retorna tensor nulo.
            return torch.zeros((3, 224, 224), dtype=torch.float32)

        # Carrega grayscale
        # Removemos resize forçado para deixar a cargo do 'transform'
        img = ImageUtils.load_image_grayscale(path).convert("L")
        img = img.convert("RGB")  # backbone pretreinada espera 3 canais

        if self.transform is not None:
            return self.transform(img)

        # fallback mínimo se não houver transform (aí sim fazemos resize padrão e to_tensor)
        from torchvision.transforms.functional import to_tensor, normalize, resize
        x = resize(img, [224, 224])
        x = to_tensor(x)
        x = normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return x

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        orient_paths = self._parse_orient_paths(row)
        if not orient_paths:
            # sem lista: usa path_col como axial e replica (pelo menos não quebra)
            base = str(row.get(self.path_col, ""))
            orient_paths = [base]

        p_axl = self._pick_path(orient_paths, "axl")
        p_cor = self._pick_path(orient_paths, "cor")
        p_sag = self._pick_path(orient_paths, "sag")

        x = {
            "axl": self._load_as_rgb_tensor(p_axl),
            "cor": self._load_as_rgb_tensor(p_cor),
            "sag": self._load_as_rgb_tensor(p_sag),
        }

        # label
        y_raw = row.get(self.label_col)

        # classificação (strings da class_map) ou regressão (float/int)
        if isinstance(y_raw, str) and y_raw in self.class_map:
            y = torch.tensor(self.class_map[y_raw], dtype=torch.long)
        elif self.label_col == "Final_Group":
            expected = ", ".join(sorted(self.class_map))
            raise ValueError(
                f"Invalid Final_Group label in MultiOrientMRIDataset ({self._sample_context(idx, row)}): "
                f"{y_raw!r}. Expected one of: {expected}."
            )
        else:
            # Tenta converter para float (caso seja regressão ou label numérico)
            try:
                y_val = float(y_raw)
                if math.isnan(y_val):
                    raise ValueError
                y = torch.tensor(y_val, dtype=torch.float32)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Invalid label for column {self.label_col!r} in MultiOrientMRIDataset "
                    f"({self._sample_context(idx, row)}): {y_raw!r}."
                ) from exc

        # tabular
        if self.clinical_features:
            # Garante float32
            vals = row[self.clinical_features].fillna(0.0).values.astype("float32")
            x["clin"] = torch.tensor(vals, dtype=torch.float32)

        return x, y
