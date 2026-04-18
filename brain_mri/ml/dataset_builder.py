import os
import logging
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

try:
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    train_test_split = None
    SKLEARN_AVAILABLE = False


VALID_EXTS = (".nii.gz", ".nii", ".png", ".jpg", ".jpeg")
ORIENTATIONS = ("axl", "cor", "sag")
DATASET_SPLIT_FILENAME = "exam_level_dataset_split.csv"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetBuilderConfig:
    dataset_dir: Path
    output_dir: Path
    csv_path: Path
    descriptors_csv: Path


def _normalize_base_dirs(base_dirs) -> list[Path]:
    """
    Normalize input into a list of pathlib.Path objects.
    
    Parameters:
        base_dirs: A single path (str or Path) or an iterable of path-like values.
    
    Returns:
        A list of `Path` objects corresponding to the input paths.
    
    Raises:
        TypeError: If `base_dirs` is neither a path-like value nor an iterable of path-like values.
    """
    if isinstance(base_dirs, (str, Path)):
        return [Path(base_dirs)]
    if isinstance(base_dirs, Iterable):
        return [Path(base_dir) for base_dir in base_dirs]
    raise TypeError("base_dirs must be a path or an iterable of paths.")


def _coerce_config(config) -> DatasetBuilderConfig:
    """
    Coerce an input configuration into a DatasetBuilderConfig with Path-typed fields.

    Accepts either a DatasetBuilderConfig or a Mapping containing the keys
    "dataset_dir", "output_dir", "csv_path", and "descriptors_csv". In both
    cases the returned DatasetBuilderConfig will have those fields converted
    to pathlib.Path instances.

    Parameters:
        config: A DatasetBuilderConfig or a mapping with the required dataset path keys.

    Returns:
        DatasetBuilderConfig: A config object whose path fields are pathlib.Path instances.

    Raises:
        TypeError: If `config` is neither a DatasetBuilderConfig nor a suitable mapping.
    """
    if isinstance(config, (DatasetBuilderConfig, Mapping)):
        return DatasetBuilderConfig(
            dataset_dir=Path(config["dataset_dir"] if isinstance(config, Mapping) else config.dataset_dir),
            output_dir=Path(config["output_dir"] if isinstance(config, Mapping) else config.output_dir),
            csv_path=Path(config["csv_path"] if isinstance(config, Mapping) else config.csv_path),
            descriptors_csv=Path(config["descriptors_csv"] if isinstance(config, Mapping) else config.descriptors_csv),
        )
    raise TypeError("config must be a DatasetBuilderConfig or mapping with dataset paths.")


def _display_path(path: Path, base_dir: Path) -> str:
    try:
        return path.relative_to(base_dir).as_posix()
    except ValueError:
        return path.as_posix()


def list_orientation_paths(mri_id: str, base_dirs):
    """
    Finds existing file paths for a given MRI exam across axial, coronal, and sagittal orientations.
    
    Parameters:
        mri_id (str): MRI exam identifier (without extension) used to build filenames like "<mri_id>_<orient><ext>".
        base_dirs (str | Path | Iterable): One or more base directories to search; each may contain axl/, cor/, sag/ subfolders.
    
    Returns:
        list[str]: Unique path strings to discovered files. Each path is relative to its base directory when possible, otherwise absolute. The search follows the module's orientation and extension order.
    """
    paths = []
    seen_files = set()
    for base_dir in _normalize_base_dirs(base_dirs):
        for orient in ORIENTATIONS:
            for ext in VALID_EXTS:
                candidate = base_dir / orient / f"{mri_id}_{orient}{ext}"
                if not candidate.exists():
                    continue
                file_key = candidate.resolve()
                path_str = _display_path(candidate, base_dir)
                if file_key not in seen_files:
                    seen_files.add(file_key)
                    paths.append(path_str)
                break
    return paths


def populate_orientation_paths(df_subset, base_dirs):
    """
    Populate each exam row with discovered orientation image paths and select an original path.
    
    For each row with a string-valued `MRI_ID`, discover available image paths across orientations and, when found, set `orientation_paths` to the list of discovered paths and `original_path` to the first axial path when available or the first discovered path otherwise. Rows with non-string or missing `MRI_ID` are omitted. If no paths are discovered for a row, the row is preserved only if it already contains a non-empty `original_path`; otherwise it is omitted.
    
    Parameters:
        df_subset (pandas.DataFrame or similar): DataFrame-like collection of exam rows; expected to support `.iterrows()` and to be constructible from a list of rows via `type(df_subset)(rows)`.
        base_dirs (str, pathlib.Path, or iterable of those): One or more base directories to search for orientation subfolders.
    
    Returns:
        Same type as `df_subset`: A new DataFrame-like object containing only rows that have a valid `MRI_ID` and either discovered orientation paths or an existing `original_path`, with populated `orientation_paths` and updated `original_path` where applicable.
    """
    if df_subset is None or df_subset.empty:
        return df_subset

    rows = []
    for _, row in df_subset.iterrows():
        mri_id = row.get("MRI_ID")
        if not isinstance(mri_id, str):
            continue
        orient_paths = list_orientation_paths(mri_id, base_dirs)
        if not orient_paths:
            if row.get("original_path"):
                rows.append(row.copy())
            continue

        enriched_row = row.copy()
        enriched_row["orientation_paths"] = orient_paths
        axial = [path for path in orient_paths if "_axl" in path]
        enriched_row["original_path"] = axial[0] if axial else orient_paths[0]
        rows.append(enriched_row)

    return type(df_subset)(rows)


def _parse_mri_subject_ids(filename: str):
    """
    Extract the MRI exam ID and subject ID from a filename that uses underscore-separated parts and one of the module's valid extensions.
    
    Parameters:
        filename (str): The filename (may include one of VALID_EXTS) to parse.
    
    Returns:
        tuple: `(mri_id, subj_id)` where `mri_id` is the first three underscore-separated parts joined by `_` and `subj_id` is the first two parts joined by `_`. Returns `(None, None)` if the filename (after removing a recognized extension) has fewer than three underscore-separated parts.
    """
    name = filename
    for ext in VALID_EXTS:
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    parts = name.split("_")
    if len(parts) < 3:
        return None, None
    subj_id = "_".join(parts[:2])
    mri_id = "_".join(parts[:3])
    return mri_id, subj_id


def _build_union_index(base_dir: Path):
    """
    Build an exam-level union index of MRI files found under axl/, cor/, and sag/ subdirectories.
    
    Scans the given base directory for files with allowed extensions inside orientation subfolders, groups file paths by MRI exam identifier, and records which orientations are present. For each MRI_ID the returned table contains all discovered orientation-relative (or absolute when necessary) paths, a chosen original_path preferring an axial entry when available, the Subject_ID, and boolean flags for has_axl, has_cor, and has_sag.
    
    Parameters:
        base_dir (Path): Root directory containing orientation subfolders (e.g., base_dir/axl, base_dir/cor, base_dir/sag).
    
    Returns:
        pd.DataFrame: A DataFrame with one row per MRI exam and columns:
            - MRI_ID: exam identifier
            - Subject_ID: subject identifier derived from filenames
            - orientation_paths: list of discovered path strings for the exam
            - original_path: selected representative path (prefers axial)
            - has_axl: `True` if an axial image was found
            - has_cor: `True` if a coronal image was found
            - has_sag: `True` if a sagittal image was found
    """
    by_mri = {}
    for orient in ORIENTATIONS:
        orient_dir = base_dir / orient
        if not orient_dir.exists():
            continue
        for ext in VALID_EXTS:
            for image_path in orient_dir.glob(f"*{ext}"):
                mri_id, subj_id = _parse_mri_subject_ids(image_path.name)
                if not mri_id or not subj_id:
                    continue
                record = by_mri.setdefault(
                    mri_id,
                    {
                        "MRI_ID": mri_id,
                        "Subject_ID": subj_id,
                        "paths": [],
                        "has_axl": False,
                        "has_cor": False,
                        "has_sag": False,
                    },
                )
                rel_path = _display_path(image_path, base_dir)
                if rel_path not in record["paths"]:
                    record["paths"].append(rel_path)
                record[f"has_{orient}"] = True

    rows = []
    for record in by_mri.values():
        paths = record["paths"]
        axial = [path for path in paths if "_axl" in path]
        original_path = axial[0] if axial else (paths[0] if paths else "")
        rows.append(
            {
                "MRI_ID": record["MRI_ID"],
                "Subject_ID": record["Subject_ID"],
                "orientation_paths": paths,
                "original_path": original_path,
                "has_axl": bool(record["has_axl"]),
                "has_cor": bool(record["has_cor"]),
                "has_sag": bool(record["has_sag"]),
            }
        )
    return pd.DataFrame(rows)


def _as_numeric(series):
    """
    Convert a pandas Series to numeric values, accepting comma or dot as the decimal separator.
    
    Parameters:
        series (pandas.Series): Series of values (often strings) to convert.
    
    Returns:
        pandas.Series: Numeric series of floats with non-convertible values set to `NaN`.
    """
    return pd.to_numeric(series.astype(str).str.replace(",", ".").str.strip(), errors="coerce")


def _resolve_final_group(row):
    """
    Determine the final diagnostic group for an exam, mapping a `Group` value of "Converted" to "Demented" or "Nondemented" based on the clinical dementia rating.
    
    Parameters:
        row (Mapping-like): A row or dict-like object containing at least the `Group` key; when `Group` is `"Converted"` the function will read `cdr` if present, otherwise `CDR`.
    
    Returns:
        str or None: `"Demented"` if `Group` is `"Converted"` and the CDR value is present and greater than 0; `"Nondemented"` if `Group` is `"Converted"` and the CDR value is missing or ≤ 0; otherwise returns the original `Group` value (which may be `None`).
    """
    grp = row.get("Group")
    if isinstance(grp, str) and grp == "Converted":
        cdr_val = row.get("cdr") if "cdr" in row else row.get("CDR")
        if pd.notna(cdr_val) and float(cdr_val) > 0:
            return "Demented"
        return "Nondemented"
    return grp


def _split_has_both(merged, subject_ids):
    """
    Check whether the specified subject IDs include at least one `Nondemented` and one `Demented` row in the `Final_Group` column of `merged`.
    
    Parameters:
        merged (pandas.DataFrame): DataFrame containing at least `Subject_ID` and `Final_Group` columns.
        subject_ids (Iterable): Collection of subject identifiers to test.
    
    Returns:
        bool: `True` if both `Nondemented` and `Demented` are present for the given subjects, `False` otherwise.
    """
    counts = merged[merged["Subject_ID"].isin(subject_ids)]["Final_Group"].value_counts()
    return counts.get("Nondemented", 0) > 0 and counts.get("Demented", 0) > 0


def create_exam_level_dataset(config):
    """
    Builds an exam-level MRI dataset by merging an image-derived union index, optional descriptor CSV, and a demographic CSV, then assigns subject-level train/validation/test splits and writes the resulting table to disk.
    
    Parameters:
        config (DatasetBuilderConfig | Mapping | str | Path): Configuration or a mapping containing the keys
            `dataset_dir`, `output_dir`, `csv_path`, and `descriptors_csv`, or a path/coercible object that
            will be converted to a DatasetBuilderConfig.
    
    Returns:
        tuple[pd.DataFrame, Path]: The final merged DataFrame (one row per exam) and the filesystem path to the
        written CSV file (output_dir/exam_level_dataset_split.csv).
    
    Raises:
        ImportError: If required dependencies (`pandas` or `scikit-learn`) are not available.
        ValueError: If no images are discovered, required demographic columns are missing, or there are fewer
        than three unique subjects to perform the split.
    """
    if not PANDAS_AVAILABLE:
        raise ImportError(
            "O módulo 'pandas' é necessário para criar o dataset.\nInstale com 'pip install pandas'."
        )
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "O módulo 'scikit-learn' é necessário para criar o split do dataset.\n"
            "Instale com 'pip install scikit-learn'."
        )

    cfg = _coerce_config(config)

    base_dir = cfg.dataset_dir.resolve()
    df_union = _build_union_index(base_dir)
    if df_union.empty:
        raise ValueError("Nenhuma imagem encontrada em axl/cor/sag para montar o dataset.")

    if cfg.descriptors_csv.exists():
        df_desc_raw = pd.read_csv(cfg.descriptors_csv)
        if df_desc_raw.empty:
            warnings.warn(
                "CSV de descritores vazio. O dataset será criado sem descritores.",
                stacklevel=2,
            )
            df_desc = df_union.copy()
            df_desc["viable"] = True
        else:
            if "MRI_ID" not in df_desc_raw.columns:
                raise ValueError(f"CSV de descritores {cfg.descriptors_csv} deve conter a coluna obrigatória MRI_ID.")
            if "viable" not in df_desc_raw.columns:
                df_desc_raw["viable"] = True
            df_desc = pd.merge(df_union, df_desc_raw, on="MRI_ID", how="left", suffixes=("", "_desc"))
            df_desc["viable"] = df_desc.get("viable", True).fillna(True)
    else:
        df_desc = df_union.copy()
        df_desc["viable"] = True

    df_demo = pd.read_csv(cfg.csv_path, sep=";", decimal=",")
    df_demo.columns = [column.strip() for column in df_demo.columns]
    if "MRI ID" in df_demo.columns:
        df_demo.rename(columns={"MRI ID": "MRI_ID"}, inplace=True)
    if "Subject ID" in df_demo.columns:
        df_demo.rename(columns={"Subject ID": "Subject_ID"}, inplace=True)

    numeric_map = {
        "Age": "age",
        "EDUC": "education",
        "MMSE": "mmse",
        "CDR": "cdr",
        "eTIV": "etiv",
        "nWBV": "nwbv",
        "ASF": "asf",
    }
    for source, destination in numeric_map.items():
        if source in df_demo.columns:
            df_demo[destination] = _as_numeric(df_demo[source])

    missing_demo_columns = [] if "MRI_ID" in df_demo.columns else ["MRI_ID"]
    if "Group" not in df_demo.columns:
        if "Final_Group" in df_demo.columns:
            df_demo["Group"] = df_demo["Final_Group"]
        else:
            missing_demo_columns.append("Group")
    if missing_demo_columns:
        missing = ", ".join(missing_demo_columns)
        raise ValueError(f"CSV demográfico sem colunas obrigatórias: {missing}")

    if "M/F" in df_demo.columns:
        df_demo["sex"] = df_demo["M/F"].map({"M": 0, "F": 1})

    merged = pd.merge(df_desc, df_demo, on="MRI_ID", how="left", suffixes=("", "_demo"))
    merged["viable"] = merged["viable"].fillna(True)
    merged = merged[merged["viable"]]

    if "Subject_ID_x" in merged.columns:
        merged["Subject_ID"] = merged["Subject_ID_x"]
    if "Subject_ID_y" in merged.columns:
        merged["Subject_ID"] = merged["Subject_ID"].fillna(merged["Subject_ID_y"])
        merged.drop(columns=["Subject_ID_y"], inplace=True)
    if "Subject_ID_x" in merged.columns:
        merged.drop(columns=["Subject_ID_x"], inplace=True)

    merged["Original_Group"] = merged.get("Group")
    merged["Final_Group"] = merged.apply(_resolve_final_group, axis=1)
    merged["Final_Group"] = merged["Final_Group"].fillna(merged["Original_Group"])

    if "original_path" in merged.columns:
        merged["original_path"] = merged["original_path"].fillna("")
    else:
        merged["original_path"] = ""
    merged = merged[merged["original_path"] != ""]

    descriptor_cols = [column for column in merged.columns if column.startswith("ventricle_")]
    if descriptor_cols:
        merged["has_descriptors"] = merged[descriptor_cols].notna().any(axis=1)
    else:
        merged["has_descriptors"] = False
    merged = merged[merged["Subject_ID"].notna()]
    merged = merged[merged["Final_Group"].notna()]

    subjects = merged["Subject_ID"].dropna().unique()
    if len(subjects) < 3:
        raise ValueError("Dados insuficientes para split (mínimo 3 sujeitos).")

    split_seed = int(os.getenv("DENSENET_SEED", "42"))

    subj_label = (
        merged[["Subject_ID", "Final_Group"]]
        .dropna(subset=["Subject_ID", "Final_Group"])
        .groupby("Subject_ID")["Final_Group"]
        .apply(lambda values: int((values == "Demented").mean() >= 0.5))
    )
    subj_ids = subj_label.index.to_numpy()
    subj_y = subj_label.values

    def _try_split(n_attempts, require_test_both):
        """Attempt stratified subject-level splits; return (train, val, test) or None."""
        for attempt in range(n_attempts):
            random_state = split_seed + attempt
            try:
                train_ids, test_ids = train_test_split(
                    subj_ids,
                    test_size=0.2,
                    random_state=random_state,
                    stratify=subj_y if len(np.unique(subj_y)) > 1 else None,
                )
                train_labels = np.array([subj_label[sid] for sid in train_ids])
                train_ids, val_ids = train_test_split(
                    train_ids,
                    test_size=0.2,
                    random_state=random_state,
                    stratify=train_labels if len(np.unique(train_labels)) > 1 else None,
                )
            except ValueError:
                continue
            has_train = _split_has_both(merged, train_ids)
            has_val = _split_has_both(merged, val_ids)
            has_test = _split_has_both(merged, test_ids)
            if has_train and has_val and (not require_test_both or has_test):
                return train_ids, val_ids, test_ids
        return None

    train_sub = val_sub = test_sub = None
    result = _try_split(500, require_test_both=True)
    if result is not None:
        train_sub, val_sub, test_sub = result

    if train_sub is None:
        result = _try_split(500, require_test_both=False)
        if result is not None:
            train_sub, val_sub, test_sub = result
            print(
                "[WARN] Não foi possível garantir ambas as classes no TESTE; "
                "mantendo split com ambas em TREINO/VAL."
            )

    if train_sub is None:
        print("[WARN] Split estratificado por sujeito falhou; usando split aleatório simples.")
        train_sub, test_sub = train_test_split(subjects, test_size=0.2, random_state=split_seed)
        train_sub, val_sub = train_test_split(train_sub, test_size=0.2, random_state=split_seed)

    merged = merged[merged["Final_Group"].notna()]
    val_sub_set = set(val_sub)
    test_sub_set = set(test_sub)

    def get_split(subject_id):
        """
        Return the dataset split name for the given subject.
        
        Parameters:
            subject_id: Identifier of the subject (e.g., value from `Subject_ID` column).
        
        Returns:
            str: "validation" if the subject is in the validation set, "test" if in the test set, "train" otherwise.
        """
        if subject_id in val_sub_set:
            return "validation"
        if subject_id in test_sub_set:
            return "test"
        return "train"

    merged["split"] = merged["Subject_ID"].apply(get_split)

    try:
        print("\n[Diagnóstico] Final_Group por split:")
        print(merged.groupby("split")["Final_Group"].value_counts())
    except Exception as exc:
        logger.exception("Error computing Final_Group counts by split: %s", exc)

    cols_to_drop = ["Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF", "Visit", "MR Delay", "M/F"]
    cols_to_drop = [column for column in cols_to_drop if column in merged.columns]
    if cols_to_drop:
        merged.drop(columns=cols_to_drop, inplace=True)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = cfg.output_dir / DATASET_SPLIT_FILENAME
    merged.to_csv(output_path, index=False)
    return merged, output_path
