import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


PANDAS_AVAILABLE = _module_available("pandas")
PIL_AVAILABLE = _module_available("PIL.Image")


def _install_tkinter_stub() -> None:
    if "tkinter" in sys.modules:
        return
    tk = types.ModuleType("tkinter")
    messagebox = types.ModuleType("tkinter.messagebox")
    messagebox.showinfo = lambda *args, **kwargs: None
    messagebox.showwarning = lambda *args, **kwargs: None
    messagebox.showerror = lambda *args, **kwargs: None
    messagebox.askyesno = lambda *args, **kwargs: True
    tk.messagebox = messagebox
    sys.modules["tkinter"] = tk
    sys.modules["tkinter.messagebox"] = messagebox


_install_tkinter_stub()

try:
    from torchvision import transforms
except ImportError:
    transforms = None

from brain_mri.ml import deep_training
from brain_mri.ml.deep_training import DeepTrainingConfig, _coerce_config, _resolve_dataset_dir

TRAINING_TEST_DEPS_MISSING = (
    not deep_training._require_dependencies(raise_on_missing=False)
    or transforms is None
    or not PANDAS_AVAILABLE
    or not PIL_AVAILABLE
)


def _write_png(path: Path, value: int) -> None:
    image_module = pytest.importorskip("PIL.Image")
    path.parent.mkdir(parents=True, exist_ok=True)
    image_module.new("L", (24, 24), color=int(value)).save(path)


def _write_split_csv(tmp_path: Path) -> tuple[Path, Path]:
    pd = pytest.importorskip("pandas")
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    split_csv_path = output_dir / "synthetic_split.csv"

    rows = []
    split_sequence = ["train"] * 8 + ["validation"] * 2 + ["test"] * 2
    for index, split_name in enumerate(split_sequence):
        subject_id = f"OAS2_{index + 1:04d}"
        mri_id = f"{subject_id}_MR1"
        label = "Demented" if index % 2 else "Nondemented"
        for orient in ["axl", "cor", "sag"]:
            _write_png(tmp_path / orient / f"{mri_id}_{orient}.png", value=32 + index * 4)
        rows.append(
            {
                "MRI_ID": mri_id,
                "Subject_ID": subject_id,
                "split": split_name,
                "Final_Group": label,
                "original_path": f"axl/{mri_id}_axl.png",
                "age": 60.0 + index,
                "education": 12.0 + (index % 4),
                "nwbv": 0.72 - index * 0.005,
                "etiv": 1500.0 + index * 5.0,
                "asf": 1.10 + index * 0.01,
            }
        )

    pd.DataFrame(rows).to_csv(split_csv_path, index=False)
    return split_csv_path, output_dir


def _small_transforms():
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    simple = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), normalize])
    return simple, simple


@pytest.mark.skipif(
    TRAINING_TEST_DEPS_MISSING,
    reason="torch/torchvision, scikit-learn, pandas or Pillow not available",
)
@pytest.mark.parametrize("mode", ["classification", "regression"])
def test_train_pytorch_model_smoke(tmp_path, monkeypatch, mode):
    split_csv_path, output_dir = _write_split_csv(tmp_path)
    monkeypatch.setattr(deep_training, "build_transforms", _small_transforms)
    monkeypatch.setattr(deep_training, "_export_embeddings", lambda **kwargs: None)
    monkeypatch.setattr(deep_training, "Figure", None)

    result = deep_training.train_pytorch_model(
        {
            "split_csv_path": split_csv_path,
            "output_dir": output_dir,
            "dataset_dir": tmp_path / "axl",
            "mode": mode,
            "backbone": "medicalnet",
            "headless": True,
            "hyperparameters": {
                "max_epochs": 1,
                "batch_size": 2,
                "medicalnet_depth": 10,
                "pretrained": False,
                "clinical_features": ["age", "education", "nwbv", "etiv", "asf"],
            },
        }
    )

    assert result.mode == mode
    assert result.backbone == "medicalnet"
    assert result.best_checkpoint_path.exists()
    assert result.legacy_checkpoint_path.exists()
    assert result.metrics["best_epoch"] == 1
    if result.mode == "regression":
        used_clinical_features = result.experiment_payload["hparams"]["clinical_features"] or []
        assert "age" not in used_clinical_features


# ---------------------------------------------------------------------------
# Additional unit tests for config helpers and result structure
# ---------------------------------------------------------------------------


def test_coerce_config_from_dict(tmp_path):
    """_coerce_config must accept a plain dict and return a DeepTrainingConfig."""
    mapping = {
        "split_csv_path": str(tmp_path / "split.csv"),
        "output_dir": str(tmp_path / "out"),
        "mode": "classification",
        "backbone": "medicalnet",
    }
    cfg = _coerce_config(mapping)
    assert isinstance(cfg, DeepTrainingConfig)
    assert cfg.mode == "classification"
    assert cfg.backbone == "medicalnet"
    assert cfg.headless is False
    assert cfg.hyperparameters == {}


def test_coerce_config_parses_headless_string_false(tmp_path):
    mapping = {
        "split_csv_path": str(tmp_path / "split.csv"),
        "output_dir": str(tmp_path / "out"),
        "mode": "classification",
        "backbone": "medicalnet",
        "headless": "false",
    }
    cfg = _coerce_config(mapping)
    assert cfg.headless is False


def test_coerce_config_from_dataclass(tmp_path):
    """_coerce_config must round-trip a DeepTrainingConfig without data loss."""
    original = DeepTrainingConfig(
        split_csv_path=tmp_path / "split.csv",
        output_dir=tmp_path / "out",
        mode="regression",
        backbone="medicalnet",
        hyperparameters={"lr": 0.01},
        headless=True,
        dataset_dir=tmp_path / "axl",
    )
    coerced = _coerce_config(original)
    assert coerced.mode == "regression"
    assert coerced.backbone == "medicalnet"
    assert coerced.hyperparameters == {"lr": 0.01}
    assert coerced.headless is True
    assert coerced.dataset_dir == tmp_path / "axl"


def test_coerce_config_invalid_type():
    """_coerce_config must raise TypeError for unsupported config types."""
    with pytest.raises(TypeError):
        _coerce_config(["not", "a", "config"])


def test_require_dependencies_reports_missing_pillow(monkeypatch):
    monkeypatch.setattr(deep_training, "SKLEARN_AVAILABLE", True)
    monkeypatch.setattr(deep_training, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(deep_training, "PANDAS_AVAILABLE", True)
    monkeypatch.setattr(deep_training, "PIL_AVAILABLE", False)

    assert deep_training._require_dependencies(raise_on_missing=False) is False
    with pytest.raises(ImportError, match=r"pillow"):
        deep_training._require_dependencies()


def test_validate_final_group_labels_rejects_invalid_values():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "split": ["train"],
            "MRI_ID": ["OAS2_bad_MR1"],
            "Final_Group": ["Unknown"],
        }
    )
    with pytest.raises(ValueError, match=r"Invalid Final_Group.*Unknown.*redacted") as exc_info:
        deep_training._validate_final_group_labels(df, "train")
    assert "OAS2_bad_MR1" not in str(exc_info.value)


def test_validate_final_group_labels_debug_mode_includes_full_mri_id(monkeypatch):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "split": ["train"],
            "MRI_ID": ["OAS2_bad_MR1"],
            "Final_Group": ["Unknown"],
        }
    )
    monkeypatch.setenv("BRAIN_MRI_DEBUG_IDENTIFIERS", "1")
    with pytest.raises(ValueError, match=r"Invalid Final_Group.*Unknown.*OAS2_bad_MR1"):
        deep_training._validate_final_group_labels(df, "train")


@pytest.mark.skipif(not deep_training.TORCH_AVAILABLE, reason="torch not available")
def test_multi_orient_dataset_rejects_invalid_final_group(tmp_path):
    pd = pytest.importorskip("pandas")
    from brain_mri.ml.datasets import MultiOrientMRIDataset

    df = pd.DataFrame(
        {
            "split": ["train"],
            "MRI_ID": ["OAS2_bad_MR1"],
            "Final_Group": [None],
            "orientation_paths": [[]],
        }
    )
    with pytest.raises(ValueError, match=r"Invalid Final_Group.*redacted") as exc_info:
        MultiOrientMRIDataset(df, root_dir=tmp_path)
    assert "OAS2_bad_MR1" not in str(exc_info.value)


@pytest.mark.skipif(not deep_training.TORCH_AVAILABLE, reason="torch not available")
def test_multi_orient_dataset_rejects_nan_regression_label(tmp_path):
    pd = pytest.importorskip("pandas")
    from brain_mri.ml.datasets import MultiOrientMRIDataset

    df = pd.DataFrame(
        {
            "split": ["train"],
            "MRI_ID": ["OAS2_bad_MR1"],
            "age_normalized": [float("nan")],
            "orientation_paths": [[]],
        }
    )
    ds = MultiOrientMRIDataset(df, root_dir=tmp_path, label_col="age_normalized")
    with pytest.raises(ValueError, match=r"Invalid label for column 'age_normalized'.*redacted") as exc_info:
        ds[0]
    assert "OAS2_bad_MR1" not in str(exc_info.value)


def test_dataset_parse_bool_unknown_string_fails_closed():
    from brain_mri.ml import datasets

    assert datasets._parse_bool("flase") is False
    assert datasets._parse_bool(object(), default=True) is True


def test_resolve_dataset_dir_from_config(tmp_path):
    """_resolve_dataset_dir uses dataset_dir when explicitly provided."""
    cfg = DeepTrainingConfig(
        split_csv_path=tmp_path / "split.csv",
        output_dir=tmp_path / "out",
        mode="classification",
        backbone="medicalnet",
        dataset_dir=tmp_path / "explicit_axl",
    )
    resolved = _resolve_dataset_dir(cfg)
    assert resolved == tmp_path / "explicit_axl"


def test_resolve_dataset_dir_fallback(tmp_path):
    """_resolve_dataset_dir falls back through <output_dir>/../axl to the dataset root."""
    cfg = DeepTrainingConfig(
        split_csv_path=tmp_path / "split.csv",
        output_dir=tmp_path / "subdir" / "out",
        mode="classification",
        backbone="medicalnet",
        dataset_dir=None,
    )
    resolved = _resolve_dataset_dir(cfg)
    assert resolved == tmp_path / "subdir"


def test_deep_training_config_is_frozen(tmp_path):
    """DeepTrainingConfig is a frozen dataclass and must not allow mutation."""
    cfg = DeepTrainingConfig(
        split_csv_path=tmp_path / "split.csv",
        output_dir=tmp_path / "out",
        mode="classification",
        backbone="medicalnet",
    )
    with pytest.raises((AttributeError, TypeError)):
        cfg.mode = "regression"


@pytest.mark.skipif(
    TRAINING_TEST_DEPS_MISSING,
    reason="torch/torchvision, scikit-learn, pandas or Pillow not available",
)
def test_deep_training_result_attributes(tmp_path, monkeypatch):
    """DeepTrainingResult must expose all documented attributes after training."""
    split_csv_path, output_dir = _write_split_csv(tmp_path)
    monkeypatch.setattr(deep_training, "build_transforms", _small_transforms)
    monkeypatch.setattr(deep_training, "_export_embeddings", lambda **kwargs: None)
    monkeypatch.setattr(deep_training, "Figure", None)

    result = deep_training.train_pytorch_model(
        {
            "split_csv_path": split_csv_path,
            "output_dir": output_dir,
            "dataset_dir": tmp_path / "axl",
            "mode": "classification",
            "backbone": "medicalnet",
            "headless": True,
            "hyperparameters": {
                "max_epochs": 1,
                "batch_size": 2,
                "medicalnet_depth": 10,
                "pretrained": False,
                "clinical_features": None,
            },
        }
    )

    assert hasattr(result, "learning_curves")
    assert hasattr(result, "metrics")
    assert hasattr(result, "experiment_payload")
    assert hasattr(result, "summary_message")
    assert hasattr(result, "artifact_paths")
    assert "train_loss" in result.learning_curves
    assert "val_loss" in result.learning_curves


@pytest.mark.skipif(
    TRAINING_TEST_DEPS_MISSING,
    reason="torch/torchvision, scikit-learn, pandas or Pillow not available",
)
def test_train_pytorch_model_experiment_payload_contains_hparams(tmp_path, monkeypatch):
    """The experiment payload must include all hyperparameters used during training."""
    split_csv_path, output_dir = _write_split_csv(tmp_path)
    monkeypatch.setattr(deep_training, "build_transforms", _small_transforms)
    monkeypatch.setattr(deep_training, "_export_embeddings", lambda **kwargs: None)
    monkeypatch.setattr(deep_training, "Figure", None)
    monkeypatch.setattr(deep_training, "debug_batch", None)
    monkeypatch.setenv("USE_MULTIMODAL", "0")

    result = deep_training.train_pytorch_model(
        {
            "split_csv_path": split_csv_path,
            "output_dir": output_dir,
            "dataset_dir": tmp_path / "axl",
            "mode": "classification",
            "backbone": "medicalnet",
            "headless": True,
            "hyperparameters": {
                "max_epochs": 1,
                "batch_size": 2,
                "medicalnet_depth": 10,
                "pretrained": "false",
                "freeze_backbone": "0",
                "mixup_alpha": 0.2,
                "clinical_features": None,
            },
        }
    )

    hparams = result.experiment_payload["hparams"]
    for key in ("lr", "weight_decay", "dropout", "batch_size", "epochs", "seed"):
        assert key in hparams, f"Missing hparam key: {key}"
    assert hparams["pretrained"] is False
    assert hparams["freeze_backbone"] is False
    assert hparams["mixup_alpha"] == 0.2
    assert result.learning_curves["train_accuracy"] == []


@pytest.mark.skipif(
    TRAINING_TEST_DEPS_MISSING,
    reason="torch/torchvision, scikit-learn, pandas or Pillow not available",
)
def test_train_pytorch_model_save_experiment_fn_called(tmp_path, monkeypatch):
    """save_experiment_fn callback must be called exactly once after training."""
    from unittest.mock import MagicMock

    split_csv_path, output_dir = _write_split_csv(tmp_path)
    monkeypatch.setattr(deep_training, "build_transforms", _small_transforms)
    monkeypatch.setattr(deep_training, "_export_embeddings", lambda **kwargs: None)
    monkeypatch.setattr(deep_training, "Figure", None)

    save_fn = MagicMock()

    deep_training.train_pytorch_model(
        {
            "split_csv_path": split_csv_path,
            "output_dir": output_dir,
            "dataset_dir": tmp_path / "axl",
            "mode": "classification",
            "backbone": "medicalnet",
            "headless": True,
            "save_experiment_fn": save_fn,
            "hyperparameters": {
                "max_epochs": 1,
                "batch_size": 2,
                "medicalnet_depth": 10,
                "pretrained": False,
                "clinical_features": None,
            },
        }
    )

    save_fn.assert_called_once()
    payload = save_fn.call_args.args[0]
    assert "metrics" in payload
    assert "hparams" in payload
