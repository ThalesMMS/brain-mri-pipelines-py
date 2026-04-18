"""Tests for the refactored brain_mri/ml/ml_training.py (MLTrainingMixin)."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def tkinter_stub(monkeypatch):
    """Keep MLTrainingMixin tests headless with a scoped tkinter stub."""
    tk = types.ModuleType("tkinter")
    messagebox = types.ModuleType("tkinter.messagebox")
    messagebox.showinfo = lambda *args, **kwargs: None
    messagebox.showwarning = lambda *args, **kwargs: None
    messagebox.showerror = lambda *args, **kwargs: None
    messagebox.askyesno = lambda *args, **kwargs: True
    tk.messagebox = messagebox
    monkeypatch.setitem(sys.modules, "tkinter", tk)
    monkeypatch.setitem(sys.modules, "tkinter.messagebox", messagebox)


def _dataset_split_filename() -> str:
    from brain_mri.ml.dataset_builder import DATASET_SPLIT_FILENAME

    return DATASET_SPLIT_FILENAME


# ---------------------------------------------------------------------------
# Helper: concrete subclass with required attributes
# ---------------------------------------------------------------------------


def _App(tmp_path: Path):
    """Minimal concrete implementation for testing the mixin."""
    from brain_mri.ml.ml_training import MLTrainingMixin

    class _ConcreteApp(MLTrainingMixin):
        def __init__(self, tmp_path: Path):
            self.dataset_dir = str(tmp_path / "axl")
            self.output_dir = str(tmp_path / "output")
            self.csv_path = str(tmp_path / "demo.csv")
            self.descriptors_csv = str(tmp_path / "desc.csv")
            self.root = None  # headless by default
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    return _ConcreteApp(tmp_path)


# ---------------------------------------------------------------------------
# _resolve_split_csv_path
# ---------------------------------------------------------------------------


def test_resolve_split_csv_path_default(tmp_path):
    """Without SPLIT_CSV_PATH env var, must return output_dir/DATASET_SPLIT_FILENAME."""
    app = _App(tmp_path)
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("SPLIT_CSV_PATH", None)
        result = app._resolve_split_csv_path()
    assert result == Path(app.output_dir) / _dataset_split_filename()


def test_resolve_split_csv_path_env_override(tmp_path):
    """When SPLIT_CSV_PATH is set, the env-var path must be returned."""
    app = _App(tmp_path)
    override = str(tmp_path / "custom_split.csv")
    with patch.dict(os.environ, {"SPLIT_CSV_PATH": override}):
        result = app._resolve_split_csv_path()
    assert result == Path(override)


def test_resolve_split_csv_path_env_whitespace_stripped(tmp_path):
    """Whitespace around the env-var value must be stripped."""
    app = _App(tmp_path)
    override = str(tmp_path / "split.csv")
    with patch.dict(os.environ, {"SPLIT_CSV_PATH": f"  {override}  "}):
        result = app._resolve_split_csv_path()
    assert result == Path(override)


def test_resolve_split_csv_path_empty_env_uses_default(tmp_path):
    """An empty SPLIT_CSV_PATH env-var must fall through to the default."""
    app = _App(tmp_path)
    with patch.dict(os.environ, {"SPLIT_CSV_PATH": ""}):
        result = app._resolve_split_csv_path()
    assert result == Path(app.output_dir) / _dataset_split_filename()


# ---------------------------------------------------------------------------
# _save_experiment_callback
# ---------------------------------------------------------------------------


def test_save_experiment_callback_returns_none_when_not_defined(tmp_path):
    """When the app has no _save_experiment attribute, callback must be None."""
    app = _App(tmp_path)
    assert app._save_experiment_callback() is None


def test_save_experiment_callback_returns_method_when_defined(tmp_path):
    """When _save_experiment exists, it must be returned by _save_experiment_callback."""
    app = _App(tmp_path)
    mock_fn = MagicMock()
    app._save_experiment = mock_fn
    result = app._save_experiment_callback()
    assert result is mock_fn


# ---------------------------------------------------------------------------
# _is_headless
# ---------------------------------------------------------------------------


def test_is_headless_when_root_is_none(tmp_path):
    app = _App(tmp_path)
    app.root = None
    assert app._is_headless() is True


def test_is_headless_when_root_is_not_none(tmp_path):
    app = _App(tmp_path)
    app.root = object()  # any non-None value simulates an active Tk root
    assert app._is_headless() is False


def test_is_headless_when_root_attribute_absent(tmp_path):
    app = _App(tmp_path)
    del app.root
    assert app._is_headless() is True


# ---------------------------------------------------------------------------
# _dataset_builder_config
# ---------------------------------------------------------------------------


def test_dataset_builder_config_creates_correct_paths(tmp_path):
    """_dataset_builder_config must create a DatasetBuilderConfig with the app's paths."""
    from brain_mri.ml.dataset_builder import DatasetBuilderConfig

    app = _App(tmp_path)
    cfg = app._dataset_builder_config()

    assert isinstance(cfg, DatasetBuilderConfig)
    assert cfg.dataset_dir == Path(app.dataset_dir)
    assert cfg.output_dir == Path(app.output_dir)
    assert cfg.csv_path == Path(app.csv_path)
    assert cfg.descriptors_csv == Path(app.descriptors_csv)


# ---------------------------------------------------------------------------
# _plot_confusion_figure
# ---------------------------------------------------------------------------


def test_plot_confusion_figure_returns_none_when_figure_unavailable(tmp_path, monkeypatch):
    """When matplotlib Figure is None, _plot_confusion_figure must return None."""
    import brain_mri.ml.ml_training as ml_mod
    monkeypatch.setattr(ml_mod, "Figure", None)

    app = _App(tmp_path)
    result = app._plot_confusion_figure([[1, 0], [0, 1]], ["A", "B"])
    assert result is None


def test_plot_confusion_figure_returns_none_when_plot_method_missing(tmp_path):
    """Without a plot_confusion_matrix method, must return None."""
    app = _App(tmp_path)
    # _App does not have plot_confusion_matrix
    result = app._plot_confusion_figure([[1, 0], [0, 1]], ["A", "B"])
    assert result is None


# ---------------------------------------------------------------------------
# _save_and_maybe_show_figure
# ---------------------------------------------------------------------------


def test_save_and_maybe_show_figure_returns_none_for_none_figure(tmp_path):
    """When figure is None, _save_and_maybe_show_figure must return None."""
    app = _App(tmp_path)
    result = app._save_and_maybe_show_figure(None, tmp_path / "fig.png", "Title")
    assert result is None


def test_save_and_maybe_show_figure_saves_and_returns_path(tmp_path):
    """When a figure is provided, it must be saved and the path returned."""
    app = _App(tmp_path)
    fig = MagicMock()
    out_path = tmp_path / "output" / "fig.png"
    result = app._save_and_maybe_show_figure(fig, out_path, "Title")
    fig.savefig.assert_called_once_with(out_path, dpi=300, bbox_inches="tight")
    assert result == out_path


# ---------------------------------------------------------------------------
# _resolve_backbone_checkpoint
# ---------------------------------------------------------------------------


def test_resolve_backbone_checkpoint_returns_none_when_no_files(tmp_path):
    """When no checkpoint files exist, must return None."""
    app = _App(tmp_path)
    result = app._resolve_backbone_checkpoint("medicalnet")
    assert result is None


def test_resolve_backbone_checkpoint_returns_first_existing(tmp_path):
    """Must return the first existing candidate checkpoint path."""
    app = _App(tmp_path)
    output_dir = Path(app.output_dir)
    ckpt = output_dir / "best_medicalnet_classifier.pth"
    ckpt.touch()

    result = app._resolve_backbone_checkpoint("medicalnet")
    assert result == ckpt


# ---------------------------------------------------------------------------
# train_svm_classifier (delegation / error path)
# ---------------------------------------------------------------------------


def test_train_svm_classifier_returns_none_when_no_dataset(tmp_path):
    """When the split CSV does not exist, train_svm_classifier must return None."""
    app = _App(tmp_path)
    # No split CSV has been created
    result = app.train_svm_classifier(features=["age"], scenario="test")
    assert result is None


def test_train_xgboost_regressor_returns_none_when_no_dataset(tmp_path):
    """When the split CSV does not exist, train_xgboost_regressor must return None."""
    app = _App(tmp_path)
    result = app.train_xgboost_regressor(features=["age"], scenario="test")
    assert result is None


# ---------------------------------------------------------------------------
# _train_pytorch_model (no-split-csv guard)
# ---------------------------------------------------------------------------


def test_train_pytorch_model_returns_none_when_no_split_csv(tmp_path):
    """_train_pytorch_model must return None when split CSV is absent."""
    app = _App(tmp_path)
    result = app._train_pytorch_model(mode="classification", backbone="medicalnet")
    assert result is None


# ---------------------------------------------------------------------------
# Delegation: _list_orientation_paths and _populate_orientation_paths
# ---------------------------------------------------------------------------


def test_list_orientation_paths_delegates_to_dataset_builder(tmp_path, monkeypatch):
    """_list_orientation_paths must delegate to dataset_builder.list_orientation_paths."""
    import brain_mri.ml.ml_training as ml_mod

    fake_result = ["axl/OAS2_0001_MR1_axl.png"]
    mock_fn = MagicMock(return_value=fake_result)
    monkeypatch.setattr(ml_mod, "list_orientation_paths", mock_fn)

    app = _App(tmp_path)
    result = app._list_orientation_paths("OAS2_0001_MR1", tmp_path)

    mock_fn.assert_called_once_with("OAS2_0001_MR1", tmp_path)
    assert result == fake_result


def test_populate_orientation_paths_delegates(tmp_path, monkeypatch):
    """_populate_orientation_paths must delegate to dataset_builder.populate_orientation_paths."""
    import brain_mri.ml.ml_training as ml_mod

    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"MRI_ID": ["OAS2_0001_MR1"]})
    enriched_df = pd.DataFrame({"MRI_ID": ["OAS2_0001_MR1"], "orientation_paths": [["axl/OAS2_0001_MR1_axl.png"]]})
    mock_fn = MagicMock(return_value=enriched_df)
    monkeypatch.setattr(ml_mod, "populate_orientation_paths", mock_fn)

    app = _App(tmp_path)
    result = app._populate_orientation_paths(df, tmp_path)

    mock_fn.assert_called_once_with(df, tmp_path)
    pd.testing.assert_frame_equal(result, enriched_df)


# ---------------------------------------------------------------------------
# create_exam_level_dataset error handling
# ---------------------------------------------------------------------------


def test_create_exam_level_dataset_returns_none_on_import_error(tmp_path, monkeypatch):
    """create_exam_level_dataset must return None when an ImportError is raised."""
    import brain_mri.ml.ml_training as ml_mod

    monkeypatch.setattr(ml_mod, "build_exam_level_dataset", lambda cfg: (_ for _ in ()).throw(ImportError("pandas missing")))

    app = _App(tmp_path)
    result = app.create_exam_level_dataset()
    assert result is None


def test_create_exam_level_dataset_returns_none_on_value_error(tmp_path, monkeypatch):
    """create_exam_level_dataset must return None when a ValueError is raised."""
    import brain_mri.ml.ml_training as ml_mod

    def _raise_value_error(cfg):
        raise ValueError("Dados insuficientes")

    monkeypatch.setattr(ml_mod, "build_exam_level_dataset", _raise_value_error)

    app = _App(tmp_path)
    result = app.create_exam_level_dataset()
    assert result is None


def test_create_exam_level_dataset_returns_tuple_on_success(tmp_path, monkeypatch):
    """create_exam_level_dataset must return (df, path) when build succeeds."""
    import brain_mri.ml.ml_training as ml_mod

    pd = pytest.importorskip("pandas")
    fake_df = pd.DataFrame({"MRI_ID": ["X"]})
    fake_path = tmp_path / "output.csv"
    fake_path.touch()

    monkeypatch.setattr(ml_mod, "build_exam_level_dataset", lambda cfg: (fake_df, fake_path))

    app = _App(tmp_path)
    result = app.create_exam_level_dataset()

    assert result is not None
    df_out, path_out = result
    assert len(df_out) == 1
    assert path_out == fake_path
