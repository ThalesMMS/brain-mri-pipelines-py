"""Tests for brain_mri/ml/training_utils.py (new/modified functions in this PR)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from brain_mri.ml import training_utils
from brain_mri.ml.training_utils import (
    build_artifact_path,
    ensure_required_columns,
    load_split_dataframe,
    sanitize_artifact_label,
)


# ---------------------------------------------------------------------------
# sanitize_artifact_label
# ---------------------------------------------------------------------------


def test_sanitize_artifact_label_replaces_spaces():
    assert sanitize_artifact_label("hello world") == "hello_world"


def test_sanitize_artifact_label_replaces_slashes():
    assert sanitize_artifact_label("a/b/c") == "a_b_c"


def test_sanitize_artifact_label_strips_whitespace():
    assert sanitize_artifact_label("  label  ") == "label"


def test_sanitize_artifact_label_empty_string():
    assert sanitize_artifact_label("") == "artifact"


def test_sanitize_artifact_label_all_unsafe_falls_back():
    assert sanitize_artifact_label("///...") == "artifact"


def test_sanitize_artifact_label_non_string_input():
    # Must coerce to str without raising
    result = sanitize_artifact_label(42)
    assert result == "42"


def test_sanitize_artifact_label_combined():
    result = sanitize_artifact_label(" my label/v2 ")
    assert result == "my_label_v2"


# ---------------------------------------------------------------------------
# build_artifact_path
# ---------------------------------------------------------------------------


def test_build_artifact_path_no_label(tmp_path):
    result = build_artifact_path(tmp_path, "model.pkl")
    assert result == tmp_path / "model.pkl"


def test_build_artifact_path_with_label(tmp_path):
    result = build_artifact_path(tmp_path, "model.pkl", "my_scenario")
    assert result == tmp_path / "model_my_scenario.pkl"


def test_build_artifact_path_label_none(tmp_path):
    result = build_artifact_path(tmp_path, "scaler.pkl", None)
    assert result == tmp_path / "scaler.pkl"


def test_build_artifact_path_label_empty_string(tmp_path):
    result = build_artifact_path(tmp_path, "scaler.pkl", "")
    assert result == tmp_path / "scaler.pkl"


def test_build_artifact_path_label_with_spaces(tmp_path):
    result = build_artifact_path(tmp_path, "xgb.pkl", "label with spaces")
    # Spaces should be sanitised in the final name
    assert " " not in result.name


def test_build_artifact_path_preserves_extension(tmp_path):
    result = build_artifact_path(tmp_path, "model.tar.gz", "v1")
    # The label is appended before the last extension only
    assert result.name.endswith(".gz")


def test_build_artifact_path_returns_path_instance(tmp_path):
    result = build_artifact_path(tmp_path, "model.pkl", "label")
    assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# ensure_required_columns
# ---------------------------------------------------------------------------


def test_ensure_required_columns_all_present():
    """Must return the dataframe unchanged when all columns are present."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": [1], "b": [2]})
    result = ensure_required_columns(df, ["a", "b"])
    assert list(result.columns) == ["a", "b"]


def test_ensure_required_columns_missing_raises():
    """Must raise ValueError listing the missing columns."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="b"):
        ensure_required_columns(df, ["a", "b"])


def test_ensure_required_columns_multiple_missing_raises():
    """All missing column names must appear in the error message."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError) as exc_info:
        ensure_required_columns(df, ["b", "c"], context="MyDF")
    assert "b" in str(exc_info.value)
    assert "c" in str(exc_info.value)


def test_ensure_required_columns_context_in_error():
    """The context string must appear in the ValueError message."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(ValueError, match="MyContext"):
        ensure_required_columns(df, ["missing"], context="MyContext")


def test_ensure_required_columns_empty_list():
    """Empty required list must not raise."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": [1]})
    result = ensure_required_columns(df, [])
    assert len(result) == 1


# ---------------------------------------------------------------------------
# load_split_dataframe
# ---------------------------------------------------------------------------


def test_load_split_dataframe_file_not_found(tmp_path):
    """Missing CSV must raise FileNotFoundError (requires pandas to be importable)."""
    pytest.importorskip("pandas")
    with pytest.raises(FileNotFoundError):
        load_split_dataframe(tmp_path / "nonexistent.csv")


def test_load_split_dataframe_loads_correctly(tmp_path):
    """A valid CSV must load with the expected rows and columns."""
    pd = pytest.importorskip("pandas")
    csv_path = tmp_path / "split.csv"
    pd.DataFrame({"split": ["train", "validation"], "col": [1, 2]}).to_csv(csv_path, index=False)

    df = load_split_dataframe(csv_path)
    assert len(df) == 2
    assert "split" in df.columns


def test_load_split_dataframe_validates_required_columns(tmp_path):
    """Requesting a column not in the CSV must raise ValueError."""
    pd = pytest.importorskip("pandas")
    csv_path = tmp_path / "split.csv"
    pd.DataFrame({"split": ["train"]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Final_Group"):
        load_split_dataframe(csv_path, required_columns=["split", "Final_Group"])


def test_load_split_dataframe_no_required_columns(tmp_path):
    """Not specifying required_columns must not raise even for a minimal CSV."""
    pd = pytest.importorskip("pandas")
    csv_path = tmp_path / "split.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(csv_path, index=False)

    df = load_split_dataframe(csv_path)
    assert len(df) == 2


# ---------------------------------------------------------------------------
# ExponentialMovingAverage
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not training_utils.TORCH_AVAILABLE, reason="torch not available")
def test_ema_update_changes_shadow_weights():
    """After update(), shadow weights must differ from initialisation when model changes."""
    import torch
    import torch.nn as nn

    model = nn.Linear(4, 2)
    ema = training_utils.ExponentialMovingAverage(model, decay=0.9)

    # Store initial shadows
    initial_shadows = {k: v.clone() for k, v in ema.shadow.items()}

    # Modify model weights
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(99.0)

    ema.update(model)

    for name, shadow in ema.shadow.items():
        # After one update with heavy model change, shadow should shift toward 99
        assert not torch.allclose(shadow, initial_shadows[name])


@pytest.mark.skipif(not training_utils.TORCH_AVAILABLE, reason="torch not available")
def test_ema_apply_shadow_and_restore():
    """apply_shadow() replaces weights; restore() reverts them."""
    import torch
    import torch.nn as nn

    model = nn.Linear(4, 2)
    original_weight = model.weight.data.clone()

    ema = training_utils.ExponentialMovingAverage(model, decay=0.99)

    # Drastically change model weights
    with torch.no_grad():
        model.weight.fill_(0.0)

    ema.apply_shadow(model)
    # After applying shadow, weights should NOT equal the zeros we set
    assert not torch.allclose(model.weight.data, torch.zeros_like(model.weight.data))

    ema.restore(model)
    # Restore should bring weights back to the zeroed version (our backup)
    assert torch.allclose(model.weight.data, torch.zeros(model.weight.shape))


@pytest.mark.skipif(not training_utils.TORCH_AVAILABLE, reason="torch not available")
def test_ema_restore_noop_without_backup():
    """restore() must be a no-op (no error) when apply_shadow was never called."""
    import torch.nn as nn

    model = nn.Linear(2, 2)
    ema = training_utils.ExponentialMovingAverage(model, decay=0.9)
    # Should not raise
    ema.restore(model)


@pytest.mark.skipif(not training_utils.TORCH_AVAILABLE, reason="torch not available")
def test_ema_decay_bounds():
    """EMA decay must be stored correctly as a float."""
    import torch.nn as nn

    model = nn.Linear(2, 2)
    ema = training_utils.ExponentialMovingAverage(model, decay=0.5)
    assert ema.decay == 0.5
