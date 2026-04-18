"""Tests for brain_mri/ml/rl_refinement.py (new module in this PR)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Stub tkinter for headless environments
# ---------------------------------------------------------------------------

def _install_tkinter_stub() -> None:
    if "tkinter" in sys.modules:
        return
    tk = types.ModuleType("tkinter")
    messagebox = types.ModuleType("tkinter.messagebox")
    messagebox.showinfo = lambda *args, **kwargs: None
    messagebox.showwarning = lambda *args, **kwargs: None
    messagebox.showerror = lambda *args, **kwargs: None
    tk.messagebox = messagebox
    sys.modules["tkinter"] = tk
    sys.modules["tkinter.messagebox"] = messagebox


_install_tkinter_stub()

from brain_mri.ml import rl_refinement
from brain_mri.ml.rl_refinement import (
    ActionSpec,
    RLRefinementConfig,
    RLRefinementResult,
    _coerce_refinement_config,
    _confusion_2x2,
    _default_actions,
    _filter_valid_mri_ids,
    _sample_dataframe,
    balanced_accuracy_from_cm,
    dump_json,
)


# ---------------------------------------------------------------------------
# _confusion_2x2
# ---------------------------------------------------------------------------


def test_confusion_2x2_all_correct():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    tn, fp, fn, tp = _confusion_2x2(y_true, y_pred)
    assert tn == 2 and fp == 0 and fn == 0 and tp == 2


def test_confusion_2x2_all_wrong():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    tn, fp, fn, tp = _confusion_2x2(y_true, y_pred)
    assert tn == 0 and fp == 2 and fn == 2 and tp == 0


def test_confusion_2x2_empty():
    tn, fp, fn, tp = _confusion_2x2(np.array([], dtype=int), np.array([], dtype=int))
    assert tn == 0 and fp == 0 and fn == 0 and tp == 0


def test_confusion_2x2_all_negative_class():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 1])
    tn, fp, fn, tp = _confusion_2x2(y_true, y_pred)
    assert tn == 2 and fp == 1 and fn == 0 and tp == 0


def test_confusion_2x2_returns_ints():
    y_true = np.array([0, 1])
    y_pred = np.array([1, 0])
    result = _confusion_2x2(y_true, y_pred)
    for val in result:
        assert isinstance(val, int)


# ---------------------------------------------------------------------------
# balanced_accuracy_from_cm
# ---------------------------------------------------------------------------


def test_balanced_accuracy_perfect():
    assert balanced_accuracy_from_cm(tn=5, fp=0, fn=0, tp=5) == 1.0


def test_balanced_accuracy_random_classifier():
    # 50% sensitivity, 50% specificity → balanced acc = 0.5
    score = balanced_accuracy_from_cm(tn=5, fp=5, fn=5, tp=5)
    assert abs(score - 0.5) < 1e-9


def test_balanced_accuracy_zero_positive_class():
    # tp + fn == 0 → sensitivity undefined → treated as 0
    score = balanced_accuracy_from_cm(tn=5, fp=0, fn=0, tp=0)
    # specificity = 5/(5+0) = 1.0, sensitivity = 0/(0+0) = 0.0 → ba = 0.5
    assert abs(score - 0.5) < 1e-9


def test_balanced_accuracy_zero_negative_class():
    # tn + fp == 0 → specificity undefined → treated as 0
    score = balanced_accuracy_from_cm(tn=0, fp=0, fn=0, tp=5)
    # sensitivity = 5/(5+0) = 1.0, specificity = 0 → ba = 0.5
    assert abs(score - 0.5) < 1e-9


def test_balanced_accuracy_all_zeros():
    score = balanced_accuracy_from_cm(tn=0, fp=0, fn=0, tp=0)
    assert score == 0.0


# ---------------------------------------------------------------------------
# _default_actions
# ---------------------------------------------------------------------------


def test_default_actions_returns_five_items():
    actions = _default_actions()
    assert len(actions) == 5


def test_default_actions_all_action_spec():
    for action in _default_actions():
        assert isinstance(action, ActionSpec)


def test_default_actions_positive_lr():
    for action in _default_actions():
        assert action.lr > 0


def test_default_actions_non_negative_weight_decay():
    for action in _default_actions():
        assert action.weight_decay >= 0


# ---------------------------------------------------------------------------
# ActionSpec dataclass
# ---------------------------------------------------------------------------


def test_action_spec_is_frozen():
    spec = ActionSpec(lr=1e-4, weight_decay=1e-5)
    with pytest.raises((AttributeError, TypeError)):
        spec.lr = 0.1


def test_action_spec_values_stored():
    spec = ActionSpec(lr=2e-4, weight_decay=3e-5)
    assert spec.lr == 2e-4
    assert spec.weight_decay == 3e-5


# ---------------------------------------------------------------------------
# _sample_dataframe
# ---------------------------------------------------------------------------


def test_sample_dataframe_limit_none():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": range(20)})
    result = _sample_dataframe(df, limit=None, seed=0)
    assert len(result) == 20


def test_sample_dataframe_limit_larger_than_df():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": range(5)})
    result = _sample_dataframe(df, limit=100, seed=0)
    assert len(result) == 5


def test_sample_dataframe_limit_smaller_than_df():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": range(20)})
    result = _sample_dataframe(df, limit=8, seed=42)
    assert len(result) == 8


def test_sample_dataframe_limit_zero():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": range(10)})
    # limit <= 0 means return full df
    result = _sample_dataframe(df, limit=0, seed=0)
    assert len(result) == 10


def test_sample_dataframe_is_copy():
    """Modifying the result must not affect the original dataframe."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": range(5)})
    result = _sample_dataframe(df, limit=None, seed=0)
    result["a"] = 999
    assert df["a"].iloc[0] != 999


# ---------------------------------------------------------------------------
# _filter_valid_mri_ids
# ---------------------------------------------------------------------------


def test_filter_valid_mri_ids_keeps_valid_rows():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"MRI_ID": ["OAS2_0001_MR1", "OAS2_0002_MR1"], "x": [1, 2]})
    result = _filter_valid_mri_ids(df, "train")
    assert len(result) == 2


def test_filter_valid_mri_ids_removes_nan():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"MRI_ID": ["OAS2_0001_MR1", None, "OAS2_0003_MR1"]})
    result = _filter_valid_mri_ids(df, "train")
    assert len(result) == 2


def test_filter_valid_mri_ids_removes_empty_string():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"MRI_ID": ["OAS2_0001_MR1", "  ", "OAS2_0003_MR1"]})
    result = _filter_valid_mri_ids(df, "train")
    assert len(result) == 2


def test_filter_valid_mri_ids_all_invalid_raises():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"MRI_ID": [None, "", "  "]})
    with pytest.raises(ValueError, match="no rows with valid MRI_ID"):
        _filter_valid_mri_ids(df, "train")


def test_filter_valid_mri_ids_missing_column_raises():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"other_col": [1, 2]})
    with pytest.raises(ValueError, match="MRI_ID"):
        _filter_valid_mri_ids(df, "train")


# ---------------------------------------------------------------------------
# dump_json
# ---------------------------------------------------------------------------


def test_dump_json_creates_file(tmp_path):
    path = tmp_path / "output.json"
    dump_json(path, {"key": "value", "num": 42})
    assert path.exists()


def test_dump_json_valid_json(tmp_path):
    path = tmp_path / "output.json"
    payload = {"a": 1, "b": [1, 2, 3], "c": {"nested": True}}
    dump_json(path, payload)
    with open(path) as f:
        loaded = json.load(f)
    assert loaded == payload


def test_dump_json_sorted_keys(tmp_path):
    path = tmp_path / "output.json"
    dump_json(path, {"z": 1, "a": 2})
    content = path.read_text()
    # 'a' should appear before 'z' when keys are sorted
    assert content.index('"a"') < content.index('"z"')


def test_dump_json_ends_with_newline(tmp_path):
    path = tmp_path / "output.json"
    dump_json(path, {"x": 1})
    content = path.read_text()
    assert content.endswith("\n")


# ---------------------------------------------------------------------------
# _coerce_refinement_config
# ---------------------------------------------------------------------------


def test_coerce_refinement_config_from_mapping(tmp_path):
    mapping = {
        "checkpoint_path": str(tmp_path / "ckpt.pth"),
        "split_csv_path": str(tmp_path / "split.csv"),
        "output_dir": str(tmp_path / "out"),
    }
    cfg = _coerce_refinement_config(mapping)
    assert isinstance(cfg, RLRefinementConfig)
    assert cfg.episodes == 8  # default
    assert cfg.horizon == 4   # default
    assert cfg.seed == 42     # default


def test_coerce_refinement_config_from_mapping_custom_values(tmp_path):
    mapping = {
        "checkpoint_path": str(tmp_path / "ckpt.pth"),
        "split_csv_path": str(tmp_path / "split.csv"),
        "output_dir": str(tmp_path / "out"),
        "episodes": 3,
        "horizon": 2,
        "seed": 7,
        "backbone": "medicalnet",
    }
    cfg = _coerce_refinement_config(mapping)
    assert cfg.episodes == 3
    assert cfg.horizon == 2
    assert cfg.seed == 7
    assert cfg.backbone == "medicalnet"


def test_coerce_refinement_config_from_dataclass(tmp_path):
    original = RLRefinementConfig(
        checkpoint_path=tmp_path / "ckpt.pth",
        split_csv_path=tmp_path / "split.csv",
        output_dir=tmp_path / "out",
        episodes=5,
        seed=99,
    )
    cfg = _coerce_refinement_config(original)
    assert cfg.episodes == 5
    assert cfg.seed == 99
    assert cfg.checkpoint_path == tmp_path / "ckpt.pth"


def test_coerce_refinement_config_invalid_type():
    with pytest.raises(TypeError):
        _coerce_refinement_config([1, 2, 3])


def test_coerce_refinement_config_dataset_dir_optional(tmp_path):
    """When dataset_dir is not specified it defaults to None."""
    mapping = {
        "checkpoint_path": str(tmp_path / "ckpt.pth"),
        "split_csv_path": str(tmp_path / "split.csv"),
        "output_dir": str(tmp_path / "out"),
    }
    cfg = _coerce_refinement_config(mapping)
    assert cfg.dataset_dir is None


def test_coerce_refinement_config_clinical_features_list(tmp_path):
    mapping = {
        "checkpoint_path": str(tmp_path / "ckpt.pth"),
        "split_csv_path": str(tmp_path / "split.csv"),
        "output_dir": str(tmp_path / "out"),
        "clinical_features": ["age", "education"],
    }
    cfg = _coerce_refinement_config(mapping)
    assert cfg.clinical_features == ["age", "education"]


# ---------------------------------------------------------------------------
# RLRefinementConfig dataclass
# ---------------------------------------------------------------------------


def test_rl_refinement_config_frozen(tmp_path):
    cfg = RLRefinementConfig(
        checkpoint_path=tmp_path / "ckpt.pth",
        split_csv_path=tmp_path / "split.csv",
        output_dir=tmp_path / "out",
    )
    with pytest.raises((AttributeError, TypeError)):
        cfg.episodes = 99


# ---------------------------------------------------------------------------
# RLRefinementResult dataclass
# ---------------------------------------------------------------------------


def test_rl_refinement_result_stores_fields(tmp_path):
    result = RLRefinementResult(
        best_hyperparameters={"lr": 1e-4, "weight_decay": 1e-5},
        refined_checkpoint_path=tmp_path / "refined.pth",
        policy_path=tmp_path / "policy.pth",
        history_path=tmp_path / "history.json",
        metrics={"baseline_val_balanced_accuracy": 0.5, "refined_val_balanced_accuracy": 0.6},
    )
    assert result.best_hyperparameters["lr"] == 1e-4
    assert result.metrics["refined_val_balanced_accuracy"] == 0.6


# ---------------------------------------------------------------------------
# PPOAgent and ActorCritic (require torch)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not rl_refinement.TORCH_AVAILABLE, reason="torch not available")
def test_actor_critic_forward_output_shapes():
    """ActorCritic.forward must return (logits, value) with correct shapes."""
    import torch
    from brain_mri.ml.rl_refinement import ActorCritic

    model = ActorCritic(state_dim=4, action_dim=3, hidden=16)
    state = torch.zeros(2, 4)  # batch_size=2
    logits, value = model(state)
    assert logits.shape == (2, 3)
    assert value.shape == (2,)


@pytest.mark.skipif(not rl_refinement.TORCH_AVAILABLE, reason="torch not available")
def test_ppo_agent_select_action_valid_index():
    """select_action must return a valid action index within [0, action_dim)."""
    import torch
    from brain_mri.ml.rl_refinement import PPOAgent

    device = torch.device("cpu")
    agent = PPOAgent(state_dim=3, action_dim=5, device=device)
    state = np.array([0.5, 0.1, 0.0], dtype=np.float32)
    action_index, logp, value = agent.select_action(state)
    assert 0 <= action_index < 5
    assert isinstance(logp, float)
    assert isinstance(value, float)


@pytest.mark.skipif(not rl_refinement.TORCH_AVAILABLE, reason="torch not available")
def test_ppo_agent_store_and_update_clears_buffer():
    """Calling update() must clear the internal experience buffer."""
    import torch
    from brain_mri.ml.rl_refinement import PPOAgent

    device = torch.device("cpu")
    agent = PPOAgent(state_dim=3, action_dim=5, device=device)
    state = np.zeros(3, dtype=np.float32)

    for _ in range(4):
        action_index, logp, value = agent.select_action(state)
        agent.store(state=state, action=action_index, logp=logp, value=value, reward=0.1, done=False)

    assert len(agent._states) == 4
    agent.update()
    assert len(agent._states) == 0


@pytest.mark.skipif(not rl_refinement.TORCH_AVAILABLE, reason="torch not available")
def test_ppo_agent_update_empty_buffer_returns_loss_zero():
    """update() on an empty buffer must return {'loss': 0.0} without error."""
    import torch
    from brain_mri.ml.rl_refinement import PPOAgent

    device = torch.device("cpu")
    agent = PPOAgent(state_dim=3, action_dim=5, device=device)
    result = agent.update()
    assert result == {"loss": 0.0}


@pytest.mark.skipif(not rl_refinement.TORCH_AVAILABLE, reason="torch not available")
def test_ppo_agent_gae_single_step():
    """_compute_gae must not raise for a single-step experience buffer."""
    import torch
    from brain_mri.ml.rl_refinement import PPOAgent

    device = torch.device("cpu")
    agent = PPOAgent(state_dim=3, action_dim=5, device=device)
    state = np.zeros(3, dtype=np.float32)
    action_index, logp, value = agent.select_action(state)
    agent.store(state=state, action=action_index, logp=logp, value=value, reward=1.0, done=True)

    returns, advantages = agent._compute_gae(next_value=0.0)
    assert returns.shape == (1,)
    assert advantages.shape == (1,)


# ---------------------------------------------------------------------------
# RLRefineEnv
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not rl_refinement.TORCH_AVAILABLE, reason="torch not available")
def test_rl_refine_env_reset_returns_correct_shape():
    """reset() must return a numpy array with shape (3,)."""
    import torch
    import torch.nn as nn
    from brain_mri.ml.rl_refinement import RLRefineEnv

    device = torch.device("cpu")

    def build_model():
        return nn.Linear(4, 2)

    dummy_sd = nn.Linear(4, 2).state_dict()

    def fake_finetune(model, train_loader, val_loader, device, **kwargs):
        return {"loss": 0.1}, {"loss": 0.2, "accuracy": 0.6, "balanced_accuracy": 0.55}

    env = RLRefineEnv(
        build_model=build_model,
        base_state_dict=dummy_sd,
        train_loader=[],
        val_loader=[],
        device=device,
        actions=_default_actions(),
        micro_epochs=1,
        max_batches_per_epoch=1,
        class_weights=None,
        seed=0,
        train_pytorch_model_fn=fake_finetune,
    )
    state = env.reset()
    assert state.shape == (3,)
    assert state[0] == env.baseline_val_bal_acc


@pytest.mark.skipif(not rl_refinement.TORCH_AVAILABLE, reason="torch not available")
def test_rl_refine_env_step_returns_correct_types():
    """step() must return (next_state, reward, info) with correct types."""
    import torch
    import torch.nn as nn
    from brain_mri.ml.rl_refinement import RLRefineEnv

    device = torch.device("cpu")

    def build_model():
        return nn.Linear(4, 2)

    dummy_sd = nn.Linear(4, 2).state_dict()

    def fake_finetune(model, train_loader, val_loader, device, **kwargs):
        return {"loss": 0.1}, {"loss": 0.2, "accuracy": 0.7, "balanced_accuracy": 0.65}

    env = RLRefineEnv(
        build_model=build_model,
        base_state_dict=dummy_sd,
        train_loader=[],
        val_loader=[],
        device=device,
        actions=_default_actions(),
        micro_epochs=1,
        max_batches_per_epoch=1,
        class_weights=None,
        seed=0,
        train_pytorch_model_fn=fake_finetune,
    )
    env.reset()
    next_state, reward, info = env.step(0)

    assert isinstance(next_state, np.ndarray)
    assert next_state.shape == (3,)
    assert isinstance(reward, float)
    assert isinstance(info, dict)
    assert "val_balanced_accuracy" in info


@pytest.mark.skipif(not rl_refinement.TORCH_AVAILABLE, reason="torch not available")
def test_rl_refine_env_step_invalid_action_raises():
    """step() with an out-of-range action must raise ValueError."""
    import torch
    import torch.nn as nn
    from brain_mri.ml.rl_refinement import RLRefineEnv

    device = torch.device("cpu")

    def build_model():
        return nn.Linear(4, 2)

    env = RLRefineEnv(
        build_model=build_model,
        base_state_dict=nn.Linear(4, 2).state_dict(),
        train_loader=[],
        val_loader=[],
        device=device,
        actions=_default_actions(),
        micro_epochs=1,
        max_batches_per_epoch=1,
        class_weights=None,
        seed=0,
    )
    env.reset()
    with pytest.raises(ValueError, match="Invalid action index"):
        env.step(999)


@pytest.mark.skipif(not rl_refinement.TORCH_AVAILABLE, reason="torch not available")
def test_rl_refine_env_to_jsonable():
    """to_jsonable() must return a dict with expected keys and JSON-serialisable values."""
    import torch
    import torch.nn as nn
    from brain_mri.ml.rl_refinement import RLRefineEnv

    device = torch.device("cpu")
    env = RLRefineEnv(
        build_model=lambda: nn.Linear(4, 2),
        base_state_dict=nn.Linear(4, 2).state_dict(),
        train_loader=[],
        val_loader=[],
        device=device,
        actions=_default_actions(),
        micro_epochs=2,
        max_batches_per_epoch=3,
        class_weights=None,
        seed=7,
    )
    info = env.to_jsonable()
    assert info["state_dim"] == 3
    assert info["action_dim"] == len(_default_actions())
    assert info["micro_epochs"] == 2
    # Must be JSON-serialisable
    json.dumps(info)