from __future__ import annotations

import hashlib
import json
import sys
import types
from types import SimpleNamespace

import run_baselines_cli
import run_deep_models_cli


REQUIRED_MANIFEST_KEYS = {"cli", "timestamp", "git_commit", "command", "inputs", "outputs"}


def _install_history_module(monkeypatch):
    history_module = types.ModuleType("brain_mri.experiments.history")

    class FakeExperimentHistoryMixin:
        pass

    history_module.ExperimentHistoryMixin = FakeExperimentHistoryMixin
    monkeypatch.setitem(sys.modules, "brain_mri.experiments.history", history_module)


def test_run_baselines_cli_writes_manifest(monkeypatch, tmp_path):
    _install_history_module(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "oasis_longitudinal_demographic.csv").write_text("demo", encoding="utf-8")

    class FakeMLTrainingMixin:
        def create_exam_level_dataset(self):
            self.output_dir.mkdir(exist_ok=True)
            (self.output_dir / "exam_level_dataset_split.csv").write_text("split", encoding="utf-8")
            self.descriptors_csv.write_text("descriptors", encoding="utf-8")

        def train_svm_classifier(self, features=None, scenario=None):
            model_path = self.output_dir / f"svm_model_{scenario}.pkl"
            scaler_path = self.output_dir / f"svm_scaler_{scenario}.pkl"
            model_path.write_text(f"model:{scenario}", encoding="utf-8")
            scaler_path.write_text(f"scaler:{scenario}", encoding="utf-8")
            self.experiment_history_path.write_text("[]", encoding="utf-8")
            return SimpleNamespace(model_path=model_path, scaler_path=scaler_path)

        def train_xgboost_regressor(self, features=None, scenario=None, seed=None):
            model_path = self.output_dir / f"xgb_age_{scenario}.pkl"
            model_path.write_text(f"xgb:{seed}", encoding="utf-8")
            self.experiment_history_path.write_text('[{"model": "XGBoost"}]', encoding="utf-8")
            return SimpleNamespace(model_path=model_path)

    ml_training_module = types.ModuleType("brain_mri.ml.ml_training")
    ml_training_module.MLTrainingMixin = FakeMLTrainingMixin
    monkeypatch.setitem(sys.modules, "brain_mri.ml.ml_training", ml_training_module)

    monkeypatch.setattr(run_baselines_cli, "_patch_messagebox", lambda: None)
    monkeypatch.setattr(
        run_baselines_cli,
        "_write_dataset_stats",
        lambda app: (app.output_dir / "dataset_stats.json").write_text("{}", encoding="utf-8"),
    )
    monkeypatch.setattr(run_baselines_cli, "capture_pip_freeze", lambda: ["example==1.0"])
    monkeypatch.setattr(run_baselines_cli, "git_commit", lambda: "abc123")
    monkeypatch.setattr(run_baselines_cli, "git_is_dirty", lambda: False)
    monkeypatch.setattr(run_baselines_cli, "generate_timestamp", lambda: "20250101_120000")
    monkeypatch.setattr(sys, "argv", ["run_baselines_cli.py", "--xgb", "train", "--seed", "7"])

    run_baselines_cli.main()

    manifest_files = list((tmp_path / "output" / "manifests" / "baselines").glob("*.json"))
    assert len(manifest_files) == 1
    data = json.loads(manifest_files[0].read_text(encoding="utf-8"))
    assert REQUIRED_MANIFEST_KEYS.issubset(data)
    assert data["cli"] == "baselines"
    assert data["timestamp"] == "20250101_120000"
    assert data["git_commit"] == "abc123"
    assert data["args"] == {"seed": 7, "xgb_mode": "train"}
    assert data["dependencies"] == ["example==1.0"]
    assert data["outputs"]["xgboost"]["trained_model"]["sha256"] == hashlib.sha256(b"xgb:7").hexdigest()


def test_run_deep_models_cli_writes_manifest(monkeypatch, tmp_path):
    _install_history_module(monkeypatch)
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    split_csv = output_dir / "exam_level_dataset_split.csv"
    experiments_path = output_dir / "training_experiments.json"
    split_csv.write_text("split", encoding="utf-8")
    experiments_path.write_text("before", encoding="utf-8")
    before_hash = hashlib.sha256(b"before").hexdigest()
    after_hash = hashlib.sha256(b"after").hexdigest()

    class FakeMLTrainingMixin:
        def _train_pytorch_model(self, mode="classification", backbone="medicalnet", hparams=None):
            artifact_path = self.output_dir / f"best_{backbone}_classifier.pth"
            artifact_path.write_text(f"weights:{backbone}", encoding="utf-8")
            self.experiment_history_path.write_text("after", encoding="utf-8")
            return SimpleNamespace(artifact_paths={"best_checkpoint": artifact_path})

    ml_training_module = types.ModuleType("brain_mri.ml.ml_training")
    ml_training_module.MLTrainingMixin = FakeMLTrainingMixin
    monkeypatch.setitem(sys.modules, "brain_mri.ml.ml_training", ml_training_module)

    monkeypatch.delenv("DEEP_SCENARIO", raising=False)
    monkeypatch.setattr(run_deep_models_cli, "capture_pip_freeze", lambda: [])
    monkeypatch.setattr(run_deep_models_cli, "git_commit", lambda: "abc123")
    monkeypatch.setattr(run_deep_models_cli, "git_is_dirty", lambda: False)
    monkeypatch.setattr(run_deep_models_cli, "generate_timestamp", lambda: "20250101_120000")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_deep_models_cli.py",
            "--seed",
            "7",
            "--epochs",
            "2",
            "--backbones",
            "efficientnet,densenet",
            "--multimodal",
        ],
    )

    run_deep_models_cli.main()

    manifest_files = list((output_dir / "manifests" / "deep_models").glob("*.json"))
    assert len(manifest_files) == 1
    data = json.loads(manifest_files[0].read_text(encoding="utf-8"))
    assert REQUIRED_MANIFEST_KEYS.issubset(data)
    assert data["cli"] == "deep_models"
    assert data["timestamp"] == "20250101_120000"
    assert data["git_commit"] == "abc123"
    assert data["args"] == {
        "seed": 7,
        "epochs": 2,
        "backbones": ["efficientnet", "densenet"],
        "multimodal": True,
    }
    assert data["inputs"]["training_experiments"]["sha256"] == before_hash
    assert set(data["outputs"]) == {"efficientnet", "densenet", "training_experiments"}
    assert data["outputs"]["training_experiments"]["sha256"] == after_hash
    assert data["dependencies"] == []


# ---------------------------------------------------------------------------
# Unit tests for helper functions in run_baselines_cli
# ---------------------------------------------------------------------------


def test_unique_manifest_path_returns_timestamp_path_when_dir_empty(tmp_path):
    manifest_dir = tmp_path / "manifests" / "baselines"
    manifest_dir.mkdir(parents=True)

    result = run_baselines_cli._unique_manifest_path(manifest_dir, "20250101_120000")

    assert result == manifest_dir / "20250101_120000.json"


def test_unique_manifest_path_returns_01_suffix_on_first_collision(tmp_path):
    manifest_dir = tmp_path / "manifests" / "baselines"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "20250101_120000.json").write_text("{}", encoding="utf-8")

    result = run_baselines_cli._unique_manifest_path(manifest_dir, "20250101_120000")

    assert result == manifest_dir / "20250101_120000_01.json"


def test_unique_manifest_path_returns_02_suffix_on_second_collision(tmp_path):
    manifest_dir = tmp_path / "manifests" / "baselines"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "20250101_120000.json").write_text("{}", encoding="utf-8")
    (manifest_dir / "20250101_120000_01.json").write_text("{}", encoding="utf-8")

    result = run_baselines_cli._unique_manifest_path(manifest_dir, "20250101_120000")

    assert result == manifest_dir / "20250101_120000_02.json"


def test_baselines_manifest_file_none_path_returns_null_fields(tmp_path):
    result = run_baselines_cli._manifest_file(None, tmp_path)

    assert result == {"path": None, "sha256": None}


def test_baselines_manifest_file_existing_path_returns_correct_hash(tmp_path):
    import hashlib

    target = tmp_path / "artifact.pkl"
    target.write_bytes(b"model bytes")
    expected = hashlib.sha256(b"model bytes").hexdigest()

    result = run_baselines_cli._manifest_file(target, tmp_path)

    assert result["sha256"] == expected
    assert result["path"] is not None


def test_baselines_manifest_file_nonexistent_path_returns_none_sha256(tmp_path):
    missing = tmp_path / "does_not_exist.pkl"

    result = run_baselines_cli._manifest_file(missing, tmp_path)

    assert result["sha256"] is None
    assert result["path"] is not None


def test_baselines_dependencies_snapshot_returns_empty_list_when_success_empty(monkeypatch):
    monkeypatch.setattr(run_baselines_cli, "capture_pip_freeze", lambda: [])

    assert run_baselines_cli._dependencies_snapshot() == []


def test_baselines_dependencies_snapshot_returns_none_when_capture_fails(monkeypatch):
    monkeypatch.setattr(run_baselines_cli, "capture_pip_freeze", lambda: None)

    assert run_baselines_cli._dependencies_snapshot() is None


def test_baselines_dependencies_snapshot_returns_list_when_nonempty(monkeypatch):
    monkeypatch.setattr(run_baselines_cli, "capture_pip_freeze", lambda: ["pkg==1.0"])

    assert run_baselines_cli._dependencies_snapshot() == ["pkg==1.0"]


# ---------------------------------------------------------------------------
# Unit tests for helper functions in run_deep_models_cli
# ---------------------------------------------------------------------------


def test_deep_models_sha256_if_exists_returns_none_for_missing_file(tmp_path):
    result = run_deep_models_cli._sha256_if_exists(tmp_path / "no_such_file.pth")

    assert result is None


def test_deep_models_sha256_if_exists_returns_correct_hash_for_existing_file(tmp_path):
    import hashlib

    artifact = tmp_path / "weights.pth"
    artifact.write_bytes(b"checkpoint data")
    expected = hashlib.sha256(b"checkpoint data").hexdigest()

    assert run_deep_models_cli._sha256_if_exists(artifact) == expected


def test_deep_models_manifest_file_none_path_returns_null_fields(tmp_path):
    result = run_deep_models_cli._manifest_file(None, tmp_path)

    assert result == {"path": None, "sha256": None}


def test_deep_models_unique_manifest_path_returns_01_suffix_on_collision(tmp_path):
    manifest_dir = tmp_path / "manifests" / "deep_models"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "20250202_080000.json").write_text("{}", encoding="utf-8")

    result = run_deep_models_cli._unique_manifest_path(manifest_dir, "20250202_080000")

    assert result == manifest_dir / "20250202_080000_01.json"


def test_deep_models_dependencies_snapshot_returns_empty_list_when_success_empty(monkeypatch):
    monkeypatch.setattr(run_deep_models_cli, "capture_pip_freeze", lambda: [])

    assert run_deep_models_cli._dependencies_snapshot() == []


def test_deep_models_dependencies_snapshot_returns_none_when_capture_fails(monkeypatch):
    monkeypatch.setattr(run_deep_models_cli, "capture_pip_freeze", lambda: None)

    assert run_deep_models_cli._dependencies_snapshot() is None


# ---------------------------------------------------------------------------
# Integration: baselines CLI with xgb="eval-existing" and no existing model
# ---------------------------------------------------------------------------


def test_run_baselines_cli_eval_existing_no_model_writes_manifest(monkeypatch, tmp_path):
    """eval-existing mode when xgb_age.pkl does not exist: evaluated flag=False -> path=None."""
    _install_history_module(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "oasis_longitudinal_demographic.csv").write_text("demo", encoding="utf-8")

    class FakeMLTrainingMixin:
        def create_exam_level_dataset(self):
            self.output_dir.mkdir(exist_ok=True)
            (self.output_dir / "exam_level_dataset_split.csv").write_text("split", encoding="utf-8")
            self.descriptors_csv.write_text("descriptors", encoding="utf-8")

        def train_svm_classifier(self, features=None, scenario=None):
            model_path = self.output_dir / f"svm_model_{scenario}.pkl"
            scaler_path = self.output_dir / f"svm_scaler_{scenario}.pkl"
            model_path.write_text(f"model:{scenario}", encoding="utf-8")
            scaler_path.write_text(f"scaler:{scenario}", encoding="utf-8")
            self.experiment_history_path.write_text("[]", encoding="utf-8")
            return types.SimpleNamespace(model_path=model_path, scaler_path=scaler_path)

    ml_training_module = types.ModuleType("brain_mri.ml.ml_training")
    ml_training_module.MLTrainingMixin = FakeMLTrainingMixin
    monkeypatch.setitem(sys.modules, "brain_mri.ml.ml_training", ml_training_module)

    monkeypatch.setattr(run_baselines_cli, "_patch_messagebox", lambda: None)
    monkeypatch.setattr(
        run_baselines_cli,
        "_write_dataset_stats",
        lambda app: (app.output_dir / "dataset_stats.json").write_text("{}", encoding="utf-8"),
    )
    # Patch _evaluate_existing_xgboost to return False (model not found)
    monkeypatch.setattr(run_baselines_cli, "_evaluate_existing_xgboost", lambda app, features: False)
    monkeypatch.setattr(run_baselines_cli, "capture_pip_freeze", lambda: [])
    monkeypatch.setattr(run_baselines_cli, "git_commit", lambda: "fff000")
    monkeypatch.setattr(run_baselines_cli, "git_is_dirty", lambda: True)
    monkeypatch.setattr(run_baselines_cli, "generate_timestamp", lambda: "20250202_080000")
    monkeypatch.setattr(
        sys, "argv", ["run_baselines_cli.py", "--xgb", "eval-existing", "--seed", "0"]
    )

    run_baselines_cli.main()

    manifest_files = list((tmp_path / "output" / "manifests" / "baselines").glob("*.json"))
    assert len(manifest_files) == 1
    data = json.loads(manifest_files[0].read_text(encoding="utf-8"))
    assert data["cli"] == "baselines"
    assert data["git_dirty"] is True
    assert data["git_commit"] == "fff000"
    assert data["args"]["xgb_mode"] == "eval-existing"
    # xgb_result is None (no training) -> trained_model path and sha256 are None
    assert data["outputs"]["xgboost"]["trained_model"] == {"path": None, "sha256": None}
    # evaluated_existing_xgb=False -> evaluated_existing_model path and sha256 are None
    assert data["outputs"]["xgboost"]["evaluated_existing_model"] == {"path": None, "sha256": None}
    assert data["dependencies"] == []
