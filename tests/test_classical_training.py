import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
pd = pytest.importorskip("pandas")


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
    from sklearn.model_selection import GridSearchCV as SklearnGridSearchCV
except ImportError:
    SklearnGridSearchCV = None

from brain_mri.ml import classical_training


@pytest.fixture
def training_dataframe():
    rows = []
    split_sequence = ["train"] * 6 + ["validation"] * 3 + ["test"] * 3
    for index, split_name in enumerate(split_sequence):
        label = "Demented" if index % 2 else "Nondemented"
        subject_id = f"OAS2_{index + 1:04d}"
        if index == 1:
            subject_id = "OAS2_0001"
        rows.append(
            {
                "MRI_ID": f"OAS2_{index + 1:04d}_MR1",
                "Subject_ID": subject_id,
                "split": split_name,
                "Final_Group": label,
                "has_descriptors": True,
                "ventricle_area": 100.0 + index * 10.0,
                "ventricle_perimeter": 50.0 + index * 2.0,
                "ventricle_circularity": 0.75 - index * 0.01,
                "ventricle_eccentricity": 0.20 + index * 0.01,
                "mmse": 29 - (index % 5),
                "cdr": 1.0 if label == "Demented" else 0.0,
                "age": 60.0 + index,
                "nwbv": 0.72 - index * 0.005,
                "etiv": 1500.0 + index * 5.0,
                "asf": 1.10 + index * 0.01,
                "sex": index % 2,
                "education": 12 + (index % 4),
            }
        )
    return pd.DataFrame(rows)


class _FastGridSearchCV:
    def __init__(self, estimator, param_grid, cv, scoring=None, n_jobs=None, verbose=0):
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.best_estimator_ = None
        self.best_params_ = None

    def fit(self, x, y, groups=None):
        estimator_steps = getattr(self.estimator, "named_steps", {})
        if self.estimator.__class__.__name__ == "SVC" or "svc" in estimator_steps:
            if groups is None:
                raise AssertionError("SVM grid search must receive subject groups")
            prefix = "svc__" if "svc" in estimator_steps else ""
            param_grid = {f"{prefix}C": [1.0], f"{prefix}gamma": ["scale"], f"{prefix}kernel": ["linear"]}
        else:
            prefix = "xgb__" if "xgb" in estimator_steps else ""
            param_grid = {
                f"{prefix}n_estimators": [8],
                f"{prefix}max_depth": [2],
                f"{prefix}learning_rate": [0.1],
                f"{prefix}min_child_weight": [1],
                f"{prefix}subsample": [1.0],
                f"{prefix}colsample_bytree": [1.0],
            }

        search = SklearnGridSearchCV(
            estimator=self.estimator,
            param_grid=param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=1,
            verbose=0,
        )
        if groups is None:
            search.fit(x, y)
        else:
            search.fit(x, y, groups=groups)
        self.best_estimator_ = search.best_estimator_
        self.best_params_ = search.best_params_
        return self


@pytest.mark.skipif(not classical_training.SKLEARN_AVAILABLE, reason="scikit-learn not available")
def test_train_svm_classifier_smoke(tmp_path, training_dataframe, monkeypatch):
    monkeypatch.setattr(classical_training, "GridSearchCV", _FastGridSearchCV)
    save_experiment = MagicMock()

    result = classical_training.train_svm_classifier(
        df=training_dataframe,
        features=["ventricle_area", "ventricle_perimeter", "age"],
        scenario="svm_smoke",
        output_dir=tmp_path,
        save_experiment_fn=save_experiment,
    )

    assert result.model_name == "SVM"
    assert result.model_path.exists()
    assert result.scaler_path.exists()
    assert "val_accuracy" in result.metrics
    assert "test_accuracy" in result.metrics
    save_experiment.assert_called_once()
    payload = save_experiment.call_args.args[0]
    assert payload["model"] == "SVM"
    assert payload["scenario"] == "svm_smoke"
    assert "test_confusion_matrix" in payload


@pytest.mark.skipif(
    not classical_training.SKLEARN_AVAILABLE or not classical_training.XGBOOST_AVAILABLE,
    reason="scikit-learn or xgboost not available",
)
def test_train_xgboost_regressor_smoke(tmp_path, training_dataframe, monkeypatch):
    monkeypatch.setattr(classical_training, "GridSearchCV", _FastGridSearchCV)
    save_experiment = MagicMock()

    result = classical_training.train_xgboost_regressor(
        df=training_dataframe,
        features=["ventricle_area", "ventricle_perimeter", "age", "education"],
        scenario="xgb_smoke",
        output_dir=tmp_path,
        seed=7,
        save_experiment_fn=save_experiment,
    )

    assert result.model_name == "XGBoost"
    assert result.model_path == tmp_path / "xgb_age_xgb_smoke.pkl"
    assert result.model_path.exists()
    assert "val_mae" in result.metrics
    assert "test_mae" in result.metrics
    save_experiment.assert_called_once()
    payload = save_experiment.call_args.args[0]
    assert payload["model"] == "XGBoost"
    assert payload["scenario"] == "xgb_smoke"
    assert payload["seed"] == 7
    assert payload["split_csv_sha256"] is not None


# ---------------------------------------------------------------------------
# Additional edge-case and regression tests
# ---------------------------------------------------------------------------


@pytest.fixture
def training_dataframe_no_test():
    """Dataset with only train and validation splits (no test rows)."""
    rows = []
    split_sequence = ["train"] * 8 + ["validation"] * 4
    for index, split_name in enumerate(split_sequence):
        label = "Demented" if index % 2 else "Nondemented"
        rows.append(
            {
                "MRI_ID": f"OAS2_{index + 1:04d}_MR1",
                "Subject_ID": f"OAS2_{index + 1:04d}",
                "split": split_name,
                "Final_Group": label,
                "has_descriptors": True,
                "ventricle_area": 100.0 + index * 10.0,
                "ventricle_perimeter": 50.0 + index * 2.0,
                "ventricle_circularity": 0.75,
                "ventricle_eccentricity": 0.20,
                "mmse": 28,
                "cdr": 1.0 if label == "Demented" else 0.0,
                "age": 65.0 + index,
                "nwbv": 0.72,
                "etiv": 1500.0,
                "asf": 1.10,
                "sex": index % 2,
                "education": 14,
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.skipif(not classical_training.SKLEARN_AVAILABLE, reason="scikit-learn not available")
def test_train_svm_no_test_split(tmp_path, training_dataframe_no_test, monkeypatch):
    """SVM training succeeds even when there is no test split; no test metrics in result."""
    monkeypatch.setattr(classical_training, "GridSearchCV", _FastGridSearchCV)

    result = classical_training.train_svm_classifier(
        df=training_dataframe_no_test,
        features=["ventricle_area", "ventricle_perimeter", "age"],
        scenario="svm_no_test",
        output_dir=tmp_path,
    )

    assert result.model_name == "SVM"
    assert result.model_path.exists()
    assert result.scaler_path is not None and result.scaler_path.exists()
    assert "val_accuracy" in result.metrics
    assert "test_accuracy" not in result.metrics
    assert result.confusion_matrix is None


@pytest.mark.skipif(not classical_training.SKLEARN_AVAILABLE, reason="scikit-learn not available")
def test_train_svm_classifier_no_scenario_uses_default_paths(tmp_path, training_dataframe, monkeypatch):
    """When scenario=None, SVM saves to the plain default artifact names."""
    monkeypatch.setattr(classical_training, "GridSearchCV", _FastGridSearchCV)

    result = classical_training.train_svm_classifier(
        df=training_dataframe,
        features=["age"],
        scenario=None,
        output_dir=tmp_path,
    )

    assert result.model_path == tmp_path / "svm_model.pkl"
    assert result.scaler_path == tmp_path / "svm_scaler.pkl"


@pytest.mark.skipif(not classical_training.SKLEARN_AVAILABLE, reason="scikit-learn not available")
def test_train_svm_classifier_scenario_label_derives_from_features(tmp_path, training_dataframe, monkeypatch):
    """Scenario label is auto-derived based on whether mmse/cdr are in features."""
    monkeypatch.setattr(classical_training, "GridSearchCV", _FastGridSearchCV)

    result_with = classical_training.train_svm_classifier(
        df=training_dataframe,
        features=["age", "mmse"],
        scenario=None,
        output_dir=tmp_path,
    )
    result_without = classical_training.train_svm_classifier(
        df=training_dataframe,
        features=["age"],
        scenario=None,
        output_dir=tmp_path,
    )

    assert result_with.scenario == "svm_with_mmse_cdr"
    assert result_without.scenario == "svm_without_mmse_cdr"


@pytest.mark.skipif(not classical_training.SKLEARN_AVAILABLE, reason="scikit-learn not available")
def test_train_svm_classifier_missing_feature_raises(tmp_path, training_dataframe, monkeypatch):
    """Requesting a column that does not exist should raise ValueError."""
    monkeypatch.setattr(classical_training, "GridSearchCV", _FastGridSearchCV)

    with pytest.raises(ValueError, match="Colunas ausentes"):
        classical_training.train_svm_classifier(
            df=training_dataframe,
            features=["nonexistent_column"],
            scenario="svm_bad",
            output_dir=tmp_path,
        )


@pytest.mark.skipif(not classical_training.SKLEARN_AVAILABLE, reason="scikit-learn not available")
def test_train_svm_classifier_empty_validation_raises(tmp_path, training_dataframe, monkeypatch):
    """Empty validation split must raise ValueError, not silently train."""
    monkeypatch.setattr(classical_training, "GridSearchCV", _FastGridSearchCV)
    no_val_df = training_dataframe[training_dataframe["split"] != "validation"].copy()

    with pytest.raises(ValueError, match="validação"):
        classical_training.train_svm_classifier(
            df=no_val_df,
            features=["age"],
            scenario="svm_no_val",
            output_dir=tmp_path,
        )


@pytest.mark.skipif(not classical_training.SKLEARN_AVAILABLE, reason="scikit-learn not available")
def test_train_svm_sex_encoding_from_mf_column(tmp_path, training_dataframe, monkeypatch):
    """When 'sex' is not in df but 'M/F' is, the sex column must be derived."""
    monkeypatch.setattr(classical_training, "GridSearchCV", _FastGridSearchCV)
    df = training_dataframe.drop(columns=["sex"]).copy()
    df["M/F"] = ["M" if i % 2 else "F" for i in range(len(df))]

    result = classical_training.train_svm_classifier(
        df=df,
        features=["age", "sex"],
        scenario="svm_sex_mf",
        output_dir=tmp_path,
    )

    assert result.model_name == "SVM"
    assert result.model_path.exists()


@pytest.mark.skipif(not classical_training.SKLEARN_AVAILABLE, reason="scikit-learn not available")
def test_train_svm_metrics_are_floats(tmp_path, training_dataframe, monkeypatch):
    """All scalar metrics in result.metrics must be plain Python floats."""
    monkeypatch.setattr(classical_training, "GridSearchCV", _FastGridSearchCV)

    result = classical_training.train_svm_classifier(
        df=training_dataframe,
        features=["age"],
        scenario="svm_types",
        output_dir=tmp_path,
    )

    for key, value in result.metrics.items():
        if key == "best_params":
            continue
        assert isinstance(value, float), f"Expected float for metric '{key}', got {type(value)}"


@pytest.mark.skipif(
    not classical_training.SKLEARN_AVAILABLE or not classical_training.XGBOOST_AVAILABLE,
    reason="scikit-learn or xgboost not available",
)
def test_train_xgboost_regressor_default_seed(tmp_path, training_dataframe, monkeypatch):
    """When seed is None the function uses 42 by default and records it in metrics."""
    monkeypatch.setattr(classical_training, "GridSearchCV", _FastGridSearchCV)

    result = classical_training.train_xgboost_regressor(
        df=training_dataframe,
        features=["ventricle_area", "age", "education"],
        scenario="xgb_default_seed",
        output_dir=tmp_path,
        seed=None,
    )

    assert result.metrics["seed"] == 42


@pytest.mark.skipif(
    not classical_training.SKLEARN_AVAILABLE or not classical_training.XGBOOST_AVAILABLE,
    reason="scikit-learn or xgboost not available",
)
def test_train_xgboost_regressor_validation_metrics_present(tmp_path, training_dataframe, monkeypatch):
    """Validation regression metrics (mae, mse, rmse, r2) must all be present."""
    monkeypatch.setattr(classical_training, "GridSearchCV", _FastGridSearchCV)

    result = classical_training.train_xgboost_regressor(
        df=training_dataframe,
        features=["ventricle_area", "age"],
        scenario="xgb_val_metrics",
        output_dir=tmp_path,
        seed=0,
    )

    for metric_key in ("val_mae", "val_mse", "val_rmse", "val_r2"):
        assert metric_key in result.metrics, f"Missing metric: {metric_key}"
    assert result.metrics["val_rmse"] >= 0.0


@pytest.mark.skipif(
    not classical_training.SKLEARN_AVAILABLE or not classical_training.XGBOOST_AVAILABLE,
    reason="scikit-learn or xgboost not available",
)
def test_train_xgboost_regressor_empty_validation_raises(tmp_path, training_dataframe, monkeypatch):
    """Empty validation split must raise ValueError for XGBoost as well."""
    monkeypatch.setattr(classical_training, "GridSearchCV", _FastGridSearchCV)
    no_val_df = training_dataframe[training_dataframe["split"] != "validation"].copy()

    with pytest.raises(ValueError, match="validação"):
        classical_training.train_xgboost_regressor(
            df=no_val_df,
            features=["age"],
            scenario="xgb_no_val",
            output_dir=tmp_path,
            seed=1,
        )


def test_ensure_classical_dependencies_raises_without_sklearn(monkeypatch):
    """_ensure_classical_dependencies must raise ImportError when sklearn is absent."""
    monkeypatch.setattr(classical_training, "SKLEARN_AVAILABLE", False)
    monkeypatch.setattr(classical_training, "PANDAS_AVAILABLE", True)

    with pytest.raises(ImportError, match="scikit-learn"):
        classical_training._ensure_classical_dependencies()


def test_ensure_classical_dependencies_raises_without_pandas(monkeypatch):
    """_ensure_classical_dependencies must raise ImportError when pandas is absent."""
    monkeypatch.setattr(classical_training, "SKLEARN_AVAILABLE", True)
    monkeypatch.setattr(classical_training, "PANDAS_AVAILABLE", False)

    with pytest.raises(ImportError, match="pandas"):
        classical_training._ensure_classical_dependencies()


def test_ensure_classical_dependencies_raises_for_xgboost(monkeypatch):
    """_ensure_classical_dependencies must raise ImportError when xgboost required but absent."""
    monkeypatch.setattr(classical_training, "SKLEARN_AVAILABLE", True)
    monkeypatch.setattr(classical_training, "PANDAS_AVAILABLE", True)
    monkeypatch.setattr(classical_training, "XGBOOST_AVAILABLE", False)

    with pytest.raises(ImportError, match="xgboost"):
        classical_training._ensure_classical_dependencies(require_xgboost=True)


@pytest.mark.skipif(not classical_training.PANDAS_AVAILABLE, reason="pandas not available")
def test_dataframe_sha256_returns_string(training_dataframe):
    """_dataframe_sha256 must return a non-empty hex string for a normal dataframe."""
    sha = classical_training._dataframe_sha256(training_dataframe)
    assert isinstance(sha, str)
    assert len(sha) == 64
    # Verify it is hexadecimal
    int(sha, 16)


@pytest.mark.skipif(not classical_training.PANDAS_AVAILABLE, reason="pandas not available")
def test_dataframe_sha256_deterministic(training_dataframe):
    """_dataframe_sha256 must return the same hash for the same dataframe content."""
    sha1 = classical_training._dataframe_sha256(training_dataframe)
    sha2 = classical_training._dataframe_sha256(training_dataframe.copy())
    assert sha1 == sha2


@pytest.mark.skipif(not classical_training.PANDAS_AVAILABLE, reason="pandas not available")
def test_prepare_feature_frame_filters_by_has_descriptors(training_dataframe):
    """Rows with has_descriptors=False must be excluded when ventricle features requested."""
    df = training_dataframe.copy()
    # Mark half of train rows as not having descriptors
    df.loc[df.index[:3], "has_descriptors"] = False

    result = classical_training._prepare_feature_frame(df, features=["ventricle_area", "age"])

    # Only rows with has_descriptors=True should remain
    assert result["has_descriptors"].all()


@pytest.mark.skipif(not classical_training.PANDAS_AVAILABLE, reason="pandas not available")
def test_prepare_feature_frame_raises_on_missing_column(training_dataframe):
    """_prepare_feature_frame must raise ValueError for completely missing columns."""
    with pytest.raises(ValueError, match="Colunas ausentes"):
        classical_training._prepare_feature_frame(training_dataframe, features=["does_not_exist"])


def test_classical_training_result_is_frozen():
    """ClassicalTrainingResult is a frozen dataclass and must not allow mutation."""
    result = classical_training.ClassicalTrainingResult(
        model_name="SVM",
        scenario="test",
        message="ok",
        metrics={"best_params": {"C": 1.0}},
        model_path=Path("model.pkl"),
    )
    with pytest.raises((AttributeError, TypeError)):
        result.model_name = "changed"
    with pytest.raises(TypeError):
        result.metrics["new_metric"] = 1.0
    with pytest.raises(TypeError):
        result.metrics["best_params"]["C"] = 2.0
