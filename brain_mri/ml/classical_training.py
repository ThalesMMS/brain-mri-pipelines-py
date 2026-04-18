import hashlib
import pickle
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None
    XGBOOST_AVAILABLE = False

try:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        r2_score,
        recall_score,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    SKLEARN_AVAILABLE = True
except ImportError:
    accuracy_score = confusion_matrix = f1_score = mean_absolute_error = mean_squared_error = None
    precision_score = r2_score = recall_score = None
    GridSearchCV = GroupKFold = Pipeline = SimpleImputer = StandardScaler = SVC = None
    SKLEARN_AVAILABLE = False

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None

from .training_utils import build_artifact_path, ensure_required_columns


DEFAULT_SVM_FEATURES = [
    "ventricle_area",
    "ventricle_perimeter",
    "ventricle_circularity",
    "ventricle_eccentricity",
    "mmse",
    "cdr",
    "age",
]

DEFAULT_XGB_FEATURES = [
    "ventricle_area",
    "ventricle_perimeter",
    "ventricle_circularity",
    "ventricle_eccentricity",
    "mmse",
    "cdr",
    "nwbv",
    "etiv",
    "asf",
    "sex",
    "education",
]


@dataclass(frozen=True)
class ClassicalTrainingResult:
    model_name: str
    scenario: str
    message: str
    metrics: Mapping[str, Any]
    model_path: Path
    scaler_path: Path | None = None
    confusion_matrix: tuple[tuple[Any, ...], ...] | np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType({key: _freeze_result_value(value) for key, value in self.metrics.items()}))
        if self.confusion_matrix is not None:
            object.__setattr__(self, "confusion_matrix", _freeze_result_value(self.confusion_matrix))


def _freeze_result_value(value):
    if isinstance(value, np.ndarray):
        frozen = np.array(value, copy=True)
        frozen.flags.writeable = False
        return frozen
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_result_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_result_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_result_value(item) for item in value)
    return value


def _ensure_classical_dependencies(*, require_xgboost: bool = False) -> None:
    """
    Verify runtime dependencies required for classical training routines.
    
    If `require_xgboost` is True, also require the XGBoost package.
    
    Parameters:
        require_xgboost (bool): When True, check that XGBoost is available in addition to scikit-learn and pandas.
    
    Raises:
        ImportError: If scikit-learn or pandas are not available, or if `require_xgboost` is True and xgboost is not available.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "O módulo 'scikit-learn' é necessário para este treino.\n"
            "Instale com 'pip install scikit-learn'."
        )
    if require_xgboost and not XGBOOST_AVAILABLE:
        raise ImportError(
            "O módulo 'xgboost' é necessário para este treino.\n"
            "Instale com 'pip install xgboost'."
        )
    if not PANDAS_AVAILABLE:
        raise ImportError(
            "O módulo 'pandas' é necessário para este treino.\n"
            "Instale com 'pip install pandas'."
        )


def _prepare_feature_frame(df, features):
    """
    Prepare and return a dataframe containing the requested feature columns and any row filtering required by those features.
    
    This function ensures the input dataframe contains a "split" column, filters out rows that lack ventricle descriptors when any requested feature name starts with "ventricle_", and synthesizes a numeric "sex" column from an "M/F" column when "sex" is requested but absent. It validates that all requested feature columns are present in the returned dataframe.
    
    Parameters:
        df (pandas.DataFrame): Input dataset containing at least a "split" column.
        features (Iterable[str]): Sequence of feature column names required for downstream training.
    
    Returns:
        pandas.DataFrame: A copy of the input dataframe filtered/augmented to include the requested features.
    
    Raises:
        ValueError: If any requested feature names are missing from the prepared dataframe.
    """
    ensure_required_columns(df, ["split"], context="training dataframe")
    tmp = df.copy()
    uses_descriptors = any(str(feature).startswith("ventricle_") for feature in features)
    if uses_descriptors:
        if "has_descriptors" in tmp.columns:
            tmp = tmp[tmp["has_descriptors"]]
        else:
            descriptor_columns = [column for column in tmp.columns if column.startswith("ventricle_")]
            if descriptor_columns:
                tmp = tmp[tmp[descriptor_columns].notna().any(axis=1)]

    if "sex" in features and "sex" not in tmp.columns:
        if "M/F" in tmp.columns:
            tmp["sex"] = tmp["M/F"].map({"M": 0, "F": 1})
        else:
            tmp["sex"] = np.nan

    missing = [feature for feature in features if feature not in tmp.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no dataset: {missing}")

    return tmp


def _dataframe_sha256(df) -> str | None:
    """
    Compute a SHA-256 checksum representing the DataFrame's contents.
    
    The checksum is produced by converting the DataFrame to CSV with columns sorted by name (no index) and hashing the UTF-8 bytes. If pandas is not available or an error occurs during serialization or hashing, returns `None`.
    
    Parameters:
        df (pandas.DataFrame): The DataFrame to hash.
    
    Returns:
        str | None: Hex-encoded SHA-256 digest of the CSV serialization when successful, or `None` if pandas is unavailable or hashing fails.
    """
    if not PANDAS_AVAILABLE:
        return None
    try:
        normalized = df.sort_index(axis=1)
        payload = normalized.to_csv(index=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
    except (AttributeError, TypeError, ValueError, KeyError):
        return None


def train_svm_classifier(df, features=None, scenario=None, output_dir=None, save_experiment_fn=None):
    """
    Train an SVM classifier to predict dementia using the provided dataset and feature set.
    
    Parameters:
        df (pandas.DataFrame): Input dataset containing at minimum the columns required for training ('Final_Group' and 'split'); additional required feature columns are validated and may be synthesized (e.g., `sex`) by the preparer.
        features (Iterable[str] | None): Ordered collection of feature column names to use; if None the module default feature list is used.
        scenario (str | None): Optional label used to name saved artifacts; when omitted a default scenario name is chosen based on whether `mmse` or `cdr` are included in `features`.
        output_dir (str | pathlib.Path): Directory where scaler and model pickle files will be written; created if necessary. Must not be None.
        save_experiment_fn (Callable[[dict], Any] | None): Optional callback invoked with a dictionary summarizing the experiment (model, scenario, features, best parameters, accuracies, training time and test metrics/confusion matrix when available).
    
    Returns:
        ClassicalTrainingResult: Immutable summary of the training run containing model name, scenario label, human-readable message, metrics dictionary, paths to saved model and scaler artifacts, and the test confusion matrix when a test split was present.
    """
    _ensure_classical_dependencies()
    ensure_required_columns(df, ["Final_Group", "Subject_ID"], context="training dataframe")
    if output_dir is None:
        raise ValueError("output_dir must be a path, got None")
    output_dir = Path(output_dir)

    start_time = time.time()
    features = list(features or DEFAULT_SVM_FEATURES)
    uses_mmse = any(str(feature).lower() == "mmse" for feature in features)
    uses_cdr = any(str(feature).lower() == "cdr" for feature in features)
    scenario_label = scenario or ("svm_with_mmse_cdr" if (uses_mmse or uses_cdr) else "svm_without_mmse_cdr")

    tmp = _prepare_feature_frame(df, features)
    train_mask = tmp["split"] == "train"
    val_mask = tmp["split"] == "validation"
    test_mask = tmp["split"] == "test"
    if not val_mask.any():
        raise ValueError("Split de validação vazio.")
    if not train_mask.any():
        raise ValueError("Split de treino vazio após preparar features.")

    x = tmp[features].copy().values
    y = (tmp["Final_Group"] == "Demented").astype(int).values

    x_train = x[train_mask]
    x_val = x[val_mask]
    x_test = x[test_mask] if test_mask.any() else None

    grid = {
        "svc__C": [0.1, 1, 10, 100],
        "svc__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        "svc__kernel": ["rbf", "linear"],
    }
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("svc", SVC()),
        ]
    )
    train_labels = y[train_mask]
    groups_train = tmp.loc[train_mask, "Subject_ID"].values
    train_subject_labels = tmp.loc[train_mask, ["Subject_ID", "Final_Group"]].drop_duplicates()
    class_subject_counts = train_subject_labels.groupby("Final_Group")["Subject_ID"].nunique()
    if len(class_subject_counts) < 2:
        raise ValueError("SVM grid search requires at least 2 distinct training classes for cross-validation.")
    min_class_subjects = int(class_subject_counts.min()) if not class_subject_counts.empty else 0
    if min_class_subjects < 2:
        raise ValueError("SVM training requires at least 2 training subjects per class for cross-validation.")
    unique_group_count = int(tmp.loc[train_mask, "Subject_ID"].nunique(dropna=True))
    if unique_group_count < 2:
        raise ValueError("SVM training requires at least 2 training subjects for GroupKFold.")
    n_splits = min(3, min_class_subjects, unique_group_count)
    if StratifiedGroupKFold is None:
        raise ValueError("SVM training requires scikit-learn with StratifiedGroupKFold support.")
    cv_obj = StratifiedGroupKFold(n_splits=n_splits)
    gs = GridSearchCV(pipeline, grid, cv=cv_obj, scoring="accuracy", n_jobs=-1, verbose=1)
    gs.fit(x_train, train_labels, groups=groups_train)
    clf = gs.best_estimator_

    y_train_pred = clf.predict(x_train)
    y_val_pred = clf.predict(x_val)
    acc_tr = accuracy_score(y[train_mask], y_train_pred)
    acc_val = accuracy_score(y[val_mask], y_val_pred)

    test_cm = None
    metrics = {
        "train_accuracy": float(acc_tr),
        "val_accuracy": float(acc_val),
        "best_params": gs.best_params_,
    }
    message = (
        f"Acurácia (Treino): {acc_tr:.2%}\n"
        f"Acurácia (Val): {acc_val:.2%}\n"
        f"Melhor: {gs.best_params_}"
    )
    if x_test is not None:
        y_test_pred = clf.predict(x_test)
        acc_test = accuracy_score(y[test_mask], y_test_pred)
        test_cm = confusion_matrix(y[test_mask], y_test_pred)
        test_precision = precision_score(y[test_mask], y_test_pred, average="binary", zero_division=0)
        test_recall = recall_score(y[test_mask], y_test_pred, average="binary", zero_division=0)
        test_f1 = f1_score(y[test_mask], y_test_pred, average="binary", zero_division=0)
        metrics.update(
            {
                "test_accuracy": float(acc_test),
                "test_precision": float(test_precision),
                "test_recall": float(test_recall),
                "test_f1": float(test_f1),
            }
        )
        message += (
            f"\n\n=== TESTE ===\nAcurácia: {acc_test:.2%}\n"
            f"Precision: {test_precision:.2%}\n"
            f"Recall: {test_recall:.2%}\n"
            f"F1-Score: {test_f1:.2%}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if scenario is None:
        scaler_path = output_dir / "svm_scaler.pkl"
        model_path = output_dir / "svm_model.pkl"
    else:
        scaler_path = build_artifact_path(output_dir, "svm_scaler.pkl", scenario_label)
        model_path = build_artifact_path(output_dir, "svm_model.pkl", scenario_label)

    with open(scaler_path, "wb") as file_obj:
        pickle.dump(clf.named_steps["scaler"], file_obj)
    with open(model_path, "wb") as file_obj:
        pickle.dump(clf, file_obj)

    training_time = time.time() - start_time
    exp_data = {
        "model": "SVM",
        "scenario": scenario_label,
        "features": features,
        "best_params": gs.best_params_,
        "train_accuracy": float(acc_tr),
        "val_accuracy": float(acc_val),
        "training_time_seconds": float(training_time),
    }
    if test_cm is not None:
        exp_data.update({k: metrics[k] for k in ("test_accuracy", "test_precision", "test_recall", "test_f1")})
        exp_data["test_confusion_matrix"] = test_cm.tolist()
    if save_experiment_fn is not None:
        save_experiment_fn(exp_data)

    return ClassicalTrainingResult(
        model_name="SVM",
        scenario=scenario_label,
        message=message,
        metrics=metrics,
        model_path=model_path,
        scaler_path=scaler_path,
        confusion_matrix=test_cm,
    )


def train_xgboost_regressor(df, features=None, scenario=None, output_dir=None, seed=None, save_experiment_fn=None):
    """
    Train an XGBoost regressor to predict subject age using group-aware cross-validation and optional test evaluation.
    
    This function validates inputs and dependencies, seeds Python and NumPy RNGs, prepares the feature frame (including imputation of missing values using training-set means), and performs a GroupKFold-based GridSearchCV (scoring by negative mean absolute error) over a predefined hyperparameter grid. It computes validation MAE, MSE, RMSE, and R²; when a test split is present, it also computes the same metrics on the test set. The fitted model is saved as a pickle in output_dir (the filename incorporates the sanitized scenario label). An experiment payload containing metrics, best parameters, training time, seed, and a SHA-256 of the input dataframe may be passed to save_experiment_fn if provided. Returns a ClassicalTrainingResult summarizing the training run.
    
    Parameters:
        df: DataFrame-like
            Dataset containing at minimum the columns "split", "Subject_ID", and "age".
        features: Iterable[str] | None
            Ordered list of feature column names to use; defaults to DEFAULT_XGB_FEATURES when None.
        scenario: str | None
            Optional scenario label used when naming saved artifacts; defaults to "xgb_train_and_test_current_split".
        output_dir: path-like
            Directory where the trained model artifact will be written; created if it does not exist. Must not be None.
        seed: int | None
            RNG seed for reproducibility; defaults to 42 when None.
        save_experiment_fn: Callable[[dict], Any] | None
            Optional callback invoked with an experiment payload dictionary after training.
    
    Returns:
        ClassicalTrainingResult:
            Immutable summary of the training run containing model_name ("XGBoost"), scenario label, human-readable message, metrics dictionary, and the path to the saved model artifact.
    
    Raises:
        ImportError: If required runtime dependencies (pandas, scikit-learn, or xgboost) are missing.
        ValueError: If the validation split is empty or if fewer than 2 unique training subjects exist for GroupKFold.
    """
    _ensure_classical_dependencies(require_xgboost=True)
    ensure_required_columns(df, ["split", "Subject_ID", "age"], context="training dataframe")
    if output_dir is None:
        raise ValueError("output_dir must be a path, got None")
    output_dir = Path(output_dir)

    if seed is None:
        seed = 42
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)

    start_time = time.time()
    features = [feature for feature in list(features or DEFAULT_XGB_FEATURES) if feature != "age"]
    scenario_label = scenario or "xgb_train_and_test_current_split"
    tmp = _prepare_feature_frame(df, features)

    train_mask = tmp["split"] == "train"
    val_mask = tmp["split"] == "validation"
    test_mask = tmp["split"] == "test"
    if not val_mask.any():
        raise ValueError("Split de validação vazio.")
    if not train_mask.any():
        raise ValueError("Split de treino vazio.")
    if not features:
        raise ValueError("XGBoost training requires at least one non-target feature.")

    x = tmp[features].copy().values
    y = tmp["age"].values
    groups = tmp.loc[train_mask, "Subject_ID"]
    unique_group_count = int(groups.nunique(dropna=True))
    if unique_group_count < 2:
        raise ValueError("XGBoost training requires at least 2 training subjects for GroupKFold.")

    base = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=1,
        verbosity=0,
        random_state=seed,
    )
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("xgb", base),
        ]
    )
    grid = {
        "xgb__n_estimators": [200, 300, 500],
        "xgb__max_depth": [6, 8, 10],
        "xgb__learning_rate": [0.05, 0.1, 0.15],
        "xgb__min_child_weight": [1, 3, 5],
        "xgb__subsample": [0.8, 0.9],
        "xgb__colsample_bytree": [0.8, 0.9],
    }

    n_splits = min(3, max(2, unique_group_count))
    gkf = GroupKFold(n_splits=n_splits)
    gs = GridSearchCV(
        pipeline,
        grid,
        cv=gkf,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(x[train_mask], y[train_mask], groups=groups)
    model = gs.best_estimator_

    val_preds = model.predict(x[val_mask])
    mae_val = mean_absolute_error(y[val_mask], val_preds)
    r2_val = r2_score(y[val_mask], val_preds)
    mse_val = mean_squared_error(y[val_mask], val_preds)
    rmse_val = float(np.sqrt(mse_val))

    metrics = {
        "val_mae": float(mae_val),
        "val_mse": float(mse_val),
        "val_rmse": float(rmse_val),
        "val_r2": float(r2_val),
        "best_params": gs.best_params_,
        "seed": seed,
    }
    test_mae = test_mse = test_rmse = test_r2 = None
    if test_mask.any():
        test_preds = model.predict(x[test_mask])
        test_mae = mean_absolute_error(y[test_mask], test_preds)
        test_mse = mean_squared_error(y[test_mask], test_preds)
        test_rmse = float(np.sqrt(test_mse))
        test_r2 = r2_score(y[test_mask], test_preds)
        metrics.update(
            {
                "test_mae": float(test_mae),
                "test_mse": float(test_mse),
                "test_rmse": float(test_rmse),
                "test_r2": float(test_r2),
            }
        )

    message = (
        f"Val MAE={mae_val:.2f} | Val RMSE={rmse_val:.2f} | "
        f"Val MSE={mse_val:.2f} | Val R²={r2_val:.4f}"
    )
    if test_mae is not None:
        message += (
            f"\nTest MAE={test_mae:.2f} | Test RMSE={test_rmse:.2f} | "
            f"Test MSE={test_mse:.2f} | Test R²={test_r2:.4f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = build_artifact_path(output_dir, "xgb_age.pkl", scenario_label)
    with open(model_path, "wb") as file_obj:
        pickle.dump(model, file_obj)

    training_time = time.time() - start_time
    exp_payload = {
        "model": "XGBoost",
        "scenario": scenario_label,
        "target": "age",
        "features": features,
        "val_mae": float(mae_val),
        "val_mse": float(mse_val),
        "val_rmse": float(rmse_val),
        "val_r2": float(r2_val),
        "best_params": gs.best_params_,
        "training_time_seconds": float(training_time),
        "seed": seed,
        "split_csv_sha256": _dataframe_sha256(df),
    }
    if test_mae is not None:
        exp_payload.update(
            {
                "test_mae": float(test_mae),
                "test_mse": float(test_mse),
                "test_rmse": float(test_rmse),
                "test_r2": float(test_r2),
            }
        )
    if save_experiment_fn is not None:
        save_experiment_fn(exp_payload)

    return ClassicalTrainingResult(
        model_name="XGBoost",
        scenario=scenario_label,
        message=message,
        metrics=metrics,
        model_path=model_path,
    )
