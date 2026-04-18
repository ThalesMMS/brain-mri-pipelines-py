import argparse
import os
import sys
from pathlib import Path
from typing import Any


# Ensure brain_mri is in path
sys.path.append(os.getcwd())

from brain_mri.experiments.run_manifest import (
    capture_pip_freeze,
    generate_timestamp,
    git_commit,
    git_is_dirty,
    manifest_file as _manifest_file,
    relativize_command,
    unique_manifest_path as _unique_manifest_path,
    write_manifest,
)


def _dependencies_snapshot() -> list[str] | None:
    return capture_pip_freeze()


def _write_run_manifest(
    *,
    app: Any,
    args: argparse.Namespace,
    base_dir: Path,
    svm_with_result: Any,
    svm_without_result: Any,
    xgb_result: Any,
    evaluated_existing_xgb: bool,
) -> Path:
    timestamp = generate_timestamp()
    manifest_path = _unique_manifest_path(app.output_dir / "manifests" / "baselines", timestamp)

    xgboost_outputs: dict[str, Any] = {
        "trained_model": _manifest_file(getattr(xgb_result, "model_path", None), base_dir),
    }
    if args.xgb in ("eval-existing", "both"):
        xgboost_outputs["evaluated_existing_model"] = _manifest_file(
            app.output_dir / "xgb_age.pkl" if evaluated_existing_xgb else None,
            base_dir,
        )

    manifest = {
        "cli": "baselines",
        "timestamp": timestamp,
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "command": relativize_command(list(sys.argv), base_dir),
        "args": {
            "seed": int(args.seed),
            "xgb_mode": args.xgb,
        },
        "inputs": {
            "split_csv": _manifest_file(app.output_dir / "exam_level_dataset_split.csv", base_dir),
            "descriptors_csv": _manifest_file(app.descriptors_csv, base_dir),
            "demographic_csv": _manifest_file(app.csv_path, base_dir),
        },
        "outputs": {
            "dataset_stats_json": _manifest_file(app.output_dir / "dataset_stats.json", base_dir),
            "svm": {
                "with_mmse_cdr": {
                    "model": _manifest_file(getattr(svm_with_result, "model_path", None), base_dir),
                    "scaler": _manifest_file(getattr(svm_with_result, "scaler_path", None), base_dir),
                },
                "without_mmse_cdr": {
                    "model": _manifest_file(getattr(svm_without_result, "model_path", None), base_dir),
                    "scaler": _manifest_file(getattr(svm_without_result, "scaler_path", None), base_dir),
                },
            },
            "xgboost": xgboost_outputs,
            "training_experiments": _manifest_file(app.experiment_history_path, base_dir),
        },
        "dependencies": _dependencies_snapshot(),
    }
    write_manifest(manifest_path, manifest)
    return manifest_path


def _patch_messagebox():
    # Patch messageboxes used by MLTrainingMixin to make this script headless.
    from brain_mri.ml import ml_training

    def showinfo(title, message):
        print(f"[INFO] {title}: {message}")

    def showwarning(title, message):
        print(f"[WARN] {title}: {message}")

    def showerror(title, message):
        print(f"[ERROR] {title}: {message}")

    def askyesno(title, message):
        print(f"[ASK] {title}: {message} -> yes")
        return True

    ml_training.messagebox.showinfo = showinfo
    ml_training.messagebox.showwarning = showwarning
    ml_training.messagebox.showerror = showerror
    ml_training.messagebox.askyesno = askyesno


def _ensure_split_csv(app):
    split_csv = app.output_dir / "exam_level_dataset_split.csv"
    if split_csv.exists():
        # Preserva histórico antes de regenerar.
        import time
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup = app.output_dir / f"exam_level_dataset_split_prev_{ts}.csv"
        try:
            backup.write_bytes(split_csv.read_bytes())
            print(f"[INFO] Backed up existing split CSV to: {backup}")
        except Exception:
            pass
    print(f"[INFO] Regenerating split CSV: {split_csv}")
    app.create_exam_level_dataset()


def _write_dataset_stats(app):
    """Gera um resumo reprodutível do dataset/splits para uso no artigo."""
    try:
        import json
        import pandas as pd
    except Exception:
        return

    split_csv = app.output_dir / "exam_level_dataset_split.csv"
    if not split_csv.exists():
        return

    df = pd.read_csv(split_csv)
    stats = {
        "split_csv": str(split_csv.relative_to(app.output_dir.parent)),
        "total_exams": int(len(df)),
        "total_subjects": int(df['Subject_ID'].nunique()) if 'Subject_ID' in df.columns else None,
        "exams_by_split": df['split'].value_counts().to_dict() if 'split' in df.columns else None,
        "subjects_by_split": df.groupby('split')['Subject_ID'].nunique().to_dict() if 'split' in df.columns and 'Subject_ID' in df.columns else None,
    }

    # Cobertura por orientação (se as colunas existirem)
    for col in ("has_axl", "has_cor", "has_sag"):
        if col in df.columns:
            stats[col] = int(df[col].sum())

    if all(c in df.columns for c in ("has_axl", "has_cor", "has_sag")):
        combo = df[["has_axl", "has_cor", "has_sag"]].astype(int).astype(str).agg(''.join, axis=1)
        stats["orientation_combo_counts"] = combo.value_counts().to_dict()

    out = app.output_dir / "dataset_stats.json"
    out.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    print(f"[OK] Wrote dataset stats: {out}")


def _evaluate_existing_xgboost(app, features):
    """Avalia um modelo XGBoost existente (se houver) e registra métricas de teste no histórico."""
    if not app.output_dir.exists():
        return False

    model_path = app.output_dir / "xgb_age.pkl"
    split_csv = app.output_dir / "exam_level_dataset_split.csv"
    if not model_path.exists() or not split_csv.exists():
        return False

    try:
        import pickle
        import numpy as np
        import pandas as pd
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    except Exception:
        return False

    df = pd.read_csv(split_csv)
    if 'has_descriptors' in df.columns and any(str(f).startswith('ventricle_') for f in features):
        df = df[df['has_descriptors'] == True]

    test_mask = df['split'] == 'test'
    train_mask = df['split'] == 'train'
    if not test_mask.any() or not train_mask.any():
        return False

    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"[WARN] XGBoost eval: missing columns: {missing}")
        return False

    X_df = df[features].copy()
    train_means = X_df.loc[train_mask].mean(numeric_only=True).fillna(0.0)
    X_df = X_df.fillna(train_means).fillna(0.0)
    X = X_df.values
    y = df['age'].values

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    preds = model.predict(X[test_mask])
    test_mae = mean_absolute_error(y[test_mask], preds)
    test_mse = mean_squared_error(y[test_mask], preds)
    test_rmse = float(np.sqrt(test_mse))
    test_r2 = r2_score(y[test_mask], preds)

    app._save_experiment({
        'model': 'XGBoost',
        'scenario': 'xgb_test_eval_existing',
        'target': 'age',
        'features': features,
        'test_mae': float(test_mae),
        'test_mse': float(test_mse),
        'test_rmse': float(test_rmse),
        'test_r2': float(test_r2),
        'source_model_path': str(model_path.relative_to(app.output_dir.parent)),
    })
    print("[OK] Logged XGBoost test metrics from existing model.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run baselines headlessly and update artifacts.")
    parser.add_argument(
        "--xgb",
        choices=("train", "eval-existing", "both"),
        default="train",
        help="XGBoost mode: train from scratch (default), evaluate existing model, or both.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.getenv("XGB_SEED", "42")),
        help="Seed for deterministic XGBoost training (also exported to XGB_SEED).",
    )
    args = parser.parse_args()

    base_dir = Path(os.getcwd())
    print(f"Running baselines headlessly in {base_dir}")

    _patch_messagebox()

    from brain_mri.experiments.history import ExperimentHistoryMixin
    from brain_mri.ml.ml_training import MLTrainingMixin

    class HeadlessApp(ExperimentHistoryMixin, MLTrainingMixin):
        def __init__(self, base_dir: Path):
            self.base_dir = base_dir
            self.dataset_dir = self.base_dir / "axl"
            self.output_dir = self.base_dir / "output"
            self.output_dir.mkdir(exist_ok=True)
            self.descriptors_csv = self.output_dir / "ventricle_descriptors.csv"
            self.csv_path = self.base_dir / "oasis_longitudinal_demographic.csv"
            self.experiment_history_path = self.output_dir / "training_experiments.json"

    app = HeadlessApp(base_dir)

    _ensure_split_csv(app)
    _write_dataset_stats(app)

    # SVM: cenário histórico (com MMSE/CDR)
    svm_features_with = [
        'ventricle_area',
        'ventricle_perimeter',
        'ventricle_circularity',
        'ventricle_eccentricity',
        'mmse',
        'cdr',
        'age',
    ]
    print("\n--- SVM (with MMSE/CDR) ---")
    svm_with_result = app.train_svm_classifier(features=svm_features_with, scenario="svm_with_mmse_cdr")

    # SVM: cenário metodologicamente consistente (sem MMSE/CDR)
    svm_features_without = [
        'ventricle_area',
        'ventricle_perimeter',
        'ventricle_circularity',
        'ventricle_eccentricity',
        'age',
    ]
    print("\n--- SVM (without MMSE/CDR) ---")
    svm_without_result = app.train_svm_classifier(features=svm_features_without, scenario="svm_without_mmse_cdr")

    # XGBoost: tenta avaliar modelo existente (rápido); se não houver, treina e registra.
    xgb_features = [
        'ventricle_area',
        'ventricle_perimeter',
        'ventricle_circularity',
        'ventricle_eccentricity',
        'mmse',
        'cdr',
        'nwbv',
        'etiv',
        'asf',
        'sex',
        'education',
    ]
    print("\n--- XGBoost (age) ---")
    os.environ["XGB_SEED"] = str(args.seed)

    xgb_result = None
    if args.xgb in ("train", "both"):
        # Treino reprodutível no split atual: registra val_* e test_* no mesmo experimento.
        xgb_result = app.train_xgboost_regressor(
            features=xgb_features,
            scenario="xgb_train_and_test_current_split",
            seed=args.seed,
        )

    evaluated_existing_xgb = False
    if args.xgb in ("eval-existing", "both"):
        # Avaliação de um modelo serializado existente (histórico).
        evaluated_existing_xgb = _evaluate_existing_xgboost(app, xgb_features)

    manifest_path = _write_run_manifest(
        app=app,
        args=args,
        base_dir=base_dir,
        svm_with_result=svm_with_result,
        svm_without_result=svm_without_result,
        xgb_result=xgb_result,
        evaluated_existing_xgb=evaluated_existing_xgb,
    )
    print(f"[OK] Wrote run manifest: {manifest_path}")


if __name__ == "__main__":
    main()
