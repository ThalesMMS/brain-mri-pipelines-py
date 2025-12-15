"""PC1 (Etapa 1) — Embeddings DL vs tradicionais + método não-profundo.

Este script gera evidências reprodutíveis para o PC1 sem execução por células:

- Embeddings DL: consome CSVs exportados pelo pipeline (ex.: EfficientNet).
- Embeddings tradicionais: descritores ventriculares (e covariáveis clínicas simples) via CSV.
- Método não-profundo: regressão logística (class_weight='balanced') em cada espaço.

Saídas (pequenas, versionáveis):

- output/etapa1/metrics.csv
- output/etapa1/manifest.json
- output/etapa1/plots/pca_dl_vs_trad_test.png
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class _Paths:
    split_csv: Path
    descriptors_csv: Path
    dl_train_csv: Path
    dl_val_csv: Path
    dl_test_csv: Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_split(split_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(split_csv)
    required = {"MRI_ID", "split", "Final_Group"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"split CSV missing columns: {sorted(missing)}")
    df = df[["MRI_ID", "split", "Final_Group"]].copy()
    df["Final_Group"] = df["Final_Group"].astype(str)
    return df


def _label_to_int(series: pd.Series) -> np.ndarray:
    mapping = {"Nondemented": 0, "Demented": 1}
    try:
        return series.map(mapping).astype(int).to_numpy()
    except Exception as e:
        raise ValueError(f"Unexpected labels in Final_Group: {sorted(series.unique())}") from e


def _load_dl_embeddings(paths: _Paths, split_df: pd.DataFrame) -> pd.DataFrame:
    dfs = []
    for p in (paths.dl_train_csv, paths.dl_val_csv, paths.dl_test_csv):
        df = pd.read_csv(p)
        if "MRI_ID" not in df.columns:
            raise ValueError(f"DL embeddings CSV missing MRI_ID: {p}")
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)

    # Keep the first occurrence per MRI_ID (train/val/test files are disjoint in normal runs).
    all_df = all_df.drop_duplicates(subset=["MRI_ID"], keep="first")
    merged = all_df.merge(split_df, on="MRI_ID", how="inner")
    return merged


def _load_traditional_features(descriptors_csv: Path, split_df: pd.DataFrame) -> pd.DataFrame:
    desc = pd.read_csv(descriptors_csv)
    if "MRI_ID" not in desc.columns:
        raise ValueError(f"Descriptors CSV missing MRI_ID: {descriptors_csv}")

    # Add a small set of demographic covariates if available in split_df (already merged in split csv).
    # For reproducibility and simplicity, only use what's present after merging with the full split CSV.
    full_split = pd.read_csv(Path("output") / "exam_level_dataset_split.csv")
    demo_cols = [c for c in ["age", "education", "etiv", "nwbv", "asf", "sex"] if c in full_split.columns]
    demo = full_split[["MRI_ID", *demo_cols]].copy() if demo_cols else full_split[["MRI_ID"]].copy()

    merged = desc.merge(demo, on="MRI_ID", how="left").merge(split_df, on="MRI_ID", how="inner")
    return merged


def _select_numeric_features(df: pd.DataFrame, *, drop: set[str]) -> tuple[np.ndarray, list[str]]:
    cols = [c for c in df.columns if c not in drop]
    numeric_cols: list[str] = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
    if not numeric_cols:
        raise ValueError("No numeric feature columns found")

    X = df[numeric_cols].copy()
    # Simple, deterministic imputation.
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    return X.to_numpy(dtype=np.float32), numeric_cols


def _fit_eval_logreg(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }


def _pca_plot(
    *,
    out_path: Path,
    X_dl: np.ndarray,
    y_dl: np.ndarray,
    X_trad: np.ndarray,
    y_trad: np.ndarray,
    title_dl: str,
    title_trad: str,
) -> None:
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    def _proj(X: np.ndarray) -> np.ndarray:
        Xs = StandardScaler().fit_transform(X.astype(np.float64, copy=False))
        Xs[~np.isfinite(Xs)] = 0.0
        # Keep projections numerically stable for plotting.
        Xs = np.clip(Xs, -10.0, 10.0)
        return PCA(n_components=2, svd_solver="full", random_state=42).fit_transform(Xs)

    P_dl = _proj(X_dl)
    P_tr = _proj(X_trad)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=130)
    for ax, P, y, ttl in [
        (axes[0], P_dl, y_dl, title_dl),
        (axes[1], P_tr, y_trad, title_trad),
    ]:
        colors = np.where(y == 1, "tab:red", "tab:blue")
        ax.scatter(P[:, 0], P[:, 1], c=colors, alpha=0.75, s=28, edgecolors="none")
        ax.set_title(ttl)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.25)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label="Nondemented", markerfacecolor="tab:blue", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label="Demented", markerfacecolor="tab:red", markersize=8),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PC1 embeddings analysis (DL vs traditional) and export evidence.")
    parser.add_argument(
        "--dl-backbone",
        type=str,
        default="efficientnet",
        choices=["efficientnet", "medicalnet", "densenet"],
        help="Which exported DL embeddings to use (expects output/<name>_embeddings_classification_{split}.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Root output directory.",
    )
    args = parser.parse_args()

    out_dir: Path = args.output_dir
    etapa_dir = out_dir / "etapa1"
    plots_dir = etapa_dir / "plots"

    split_csv = out_dir / "exam_level_dataset_split.csv"
    descriptors_csv = out_dir / "ventricle_descriptors.csv"

    prefix = f"{args.dl_backbone}_embeddings_classification"
    dl_train_csv = out_dir / f"{prefix}_train.csv"
    dl_val_csv = out_dir / f"{prefix}_val.csv"
    dl_test_csv = out_dir / f"{prefix}_test.csv"

    for p in [split_csv, descriptors_csv, dl_train_csv, dl_val_csv, dl_test_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    split_df = _load_split(split_csv)
    dl_df = _load_dl_embeddings(
        _Paths(
            split_csv=split_csv,
            descriptors_csv=descriptors_csv,
            dl_train_csv=dl_train_csv,
            dl_val_csv=dl_val_csv,
            dl_test_csv=dl_test_csv,
        ),
        split_df,
    )
    trad_df = _load_traditional_features(descriptors_csv, split_df)

    # Intersect to compare on the same set of exams.
    common_ids = set(dl_df["MRI_ID"]).intersection(set(trad_df["MRI_ID"]))
    dl_df = dl_df[dl_df["MRI_ID"].isin(common_ids)].copy()
    trad_df = trad_df[trad_df["MRI_ID"].isin(common_ids)].copy()

    y_dl = _label_to_int(dl_df["Final_Group"])
    y_tr = _label_to_int(trad_df["Final_Group"])

    dl_train = dl_df[dl_df["split"] == "train"]
    dl_test = dl_df[dl_df["split"] == "test"]
    tr_train = trad_df[trad_df["split"] == "train"]
    tr_test = trad_df[trad_df["split"] == "test"]

    X_dl_train, dl_cols = _select_numeric_features(dl_train, drop={"MRI_ID", "target", "split", "Final_Group"})
    X_dl_test, _ = _select_numeric_features(dl_test, drop={"MRI_ID", "target", "split", "Final_Group"})
    y_dl_train = _label_to_int(dl_train["Final_Group"])
    y_dl_test = _label_to_int(dl_test["Final_Group"])

    X_tr_train, tr_cols = _select_numeric_features(tr_train, drop={"MRI_ID", "Subject_ID", "segmented_path", "split", "Final_Group"})
    X_tr_test, _ = _select_numeric_features(tr_test, drop={"MRI_ID", "Subject_ID", "segmented_path", "split", "Final_Group"})
    y_tr_train = _label_to_int(tr_train["Final_Group"])
    y_tr_test = _label_to_int(tr_test["Final_Group"])

    dl_metrics = _fit_eval_logreg(X_dl_train, y_dl_train, X_dl_test, y_dl_test)
    tr_metrics = _fit_eval_logreg(X_tr_train, y_tr_train, X_tr_test, y_tr_test)

    etapa_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = etapa_dir / "metrics.csv"
    rows = [
        {
            "run_id": "pc1",
            "embedding_space": "dl",
            "dl_backbone": args.dl_backbone,
            "model": "logreg_balanced",
            "train_rows": int(X_dl_train.shape[0]),
            "test_rows": int(X_dl_test.shape[0]),
            **{k: dl_metrics[k] for k in ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]},
            "notes": f"DL cols={len(dl_cols)}",
        },
        {
            "run_id": "pc1",
            "embedding_space": "traditional",
            "dl_backbone": args.dl_backbone,
            "model": "logreg_balanced",
            "train_rows": int(X_tr_train.shape[0]),
            "test_rows": int(X_tr_test.shape[0]),
            **{k: tr_metrics[k] for k in ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]},
            "notes": f"traditional cols={len(tr_cols)}",
        },
    ]
    pd.DataFrame(rows).to_csv(metrics_path, index=False)

    plot_path = plots_dir / "pca_dl_vs_trad_test.png"
    _pca_plot(
        out_path=plot_path,
        X_dl=X_dl_test,
        y_dl=y_dl_test,
        X_trad=X_tr_test,
        y_trad=y_tr_test,
        title_dl=f"DL ({args.dl_backbone}) — PCA (teste)",
        title_trad="Tradicional (descritores) — PCA (teste)",
    )

    manifest = {
        "pc": "PC1",
        "dl_backbone": args.dl_backbone,
        "canonical_inputs": {
            "split_csv": {"path": str(split_csv), "sha256": _sha256(split_csv)},
            "dataset_stats_json": {"path": str(out_dir / "dataset_stats.json"), "sha256": _sha256(out_dir / "dataset_stats.json")},
            "descriptors_csv": {"path": str(descriptors_csv), "sha256": _sha256(descriptors_csv)},
            "dl_train_csv": {"path": str(dl_train_csv), "sha256": _sha256(dl_train_csv)},
            "dl_val_csv": {"path": str(dl_val_csv), "sha256": _sha256(dl_val_csv)},
            "dl_test_csv": {"path": str(dl_test_csv), "sha256": _sha256(dl_test_csv)},
        },
        "outputs": {
            "metrics_csv": {"path": str(metrics_path), "sha256": _sha256(metrics_path)},
            "plot": {"path": str(plot_path), "sha256": _sha256(plot_path)},
        },
        "notes": {
            "comparison_intersection": int(len(common_ids)),
            "dl_confusion_matrix": dl_metrics["confusion_matrix"],
            "traditional_confusion_matrix": tr_metrics["confusion_matrix"],
        },
    }
    (etapa_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[OK] Wrote: {metrics_path}")
    print(f"[OK] Wrote: {plot_path}")
    print(f"[OK] Wrote: {etapa_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
