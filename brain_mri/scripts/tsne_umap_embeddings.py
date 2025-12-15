"""
Gera visualizações 2D (t-SNE e UMAP) para embeddings salvos em CSV.

Uso:
    python tsne_umap_embeddings.py --embeddings output/densenet_embeddings_classification_train.csv --target-col target --out-prefix output/tsne_densenet_class_train

Requisitos: pandas, numpy, matplotlib, seaborn, scikit-learn, umap-learn (opcional para UMAP).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

try:
    import umap  # type: ignore

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def load_embeddings(path: Path, target_col: str):
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"Coluna alvo '{target_col}' não encontrada em {path}")

    meta_cols = {'MRI_ID', target_col}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    if not feature_cols:
        raise ValueError("Nenhuma coluna de feature encontrada no CSV de embeddings.")

    X = df[feature_cols].values
    y = df[target_col].values
    return X, y, feature_cols


def plot_scatter(df_plot: pd.DataFrame, x_col: str, y_col: str, hue_col: str, title: str, out_path: Path):
    plt.figure(figsize=(8, 6))
    # Usa mapa contínuo se alvo for numérico contínuo
    if pd.api.types.is_numeric_dtype(df_plot[hue_col]) and df_plot[hue_col].nunique() > 10:
        scatter = plt.scatter(df_plot[x_col], df_plot[y_col], c=df_plot[hue_col], cmap="viridis", alpha=0.8, s=35)
        cbar = plt.colorbar(scatter)
        cbar.set_label(hue_col)
    else:
        sns.scatterplot(data=df_plot, x=x_col, y=y_col, hue=hue_col, palette="tab10", alpha=0.85, s=40, edgecolor="none")
        plt.legend(title=hue_col)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def run_tsne_umap(emb_path: Path, target_col: str, out_prefix: Path):
    print(f"Lendo embeddings de {emb_path}")
    X, y, _ = load_embeddings(emb_path, target_col)

    # Escala antes de reduzir
    X_scaled = StandardScaler().fit_transform(X)
    n_samples = len(X_scaled)
    perplexity = max(5, min(30, n_samples - 1))

    print(f"Executando t-SNE (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")
    X_tsne = tsne.fit_transform(X_scaled)

    df_tsne = pd.DataFrame({"x": X_tsne[:, 0], "y": X_tsne[:, 1], "target": y})
    plot_scatter(df_tsne, "x", "y", "target", "t-SNE dos embeddings", out_prefix.with_suffix(".tsne.png"))

    if UMAP_AVAILABLE:
        print("Executando UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=42)
        X_umap = reducer.fit_transform(X_scaled)
        df_umap = pd.DataFrame({"x": X_umap[:, 0], "y": X_umap[:, 1], "target": y})
        plot_scatter(df_umap, "x", "y", "target", "UMAP dos embeddings", out_prefix.with_suffix(".umap.png"))
    else:
        print("UMAP não disponível (instale com 'pip install umap-learn' para gerar essa visualização).")


def main():
    parser = argparse.ArgumentParser(description="Gera t-SNE e UMAP para embeddings.")
    parser.add_argument("--embeddings", type=Path, required=True, help="Caminho para CSV de embeddings (ex: output/densenet_embeddings_classification_train.csv)")
    parser.add_argument("--target-col", type=str, default="target", help="Nome da coluna alvo no CSV")
    parser.add_argument("--out-prefix", type=Path, default=Path("output/tsne_umap"), help="Prefixo do arquivo de saída (sem extensão)")
    args = parser.parse_args()

    run_tsne_umap(args.embeddings, args.target_col, args.out_prefix)


if __name__ == "__main__":
    main()
