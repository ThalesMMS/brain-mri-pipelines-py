"""Experimento: Refinamento RL com fusão multimodal (variáveis clínicas).

Compara o refinamento RL com e sem fusão de dados tabulares clínicos.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

from brain_mri.ml.datasets import MultiOrientMRIDataset
from brain_mri.ml.multistream_models import MultiOrientTabularFusionNet
from brain_mri.ml.rl_refinement import (
    ActionSpec,
    PPOAgent,
    RLRefineEnv,
    evaluate_classifier,
    micro_finetune,
    set_global_seed,
)
from brain_mri.ml.training_utils import build_transforms, select_device

OUTPUT_DIR = REPO_ROOT / "output"
EXPERIMENT_DIR = OUTPUT_DIR / "multimodal_experiment"

# Variáveis clínicas disponíveis no dataset (excluindo CDR e MMSE que são proxies do target)
CLINICAL_FEATURES = ["age", "education", "nwbv", "etiv", "asf"]


def _sample_df(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed).copy()


def _class_weights_from_df(df_train: pd.DataFrame) -> torch.Tensor:
    vc = df_train["Final_Group"].value_counts().to_dict()
    n0 = float(vc.get("Nondemented", 1))
    n1 = float(vc.get("Demented", 1))
    total = n0 + n1
    w0 = total / (2.0 * n0) if n0 else 1.0
    w1 = total / (2.0 * n1) if n1 else 1.0
    return torch.tensor([w0, w1], dtype=torch.float32)


def _make_actions() -> list[ActionSpec]:
    lrs = [1e-5, 3e-5, 1e-4, 3e-4]
    wds = [0.0, 1e-6, 1e-5, 1e-4]
    actions: list[ActionSpec] = []
    for lr in lrs:
        for wd in wds:
            actions.append(ActionSpec(lr=lr, weight_decay=wd))
    return actions


def _normalize_clinical_features(df: pd.DataFrame, clinical_cols: list[str]) -> pd.DataFrame:
    """Normaliza variáveis clínicas usando z-score."""
    df = df.copy()
    for col in clinical_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                df[col] = (df[col] - mean_val) / std_val
            else:
                df[col] = 0.0
    return df


def _prepare_clinical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara colunas clínicas: renomeia e preenche NaNs."""
    df = df.copy()
    
    # Mapeia nomes de colunas do CSV original para lowercase
    col_map = {
        "Age": "age",
        "EDUC": "education",
        "nWBV": "nwbv",
        "eTIV": "etiv",
        "ASF": "asf",
    }
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    
    # Preenche NaNs com mediana
    for col in CLINICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    return df


def run_experiment(
    *,
    use_multimodal: bool,
    backbone: str,
    seed: int,
    episodes: int,
    horizon: int,
    micro_epochs: int,
    max_batches: int,
    train_subset: int,
    val_subset: int,
    dropout: float,
) -> dict[str, Any]:
    """Executa um experimento de refinamento RL."""
    
    set_global_seed(seed)
    device = select_device()
    
    split_csv = OUTPUT_DIR / "exam_level_dataset_split.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"Missing split CSV: {split_csv}")
    
    df = pd.read_csv(split_csv)
    df = _prepare_clinical_columns(df)
    
    df_train = df[df["split"] == "train"].copy()
    df_val = df[df["split"] == "validation"].copy()
    df_test = df[df["split"] == "test"].copy()
    
    # Normaliza features clínicas usando estatísticas do treino
    if use_multimodal:
        train_stats = {}
        for col in CLINICAL_FEATURES:
            if col in df_train.columns:
                train_stats[col] = {
                    "mean": df_train[col].mean(),
                    "std": df_train[col].std() or 1.0,
                }
        
        for split_df in [df_train, df_val, df_test]:
            for col, stats in train_stats.items():
                split_df[col] = (split_df[col] - stats["mean"]) / stats["std"]
    
    clinical_features = CLINICAL_FEATURES if use_multimodal else None
    num_tabular = len(CLINICAL_FEATURES) if use_multimodal else 0
    
    train_tf, val_tf = build_transforms()
    
    df_train_small = _sample_df(df_train, train_subset, seed)
    df_val_small = _sample_df(df_val, val_subset, seed)
    
    train_loader_small = torch.utils.data.DataLoader(
        MultiOrientMRIDataset(
            df_train_small, train_tf, REPO_ROOT, "original_path", "Final_Group",
            clinical_features=clinical_features
        ),
        batch_size=8,
        shuffle=True,
    )
    val_loader_small = torch.utils.data.DataLoader(
        MultiOrientMRIDataset(
            df_val_small, val_tf, REPO_ROOT, "original_path", "Final_Group",
            clinical_features=clinical_features
        ),
        batch_size=16,
        shuffle=False,
    )
    val_loader_full = torch.utils.data.DataLoader(
        MultiOrientMRIDataset(
            df_val, val_tf, REPO_ROOT, "original_path", "Final_Group",
            clinical_features=clinical_features
        ),
        batch_size=16,
        shuffle=False,
    )
    test_loader_full = torch.utils.data.DataLoader(
        MultiOrientMRIDataset(
            df_test, val_tf, REPO_ROOT, "original_path", "Final_Group",
            clinical_features=clinical_features
        ),
        batch_size=16,
        shuffle=False,
    )
    
    class_weights = _class_weights_from_df(df_train_small)
    
    def build_model() -> MultiOrientTabularFusionNet:
        return MultiOrientTabularFusionNet(
            backbone=backbone,
            mode="classification",
            num_tabular_features=num_tabular,
            medicalnet_depth=18,
            pretrained=True,
            share_encoder=True,
            dropout=dropout,
        )
    
    # Baseline: modelo sem treino adicional
    baseline_model = build_model().to(device)
    baseline_val = evaluate_classifier(baseline_model, val_loader_full, device, class_weights=class_weights)
    baseline_test = evaluate_classifier(baseline_model, test_loader_full, device, class_weights=class_weights)
    
    # Captura state_dict inicial para o ambiente RL
    base_sd = {k: v.cpu().clone() for k, v in baseline_model.state_dict().items()}
    
    actions = _make_actions()
    env = RLRefineEnv(
        build_model=build_model,
        base_state_dict=base_sd,
        train_loader=train_loader_small,
        val_loader=val_loader_small,
        device=device,
        actions=actions,
        micro_epochs=micro_epochs,
        max_batches_per_epoch=max_batches,
        class_weights=class_weights,
        seed=seed,
    )
    env.baseline_val_bal_acc = float(baseline_val["balanced_accuracy"])
    
    agent = PPOAgent(state_dim=env.state_dim, action_dim=env.action_dim, device=device)
    
    steps_total = episodes * horizon
    rl_steps: list[dict[str, Any]] = []
    rewards: list[float] = []
    
    state = env.reset()
    for ep in range(episodes):
        for t in range(horizon):
            action_index, logp, value = agent.select_action(state)
            next_state, reward, info = env.step(action_index)
            done = (t == horizon - 1)
            agent.store(state=state, action=action_index, logp=logp, value=value, reward=reward, done=done)
            rl_steps.append(info)
            rewards.append(reward)
            state = next_state
        agent.update()
        state = env.reset()
    
    # Avalia melhor modelo RL
    if env.best_state_dict is None:
        raise RuntimeError("RL não produziu checkpoint")
    
    rl_best_model = build_model().to(device)
    rl_best_model.load_state_dict(env.best_state_dict, strict=False)
    rl_val_full = evaluate_classifier(rl_best_model, val_loader_full, device, class_weights=class_weights)
    rl_test = evaluate_classifier(rl_best_model, test_loader_full, device, class_weights=class_weights)
    
    return {
        "config": {
            "use_multimodal": use_multimodal,
            "backbone": backbone,
            "seed": seed,
            "episodes": episodes,
            "horizon": horizon,
            "micro_epochs": micro_epochs,
            "max_batches": max_batches,
            "train_subset": len(df_train_small),
            "val_subset": len(df_val_small),
            "num_tabular_features": num_tabular,
            "clinical_features": clinical_features,
        },
        "baseline": {
            "val_balanced_accuracy": baseline_val["balanced_accuracy"],
            "test_balanced_accuracy": baseline_test["balanced_accuracy"],
            "test_accuracy": baseline_test["accuracy"],
        },
        "rl_refinement": {
            "best_val_balanced_accuracy": env.best_val_bal_acc,
            "val_balanced_accuracy": rl_val_full["balanced_accuracy"],
            "test_balanced_accuracy": rl_test["balanced_accuracy"],
            "test_accuracy": rl_test["accuracy"],
            "best_action": {
                "lr": actions[env.best_action_index].lr,
                "weight_decay": actions[env.best_action_index].weight_decay,
            } if env.best_action_index is not None else None,
        },
        "rl_steps": rl_steps,
        "rewards": rewards,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimento: RL com/sem fusão multimodal")
    parser.add_argument("--backbone", default="efficientnet", choices=["efficientnet", "medicalnet", "densenet"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--micro-epochs", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=3)
    parser.add_argument("--train-subset", type=int, default=120)
    parser.add_argument("--val-subset", type=int, default=80)
    parser.add_argument("--dropout", type=float, default=0.25)
    args = parser.parse_args()
    
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Experimento 1: SEM fusão multimodal (baseline)
    print("\n" + "="*60)
    print("Experimento 1: SEM fusão multimodal")
    print("="*60)
    results["sem_multimodal"] = run_experiment(
        use_multimodal=False,
        backbone=args.backbone,
        seed=args.seed,
        episodes=args.episodes,
        horizon=args.horizon,
        micro_epochs=args.micro_epochs,
        max_batches=args.max_batches,
        train_subset=args.train_subset,
        val_subset=args.val_subset,
        dropout=args.dropout,
    )
    print(f"  Baseline test bal_acc: {results['sem_multimodal']['baseline']['test_balanced_accuracy']:.4f}")
    print(f"  RL test bal_acc:       {results['sem_multimodal']['rl_refinement']['test_balanced_accuracy']:.4f}")
    
    # Experimento 2: COM fusão multimodal
    print("\n" + "="*60)
    print("Experimento 2: COM fusão multimodal")
    print(f"  Features: {CLINICAL_FEATURES}")
    print("="*60)
    results["com_multimodal"] = run_experiment(
        use_multimodal=True,
        backbone=args.backbone,
        seed=args.seed,
        episodes=args.episodes,
        horizon=args.horizon,
        micro_epochs=args.micro_epochs,
        max_batches=args.max_batches,
        train_subset=args.train_subset,
        val_subset=args.val_subset,
        dropout=args.dropout,
    )
    print(f"  Baseline test bal_acc: {results['com_multimodal']['baseline']['test_balanced_accuracy']:.4f}")
    print(f"  RL test bal_acc:       {results['com_multimodal']['rl_refinement']['test_balanced_accuracy']:.4f}")
    
    # Resumo comparativo
    print("\n" + "="*60)
    print("RESUMO COMPARATIVO")
    print("="*60)
    print(f"{'Configuração':<25} {'Baseline':<12} {'RL Refinado':<12} {'Delta':<10}")
    print("-"*60)
    
    for name, res in results.items():
        baseline_acc = res["baseline"]["test_balanced_accuracy"]
        rl_acc = res["rl_refinement"]["test_balanced_accuracy"]
        delta = rl_acc - baseline_acc
        print(f"{name:<25} {baseline_acc:>10.2%}   {rl_acc:>10.2%}   {delta:>+8.2%}")
    
    # Compara multimodal vs não-multimodal
    delta_multimodal = (
        results["com_multimodal"]["rl_refinement"]["test_balanced_accuracy"] -
        results["sem_multimodal"]["rl_refinement"]["test_balanced_accuracy"]
    )
    print("-"*60)
    print(f"Ganho da fusão multimodal: {delta_multimodal:+.2%}")
    
    # Salva resultados
    output_file = EXPERIMENT_DIR / "multimodal_comparison.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResultados salvos em: {output_file}")


if __name__ == "__main__":
    main()
