"""Run PC3 end-to-end: RL (actor-critic / PPO) refinement + comparisons.

PC3 requirements:
- Use validation (only) for reward/selection (no test leakage)
- Compare against PC2 baseline and a traditional tuning strategy under similar budget
- Produce only small, versionable artifacts under output/etapa3/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Ensure brain_mri is importable when running as a script from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

if TYPE_CHECKING:
    import numpy as np  # noqa: F401
    import pandas as pd  # noqa: F401
    import torch  # noqa: F401

OUTPUT_DIR = REPO_ROOT / "output"
ETAPA3_DIR = OUTPUT_DIR / "etapa3"


def _load_ml_deps() -> dict[str, Any]:
    """Import ML dependencies lazily.

    This keeps the `--skip-run` path fast and avoids importing modules that may
    touch datasets/checkpoints during import in some environments.
    """

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

    return {
        "np": np,
        "pd": pd,
        "torch": torch,
        "MultiOrientMRIDataset": MultiOrientMRIDataset,
        "MultiOrientTabularFusionNet": MultiOrientTabularFusionNet,
        "ActionSpec": ActionSpec,
        "PPOAgent": PPOAgent,
        "RLRefineEnv": RLRefineEnv,
        "evaluate_classifier": evaluate_classifier,
        "micro_finetune": micro_finetune,
        "set_global_seed": set_global_seed,
        "build_transforms": build_transforms,
        "select_device": select_device,
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def _git_is_dirty() -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT)
        return bool(out.strip())
    except Exception:
        return True


def _relativize_cmd(cmd: list[str]) -> list[str]:
    display: list[str] = []
    for raw in cmd:
        try:
            p = Path(raw)
        except Exception:
            display.append(raw)
            continue

        if not p.is_absolute():
            display.append(raw)
            continue

        try:
            display.append(str(p.relative_to(REPO_ROOT)))
        except Exception:
            display.append(p.name)
    return display


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_pc2_entry(experiments: list[dict[str, Any]], *, backbone: str, seed: int) -> dict[str, Any]:
    # PC2 may have a customized DEEP_SCENARIO prefix. Identify it by the fine-tuning contract.
    def _is_pc2(entry: dict[str, Any]) -> bool:
        if str(entry.get("model") or "") != f"{backbone}_classification":
            return False
        if int(entry.get("seed") or -1) != int(seed):
            return False
        if not bool(entry.get("pretrained", False)):
            return False
        if not bool(entry.get("freeze_backbone_initial", False)):
            return False
        if entry.get("unfreeze_epoch") is None:
            return False
        return True

    matches = [e for e in experiments if _is_pc2(e)]
    if not matches:
        raise SystemExit(
            "Missing PC2 fine-tuning entry for the requested backbone/seed in output/training_experiments.json. "
            "Run: python3 brain_mri/scripts/run_pc2_finetune.py"
        )
    return sorted(matches, key=lambda e: str(e.get("timestamp", "")))[-1]


def _make_actions() -> list[ActionSpec]:
    # Small discrete action space to keep PC3 cheap/reproducible.
    lrs = [1e-5, 3e-5, 1e-4, 3e-4]
    wds = [0.0, 1e-6, 1e-5, 1e-4]
    actions: list[ActionSpec] = []
    for lr in lrs:
        for wd in wds:
            actions.append(ActionSpec(lr=float(lr), weight_decay=float(wd)))
    return actions


def _sample_df(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= int(n):
        return df
    return df.sample(n=int(n), random_state=int(seed)).copy()


def _class_weights_from_df(df_train: pd.DataFrame) -> torch.Tensor:
    vc = df_train["Final_Group"].value_counts().to_dict()
    n0 = float(vc.get("Nondemented", 1))
    n1 = float(vc.get("Demented", 1))
    total = n0 + n1
    w0 = total / (2.0 * n0) if n0 else 1.0
    w1 = total / (2.0 * n1) if n1 else 1.0
    return torch.tensor([w0, w1], dtype=torch.float32)


def _build_model(backbone: str, *, dropout: float) -> MultiOrientTabularFusionNet:
    return MultiOrientTabularFusionNet(
        backbone=backbone,
        mode="classification",
        num_tabular_features=0,
        medicalnet_depth=18,
        pretrained=True,
        share_encoder=True,
        dropout=float(dropout),
    )


def _plot_curves(path: Path, *, rewards: list[float], val_bacc: list[float]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.arange(1, len(rewards) + 1)
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(steps, rewards, "b-")
    ax1.set_title("Recompensa (val) por passo")
    ax1.set_xlabel("Passo")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(steps, val_bacc, "r-")
    ax2.set_title("Balanced accuracy (val) por passo")
    ax2.set_xlabel("Passo")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_bars(path: Path, *, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(labels, values, color=["#777777", "#3b82f6", "#10b981"][: len(labels)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.0)
    for i, v in enumerate(values):
        ax.text(i, min(0.98, v + 0.02), f"{100*v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_manifest(
    *,
    args: argparse.Namespace,
    split_csv: Path,
    experiments_path: Path,
    ckpt_path: Path,
    expected_outputs: dict[str, Path],
) -> None:
    manifest = {
        "pc": "PC3",
        "git_commit": _git_commit(),
        "git_dirty": _git_is_dirty(),
        "command": _relativize_cmd([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]),
        "inputs": {
            "split_csv": str(split_csv.relative_to(REPO_ROOT)),
            "split_csv_sha256": _sha256_file(split_csv),
            "training_experiments": str(experiments_path.relative_to(REPO_ROOT)),
            "training_experiments_sha256": _sha256_file(experiments_path),
            "pc2_checkpoint": str(ckpt_path.relative_to(REPO_ROOT)),
            "pc2_checkpoint_sha256": _sha256_file(ckpt_path),
        },
        "outputs": {
            name: {"path": str(path.relative_to(REPO_ROOT)), "sha256": _sha256_file(path)}
            for name, path in expected_outputs.items()
        },
        "notes": {
            "selection": "Reward and selection use validation only; test is reported after selecting best configs.",
            "budget": {
                "episodes": int(args.episodes),
                "horizon": int(args.horizon),
                "micro_epochs": int(args.micro_epochs),
                "max_batches": int(args.max_batches),
                "train_subset": int(args.train_subset),
                "val_subset": int(args.val_subset),
            },
        },
    }
    _dump_json(ETAPA3_DIR / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="PC3: RL refinement (actor-critic / PPO) + comparisons.")
    parser.add_argument("--backbone", default="efficientnet", choices=["efficientnet", "medicalnet", "densenet"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--micro-epochs", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=3)
    parser.add_argument("--train-subset", type=int, default=120)
    parser.add_argument("--val-subset", type=int, default=80)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Do not run refinement; only regenerate output/etapa3/manifest.json from existing artifacts.",
    )
    args = parser.parse_args()

    # Manifest-only path: should not depend on ML deps or evaluate datasets/models.
    if args.skip_run:
        OUTPUT_DIR.mkdir(exist_ok=True)
        ETAPA3_DIR.mkdir(parents=True, exist_ok=True)
        plots_dir = ETAPA3_DIR / "plots"

        expected = {
            "rl_history_json": ETAPA3_DIR / "rl_history.json",
            "comparativo_csv": ETAPA3_DIR / "comparativo.csv",
            "plot_reward_curve": plots_dir / "pc3_reward_and_val_curve.png",
            "plot_test_bacc": plots_dir / "pc3_test_balanced_accuracy.png",
        }
        missing = [str(p) for p in expected.values() if not p.exists()]
        if missing:
            raise SystemExit("--skip-run requested but some PC3 artifacts are missing: " + ", ".join(missing))

        split_csv = OUTPUT_DIR / "exam_level_dataset_split.csv"
        experiments_path = OUTPUT_DIR / "training_experiments.json"

        rl_hist = _read_json(expected["rl_history_json"])
        meta = (rl_hist or {}).get("meta") if isinstance(rl_hist, dict) else {}

        ckpt_rel = (meta or {}).get("base_checkpoint")
        ckpt_sha = (meta or {}).get("base_checkpoint_sha256")
        ckpt_path = (REPO_ROOT / ckpt_rel) if ckpt_rel else None
        ckpt_present = bool(ckpt_path is not None and Path(ckpt_path).exists())
        if ckpt_present:
            ckpt_sha = _sha256_file(Path(ckpt_path))

        manifest = {
            "pc": "PC3",
            "git_commit": _git_commit(),
            "git_dirty": _git_is_dirty(),
            "command": _relativize_cmd([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]),
            "inputs": {
                "split_csv": str(split_csv.relative_to(REPO_ROOT)) if split_csv.exists() else str(split_csv),
                "split_csv_sha256": _sha256_file(split_csv) if split_csv.exists() else None,
                "training_experiments": str(experiments_path.relative_to(REPO_ROOT))
                if experiments_path.exists()
                else str(experiments_path),
                "training_experiments_sha256": _sha256_file(experiments_path) if experiments_path.exists() else None,
                "pc2_checkpoint": str(Path(ckpt_path).relative_to(REPO_ROOT)) if ckpt_path is not None else None,
                "pc2_checkpoint_sha256": ckpt_sha,
                "pc2_checkpoint_present": ckpt_present,
            },
            "outputs": {
                name: {"path": str(path.relative_to(REPO_ROOT)), "sha256": _sha256_file(path)}
                for name, path in expected.items()
            },
            "notes": {
                "selection": "Reward and selection use validation only; test is reported after selecting best configs.",
                "budget": {
                    "episodes": int(args.episodes),
                    "horizon": int(args.horizon),
                    "micro_epochs": int(args.micro_epochs),
                    "max_batches": int(args.max_batches),
                    "train_subset": int(args.train_subset),
                    "val_subset": int(args.val_subset),
                },
            },
        }
        _dump_json(ETAPA3_DIR / "manifest.json", manifest)
        print("[PC3] Wrote:")
        print(f"- {ETAPA3_DIR / 'manifest.json'}")
        return

    try:
        ml = _load_ml_deps()
    except Exception as exc:
        raise SystemExit(f"Missing dependencies required to run PC3 refinement: {exc}") from exc
    globals().update(ml)

    set_global_seed(int(args.seed))
    OUTPUT_DIR.mkdir(exist_ok=True)
    ETAPA3_DIR.mkdir(parents=True, exist_ok=True)

    split_csv = OUTPUT_DIR / "exam_level_dataset_split.csv"
    if not split_csv.exists():
        raise SystemExit(f"Missing split CSV: {split_csv} (run PC0 first)")

    experiments_path = OUTPUT_DIR / "training_experiments.json"
    if not experiments_path.exists():
        raise SystemExit(f"Missing experiments JSON: {experiments_path}")
    experiments = _read_json(experiments_path)
    if not isinstance(experiments, list):
        raise SystemExit("training_experiments.json must be a list")

    pc2_entry = _latest_pc2_entry(experiments, backbone=args.backbone, seed=args.seed)
    ckpt_name = str(pc2_entry.get("best_checkpoint") or pc2_entry.get("legacy_checkpoint") or "").strip()
    if not ckpt_name:
        raise SystemExit("PC2 entry missing best_checkpoint/legacy_checkpoint")
    ckpt_path = OUTPUT_DIR / ckpt_name
    if not ckpt_path.exists():
        raise SystemExit(f"Missing PC2 checkpoint file: {ckpt_path}")

    base_sd = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(base_sd, dict):
        raise SystemExit(f"Unexpected checkpoint format: {ckpt_path}")

    df = pd.read_csv(split_csv)
    for col in ["split", "Final_Group"]:
        if col not in df.columns:
            raise SystemExit(f"Split CSV missing column: {col}")
    df_train = df[df["split"] == "train"].copy()
    df_val = df[df["split"] == "validation"].copy()
    df_test = df[df["split"] == "test"].copy()

    if df_train.empty or df_val.empty or df_test.empty:
        raise SystemExit("Invalid split: one of train/validation/test is empty")

    device = select_device()
    train_tf, val_tf = build_transforms()

    df_train_small = _sample_df(df_train, int(args.train_subset), seed=int(args.seed))
    df_val_small = _sample_df(df_val, int(args.val_subset), seed=int(args.seed))

    train_loader_small = torch.utils.data.DataLoader(
        MultiOrientMRIDataset(df_train_small, train_tf, REPO_ROOT, "original_path", "Final_Group"),
        batch_size=8,
        shuffle=True,
    )
    val_loader_small = torch.utils.data.DataLoader(
        MultiOrientMRIDataset(df_val_small, val_tf, REPO_ROOT, "original_path", "Final_Group"),
        batch_size=16,
        shuffle=False,
    )

    # Full loaders are only used for reporting after selection.
    val_loader_full = torch.utils.data.DataLoader(
        MultiOrientMRIDataset(df_val, val_tf, REPO_ROOT, "original_path", "Final_Group"),
        batch_size=16,
        shuffle=False,
    )
    test_loader_full = torch.utils.data.DataLoader(
        MultiOrientMRIDataset(df_test, val_tf, REPO_ROOT, "original_path", "Final_Group"),
        batch_size=16,
        shuffle=False,
    )

    class_weights = _class_weights_from_df(df_train_small)

    def build_model() -> MultiOrientTabularFusionNet:
        return _build_model(args.backbone, dropout=float(args.dropout))

    # Baseline: PC2 checkpoint as-is.
    baseline_model = build_model().to(device)
    baseline_model.load_state_dict(base_sd, strict=False)
    baseline_val_small = evaluate_classifier(baseline_model, val_loader_small, device, class_weights=class_weights)
    baseline_val_full = evaluate_classifier(baseline_model, val_loader_full, device, class_weights=class_weights)
    baseline_test = evaluate_classifier(baseline_model, test_loader_full, device, class_weights=class_weights)

    # expected outputs are used only after the run completes.
    plots_dir = ETAPA3_DIR / "plots"
    expected = {
        "rl_history_json": ETAPA3_DIR / "rl_history.json",
        "comparativo_csv": ETAPA3_DIR / "comparativo.csv",
        "plot_reward_curve": plots_dir / "pc3_reward_and_val_curve.png",
        "plot_test_bacc": plots_dir / "pc3_test_balanced_accuracy.png",
    }

    actions = _make_actions()
    env = RLRefineEnv(
        build_model=build_model,
        base_state_dict=base_sd,
        train_loader=train_loader_small,
        val_loader=val_loader_small,
        device=device,
        actions=actions,
        micro_epochs=int(args.micro_epochs),
        max_batches_per_epoch=int(args.max_batches),
        class_weights=class_weights,
        seed=int(args.seed),
    )
    env.baseline_val_bal_acc = float(baseline_val_small["balanced_accuracy"])

    agent = PPOAgent(state_dim=env.state_dim, action_dim=env.action_dim, device=device)

    steps_total = int(args.episodes) * int(args.horizon)
    rl_steps: list[dict[str, Any]] = []
    rewards: list[float] = []
    val_bacc_curve: list[float] = []

    state = env.reset()
    for ep in range(int(args.episodes)):
        for t in range(int(args.horizon)):
            action_index, logp, value = agent.select_action(state)
            next_state, reward, info = env.step(action_index)
            done = (t == int(args.horizon) - 1)
            agent.store(state=state, action=action_index, logp=logp, value=value, reward=float(reward), done=done)
            rl_steps.append(info)
            rewards.append(float(reward))
            val_bacc_curve.append(float(info["val_balanced_accuracy"]))
            state = next_state
        update_stats = agent.update()
        rl_steps[-1]["ppo_update"] = update_stats
        state = env.reset()

    # Traditional tuning: random search over the same budget.
    rng = random.Random(int(args.seed))
    best_trad_val = -1.0
    best_trad_action: ActionSpec | None = None
    best_trad_sd: dict[str, Any] | None = None
    trad_trials: list[dict[str, Any]] = []

    for i in range(steps_total):
        a_idx = rng.randrange(0, len(actions))
        a = actions[a_idx]
        set_global_seed(int(args.seed) + 2000 + i)
        m = build_model().to(device)
        m.load_state_dict(base_sd, strict=False)
        train_summary, val_summary = micro_finetune(
            m,
            train_loader_small,
            val_loader_small,
            device,
            action=a,
            micro_epochs=int(args.micro_epochs),
            max_batches_per_epoch=int(args.max_batches),
            class_weights=class_weights,
        )
        trad_trials.append(
            {
                "trial": int(i),
                "action_index": int(a_idx),
                "action": {"lr": float(a.lr), "weight_decay": float(a.weight_decay)},
                "train_loss": float(train_summary.get("loss", 0.0)),
                "val_balanced_accuracy": float(val_summary.get("balanced_accuracy", 0.0)),
                "val_loss": float(val_summary.get("loss", 0.0)),
            }
        )
        if float(val_summary.get("balanced_accuracy", 0.0)) > best_trad_val:
            best_trad_val = float(val_summary.get("balanced_accuracy", 0.0))
            best_trad_action = a
            best_trad_sd = {k: v.detach().cpu() for k, v in m.state_dict().items()}

    if env.best_state_dict is None or env.best_action_index is None:
        raise SystemExit("RL did not produce a best checkpoint")
    rl_best_action = actions[int(env.best_action_index)]

    rl_best_model = build_model().to(device)
    rl_best_model.load_state_dict(env.best_state_dict, strict=False)
    rl_val_full = evaluate_classifier(rl_best_model, val_loader_full, device, class_weights=class_weights)
    rl_test = evaluate_classifier(rl_best_model, test_loader_full, device, class_weights=class_weights)

    if best_trad_sd is None or best_trad_action is None:
        raise SystemExit("Traditional tuning did not produce a best checkpoint")
    trad_best_model = build_model().to(device)
    trad_best_model.load_state_dict(best_trad_sd, strict=False)
    trad_val_full = evaluate_classifier(trad_best_model, val_loader_full, device, class_weights=class_weights)
    trad_test = evaluate_classifier(trad_best_model, test_loader_full, device, class_weights=class_weights)

    rl_history = {
        "meta": {
            "pc": "PC3",
            "seed": int(args.seed),
            "backbone": str(args.backbone),
            "base_checkpoint": str(ckpt_path.relative_to(REPO_ROOT)),
            "base_checkpoint_sha256": _sha256_file(ckpt_path),
            "split_csv": str(split_csv.relative_to(REPO_ROOT)),
            "split_csv_sha256": _sha256_file(split_csv),
            "episodes": int(args.episodes),
            "horizon": int(args.horizon),
            "budget_evaluations": int(steps_total),
            "micro_epochs": int(args.micro_epochs),
            "max_batches": int(args.max_batches),
            "train_subset": int(len(df_train_small)),
            "val_subset": int(len(df_val_small)),
            "selection_rule": "selection_by_validation_only",
        },
        "env": env.to_jsonable(),
        "baseline_val_small": baseline_val_small,
        "rl_steps": rl_steps,
        "traditional_trials": trad_trials,
        "best": {
            "rl": {
                "action": {"lr": float(rl_best_action.lr), "weight_decay": float(rl_best_action.weight_decay)},
                "val_small_balanced_accuracy": float(env.best_val_bal_acc),
            },
            "traditional": {
                "action": {"lr": float(best_trad_action.lr), "weight_decay": float(best_trad_action.weight_decay)},
                "val_small_balanced_accuracy": float(best_trad_val),
            },
        },
    }
    _dump_json(ETAPA3_DIR / "rl_history.json", rl_history)

    comp_rows = []
    comp_rows.append(
        {
            "method": "baseline_pc2",
            "budget_evaluations": 0,
            "selection_metric": "val_small_balanced_accuracy",
            "selection_value": float(baseline_val_small["balanced_accuracy"]),
            "val_balanced_accuracy": float(baseline_val_full["balanced_accuracy"]),
            "test_balanced_accuracy": float(baseline_test["balanced_accuracy"]),
            "test_accuracy": float(baseline_test["accuracy"]),
        }
    )
    comp_rows.append(
        {
            "method": "traditional_tuning",
            "budget_evaluations": int(steps_total),
            "selection_metric": "val_small_balanced_accuracy",
            "selection_value": float(best_trad_val),
            "val_balanced_accuracy": float(trad_val_full["balanced_accuracy"]),
            "test_balanced_accuracy": float(trad_test["balanced_accuracy"]),
            "test_accuracy": float(trad_test["accuracy"]),
            "lr": float(best_trad_action.lr),
            "weight_decay": float(best_trad_action.weight_decay),
        }
    )
    comp_rows.append(
        {
            "method": "rl_ppo_actor_critic",
            "budget_evaluations": int(steps_total),
            "selection_metric": "val_small_balanced_accuracy",
            "selection_value": float(env.best_val_bal_acc),
            "val_balanced_accuracy": float(rl_val_full["balanced_accuracy"]),
            "test_balanced_accuracy": float(rl_test["balanced_accuracy"]),
            "test_accuracy": float(rl_test["accuracy"]),
            "lr": float(rl_best_action.lr),
            "weight_decay": float(rl_best_action.weight_decay),
        }
    )

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(ETAPA3_DIR / "comparativo.csv", index=False)

    # Plots
    plots_dir = ETAPA3_DIR / "plots"
    _plot_curves(plots_dir / "pc3_reward_and_val_curve.png", rewards=rewards, val_bacc=val_bacc_curve)
    _plot_bars(
        plots_dir / "pc3_test_balanced_accuracy.png",
        labels=["PC2", "Trad.", "RL"],
        values=[float(baseline_test["balanced_accuracy"]), float(trad_test["balanced_accuracy"]), float(rl_test["balanced_accuracy"])],
        title="PC3: balanced accuracy no teste",
        ylabel="Balanced accuracy",
    )

    _write_manifest(
        args=args,
        split_csv=split_csv,
        experiments_path=experiments_path,
        ckpt_path=ckpt_path,
        expected_outputs=expected,
    )

    print("[PC3] Wrote:")
    print(f"- {ETAPA3_DIR / 'rl_history.json'}")
    print(f"- {ETAPA3_DIR / 'comparativo.csv'}")
    print(f"- {plots_dir / 'pc3_reward_and_val_curve.png'}")
    print(f"- {plots_dir / 'pc3_test_balanced_accuracy.png'}")
    print(f"- {ETAPA3_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
