from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from .training_utils import load_split_dataframe

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from .dataset_builder import populate_orientation_paths
    from .datasets import MultiOrientMRIDataset
    from .multistream_models import MultiOrientTabularFusionNet
    from .training_utils import build_transforms, select_device

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    class _NNFallback:
        Module = object

    nn = _NNFallback()
    F = None
    populate_orientation_paths = None
    MultiOrientMRIDataset = None
    MultiOrientTabularFusionNet = None
    build_transforms = None
    select_device = None
    TORCH_AVAILABLE = False


@dataclass(frozen=True)
class ActionSpec:
    lr: float
    weight_decay: float


@dataclass(frozen=True)
class StepResult:
    action_index: int
    action: ActionSpec
    reward: float
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_balanced_accuracy: float


def _require_torch() -> None:
    """
    Ensure PyTorch is available for operations that require it.
    
    Raises:
        ImportError: If PyTorch is not installed or flagged as unavailable.
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch é necessário para o refinamento com RL.\n"
            "Instale com 'pip install torch torchvision'."
        )


def set_global_seed(seed: int) -> None:
    """
    Seed Python's `random`, NumPy, and PyTorch random number generators (including all CUDA devices when available) to ensure reproducible behavior.
    
    Parameters:
        seed (int): Integer seed value to apply.
    
    Raises:
        ImportError: If PyTorch is not available.
    """
    _require_torch()
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _confusion_2x2(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    # Classes are expected to be 0/1.
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return tn, fp, fn, tp


def balanced_accuracy_from_cm(tn: int, fp: int, fn: int, tp: int) -> float:
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return 0.5 * (sens + spec)


def evaluate_classifier(
    model: nn.Module,
    loader,
    device: torch.device,
    *,
    class_weights: torch.Tensor | None = None,
) -> dict[str, Any]:
    model.eval()
    ce = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    losses: list[float] = []
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []

    with torch.no_grad():
        for batch_x, lbls in loader:
            axl = batch_x["axl"].to(device)
            cor = batch_x["cor"].to(device)
            sag = batch_x["sag"].to(device)
            clin = batch_x.get("clin")
            if clin is not None:
                clin = clin.to(device)
            lbls = lbls.to(device)
            out = model(axl, cor, sag, clin)
            loss = ce(out, lbls.long())
            losses.append(float(loss.detach().cpu().item()))
            preds = out.argmax(dim=1)
            y_true.append(lbls.detach().cpu().numpy())
            y_pred.append(preds.detach().cpu().numpy())

    y_true_np = np.concatenate(y_true) if y_true else np.array([], dtype=int)
    y_pred_np = np.concatenate(y_pred) if y_pred else np.array([], dtype=int)
    tn, fp, fn, tp = _confusion_2x2(y_true_np, y_pred_np) if y_true_np.size else (0, 0, 0, 0)
    acc = float(np.mean(y_true_np == y_pred_np)) if y_true_np.size else 0.0
    bal_acc = balanced_accuracy_from_cm(tn, fp, fn, tp) if y_true_np.size else 0.0
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "n": int(y_true_np.size),
    }


def micro_finetune(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    *,
    action: ActionSpec,
    micro_epochs: int,
    max_batches_per_epoch: int,
    class_weights: torch.Tensor | None = None,
    grad_clip: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=float(action.lr), weight_decay=float(action.weight_decay))
    ce = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)

    model.train()
    train_losses: list[float] = []
    for _ in range(int(max(1, micro_epochs))):
        for batch_i, (batch_x, lbls) in enumerate(train_loader):
            if batch_i >= int(max_batches_per_epoch):
                break
            axl = batch_x["axl"].to(device)
            cor = batch_x["cor"].to(device)
            sag = batch_x["sag"].to(device)
            clin = batch_x.get("clin")
            if clin is not None:
                clin = clin.to(device)
            lbls = lbls.to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(axl, cor, sag, clin)
            loss = ce(out, lbls.long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

    train_summary = {
        "loss": float(np.mean(train_losses)) if train_losses else 0.0,
    }
    val_summary = evaluate_classifier(model, val_loader, device, class_weights=class_weights)
    return train_summary, val_summary


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, action_dim)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.net(state)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value


class PPOAgent:
    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        device: torch.device,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        update_epochs: int = 4,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
    ):
        self.device = device
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_eps = float(clip_eps)
        self.update_epochs = int(update_epochs)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)

        self.model = ActorCritic(state_dim=state_dim, action_dim=action_dim).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=float(lr))

        self._states: list[torch.Tensor] = []
        self._actions: list[torch.Tensor] = []
        self._logps: list[torch.Tensor] = []
        self._values: list[torch.Tensor] = []
        self._rewards: list[float] = []
        self._dones: list[float] = []

    def select_action(self, state: np.ndarray) -> tuple[int, float, float]:
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.model(state_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        return int(action.item()), float(logp.item()), float(value.squeeze(0).item())

    def store(self, *, state: np.ndarray, action: int, logp: float, value: float, reward: float, done: bool) -> None:
        self._states.append(torch.tensor(state, dtype=torch.float32, device=self.device))
        self._actions.append(torch.tensor(action, dtype=torch.long, device=self.device))
        self._logps.append(torch.tensor(logp, dtype=torch.float32, device=self.device))
        self._values.append(torch.tensor(value, dtype=torch.float32, device=self.device))
        self._rewards.append(float(reward))
        self._dones.append(1.0 if done else 0.0)

    def _compute_gae(self, next_value: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        rewards = self._rewards
        values = [v.item() for v in self._values] + [float(next_value)]
        dones = self._dones
        adv = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            adv.append(gae)
        adv = list(reversed(adv))
        adv_t = torch.tensor(adv, dtype=torch.float32, device=self.device)
        val_t = torch.stack(self._values)
        returns = adv_t + val_t
        # Use unbiased=False to avoid NaNs when there is a single step in the buffer.
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)
        return returns.detach(), adv_t.detach()

    def update(self) -> dict[str, float]:
        if not self._states:
            return {"loss": 0.0}

        states = torch.stack(self._states)
        actions = torch.stack(self._actions)
        old_logps = torch.stack(self._logps)
        returns, advantages = self._compute_gae(next_value=0.0)

        total_loss = 0.0
        for _ in range(self.update_epochs):
            logits, values = self.model(states)
            dist = torch.distributions.Categorical(logits=logits)
            logps = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratios = torch.exp(logps - old_logps)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, returns)
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
            self.opt.step()

            total_loss += float(loss.detach().cpu().item())

        stats = {
            "loss": total_loss / float(self.update_epochs),
            "steps": float(len(self._states)),
        }

        self._states.clear()
        self._actions.clear()
        self._logps.clear()
        self._values.clear()
        self._rewards.clear()
        self._dones.clear()
        return stats


class RLRefineEnv:
    def __init__(
        self,
        *,
        build_model,
        base_state_dict: dict[str, Any],
        train_loader,
        val_loader,
        device: torch.device,
        actions: list[ActionSpec],
        micro_epochs: int,
        max_batches_per_epoch: int,
        class_weights: torch.Tensor | None,
        seed: int,
        train_pytorch_model_fn=None,
    ):
        """
        Create an environment that performs supervised micro-finetuning steps as RL actions to refine a base classification model.
        
        Parameters:
            build_model (callable): Factory that returns a fresh, uninitialized model instance when called.
            base_state_dict (dict[str, Any]): Model state dictionary to load into each fresh model (loaded with non-strict semantics).
            train_loader: Iterable DataLoader providing training batches for micro-finetuning.
            val_loader: Iterable DataLoader providing validation batches for evaluation.
            device (torch.device): Compute device used for model/ tensor placement during training and evaluation.
            actions (list[ActionSpec]): Discrete action specifications; each ActionSpec contains hyperparameters (e.g., `lr`, `weight_decay`).
            micro_epochs (int): Number of fine-tuning epochs to run for each action.
            max_batches_per_epoch (int): Maximum number of training batches to consume per micro-epoch.
            class_weights (torch.Tensor | None): Optional 2-element tensor of class weights placed on `device` for the loss function.
            seed (int): Base random seed; the environment derives deterministic per-step seeds from this value.
            train_pytorch_model_fn (callable | None): Optional function to perform micro-finetuning. Expected signature:
                (model, train_loader, val_loader, device, action, micro_epochs, max_batches_per_epoch, class_weights=None, ...) -> (train_summary, val_summary).
                If `None`, a default micro-finetuning function is used.
        
        Notes:
            - The environment exposes `state_dim = 3` and `action_dim = len(actions)` to describe the observation and action spaces.
            - Internal tracking fields include `baseline_val_bal_acc`, `best_val_bal_acc`, `best_action_index`, and `best_state_dict`.
        """
        self._build_model = build_model
        self._base_sd = base_state_dict
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._device = device
        self.actions = actions
        self.micro_epochs = int(micro_epochs)
        self.max_batches_per_epoch = int(max_batches_per_epoch)
        self.class_weights = class_weights
        self.seed = int(seed)
        self._train_pytorch_model_fn = micro_finetune if train_pytorch_model_fn is None else train_pytorch_model_fn

        self.state_dim = 3
        self.action_dim = len(actions)
        self._step_index = 0

        self.baseline_val_bal_acc = 0.0
        self.best_val_bal_acc = -math.inf
        self.best_action_index: int | None = None
        self.best_state_dict: dict[str, Any] | None = None

    def reset(self) -> np.ndarray:
        self._step_index = 0
        # State starts at baseline.
        return np.array([self.baseline_val_bal_acc, 0.0, 0.0], dtype=np.float32)

    def step(self, action_index: int) -> tuple[np.ndarray, float, dict[str, Any]]:
        """
        Perform one environment step: apply the discrete action at the given index to micro-finetune a fresh copy of the base model and produce the next state, scalar reward, and an info dictionary.
        
        Parameters:
            action_index (int): Index of the discrete action to apply (must be within range of `self.actions`).
        
        Returns:
            tuple[np.ndarray, float, dict[str, Any]]: 
                - next_state: A float32 1-D array [val_balanced_accuracy, val_loss, step_index].
                - reward: The validation balanced accuracy improvement over the environment baseline (`val_balanced_accuracy - baseline_val_bal_acc`).
                - info: A dictionary with keys `step`, `action_index`, `action` (with `lr` and `weight_decay`), `reward`, `train_loss`, `val_loss`, `val_accuracy`, and `val_balanced_accuracy`.
        
        Raises:
            ValueError: If `action_index` is out of bounds for `self.actions`.
        """
        action_index = int(action_index)
        if action_index < 0 or action_index >= len(self.actions):
            raise ValueError(f"Invalid action index: {action_index}")

        set_global_seed(self.seed + 1000 + self._step_index)

        model = self._build_model().to(self._device)
        model.load_state_dict(self._base_sd, strict=False)

        train_summary, val_summary = self._train_pytorch_model_fn(
            model,
            self._train_loader,
            self._val_loader,
            self._device,
            action=self.actions[action_index],
            micro_epochs=self.micro_epochs,
            max_batches_per_epoch=self.max_batches_per_epoch,
            class_weights=self.class_weights,
        )

        reward = float(val_summary["balanced_accuracy"] - float(self.baseline_val_bal_acc))

        if float(val_summary["balanced_accuracy"]) > float(self.best_val_bal_acc):
            self.best_val_bal_acc = float(val_summary["balanced_accuracy"])
            self.best_action_index = int(action_index)
            self.best_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        info = {
            "step": int(self._step_index),
            "action_index": int(action_index),
            "action": {"lr": float(self.actions[action_index].lr), "weight_decay": float(self.actions[action_index].weight_decay)},
            "reward": float(reward),
            "train_loss": float(train_summary.get("loss", 0.0)),
            "val_loss": float(val_summary.get("loss", 0.0)),
            "val_accuracy": float(val_summary.get("accuracy", 0.0)),
            "val_balanced_accuracy": float(val_summary.get("balanced_accuracy", 0.0)),
        }
        self._step_index += 1
        next_state = np.array(
            [
                float(info["val_balanced_accuracy"]),
                float(info["val_loss"]),
                float(self._step_index),
            ],
            dtype=np.float32,
        )
        done = False
        return next_state, reward, info

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "state_dim": int(self.state_dim),
            "action_dim": int(self.action_dim),
            "micro_epochs": int(self.micro_epochs),
            "max_batches_per_epoch": int(self.max_batches_per_epoch),
            "baseline_val_balanced_accuracy": float(self.baseline_val_bal_acc),
            "best_val_balanced_accuracy": float(self.best_val_bal_acc),
            "actions": [
                {"lr": float(a.lr), "weight_decay": float(a.weight_decay)} for a in self.actions
            ],
        }


def dump_json(path, payload: dict[str, Any]) -> None:
    """
    Write a JSON-serializable mapping to the given filesystem path using UTF-8 encoding, pretty-printed with sorted keys and a trailing newline.
    
    Parameters:
        path (str | os.PathLike): Destination file path.
        payload (dict[str, Any]): JSON-serializable mapping to write.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


@dataclass(frozen=True)
class RLRefinementConfig:
    checkpoint_path: Path
    split_csv_path: Path
    output_dir: Path
    episodes: int = 8
    horizon: int = 4
    micro_epochs: int = 1
    train_subset: int | None = 120
    val_subset: int | None = 80
    dataset_dir: Path | None = None
    backbone: str = "efficientnet"
    clinical_features: list[str] | None = None
    seed: int = 42
    save_experiment_fn: Any = None
    train_pytorch_model_fn: Any = None


@dataclass(frozen=True)
class RLRefinementResult:
    best_hyperparameters: dict[str, float]
    refined_checkpoint_path: Path
    policy_path: Path
    history_path: Path
    metrics: dict[str, Any]


def _coerce_refinement_config(config) -> RLRefinementConfig:
    """
    Normalize and validate a refinement configuration into an RLRefinementConfig.
    
    Converts a user-supplied configuration (either an RLRefinementConfig instance or a mapping)
    into a fully typed RLRefinementConfig with filesystem paths coerced to pathlib.Path,
    numeric fields cast to int, optional fields normalized (e.g., dataset_dir -> Path or None,
    clinical_features -> list or None), and default values applied when a mapping is provided.
    
    Parameters:
        config: An RLRefinementConfig instance or a mapping containing configuration keys.
    
    Returns:
        RLRefinementConfig: A validated and normalized configuration object ready for use.
    
    Raises:
        TypeError: If `config` is neither an RLRefinementConfig nor a mapping.
    """
    if isinstance(config, RLRefinementConfig):
        return RLRefinementConfig(
            checkpoint_path=Path(config.checkpoint_path),
            split_csv_path=Path(config.split_csv_path),
            output_dir=Path(config.output_dir),
            episodes=int(config.episodes),
            horizon=int(config.horizon),
            micro_epochs=int(config.micro_epochs),
            train_subset=config.train_subset,
            val_subset=config.val_subset,
            dataset_dir=Path(config.dataset_dir) if config.dataset_dir is not None else None,
            backbone=str(config.backbone),
            clinical_features=list(config.clinical_features) if config.clinical_features is not None else None,
            seed=int(config.seed),
            save_experiment_fn=config.save_experiment_fn,
            train_pytorch_model_fn=config.train_pytorch_model_fn,
        )
    if isinstance(config, Mapping):
        dataset_dir = config.get("dataset_dir")
        clinical_features = config.get("clinical_features")
        return RLRefinementConfig(
            checkpoint_path=Path(config["checkpoint_path"]),
            split_csv_path=Path(config["split_csv_path"]),
            output_dir=Path(config["output_dir"]),
            episodes=int(config.get("episodes", 8)),
            horizon=int(config.get("horizon", 4)),
            micro_epochs=int(config.get("micro_epochs", 1)),
            train_subset=config.get("train_subset", 120),
            val_subset=config.get("val_subset", 80),
            dataset_dir=Path(dataset_dir) if dataset_dir is not None else None,
            backbone=str(config.get("backbone", "efficientnet")),
            clinical_features=list(clinical_features) if clinical_features is not None else None,
            seed=int(config.get("seed", 42)),
            save_experiment_fn=config.get("save_experiment_fn"),
            train_pytorch_model_fn=config.get("train_pytorch_model_fn"),
        )
    raise TypeError("config must be a RLRefinementConfig or mapping.")


def _sample_dataframe(df, limit: int | None, seed: int):
    """
    Return a copy of the given DataFrame limited to at most `limit` rows.
    
    Parameters:
        df (pandas.DataFrame): Source DataFrame to copy or sample from.
        limit (int | None): Maximum number of rows to return. If `None`, <= 0, or greater than or equal to the number of rows in `df`, a full copy is returned.
        seed (int): Random seed used when sampling rows.
    
    Returns:
        pandas.DataFrame: A copy of the original DataFrame containing either all rows or a random sample of `limit` rows.
    """
    if limit is None or limit <= 0 or len(df) <= limit:
        return df.copy()
    return df.sample(n=int(limit), random_state=int(seed)).copy()


def _filter_valid_mri_ids(df, split_name: str):
    """
    Filter a DataFrame to rows that have a non-empty `MRI_ID` column.
    
    Parameters:
        df (pandas.DataFrame): DataFrame containing an `MRI_ID` column.
        split_name (str): Name of the dataset split (e.g., "train" or "validation") used in error messages.
    
    Returns:
        pandas.DataFrame: A copy of `df` containing only rows where `MRI_ID` is not null and not an empty/whitespace string.
    
    Raises:
        ValueError: If `MRI_ID` is not a column of `df`, or if no rows remain after filtering.
    """
    if "MRI_ID" not in df.columns:
        raise ValueError("Split CSV is missing required column: MRI_ID")
    valid_mask = df["MRI_ID"].notna() & (df["MRI_ID"].astype(str).str.strip() != "")
    filtered = df[valid_mask].copy()
    if filtered.empty:
        raise ValueError(f"Split {split_name} has no rows with valid MRI_ID values for RL refinement.")
    return filtered


def _default_actions() -> list[ActionSpec]:
    """
    Provide the default discrete action set of learning-rate and weight-decay pairs for RL refinement.
    
    Each action is an ActionSpec pairing a learning rate (`lr`) with a weight decay (`weight_decay`). The returned list contains five candidate hyperparameter combinations spanning low-to-moderate learning rates and two weight-decay scales.
    
    Returns:
        list[ActionSpec]: Five ActionSpec entries representing candidate (lr, weight_decay) hyperparameter choices.
    """
    return [
        ActionSpec(lr=5e-5, weight_decay=1e-5),
        ActionSpec(lr=1e-4, weight_decay=1e-5),
        ActionSpec(lr=2e-4, weight_decay=1e-5),
        ActionSpec(lr=1e-4, weight_decay=1e-4),
        ActionSpec(lr=2e-4, weight_decay=1e-4),
    ]


def _class_weights_from_df(train_df, device: torch.device):
    """
    Compute per-class weights from a training DataFrame for use with PyTorch classification losses.
    
    Parameters:
        train_df (pandas.DataFrame): Training table containing a "Final_Group" column with class labels.
            Labels may be the strings "Nondemented" and "Demented" or numeric 0 and 1.
        device (torch.device): Device on which to allocate the returned tensor.
    
    Returns:
        torch.Tensor: A 2-element float32 tensor on `device` with weights for classes
        [Nondemented, Demented], suitable for passing to loss functions like `CrossEntropyLoss`.
    """
    counts = train_df["Final_Group"].value_counts()
    n_nondemented = max(int(counts.get("Nondemented", counts.get(0, 0))), 1)
    n_demented = max(int(counts.get("Demented", counts.get(1, 0))), 1)
    total = n_nondemented + n_demented
    return torch.tensor(
        [total / (2.0 * n_nondemented), total / (2.0 * n_demented)],
        dtype=torch.float32,
        device=device,
    )


def refine_model_with_rl(config):
    """
    Run PPO-based reinforcement learning to finetune a classification model's learning-rate/weight-decay hyperparameters and save the refined checkpoint, policy, and history.
    
    Parameters:
        config (Mapping | RLRefinementConfig): Configuration or mapping convertible to RLRefinementConfig that specifies paths (checkpoint, split CSV, output dir), RL settings (episodes, horizon), finetuning options (micro_epochs, train/val subset), dataset/model options, seed, and optional hooks (`save_experiment_fn`, `train_pytorch_model_fn`).
    
    Returns:
        RLRefinementResult: Result object containing `best_hyperparameters` (`lr`, `weight_decay`), file paths (`refined_checkpoint_path`, `policy_path`, `history_path`), and `metrics` with baseline and refined validation balanced accuracy and the best action index.
    
    Raises:
        ImportError: If PyTorch is not available (via internal _require_torch).
        ValueError: If the train or validation split is empty or contains no valid MRI IDs.
        RuntimeError: If RL training completes without producing a best checkpoint.
    """
    _require_torch()
    cfg = _coerce_refinement_config(config)
    set_global_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_df = load_split_dataframe(cfg.split_csv_path, required_columns=["split", "Final_Group", "MRI_ID"])
    dataset_dir = Path(cfg.dataset_dir) if cfg.dataset_dir is not None else Path(output_dir).parent / "axl"
    dataset_root = dataset_dir.parent

    train_df = split_df[split_df["split"] == "train"].copy()
    val_df = split_df[split_df["split"] == "validation"].copy()
    if train_df.empty or val_df.empty:
        raise ValueError("Split de treino/validação vazio para refinamento por RL.")
    train_df = _filter_valid_mri_ids(train_df, "train")
    val_df = _filter_valid_mri_ids(val_df, "validation")

    train_df = _sample_dataframe(train_df, cfg.train_subset, cfg.seed)
    val_df = _sample_dataframe(val_df, cfg.val_subset, cfg.seed + 1)

    train_df = populate_orientation_paths(train_df, dataset_root)
    val_df = populate_orientation_paths(val_df, dataset_root)
    if train_df.empty:
        raise ValueError("Split de treino ficou vazio apÃ³s resolver caminhos de orientaÃ§Ã£o para refinamento por RL.")
    if val_df.empty:
        raise ValueError("Split de validaÃ§Ã£o ficou vazio apÃ³s resolver caminhos de orientaÃ§Ã£o para refinamento por RL.")
    clinical_features = cfg.clinical_features or ["age", "education", "nwbv", "etiv", "asf"]
    train_tf, val_tf = build_transforms()

    train_ds = MultiOrientMRIDataset(train_df, train_tf, dataset_root, "original_path", "Final_Group", clinical_features=clinical_features)
    val_ds = MultiOrientMRIDataset(val_df, val_tf, dataset_root, "original_path", "Final_Group", clinical_features=clinical_features)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)

    device = select_device()
    base_state_dict = torch.load(cfg.checkpoint_path, map_location="cpu")

    def build_model():
        """
        Constructs a MultiOrientTabularFusionNet configured for classification using the current experiment settings.
        
        The network is created with the module's selected backbone, classification mode, specified number of tabular (clinical) features, MedicalNet depth 18, pretrained encoder weights, shared encoder across orientations, and 0.25 dropout.
        
        Returns:
            A configured `MultiOrientTabularFusionNet` instance ready for training or evaluation.
        """
        return MultiOrientTabularFusionNet(
            backbone=cfg.backbone,
            mode="classification",
            num_tabular_features=len(clinical_features) if clinical_features else 0,
            medicalnet_depth=18,
            pretrained=True,
            share_encoder=True,
            dropout=0.25,
        )

    baseline_model = build_model().to(device)
    baseline_model.load_state_dict(base_state_dict, strict=False)
    class_weights = _class_weights_from_df(train_df, device)
    baseline_metrics = evaluate_classifier(baseline_model, val_loader, device, class_weights=class_weights)

    env = RLRefineEnv(
        build_model=build_model,
        base_state_dict=base_state_dict,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        actions=_default_actions(),
        micro_epochs=cfg.micro_epochs,
        max_batches_per_epoch=max(1, int(cfg.horizon)),
        class_weights=class_weights,
        seed=cfg.seed,
        train_pytorch_model_fn=cfg.train_pytorch_model_fn,
    )
    env.baseline_val_bal_acc = float(baseline_metrics["balanced_accuracy"])

    agent = PPOAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        device=device,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        update_epochs=4,
    )

    rl_history = []
    for episode in range(int(max(1, cfg.episodes))):
        state = env.reset()
        episode_rewards = []
        last_info = None
        for step in range(int(max(1, cfg.horizon))):
            action_index, logp, value = agent.select_action(state)
            next_state, reward, info = env.step(action_index)
            done = step == int(max(1, cfg.horizon)) - 1
            agent.store(state=state, action=action_index, logp=logp, value=value, reward=reward, done=done)
            episode_rewards.append(float(reward))
            last_info = info
            state = next_state
        ppo_stats = agent.update()
        rl_history.append(
            {
                "episode": episode + 1,
                "reward_mean": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
                "reward_sum": float(np.sum(episode_rewards)) if episode_rewards else 0.0,
                "ppo_loss": float(ppo_stats.get("loss", 0.0)),
                "last_step": last_info,
            }
        )

    if env.best_state_dict is None or env.best_action_index is None:
        raise RuntimeError("RL refinement did not produce a best checkpoint.")

    best_action = env.actions[env.best_action_index]
    best_hyperparameters = {
        "lr": float(best_action.lr),
        "weight_decay": float(best_action.weight_decay),
    }

    refined_checkpoint_path = output_dir / f"best_{cfg.backbone}_classifier_rl_refined.pth"
    policy_path = output_dir / f"{cfg.backbone}_ppo_policy.pth"
    history_path = output_dir / f"{cfg.backbone}_rl_history.json"
    torch.save(env.best_state_dict, refined_checkpoint_path)
    torch.save(agent.model.state_dict(), policy_path)
    dump_json(
        history_path,
        {
            "config": {
                "checkpoint_path": str(cfg.checkpoint_path),
                "split_csv_path": str(cfg.split_csv_path),
                "episodes": int(cfg.episodes),
                "horizon": int(cfg.horizon),
                "micro_epochs": int(cfg.micro_epochs),
                "train_subset": cfg.train_subset,
                "val_subset": cfg.val_subset,
                "seed": int(cfg.seed),
            },
            "baseline_metrics": baseline_metrics,
            "env": env.to_jsonable(),
            "history": rl_history,
        },
    )

    refined_model = build_model().to(device)
    refined_model.load_state_dict(torch.load(refined_checkpoint_path, map_location=device), strict=False)
    refined_metrics = evaluate_classifier(refined_model, val_loader, device, class_weights=class_weights)

    # TODO: Naming is inconsistent between older DenseNetRefineEnv references and this module's RLRefineEnv.
    result = RLRefinementResult(
        best_hyperparameters=best_hyperparameters,
        refined_checkpoint_path=refined_checkpoint_path,
        policy_path=policy_path,
        history_path=history_path,
        metrics={
            "baseline_val_balanced_accuracy": float(baseline_metrics["balanced_accuracy"]),
            "refined_val_balanced_accuracy": float(refined_metrics["balanced_accuracy"]),
            "best_action_index": int(env.best_action_index),
        },
    )

    if callable(cfg.save_experiment_fn):
        cfg.save_experiment_fn(
            {
                "model": f"{cfg.backbone}_rl_refinement",
                "scenario": "ppo_refinement",
                "best_hparams": best_hyperparameters,
                "baseline_val_balanced_accuracy": float(baseline_metrics["balanced_accuracy"]),
                "refined_val_balanced_accuracy": float(refined_metrics["balanced_accuracy"]),
                "checkpoint_path": str(refined_checkpoint_path),
                "policy_path": str(policy_path),
                "history_path": str(history_path),
            }
        )

    return result


def refine_densenet_with_rl(config):
    """
    Run RL-based hyperparameter refinement configured to use a DenseNet backbone.
    
    Parameters:
        config (RLRefinementConfig | Mapping): Refinement configuration or a mapping coercible to RLRefinementConfig.
    
    Returns:
        RLRefinementResult: Result object containing the best hyperparameters, file paths for the refined checkpoint, saved policy and history, and baseline/refined metrics.
    """
    return refine_model_with_rl(config)
