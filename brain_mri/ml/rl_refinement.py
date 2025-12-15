from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def set_global_seed(seed: int) -> None:
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
    ):
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
        action_index = int(action_index)
        if action_index < 0 or action_index >= len(self.actions):
            raise ValueError(f"Invalid action index: {action_index}")

        set_global_seed(self.seed + 1000 + self._step_index)

        model = self._build_model().to(self._device)
        model.load_state_dict(self._base_sd, strict=False)

        train_summary, val_summary = micro_finetune(
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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
