import os

import torch
from torchvision.utils import save_image


@torch.no_grad()
def debug_batch(x, y, out_dir="output/debug", prefix="batch"):
    """Inspeciona um batch: distribuição de y, estatísticas de x e salva grid de imagens."""
    os.makedirs(out_dir, exist_ok=True)

    y_cpu = y.detach().cpu()
    uniq = torch.unique(y_cpu).tolist()
    max_class = int(y_cpu.max().item()) if y_cpu.numel() > 0 else 0
    bincount = torch.bincount(y_cpu.to(torch.int64), minlength=max(2, max_class + 1)).tolist()
    print(f"[DBG] y unique={uniq} | bincount={bincount}")

    x_cpu = x.detach().float().cpu()
    print(
        "[DBG] x stats:",
        "min", float(x_cpu.min()),
        "max", float(x_cpu.max()),
        "mean", float(x_cpu.mean()),
        "std", float(x_cpu.std()),
        "zero_frac", float((x_cpu == 0).float().mean()),
    )

    per_std = x_cpu.view(x_cpu.size(0), -1).std(dim=1)
    print("[DBG] per-sample std: min/mean/max =",
          float(per_std.min()), float(per_std.mean()), float(per_std.max()))

    imgs = []
    for i in range(min(16, x_cpu.size(0))):
        xi = x_cpu[i]
        xi = (xi - xi.min()) / (xi.max() - xi.min() + 1e-6)
        imgs.append(xi)
    save_image(torch.stack(imgs, 0), os.path.join(out_dir, f"{prefix}.png"), nrow=8)
    print(f"[DBG] saved {out_dir}/{prefix}.png")


def debug_one_step(model, criterion, optimizer, x, y, clin=None, mode="classification",
                   use_focal=False, focal_gamma=2.0, loss_weights=None):
    """Roda um passo de treino e loga gradientes/atualização para detectar freeze."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    if clin is not None:
        logits = model(x, clin)
    else:
        logits = model(x)

    if mode == "regression":
        loss = criterion(logits.squeeze(), y)
    else:
        if use_focal:
            from .training_utils import focal_loss as focal_loss_fn
            loss = focal_loss_fn(logits, y.long(), gamma=focal_gamma, alpha=loss_weights)
        else:
            loss = criterion(logits, y.long())

    loss.backward()

    gsum = 0.0
    trainable = 0
    for p in model.parameters():
        if p.requires_grad:
            trainable += 1
            if p.grad is not None:
                gsum += float(p.grad.detach().norm().cpu())

    print(
        "[DBG] loss=", float(loss.item()),
        "| logits mean/std=", float(logits.mean().detach().cpu()), float(logits.std().detach().cpu()),
        "| trainable_params=", trainable,
        "| grad_norm_sum=", gsum,
    )

    ref = None
    for p in model.parameters():
        if p.requires_grad:
            ref = p
            break
    p0 = ref.detach().clone() if ref is not None else None
    optimizer.step()
    if ref is not None and p0 is not None:
        delta = float((ref.detach() - p0).abs().mean().cpu())
    else:
        delta = 0.0
    print("[DBG] param_delta_mean_abs=", delta)
