# utils/optimizer.py
# =====================================================
# Unified Optimizer Builder for Uni-SLM
# Supports:
#   - Backbone/Proj/Head different LR
#   - freeze skipping
#   - lr_mult
#   - AdamW or SGD
#   - optional scheduler
# =====================================================

import torch
from torch.optim import AdamW, SGD


def _collect_params(module, base_lr, lr_mult=1.0, weight_decay=0.0):
    """
    收集一个模块的所有可训练参数，并附带组别学习率。
    模块可能是 backbone / proj / head
    """
    if module is None:
        return []

    params = [p for p in module.parameters() if p.requires_grad]

    if len(params) == 0:
        return []

    return [{"params": params,
             "lr": base_lr * lr_mult,
             "weight_decay": weight_decay}]


# utils/optimizer.py
import torch


def build_optimizer(
    model,
    cfg=None,
    train_cfg=None,
    mode=None,      # ⭐ 新增：pretrain / finetune / None
):
    """
    Universal optimizer builder.

    Compatible with:
      - old finetune code
      - new pretrain code
      - future multi-mode extensions

    Args:
        model: nn.Module
        cfg: full config namespace (optional)
        train_cfg: cfg.Training namespace (preferred)
        mode: "pretrain" | "finetune" | None
    """

    # --------------------------------------------------
    # Resolve training config
    # --------------------------------------------------
    if train_cfg is not None:
        tcfg = train_cfg
    elif cfg is not None and hasattr(cfg, "Training"):
        tcfg = cfg.Training
    else:
        raise ValueError(
            "build_optimizer requires train_cfg or cfg.Training"
        )

    # --------------------------------------------------
    # Hyperparameters (defaults)
    # --------------------------------------------------
    lr = getattr(tcfg, "lr", 1e-4)
    weight_decay = getattr(tcfg, "weight_decay", 0.0)

    # Optional: mode-aware lr override
    if mode == "pretrain":
        lr = getattr(tcfg, "pretrain_lr", lr)
    elif mode == "finetune":
        lr = getattr(tcfg, "finetune_lr", lr)

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # --------------------------------------------------
    # Scheduler (optional)
    # --------------------------------------------------
    scheduler = None
    if getattr(tcfg, "use_scheduler", False):
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=getattr(tcfg, "step_size", 10),
            gamma=getattr(tcfg, "gamma", 0.1),
        )

    return optimizer, scheduler

