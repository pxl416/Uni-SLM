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


def build_optimizer(model, cfg_train):
    """
    model: MultiModalModel
    cfg_train: Training section in YAML
        必须提供:
            - optimizer: "adamw" or "sgd"
            - base_lr
            - weight_decay
            - lr_backbone
            - lr_proj
            - lr_head
            - scheduler name (optional)
    """

    opt_name = getattr(cfg_train, "optimizer", "adamw").lower()

    base_lr = getattr(cfg_train, "base_lr", 1e-4)
    lr_backbone = getattr(cfg_train, "lr_backbone", base_lr * 0.5)
    lr_proj     = getattr(cfg_train, "lr_proj", base_lr)
    lr_head     = getattr(cfg_train, "lr_head", base_lr)
    weight_decay = getattr(cfg_train, "weight_decay", 0.01)

    # ---------------------------------------------------------------------
    # 1. Collect all parameter groups
    # ---------------------------------------------------------------------
    param_groups = []

    # ---- RGB encoder
    param_groups += _collect_params(
        module=model.rgb_encoder.backbone,
        base_lr=lr_backbone,
        lr_mult=getattr(model.cfg.rgb_encoder.backbone, "lr_mult", 1.0),
        weight_decay=weight_decay,
    )
    param_groups += _collect_params(
        module=model.rgb_encoder.proj,
        base_lr=lr_proj,
        lr_mult=getattr(model.cfg.rgb_encoder.proj, "lr_mult", 1.0),
        weight_decay=weight_decay,
    )

    # ---- Text encoder
    param_groups += _collect_params(
        module=model.text_encoder.backbone,
        base_lr=lr_backbone * getattr(model.cfg.text_encoder.backbone, "lr_mult", 1.0),
        lr_mult=1.0,
        weight_decay=weight_decay,
    )
    param_groups += _collect_params(
        module=model.text_encoder.proj,
        base_lr=lr_proj,
        lr_mult=getattr(model.cfg.text_encoder.proj, "lr_mult", 1.0),
        weight_decay=weight_decay,
    )

    # ---- Recognition Head
    if hasattr(model, "recognition_head") and model.recognition_head is not None:
        param_groups += _collect_params(
            module=model.recognition_head,
            base_lr=lr_head,
            lr_mult=getattr(model.cfg.recognition_head, "lr_mult", 1.0),
            weight_decay=weight_decay,
        )

    # ---- Retrieval Head
    if hasattr(model, "retrieval_head") and model.retrieval_head is not None:
        param_groups += _collect_params(
            module=model.retrieval_head,
            base_lr=lr_head,
            lr_mult=getattr(model.cfg.retrieval_head, "lr_mult", 1.0),
            weight_decay=weight_decay,
        )

    # ---- Translation Head (MT5)
    if hasattr(model, "translation_head") and model.translation_head is not None:
        if model.translation_head.use_mt5:
            # MT5 encoder/decoder 参数分开处理
            param_groups += [{
                "params": model.translation_head.mt5.parameters(),
                "lr": lr_head * getattr(model.cfg.translation_head, "lr_mult", 0.2),
                "weight_decay": weight_decay,
            }]
        # prefix proj
        param_groups += _collect_params(
            module=model.translation_head.video_proj,
            base_lr=lr_head,
            lr_mult=getattr(model.cfg.translation_head, "lr_mult", 1.0),
            weight_decay=weight_decay,
        )

    # ---------------------------------------------------------------------
    # 2. Build optimizer
    # ---------------------------------------------------------------------
    if opt_name == "adamw":
        optimizer = AdamW(param_groups)
    elif opt_name == "sgd":
        optimizer = SGD(param_groups, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    # ---------------------------------------------------------------------
    # 3. Scheduler (optional)
    # ---------------------------------------------------------------------
    scheduler_type = getattr(cfg_train, "scheduler", None)

    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=getattr(cfg_train, "epochs", 30)
        )
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=getattr(cfg_train, "step_size", 10),
            gamma=getattr(cfg_train, "gamma", 0.1)
        )
    else:
        scheduler = None

    return optimizer, scheduler
