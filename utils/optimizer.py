# utils/optimizer.py

import torch
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.config import cfg_get


def build_optimizer(cfg, params):
    """构建优化器，支持多种类型"""
    opt_name = cfg_get(cfg, "Training.optimizer", "adam").lower()
    lr = cfg_get(cfg, "Training.learning_rate", 1e-4)
    weight_decay = cfg_get(cfg, "Training.weight_decay", 0.01)

    if opt_name == "adam":
        optimizer = Adam(params, lr=lr, weight_decay=weight_decay)
    elif opt_name == "adamw":
        optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        momentum = cfg_get(cfg, "Training.momentum", 0.9)
        optimizer = SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"[build_optimizer] Unsupported optimizer: {opt_name}")

    return optimizer


def build_scheduler(cfg, optimizer):
    """构建调度器（可选调用）"""
    sched_name = cfg_get(cfg, "Training.scheduler", "none").lower()
    if sched_name == "cosine":
        epochs = cfg_get(cfg, "Training.epochs", 20)
        lr = cfg_get(cfg, "Training.learning_rate", 1e-4)
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    else:
        return None
