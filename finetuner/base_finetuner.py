# finetuner/base_finetuner.py
import os
import torch
from torch.cuda.amp import GradScaler, autocast

from utils.optimizer import build_optimizer


class BaseFinetuner:
    """
    通用 Finetuner 基类，包括:
      - optimizer / scheduler / scaler 初始化
      - checkpoint 读写
      - 通用 train_epoch / eval_epoch 模板（子类可 override）
    """

    def __init__(self, cfg, model, device):
        self.cfg = cfg
        self.model = model.to(device)
        self.device = device

        # ---- Training configs ----
        self.max_epochs = cfg.Training.epochs
        self.grad_clip = getattr(cfg.Training, "grad_clip", 1.0)

        # ---- Optimizer & Scheduler ----
        self.optimizer, self.scheduler = build_optimizer(model, cfg.Training)

        # ---- AMP scaler ----
        self.scaler = GradScaler(enabled=True)

        # ---- Checkpoints ----
        self.save_dir = cfg.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_metric = -1e9  # e.g. for accuracy / WER(negative)
        self.global_step = 0

    # ===============================================================
    #               Checkpoint Save / Load
    # ===============================================================
    def save_checkpoint(self, name):
        path = os.path.join(self.save_dir, name)
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_metric": self.best_metric,
            "global_step": self.global_step,
        }, path)
        print(f"[Checkpoint] Saved: {path}")

    # ===============================================================
    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"], strict=False)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.best_metric = ckpt.get("best_metric", -1e9)
        self.global_step = ckpt.get("global_step", 0)
        print(f"[Checkpoint] Loaded from {path}")

    # ===============================================================
    # 子类必须实现
    # ===============================================================
    def train_epoch(self, loader):
        raise NotImplementedError

    def eval_epoch(self, loader):
        raise NotImplementedError
