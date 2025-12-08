# finetuner/base_finetuner.py
import os
import torch
from torch.cuda.amp import GradScaler

from utils.optimizer import build_optimizer


class BaseFinetuner:
    """
    通用 Finetuner 基类，包括:
      - optimizer / scheduler / scaler 初始化
      - use_amp 设置
      - checkpoint 读写
    """

    def __init__(self, cfg, model, device):
        self.cfg = cfg
        self.model = model.to(device)
        self.device = device

        # -------------------------------
        # Training configs
        # -------------------------------
        train_cfg = getattr(cfg, "Training", None)
        if train_cfg is None:
            raise ValueError("cfg.Training not found — 请在 YAML 中添加 Training 字段")

        self.max_epochs = getattr(train_cfg, "epochs", 1)
        self.grad_clip = getattr(train_cfg, "grad_clip", 1.0)

        # ⭐ AMP 开关 —— 现在 RecognitionFinetuner 就能使用 self.use_amp 了
        finetune_cfg = getattr(cfg, "Finetune", None)
        self.use_amp = getattr(finetune_cfg, "amp", True)

        # -------------------------------
        # Optimizer & Scheduler
        # -------------------------------
        self.optimizer, self.scheduler = build_optimizer(model, train_cfg)

        # -------------------------------
        # AMP Scaler
        # -------------------------------
        self.scaler = GradScaler(enabled=self.use_amp)

        # -------------------------------
        # Checkpoints
        # -------------------------------
        self.save_dir = cfg.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_metric = -1e9
        self.global_step = 0

    def _move_batch_to_device(self, batch):
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device)
            else:
                out[k] = v
        return out

    # ===============================================================
    # Save / Load
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
    def train_epoch(self, loader):
        raise NotImplementedError

    def eval_epoch(self, loader):
        raise NotImplementedError
