# pretrainer/base_pretrainer.py
import os
import torch
from torch.cuda.amp import GradScaler

from utils.optimizer import build_optimizer


class BasePretrainer:
    """
    通用 Pretrainer 基类（训练系统地基），包括：
      - optimizer / scheduler / scaler 初始化
      - AMP 开关（由 cfg.Pretrain.amp 控制）
      - checkpoint 读写

    设计原则：
      BasePretrainer = 训练系统（不含具体 pretext task 逻辑）
      子类 Pretrainer = 算法逻辑（构造 target / loss / forward）
    """

    def __init__(self, cfg, model, device):
        self.cfg = cfg
        self.model = model.to(device)
        self.device = device
        self.trainer_type = "pretrain"

        # -------------------------------
        # Training configs
        # -------------------------------
        train_cfg = getattr(cfg, "Training", None)
        if train_cfg is None:
            raise ValueError("cfg.Training not found — 请在 YAML 中添加 Training 字段")

        self.max_epochs = getattr(train_cfg, "epochs", 1)
        self.grad_clip = getattr(train_cfg, "grad_clip", 1.0)

        # -------------------------------
        # AMP 开关：只由 cfg.Pretrain.amp 控制
        # -------------------------------
        pretrain_cfg = getattr(cfg, "Pretrain", None)
        # 若不存在 Pretrain 字段，默认不开 AMP（避免 silent behavior）
        self.use_amp = bool(pretrain_cfg is not None and getattr(pretrain_cfg, "amp", True))

        # -------------------------------
        # Optimizer & Scheduler
        # -------------------------------
        # 关键：增加 mode="pretrain"
        # build_optimizer 应在该模式下默认只优化 encoder + pretext head（跳过下游 heads）
        self.optimizer, self.scheduler = build_optimizer(
            model=self.model,
            train_cfg=train_cfg,
            mode="pretrain",
        )

        # -------------------------------
        # AMP Scaler
        # -------------------------------
        self.scaler = GradScaler(enabled=self.use_amp)

        # -------------------------------
        # Checkpoints
        # -------------------------------
        self.save_dir = getattr(cfg, "save_dir", "./checkpoints/pretrainer")
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_metric = -1e9
        self.global_step = 0

    def _move_batch_to_device(self, batch):
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out

    # ===============================================================
    # Save / Load
    # ===============================================================
    def save_checkpoint(self, name: str):
        """
        预训练 checkpoint 语义：
          - 以 encoder 权重为主要产物（便于下游复用）
          - 同时保留 model 整包 state_dict 以兼容当前工程（strict=False 加载）
        """
        path = os.path.join(self.save_dir, name)

        state = {
            # 兼容整包：方便你当前结构直接加载
            "model": self.model.state_dict(),

            # 研究语义增强：更推荐下游加载 encoder
            "trainer_type": self.trainer_type,
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_metric": self.best_metric,
            "global_step": self.global_step,
        }

        # 尽量存 encoder（如果你的 MultiModalModel 暴露了 encoder / encoders）
        if hasattr(self.model, "encoder"):
            try:
                state["encoder"] = self.model.encoder.state_dict()
            except Exception:
                pass
        if hasattr(self.model, "encoders"):
            # e.g. dict-like { "rgb":..., "pose":..., "text":... }
            try:
                enc_state = {}
                for k, enc in self.model.encoders.items():
                    enc_state[k] = enc.state_dict()
                state["encoders"] = enc_state
            except Exception:
                pass

        # 如果你在 pretrain 时挂了 pretext head，也可以存一下方便恢复训练
        if hasattr(self.model, "pretext_head"):
            try:
                state["pretext_head"] = self.model.pretext_head.state_dict()
            except Exception:
                pass

        torch.save(state, path)
        print(f"[Checkpoint] Saved: {path}")

    def load_checkpoint(self, path: str, strict: bool = False, load_optimizer: bool = True):
        """
        加载 checkpoint：
          - 默认 strict=False，允许你未来改 head / vocab 仍可加载 encoder
          - load_optimizer=True 用于断点续训；仅做初始化/评估可关掉
        """
        ckpt = torch.load(path, map_location=self.device)

        # 兼容整包加载
        if "model" in ckpt:
            self.model.load_state_dict(ckpt["model"], strict=strict)
        else:
            # 兼容只存 encoder 的情况（如果未来你选择仅保存 encoder）
            if "encoder" in ckpt and hasattr(self.model, "encoder"):
                self.model.encoder.load_state_dict(ckpt["encoder"], strict=strict)
            if "encoders" in ckpt and hasattr(self.model, "encoders"):
                for k, sd in ckpt["encoders"].items():
                    if k in self.model.encoders:
                        self.model.encoders[k].load_state_dict(sd, strict=strict)

        if load_optimizer and "optimizer" in ckpt and ckpt["optimizer"] is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if load_optimizer and "scaler" in ckpt and ckpt["scaler"] is not None:
            self.scaler.load_state_dict(ckpt["scaler"])

        self.best_metric = ckpt.get("best_metric", -1e9)
        self.global_step = ckpt.get("global_step", 0)
        print(f"[Checkpoint] Loaded from {path}")

    # Hooks for subclasses
    def train_epoch(self, loader):
        raise NotImplementedError

    def eval_epoch(self, loader):
        raise NotImplementedError
