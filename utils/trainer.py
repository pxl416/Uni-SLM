# utils/trainer.py
# -*- coding: utf-8 -*-
"""
通用训练工具：
- 初始化 Trainer 通用字段（epochs / grad_clip / batch_size / num_workers / save_dir / wandb）
- 构建 train/dev DataLoader
- 统一的 RGB 编码函数（feat + mask）
- 统一的优化器构建函数（head / backbone 不同 lr）
- AMP 支持（GradScaler + autocast 上下文）

可同时服务 finetuner.py / pretrain.py / 其他训练脚本。
"""

import os
import logging
import contextlib
from types import SimpleNamespace
from typing import List, Optional, Dict, Any, Iterable, Tuple

import torch
from torch.optim import AdamW

from datasets.datasets import create_dataloader
from utils.config import cfg_get

logger = logging.getLogger(__name__)


# =========================================================
# 小工具：从 cfg 解析 wandb / amp / save_dir 等
# =========================================================
def _resolve_wandb_from_cfg(cfg, default: bool = False) -> bool:
    """
    尝试从 cfg 中解析是否启用 wandb。
    你可以根据自己的 YAML 结构继续扩展这里的规则。
    """
    # 优先从 Logging/use_wandb 读取
    logging_cfg = getattr(cfg, "Logging", None)
    if logging_cfg is not None and hasattr(logging_cfg, "use_wandb"):
        return bool(logging_cfg.use_wandb)

    # Finetune 里可能也会有 use_wandb 字段
    finetune_cfg = getattr(cfg, "Finetune", None)
    if finetune_cfg is not None and hasattr(finetune_cfg, "use_wandb"):
        return bool(finetune_cfg.use_wandb)

    # 顶层 cfg.use_wandb
    if hasattr(cfg, "use_wandb"):
        return bool(getattr(cfg, "use_wandb"))

    return bool(default)


def _resolve_save_dir(cfg, default: str = "checkpoints") -> str:
    """
    解析 checkpoint 保存目录：
    优先级示例：
      1) cfg.Finetune.save_dir / cfg.Pretrain.save_dir
      2) cfg.save_dir
      3) default
    """
    # 针对 finetuner/pretrain 任务的子配置
    for subkey in ["Finetune", "Pretrain", "Training"]:
        sub_cfg = getattr(cfg, subkey, None)
        if sub_cfg is not None and hasattr(sub_cfg, "save_dir"):
            return os.path.abspath(getattr(sub_cfg, "save_dir"))

    # 顶层
    if hasattr(cfg, "save_dir"):
        return os.path.abspath(getattr(cfg, "save_dir"))

    return os.path.abspath(default)


def _resolve_amp_flag_from_cfg(cfg, default: bool = True) -> bool:
    """
    尝试从 cfg 中解析是否启用 AMP（自动混合精度）。
    """
    # 顶层 use_amp
    if hasattr(cfg, "use_amp"):
        return bool(getattr(cfg, "use_amp"))

    # Finetune/Pretrain 层的 amp 或 use_amp
    for subkey in ["Finetune", "Pretrain", "Training"]:
        sub_cfg = getattr(cfg, subkey, None)
        if sub_cfg is None:
            continue
        if hasattr(sub_cfg, "amp"):
            return bool(getattr(sub_cfg, "amp"))
        if hasattr(sub_cfg, "use_amp"):
            return bool(getattr(sub_cfg, "use_amp"))

    return bool(default)


# =========================================================
# 1. 初始化 Trainer 的通用字段
# =========================================================
def init_trainer_common(trainer, cfg, device):
    trainer.cfg = cfg
    trainer.device = device

    # 1. save dir
    trainer.save_dir = getattr(cfg, "save_dir", "checkpoints")
    os.makedirs(trainer.save_dir, exist_ok=True)

    # 2. best metric
    trainer.best_metric = float("-inf")

    # 3. AMP 开关：从 cfg.Finetune.amp 读（没有就默认 True）
    use_amp = cfg_get(cfg, "Finetune.amp", True)
    trainer.use_amp = bool(use_amp and str(device).startswith("cuda"))

    # 4. GradScaler（用旧接口，避免报错；就算有 deprecate warning 也能跑）
    if trainer.use_amp:
        from torch.cuda.amp import GradScaler
        trainer.scaler = GradScaler(enabled=True)
    else:
        trainer.scaler = None

    # 5. checkpoint dir（如果你想用单独目录的话）
    trainer.ckpt_dir = cfg_get(cfg, "Finetune.checkpoint_dir", trainer.save_dir)




# =========================================================
# 2. DataLoader 构建
# =========================================================
def build_data_loaders(
    cfg,
    *,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    extra_args: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    统一构建 train / dev 两个 DataLoader。

    参数：
      - cfg: 全局配置(SimpleNamespace 或类似结构)
      - batch_size, num_workers: 如不提供，则从 cfg.Training 中取
      - extra_args: 额外塞进 args 的字段（如 rgb_support, pose_support 等）

    返回：
      train_loader, val_loader
    """
    train_cfg = getattr(cfg, "Training", SimpleNamespace())

    if batch_size is None:
        batch_size = getattr(train_cfg, "batch_size", 1)
    if num_workers is None:
        num_workers = getattr(train_cfg, "num_workers", 4)

    args_dict: Dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "seed": getattr(cfg, "seed", 3407),
        # 有些版本里 BaseDataset 会从 args 里读 cfg，这里保持兼容字段名称
        "cfg": None,
    }

    if extra_args:
        args_dict.update(extra_args)

    args = SimpleNamespace(**args_dict)

    train_loader = create_dataloader(args, cfg, phase="train")
    val_loader = create_dataloader(args, cfg, phase="dev")

    logger.info(f"[Data] train batches = {len(train_loader)}")
    logger.info(f"[Data] val   batches = {len(val_loader)}")

    return train_loader, val_loader


# =========================================================
# 3. RGB 编码 + mask 构建
# =========================================================
def encode_rgb_features(
    rgb_encoder: torch.nn.Module,
    rgb: torch.Tensor,
    rgb_len: torch.Tensor,
    device: torch.device,
    *,
    check_3d: bool = False,
    clip_to_len: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    统一的 RGB 特征编码函数。

    输入：
      - rgb: [B, T, 3, H, W] 或其他模型支持的形状
      - rgb_len: [B]，每个样本有效帧数
    输出：
      - feat: [B, T, D]
      - mask: [B, T]，True 表示有效帧
    """
    rgb = rgb.to(device)
    rgb_len = rgb_len.to(device)

    feat = rgb_encoder(rgb)  # 期望输出 [B, T, D]

    if check_3d and feat.ndim != 3:
        raise ValueError(
            f"[encode_rgb_features] Expected encoder output [B,T,D], got shape={feat.shape}"
        )

    if feat.ndim < 3:
        raise ValueError(
            f"[encode_rgb_features] Encoder output must have at least 3 dims [B,T,D], got {feat.shape}"
        )

    B, T = feat.shape[0], feat.shape[1]

    mask = torch.zeros((B, T), dtype=torch.bool, device=device)
    for i in range(B):
        valid = int(rgb_len[i].item())
        if clip_to_len and valid > T:
            valid = T
        mask[i, :valid] = True

    return feat, mask


# =========================================================
# 4. 参数收集 & 优化器构建
# =========================================================
def _collect_parameters(
    modules: Iterable[Any],
    recursive: bool = True,
) -> List[torch.nn.Parameter]:
    """
    递归收集参数：
      - 支持 nn.Module
      - 支持 list/tuple 嵌套
      - 支持 recursive=False 时只收集当前模块，不递归子模块
    """
    params: List[torch.nn.Parameter] = []
    for m in modules:
        if m is None:
            continue
        if isinstance(m, torch.nn.Module):
            if recursive:
                params.extend(
                    p for p in m.parameters() if p.requires_grad
                )
            else:
                params.extend(
                    p for p in m.parameters(recurse=False) if p.requires_grad
                )
        elif isinstance(m, (list, tuple)):
            params.extend(_collect_parameters(m, recursive=recursive))
        # 其他类型（比如已经是 Parameter 列表）可以在未来需要时扩展
    return params


def build_optimizer(
    head_modules: Iterable[Any],
    backbone_modules: Iterable[Any],
    train_cfg: Optional[Any] = None,
    *,
    default_lr_head: float = 3e-4,
    default_lr_backbone: float = 5e-5,
    weight_decay: float = 0.0,
    recursive: bool = True,
) -> AdamW:
    """
    通用的优化器构建函数。

    - 从 train_cfg 中读取：
        - learning_rate_head 或 lr_head
        - learning_rate_backbone 或 lr_backbone
      两者都不存在时使用默认值。

    - head_modules: 例如 [self.task_head] / [self.head]
    - backbone_modules: 例如 [self.rgb, self.text] / [self.rgb]
    """
    if train_cfg is None:
        train_cfg = SimpleNamespace()

    # 兼容两种命名：learning_rate_head / lr_head
    lr_head = getattr(
        train_cfg,
        "learning_rate_head",
        getattr(train_cfg, "lr_head", default_lr_head),
    )
    lr_backbone = getattr(
        train_cfg,
        "learning_rate_backbone",
        getattr(train_cfg, "lr_backbone", default_lr_backbone),
    )

    head_params = _collect_parameters(head_modules, recursive=recursive)
    back_params = _collect_parameters(backbone_modules, recursive=recursive)

    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": float(lr_head)})
    if back_params:
        param_groups.append({"params": back_params, "lr": float(lr_backbone)})

    if not param_groups:
        raise RuntimeError("[build_optimizer] No parameters to optimize.")

    optimizer = AdamW(param_groups, weight_decay=weight_decay)
    logger.info(
        f"[Optimizer] head lr={lr_head}, backbone lr={lr_backbone}, "
        f"weight_decay={weight_decay}"
    )
    return optimizer


# =========================================================
# 5. AMP 支持：GradScaler & autocast 上下文
# =========================================================
def get_amp_scaler_if_enabled(
    cfg,
    enabled_by_default: bool = True,
) -> Optional[torch.cuda.amp.GradScaler]:
    """
    根据配置返回 GradScaler（或 None）

    规则：
      - 优先从 cfg / cfg.Finetune / cfg.Pretrain / cfg.Training 解析 amp/use_amp
      - 如果启用且有 CUDA，可用则返回 GradScaler，否则返回 None
    """
    use_amp = _resolve_amp_flag_from_cfg(cfg, default=enabled_by_default)
    if use_amp and torch.cuda.is_available():
        logger.info("[AMP] Enabled, creating GradScaler.")
        return torch.cuda.amp.GradScaler()
    logger.info("[AMP] Disabled.")
    return None


@contextlib.contextmanager
def amp_autocast_if_enabled(
    cfg,
    enabled_by_default: bool = True,
    **autocast_kwargs,
):
    """
    AMP 自动转换上下文管理器：

    使用方式：
        with amp_autocast_if_enabled(cfg):
            outputs = model(inputs)

    可选参数例如 dtype/device_type，可通过 autocast_kwargs 传入。
    """
    use_amp = _resolve_amp_flag_from_cfg(cfg, default=enabled_by_default)
    cuda_ok = torch.cuda.is_available()

    if use_amp and cuda_ok:
        with torch.cuda.amp.autocast(**autocast_kwargs):
            yield
    else:
        # 不启用 AMP，直接执行
        yield
# utils/trainer.py 中追加

import torch
import os
import logging
from typing import Iterable, Any, List
# 如果前面已经有 logger 就不要重复
logger = logging.getLogger(__name__)


# =========================================================
# A. 全局随机数种子
# =========================================================
def set_global_seed(seed: int = 3407):
    """
    设置 Python / NumPy / PyTorch / CUDA 的随机种子。
    建议从 cfg.seed 里读取，然后传进来。
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"[Seed] Global seed set to {seed}")


# =========================================================
# B. 参数分组（兼容旧版 params_with_lr）
# =========================================================
def params_with_lr(modules: Iterable[Any], lr: float, recursive: bool = True):
    """
    兼容旧版的参数分组函数：
      - modules: [self.head] / [self.rgb, self.text] 等
      - lr: 学习率
    内部复用 _collect_parameters（如果你前面已经定义了），
    否则就做一个简单版。
    """
    # 如果你已经在本文件里定义了 _collect_parameters，就用它：
    try:
        from utils.trainer import _collect_parameters  # 仅用于类型提示，实际在本文件中
    except Exception:
        _collect_parameters = None

    params: List[torch.nn.Parameter] = []

    if _collect_parameters is not None:
        # 使用更强的递归收集版本
        params = _collect_parameters(modules, recursive=recursive)
    else:
        # 兜底实现：简单遍历
        for m in modules:
            if m is None:
                continue
            if isinstance(m, torch.nn.Module):
                if recursive:
                    params.extend(p for p in m.parameters() if p.requires_grad)
                else:
                    params.extend(p for p in m.parameters(recurse=False) if p.requires_grad)
            elif isinstance(m, (list, tuple)):
                for sub in m:
                    if isinstance(sub, torch.nn.Module):
                        params.extend(p for p in sub.parameters() if p.requires_grad)

    return {"params": params, "lr": float(lr)} if params else None


# =========================================================
# C. Checkpoint 保存 / 加载
# =========================================================
def save_checkpoint(path: str, model_dict: dict, optimizer, epoch: int, best_metric: float):
    """
    统一的 checkpoint 保存函数。
      - model_dict: {name: nn.Module, ...}
      - optimizer: torch.optim.Optimizer
    """
    state = {
        "model": {k: v.state_dict() for k, v in model_dict.items() if v is not None},
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    logger.info(f"[Checkpoint] Saved to {path}")


def load_checkpoint(path: str, model_dict: dict, optimizer=None):
    """
    从 checkpoint 恢复：
      - model_dict: {name: nn.Module, ...}
      - optimizer: 如果传入则恢复其 state_dict
    返回： (epoch, best_metric)
    """
    ckpt = torch.load(path, map_location="cpu")

    # 恢复模型
    model_state = ckpt.get("model", {})
    for name, module in model_dict.items():
        if module is not None and name in model_state:
            module.load_state_dict(model_state[name], strict=False)
            logger.info(f"[Checkpoint] Loaded weights for '{name}'")

    # 恢复优化器
    if optimizer is not None and ckpt.get("optimizer", None) is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        logger.info("[Checkpoint] Optimizer state restored.")

    return ckpt.get("epoch", 0), ckpt.get("best_metric", None)

def log_metrics_if_enabled(trainer, metrics: dict, prefix: str = ""):
    """统一的 wandb logging 帮助函数."""
    if not getattr(trainer, "wandb_enabled", False):
        return
    try:
        import wandb
    except Exception:
        return

    log_dict = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            key = f"{prefix}/{k}" if prefix else k
            log_dict[key] = v

    if log_dict:
        wandb.log(log_dict)
