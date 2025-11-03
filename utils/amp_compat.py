# utils/amp_compat.py
import inspect
import torch

try:
    from torch.amp import autocast as _autocast
    from torch.amp import GradScaler as _GradScaler
    _USE_TORCH_AMP = True
except Exception:
    from torch.cuda.amp import autocast as _autocast
    from torch.cuda.amp import GradScaler as _GradScaler
    _USE_TORCH_AMP = False

def _choose_dtype(pref: str | None) -> torch.dtype:
    # 允许外部传入 'bf16' / 'fp16' / None
    if pref:
        pref = str(pref).lower()
        if pref in {"bf16", "bfloat16"} and torch.cuda.is_available():
            # 仅在 GPU 支持时返回 bf16
            major, minor = torch.cuda.get_device_capability(0)
            if (major, minor) >= (8, 0):  # Ampere+
                return torch.bfloat16
        if pref in {"fp16", "float16", "half"}:
            return torch.float16
    # 默认：A100 优先用 bf16，否则 fp16
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        if (major, minor) >= (8, 0):
            return torch.bfloat16
    return torch.float16

def make_autocast(dtype: str | None = None):
    """返回 autocast 上下文；2.x 需要 device_type 参数，1.x 只要 dtype。"""
    dt = _choose_dtype(dtype)
    if _USE_TORCH_AMP:
        return _autocast(device_type='cuda', dtype=dt)
    else:
        return _autocast(dtype=dt)

def make_scaler(enabled: bool = True, dtype: str | None = None,
                init_scale=2.**16, growth_factor=2.0,
                backoff_factor=0.5, growth_interval=2000):
    """
    bf16 不需要/不支持梯度缩放；fp16 才启用。
    """
    dt = _choose_dtype(dtype)
    use_scaler = enabled and (dt == torch.float16)

    sig = inspect.signature(_GradScaler)
    cand = dict(
        enabled=use_scaler,
        init_scale=init_scale,
        growth_factor=growth_factor,
        backoff_factor=backoff_factor,
        growth_interval=growth_interval,
    )
    kwargs = {k: v for k, v in cand.items() if k in sig.parameters}
    return _GradScaler(**kwargs)
