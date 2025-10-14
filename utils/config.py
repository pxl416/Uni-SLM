# utils/config.py
import os
import re
import yaml
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Tuple

_INT_RE = re.compile(r"^[+-]?[0-9]+$")
_FLOAT_RE = re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$")

REQUIRED_DS_KEYS = (
    "rgb_dirs",
    "pose_dirs",
    "train_label_paths",
    "val_label_paths",
    "test_label_paths",
)

def _maybe_cast_scalar(s: str) -> Any:
    """对带引号的标量做保守转换：
    - true/false/null -> bool/None
    - 纯整数字符串（无多余前导0）-> int
    - 浮点格式 -> float
    其他保持原样
    """
    ls = s.strip().lower()
    if ls == "true":
        return True
    if ls == "false":
        return False
    if ls in ("null", "none"):
        return None

    # 纯整数字符串（允许 + / -），避免诸如 "00123" 被误转
    if _INT_RE.match(s) and (len(s) == 1 or not (s.lstrip("+-").startswith("0") and s.strip() not in ("0", "+0", "-0"))):
        try:
            return int(s)
        except Exception:
            pass

    if _FLOAT_RE.match(s):
        try:
            return float(s)
        except Exception:
            pass

    return s

def _normalize_str(s: str, base_dir: str = "") -> str:
    """展开 env 与家目录，并在需要时拼接 base_dir，最后取绝对路径"""
    # 展开环境变量与 ~
    s2 = os.path.expandvars(s)
    s2 = os.path.expanduser(s2)
    # 若是相对路径且需要基准目录，拼上
    if base_dir and not os.path.isabs(s2) and ("/" in s2 or "\\" in s2):
        s2 = os.path.join(base_dir, s2)
    return os.path.abspath(s2)

def _parse_value(v: Any, base_dir: str = "") -> Any:
    """环境变量替换（允许嵌入）、路径归一化、保守数值/布尔转换"""
    if isinstance(v, str):
        # 先展开 env 与 ~
        expanded = _normalize_str(v, base_dir=base_dir)
        # 若是路径风格（包含分隔符）或以 . 开头，我们保留字符串（不再尝试数字转换）
        if os.sep in v or "/" in v or "\\" in v or v.strip().startswith(".") or ":" in v:
            return expanded
        # 否则尝试保守转换（只对“纯标量”字符串）
        return _maybe_cast_scalar(expanded)

    if isinstance(v, dict):
        return {k: _parse_value(val, base_dir=base_dir) for k, val in v.items()}
    if isinstance(v, list):
        return [_parse_value(i, base_dir=base_dir) for i in v]
    return v

def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(x) for x in obj]
    return obj

def to_dict(ns: Any) -> Any:
    """SimpleNamespace/list -> dict/list"""
    if isinstance(ns, SimpleNamespace):
        return {k: to_dict(getattr(ns, k)) for k in vars(ns)}
    if isinstance(ns, list):
        return [to_dict(x) for x in ns]
    return ns

def load_config(config_path: str, base_dir: str = "") -> SimpleNamespace:
    """通用 YAML 加载器：
    - 支持环境变量与 ~ 展开（可嵌入）
    - 相对路径可用 base_dir 归一化
    - 递归转 SimpleNamespace（点号访问）
    - 明确抛错
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Empty YAML: {config_path}")

    # 若未显式提供 base_dir，默认用配置文件所在目录
    if not base_dir:
        base_dir = os.path.dirname(os.path.abspath(config_path))

    parsed = _parse_value(raw, base_dir=base_dir)
    cfg = _to_namespace(parsed)
    _validate_config(cfg)
    return cfg

def _validate_config(cfg: SimpleNamespace) -> None:
    """最小必需校验：datasets/active_datasets 以及关键字段完整性"""
    if not hasattr(cfg, "datasets") or not isinstance(cfg.datasets, SimpleNamespace):
        raise ValueError("`datasets` section is required and must be a mapping.")

    # 若用户没给 active_datasets，就默认全用
    if not hasattr(cfg, "active_datasets") or cfg.active_datasets is None:
        all_names = list(vars(cfg.datasets).keys())
        cfg.active_datasets = all_names
    elif isinstance(cfg.active_datasets, list):
        # 确保 active 的都在 datasets 中
        for name in cfg.active_datasets:
            if name not in vars(cfg.datasets):
                raise ValueError(f"`active_datasets` contains unknown dataset: {name}")
    else:
        raise ValueError("`active_datasets` must be a list of dataset names.")

    # 校验每个数据集的必需字段
    for name in cfg.active_datasets:
        ds = getattr(cfg.datasets, name, None)
        if not isinstance(ds, SimpleNamespace):
            raise ValueError(f"Dataset `{name}` must be a mapping.")
        missing = [k for k in REQUIRED_DS_KEYS if not hasattr(ds, k)]
        if missing:
            raise ValueError(f"Dataset `{name}` missing keys: {missing}")

def iter_active_datasets(cfg: SimpleNamespace) -> Iterable[Tuple[str, SimpleNamespace]]:
    """遍历活跃数据集 (name, ds_cfg)"""
    for name in cfg.active_datasets:
        yield name, getattr(cfg.datasets, name)

def load_train_config(config_path: str = "config/trainer.yaml") -> SimpleNamespace:
    return load_config(config_path)

