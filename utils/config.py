# utils/config.py

import os
import re
import yaml
from types import SimpleNamespace
from typing import Any, Iterable, Tuple, Dict

# ---------- 标量解析用到的正则 ----------
_INT_RE = re.compile(r"^[+-]?[0-9]+$")
_FLOAT_RE = re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$")

# ---------- 各数据集的“必需字段”定义（可扩展） ----------
REQUIRED_SCHEMAS: Dict[str, Tuple[str, ...]] = {
    "CSL_Daily": (
        "paths.root",
        "paths.rgb",
        "splits.train",
    ),
    "CSL_News": (
        "paths.root",
        "paths.rgb",
        "paths.pose",
        "splits.train",
        "splits.val",
        "splits.test",
    ),
}

# utils/config.py 中追加的工具函数

def load_yaml(path: str):
    """从 YAML 文件加载为 Python dict。"""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def dict_to_ns(d):
    """递归地把 dict 转为 SimpleNamespace，便于点号访问。"""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
    return d


# =============== 基础工具 ===============

def _maybe_cast_scalar(s: str) -> Any:
    ls = s.strip().lower()
    if ls == "true": return True
    if ls == "false": return False
    if ls in ("null", "none"): return None
    if _INT_RE.match(s) and not (s.lstrip("+-").startswith("0") and s.strip() not in ("0", "+0", "-0")):
        try: return int(s)
        except: pass
    if _FLOAT_RE.match(s):
        try: return float(s)
        except: pass
    return s

def _normalize_str(s: str, base_dir: str = "") -> str:
    raw = s
    s2 = os.path.expandvars(raw)
    s2 = os.path.expanduser(s2)
    looks_like_path = any(c in raw for c in ["/", "\\", os.sep]) or raw.strip().startswith(".")
    if looks_like_path:
        if base_dir and not os.path.isabs(s2):
            s2 = os.path.join(base_dir, s2)
        return os.path.abspath(s2)
    return raw

def _parse_value(v: Any, base_dir: str = "") -> Any:
    if isinstance(v, str):
        expanded = _normalize_str(v, base_dir=base_dir)
        if expanded == v:
            return _maybe_cast_scalar(expanded)
        return expanded
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
    if isinstance(ns, SimpleNamespace):
        return {k: to_dict(getattr(ns, k)) for k in vars(ns)}
    if isinstance(ns, list):
        return [to_dict(x) for x in ns]
    return ns

def _has_nested_key(ns: SimpleNamespace, key_path: str) -> bool:
    """检查 SimpleNamespace 中是否存在多级路径，例如 'paths.rgb'"""
    keys = key_path.split(".")
    curr = ns
    for k in keys:
        if isinstance(curr, SimpleNamespace) and hasattr(curr, k):
            curr = getattr(curr, k)
        else:
            return False
    return True

# =============== 配置加载 ===============

def load_config(config_path: str, base_dir: str = "") -> SimpleNamespace:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise ValueError(f"Empty YAML: {config_path}")
    if not base_dir:
        base_dir = os.path.dirname(os.path.abspath(config_path))
    parsed = _parse_value(raw, base_dir=base_dir)
    cfg = _to_namespace(parsed)
    _validate_config(cfg)
    return cfg

# =============== 校验 ===============

def _normalize_active_list(cfg) -> None:
    datasets_ns = getattr(cfg, "datasets", SimpleNamespace())
    valid = set(vars(datasets_ns).keys())
    act = getattr(cfg, "active_datasets", None)
    if act is None:
        cfg.active_datasets = list(valid)
        return
    if not isinstance(act, list):
        raise ValueError("`active_datasets` must be a list of dataset names.")
    normalized = []
    for name in act:
        base = os.path.basename(str(name))
        if name in valid:
            normalized.append(name)
        elif base in valid:
            normalized.append(base)
        else:
            raise ValueError(f"`active_datasets` contains unknown dataset: {name}\nAvailable: {sorted(valid)}")
    cfg.active_datasets = normalized

def _validate_per_dataset(name: str, ds_cfg: SimpleNamespace) -> None:
    schema = REQUIRED_SCHEMAS.get(name, None)
    if schema is None:
        print(f"[Config] Warning: No strict schema for dataset `{name}`.")
        return

    missing = [key for key in schema if not _has_nested_key(ds_cfg, key)]
    if missing:
        raise ValueError(
            f"Dataset `{name}` missing required keys: {missing}\n"
            f"Expected keys: {list(schema)}"
        )

def _validate_config(cfg: SimpleNamespace) -> None:
    if not hasattr(cfg, "datasets") or not isinstance(cfg.datasets, SimpleNamespace):
        raise ValueError("`datasets` section is required and must be a mapping.")
    _normalize_active_list(cfg)
    for name in cfg.active_datasets:
        ds_cfg = getattr(cfg.datasets, name, None)
        if ds_cfg is None:
            raise ValueError(f"Dataset `{name}` not found under `datasets`.")
        _validate_per_dataset(name, ds_cfg)

# =============== 用户接口 ===============

def cfg_get(cfg: SimpleNamespace, dotted_key: str, default=None):
    keys = dotted_key.split(".")
    curr = cfg
    for k in keys:
        if isinstance(curr, dict):
            curr = curr.get(k, default)
        elif isinstance(curr, SimpleNamespace):
            curr = getattr(curr, k, default)
        else:
            return default
    return curr

def load_config_with_args(args) -> SimpleNamespace:
    cfg = load_config(args.config)
    training = getattr(cfg, "Training", SimpleNamespace())
    if args.epochs is not None:
        training.epochs = args.epochs
    if args.batch_size is not None:
        training.batch_size = args.batch_size
    if args.lr_head is not None:
        training.learning_rate_head = args.lr_head
    if args.lr_backbone is not None:
        training.learning_rate_backbone = args.lr_backbone
    cfg.Training = training
    cfg.device = args.device if args.device is not None else "0"
    return cfg

def iter_active_datasets(cfg: SimpleNamespace) -> Iterable[Tuple[str, SimpleNamespace]]:
    for name in cfg.active_datasets:
        yield name, getattr(cfg.datasets, name)


def abs_path(root: str, p: str | None) -> str | None:
    """
    将相对路径安全地拼到 root 下；若 p 为 None 则返回 None。
    - 会展开环境变量（$HOME 等）
    - 会展开 ~
    - 最终返回绝对路径
    """
    if p is None:
        return None
    # 展开环境变量和 ~
    p_expanded = os.path.expanduser(os.path.expandvars(p))
    # 如果本身是绝对路径，直接返回
    if os.path.isabs(p_expanded):
        return os.path.abspath(p_expanded)
    # 否则拼到 root 下
    return os.path.abspath(os.path.join(root, p_expanded))


def dict_to_ns(d):
    """Convert nested dict → SimpleNamespace recursively."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_ns(x) for x in d]
    else:
        return d


def load_yaml_as_ns(path: str):
    """Load YAML → SimpleNamespace."""
    try:
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return dict_to_ns(raw)
    except Exception as e:
        print(f"Error loading YAML {path}: {e}")
        return None
