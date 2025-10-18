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
# 注意：这里用的是你当前 config.yaml 的键名（单数）
REQUIRED_SCHEMAS: Dict[str, Tuple[str, ...]] = {
    # 帧目录版的 CSL_Daily
    "CSL_Daily": (
        "root",
        "rgb_dir",
        "split_file",
    ),
    # 旧版 CSL_News（视频+姿态+pkl/json）
    "CSL_News": (
        "rgb_dir",
        "pose_dir",
        "train_labels",
        "val_labels",
        "test_labels",
    ),
    # 其他数据集可在此新增：
    # "WLASL": (...),
}

# =============== 基础工具 ===============

def _maybe_cast_scalar(s: str) -> Any:
    """对纯标量字符串做保守类型转换；路径风格不转。"""
    ls = s.strip().lower()
    if ls == "true":
        return True
    if ls == "false":
        return False
    if ls in ("null", "none"):
        return None

    # 纯整数字符串（允许 + / -），避免 "00123" 这类被误转
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
    """展开 env 与 ~，相对路径基于 base_dir 转绝对路径；非路径风格保持原样。"""
    # 原始值先保留
    raw = s
    # 展开环境变量与 ~
    s2 = os.path.expandvars(raw)
    s2 = os.path.expanduser(s2)
    # 若像路径（包含分隔符或以 . 开头），再处理为绝对路径
    looks_like_path = (os.sep in raw) or ("/" in raw) or ("\\" in raw) or raw.strip().startswith(".") or (":" in raw)
    if looks_like_path:
        if base_dir and not os.path.isabs(s2):
            s2 = os.path.join(base_dir, s2)
        return os.path.abspath(s2)
    return raw  # 非路径：不做路径归一化

def _parse_value(v: Any, base_dir: str = "") -> Any:
    """环境变量替换 & 路径归一化 & 保守数值/布尔转换（递归）"""
    if isinstance(v, str):
        # 先做路径/变量处理；若不是路径风格，再尝试标量转换
        expanded = _normalize_str(v, base_dir=base_dir)
        if expanded is v:  # 不是路径风格
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
    """SimpleNamespace/list -> dict/list"""
    if isinstance(ns, SimpleNamespace):
        return {k: to_dict(getattr(ns, k)) for k in vars(ns)}
    if isinstance(ns, list):
        return [to_dict(x) for x in ns]
    return ns

# =============== 配置加载 ===============

def load_config(config_path: str, base_dir: str = "") -> SimpleNamespace:
    """通用 YAML 加载器：
    - 支持环境变量与 ~ 展开（可嵌入）
    - 相对路径按配置文件所在目录归一化
    - 递归转 SimpleNamespace（点号访问）
    - 加载后做结构校验
    """
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
    """把 active_datasets 里的“路径/误写”转换为合法数据集名（取 basename）。"""
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
            raise ValueError(
                f"`active_datasets` contains unknown dataset: {name}\n"
                f"Available: {sorted(valid)}"
            )
    cfg.active_datasets = normalized

def _validate_per_dataset(name: str, ds_cfg: SimpleNamespace) -> None:
    """按数据集名检查必要键；未知数据集仅做最小校验并给出警告。"""
    schema = REQUIRED_SCHEMAS.get(name, None)
    if schema is None:
        # 最小校验：必须是映射；给出提示，方便以后扩展
        if not isinstance(ds_cfg, SimpleNamespace):
            raise ValueError(f"Dataset `{name}` must be a mapping.")
        print(f"[Config] Warning: No strict schema for dataset `{name}`. Only minimal checks applied.")
        return

    if not isinstance(ds_cfg, SimpleNamespace):
        raise ValueError(f"Dataset `{name}` must be a mapping.")

    missing = [k for k in schema if not hasattr(ds_cfg, k)]
    if missing:
        raise ValueError(
            f"Dataset `{name}` missing required keys: {missing}\n"
            f"Expected keys for `{name}`: {list(schema)}"
        )

def _validate_config(cfg: SimpleNamespace) -> None:
    """总体校验：datasets、active_datasets、每个数据集的必需键。"""
    if not hasattr(cfg, "datasets") or not isinstance(cfg.datasets, SimpleNamespace):
        raise ValueError("`datasets` section is required and must be a mapping.")

    _normalize_active_list(cfg)

    for name in cfg.active_datasets:
        ds_cfg = getattr(cfg.datasets, name, None)
        if ds_cfg is None:
            raise ValueError(f"Dataset `{name}` not found under `datasets`.")
        _validate_per_dataset(name, ds_cfg)

# =============== 便利函数 ===============

def iter_active_datasets(cfg: SimpleNamespace) -> Iterable[Tuple[str, SimpleNamespace]]:
    """遍历活跃数据集 (name, ds_cfg)"""
    for name in cfg.active_datasets:
        yield name, getattr(cfg.datasets, name)

def load_train_config(config_path: str = "config/config.yaml") -> SimpleNamespace:
    return load_config(config_path)
