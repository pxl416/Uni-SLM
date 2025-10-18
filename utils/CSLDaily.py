# utils/CSLDaily.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os, glob
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from pathlib import Path

from utils.dataset import BaseDataset


# -------------------------
# 小工具
# -------------------------
def _get(cfg: dict, dotted: str, default=None):
    cur = cfg
    for k in dotted.split('.'):
        if isinstance(cur, dict):
            cur = cur.get(k, default)
        else:
            cur = getattr(cur, k, default)
        if cur is default:
            break
    return cur

def _abs_path(root: str, p: Optional[str]) -> Optional[str]:
    if p is None:
        return None
    return p if os.path.isabs(p) else os.path.join(root, p)

def list_frame_paths(frame_dir: str) -> List[str]:
    if not frame_dir or not os.path.isdir(frame_dir):
        return []
    pats = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        pats.extend(glob.glob(os.path.join(frame_dir, ext)))
    pats = [p for p in pats if os.path.getsize(p) > 0]  # 过滤 0 字节
    if not pats:
        return []

    def key_fn(p):
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            return int(stem)
        except ValueError:
            return stem

    return sorted(pats, key=key_fn)

def _safe_open(path: str) -> Optional[Image.Image]:
    try:
        img = Image.open(path).convert("RGB")
        return img
    except (UnidentifiedImageError, OSError, ValueError):
        return None


# -------------------------
# CSLDaily 数据集（帧目录）
# -------------------------
# class CSLDailyDataset(BaseDataset):
#     """
#     读取 CSL_Daily 帧目录（sentence/）与可选 split 文件（sentence_label/split_*.txt）。
#     复用 BaseDataset.collate_fn：__getitem__ 返回
#       (name, pose_sample(dict或空), text(str), indices(np.ndarray), support(dict))
#     其中 support 至少包含：
#       - 'rgb_img': [T,3,H,W] Tensor
#       - 'rgb_img_indices': 长度 T 的采样帧索引
#     """
#     def __init__(self, args, cfg, phase: str):
#         super().__init__(args, cfg, phase)
#
#         # 1) 读取配置（每数据集块）
#         ds_cfg = _get(cfg, "datasets.CSL_Daily", {}) or {}
#         root = _get(ds_cfg, "root")
#         if not root:
#             raise ValueError("[CSL_Daily] 缺少 datasets.CSL_Daily.root")
#         self.root = os.path.abspath(root)
#
#         rgb_dir_conf    = _get(ds_cfg, "rgb_dir", "sentence")
#         split_file_conf = _get(ds_cfg, "split_file", None)
#
#         self.rgb_dir    = _abs_path(self.root, rgb_dir_conf)
#         self.split_file = _abs_path(self.root, split_file_conf) if split_file_conf else None
#
#         # 基本开关
#         self.use_rgb_ds = bool(_get(ds_cfg, "use_rgb", True))
#         # 与 Base 的 args.rgb_support 联合：只要任一为 False 就禁用 RGB
#         self.rgb_support = self.rgb_support and self.use_rgb_ds
#
#         # 文本相关（当前只返回字符串；需要 id 编码可在此扩展）
#         self.token_level  = _get(ds_cfg, "token_level", "char")
#         self.max_text_len = int(_get(ds_cfg, "max_text_len", 128))
#
#         # temporal（注意：你的 config 把 augment/augment_val 放在 temporal 下面）
#         tmp_cfg   = _get(ds_cfg, "temporal", {}) or {}
#         self.max_length = int(_get(tmp_cfg, "max_frames", _get(tmp_cfg, "T", self.max_length)))  # 优先用 max_frames/T 作为长度
#         self.jitter     = bool(_get(tmp_cfg, "jitter", True))
#         self.min_frames = int(_get(tmp_cfg, "min_frames", 1))
#
#         aug_train_cfg = _get(tmp_cfg, "augment", {}) or {}
#         aug_val_cfg   = _get(tmp_cfg, "augment_val", {}) or {}
#
#         # 2) 早失败校验
#         if not os.path.isdir(self.rgb_dir):
#             raise FileNotFoundError(f"[CSL_Daily] rgb_dir 不存在: {self.rgb_dir}")
#         if self.split_file and not os.path.exists(self.split_file):
#             print(f"[CSL_Daily][WARN] split_file 不存在，改为扫描 {self.rgb_dir}: {self.split_file}")
#
#         # 3) 构建样本列表
#         self.items: List[str] = self._collect_items()
#         if len(self.items) == 0:
#             raise RuntimeError("[CSL_Daily] 样本列表为空，请检查 rgb_dir/split_file 配置是否正确。")
#
#         # 4) 保存增强配置，供 transform 构建
#         self.aug_train_cfg = aug_train_cfg
#         self.aug_val_cfg   = aug_val_cfg
#
#         # 5) 构建 transform（通过 Base 的钩子）
#         self.data_transform = self.build_train_transform() if self.phase == "train" else self.build_val_transform()
#
#         # 6) 调试信息
#         if getattr(args, "debug", False):
#             print("[CSL_Daily][DEBUG] root:", self.root)
#             print("[CSL_Daily][DEBUG] rgb_dir:", self.rgb_dir)
#             print("[CSL_Daily][DEBUG] split_file:", self.split_file)
#             print("[CSL_Daily][DEBUG] items:", len(self.items), "例：", self.items[:5])
#             print("[CSL_Daily][DEBUG] max_length:", self.max_length, "jitter:", self.jitter, "min_frames:", self.min_frames)
#             print("[CSL_Daily][DEBUG] rgb_support:", self.rgb_support)

class CSLDailyDataset(BaseDataset):
    def __init__(self, args, cfg, phase: str):
        # 1) 先读数据集配置 & 增强配置 —— 注意：这些属性要在 super() 之前就准备好
        ds_cfg = _get(cfg, "datasets.CSL_Daily", {}) or {}
        root = _get(ds_cfg, "root")
        if not root:
            raise ValueError("[CSL_Daily] 缺少 datasets.CSL_Daily.root")
        self.root = os.path.abspath(root)

        rgb_dir_conf = _get(ds_cfg, "rgb_dir", "sentence")
        split_file_conf = _get(ds_cfg, "split_file", None)
        self.rgb_dir = _abs_path(self.root, rgb_dir_conf)
        self.split_file = _abs_path(self.root, split_file_conf) if split_file_conf else None

        # 文本/temporal/增强参数 —— 这些会在 build_*_transform 用到，所以现在就放到 self 上
        self.token_level = _get(ds_cfg, "token_level", "char")
        self.max_text_len = int(_get(ds_cfg, "max_text_len", 128))

        tmp_cfg = _get(ds_cfg, "temporal", {}) or {}
        self.jitter = bool(_get(tmp_cfg, "jitter", True))
        self.min_frames = int(_get(tmp_cfg, "min_frames", 1))
        # 先记下“想要的” max_length，super() 后再覆盖 Base 的默认
        desired_max_len = _get(tmp_cfg, "max_frames", _get(tmp_cfg, "T", None))

        # !!! 关键：增强配置要在 super() 之前就赋值，供 Base.__init__ 里调用 build_*_transform 使用
        self.aug_train_cfg = _get(tmp_cfg, "augment", {}) or {}
        self.aug_val_cfg = _get(tmp_cfg, "augment_val", {}) or {}

        # 2) 再调用 Base 初始化（这里面会调用 build_train_transform/build_val_transform）
        super().__init__(args, cfg, phase)

        # 3) 训练时是否启用 RGB（与 Base 的 args.rgb_support 联合）
        self.use_rgb_ds = bool(_get(ds_cfg, "use_rgb", True))
        self.rgb_support = self.rgb_support and self.use_rgb_ds

        # 4) 如果数据集指定了 max_length，就覆盖 Base 中的默认
        if desired_max_len is not None:
            self.max_length = int(desired_max_len)

        # 5) 基本校验
        if not os.path.isdir(self.rgb_dir):
            raise FileNotFoundError(f"[CSL_Daily] rgb_dir 不存在: {self.rgb_dir}")
        if self.split_file and not os.path.exists(self.split_file):
            print(f"[CSL_Daily][WARN] split_file 不存在，改为扫描 {self.rgb_dir}: {self.split_file}")

        # 6) 构建样本列表
        self.items: List[str] = self._collect_items()
        if len(self.items) == 0:
            raise RuntimeError("[CSL_Daily] 样本列表为空，请检查 rgb_dir/split_file 配置是否正确。")

        # 7) 调试信息（可选）
        if getattr(args, "debug", False):
            print("[CSL_Daily][DEBUG] root:", self.root)
            print("[CSL_Daily][DEBUG] rgb_dir:", self.rgb_dir)
            print("[CSL_Daily][DEBUG] split_file:", self.split_file)
            print("[CSL_Daily][DEBUG] items:", len(self.items), "例：", self.items[:5])
            print("[CSL_Daily][DEBUG] max_length:", self.max_length, "jitter:", self.jitter, "min_frames:",
                  self.min_frames)
            print("[CSL_Daily][DEBUG] rgb_support:", self.rgb_support)

    # -------- transform 构建（覆盖基类钩子） -------- #
    def _build_resize_norm(self, size: int):
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def build_train_transform(self):
        size = int(_get(self.aug_train_cfg, "size", 224))
        # 按需加入随机增强（翻转一般不用于手语）
        degrees = float(_get(self.aug_train_cfg, "degrees", 0.0))
        translate = float(_get(self.aug_train_cfg, "translate", 0.0))
        scale = _get(self.aug_train_cfg, "scale", None)  # [min, max]
        shear = float(_get(self.aug_train_cfg, "shear", 0.0))
        hue = float(_get(self.aug_train_cfg, "hue", 0.0))
        saturation = float(_get(self.aug_train_cfg, "saturation", 0.0))
        brightness = float(_get(self.aug_train_cfg, "brightness", 0.0))
        contrast = float(_get(self.aug_train_cfg, "contrast", 0.0))

        ops = [transforms.Resize((size, size))]
        # 仿射增强
        if degrees or translate or scale or shear:
            # torchvision 的 affine 中 translate 需要 (tx, ty) 相对比例；scale 为浮点或序列
            aff_scale = None
            if scale and isinstance(scale, (list, tuple)) and len(scale) == 2:
                aff_scale = scale
            ops.append(transforms.RandomAffine(
                degrees=degrees or 0.0,
                translate=(translate, translate) if translate else None,
                scale=aff_scale,
                shear=shear or 0.0,
            ))
        # 颜色增强
        if any([hue, saturation, brightness, contrast]):
            ops.append(transforms.ColorJitter(
                brightness=(1-brightness, 1+brightness) if brightness else None,
                contrast=(1-contrast, 1+contrast) if contrast else None,
                saturation=(1-saturation, 1+saturation) if saturation else None,
                hue=(-hue, hue) if hue else None,
            ))

        ops += [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]
        return transforms.Compose(ops)

    def build_val_transform(self):
        size = int(_get(self.aug_val_cfg, "size", _get(self.aug_train_cfg, "size", 224)))
        enable = bool(_get(self.aug_val_cfg, "enable", False))
        if not enable:
            return self._build_resize_norm(size)
        # 如果显式启用 val transform，就只做 resize + normalize（可按需要扩展）
        return self._build_resize_norm(size)

    # -------- 采样（分段均匀 + 抖动） -------- #
    def _sample_indices(self, duration: int) -> np.ndarray:
        L = int(self.max_length)
        if duration <= 0:
            return np.arange(1, dtype=np.int64)
        if duration <= L:
            return np.arange(duration, dtype=np.int64)

        seg = np.linspace(0, duration, num=L + 1, dtype=np.int64)
        starts, ends = seg[:-1], np.clip(seg[1:], 1, duration)
        rng = self._make_rng()
        if self.jitter:
            idx = [int(rng.integers(low=s, high=e) if e > s else min(s, duration - 1))
                   for s, e in zip(starts, ends)]
        else:
            idx = [int((s + e - 1) // 2) for s, e in zip(starts, ends)]
        return np.asarray(idx, dtype=np.int64)

    # -------- 样本列表 -------- #
    def _collect_items(self) -> List[str]:
        ids = self._read_split_ids(self.split_file) if self.split_file else None
        all_dirs = [d for d in os.listdir(self.rgb_dir) if os.path.isdir(os.path.join(self.rgb_dir, d))]
        candidates = ids if ids is not None else all_dirs

        items = []
        for vid in candidates:
            vdir = os.path.join(self.rgb_dir, vid)
            if not os.path.isdir(vdir):
                continue
            n_frames = len(list_frame_paths(vdir))
            if n_frames >= self.min_frames:
                items.append(vid)
        return items

    def _read_split_ids(self, split_txt: str) -> Optional[List[str]]:
        if not split_txt or not os.path.exists(split_txt):
            return None

        want = self.phase  # 'train' / 'val' / 'test'
        if want == "val":
            aliases = {"val", "dev", "valid", "validation"}
        elif want == "train":
            aliases = {"train", "tr", "training"}
        elif want == "test":
            aliases = {"test", "te"}
        else:
            aliases = {want}

        def norm_vid(x: str) -> str:
            return Path(x.strip().strip('",')).stem

        ids = []
        with open(split_txt, "r", encoding="utf-8") as f:
            first = True
            for raw in f:
                line = raw.strip().lstrip("\ufeff")
                if not line or line.startswith("#") or line.startswith("//"):
                    continue
                if "|" in line:
                    if first and ("name" in line and "split" in line):
                        first = False
                        continue
                    first = False
                    name, sp = [x.strip() for x in line.split("|", 1)]
                    if sp.lower() in aliases:
                        ids.append(norm_vid(name))
                    continue
                parts = line.split()
                first = False
                if len(parts) >= 2:
                    sp, vid_raw = parts[0], parts[-1]
                    if sp.lower() in aliases:
                        ids.append(norm_vid(vid_raw))
                else:
                    ids.append(norm_vid(parts[0]))
        return ids

    # -------- 文本（目前只返回原字符串） -------- #
    def _load_text(self, vid: str) -> str:
        # 优先尝试 sentence/<vid>.txt
        txt_path = os.path.join(self.rgb_dir, f"{vid}.txt")
        if os.path.exists(txt_path):
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    raw = f.read().strip()
                return raw
            except Exception:
                return ""
        return ""

    # -------- 载入 clip -------- #
    def _load_rgb_clip(self, frame_dir: str, indices: np.ndarray) -> torch.Tensor:
        if not self.rgb_support:
            # 仅返回占位，保持形状一致
            return torch.zeros((len(indices), 3, 224, 224))

        paths = list_frame_paths(frame_dir)
        if len(paths) == 0:
            return torch.zeros((len(indices), 3, 224, 224))

        out = []
        for i in range(len(indices)):
            j = int(np.clip(indices[i], 0, len(paths) - 1))
            img = _safe_open(paths[j])
            if img is None:
                # 临近回退
                found = False
                for off in range(1, min(6, len(paths))):
                    for cand in (j - off, j + off):
                        if 0 <= cand < len(paths):
                            img = _safe_open(paths[cand])
                            if img is not None:
                                found = True
                                break
                    if found:
                        break
                if img is None:
                    img = Image.new("RGB", (224, 224))
            out.append(self.data_transform(img))
        return torch.stack(out, dim=0)  # [T,3,H,W]

    # -------- Dataset 标准接口 -------- #
    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        vid = self.items[idx]
        frame_dir = os.path.join(self.rgb_dir, vid)

        text = self._load_text(vid)

        num_frames = len(list_frame_paths(frame_dir))
        indices = self._sample_indices(num_frames)

        rgb_seq = self._load_rgb_clip(frame_dir, indices)

        pose_sample = {}  # 本数据集无 pose
        support = {
            'rgb_img': rgb_seq,
            'rgb_img_indices': indices,
        }
        return vid, pose_sample, text, indices, support
