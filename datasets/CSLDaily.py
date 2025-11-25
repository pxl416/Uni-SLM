# datasets/CSLDaily.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import glob
import math
import json
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from datasets.datasets import BaseDataset
from utils.augment import SignAugment, preset_medium
from utils.plot import (
    tensor_clip_to_pil_list,
    save_keypoint_overlay_grid,
    save_timeline,
    save_attn_mask,
)


# =========================
# 小工具
# =========================
def _get(cfg: Any, dotted: str, default=None):
    """
    从 dict / namespace 里用 'a.b.c' 这种路径取值。
    """
    cur = cfg
    for k in dotted.split('.'):
        if isinstance(cur, dict):
            if k not in cur:
                return default
            cur = cur[k]
        else:
            if not hasattr(cur, k):
                return default
            cur = getattr(cur, k)
    return cur


def _abs_path(root: str, p: Optional[str]) -> Optional[str]:
    if p is None:
        return None
    return p if os.path.isabs(p) else os.path.join(root, p)


def list_frame_paths(frame_dir: str) -> List[str]:
    """
    列出目录下所有非空 jpg/png 并按文件名数字排序。
    """
    if not frame_dir or not os.path.isdir(frame_dir):
        return []
    pats: List[str] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        pats.extend(glob.glob(os.path.join(frame_dir, ext)))
    pats = [p for p in pats if os.path.getsize(p) > 0]
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
        return Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return None


# =========================
# 主类
# =========================
class CSLDailyDataset(BaseDataset):
    """
    CSL-Daily 句子级数据集。

    期望 YAML 结构大致为：
    datasets:
      CSL_Daily:
        modalities:
          rgb:  true
          pose: false
          text: true

        paths:
          root: /path/to/mini_CSL_Daily
          rgb: sentence
          pose: lmdb
          text: sentence_label/video_map.txt
          segments: sentence_label/video_map.txt   # 目前仅用 dirname

        splits:
          train: sentence_label/split_1_train.txt
          val:   sentence_label/split_1_dev.txt
          test:  sentence_label/split_1_test.txt

        temporal:
          T: 32
          jitter: false
          min_frames: 1
          max_frames: 128

        # 可选：
        # text_column: "word" | "char" | "gloss"
        # merge: { ... }
    """

    def __init__(self, args, cfg, phase: str):
        # -------- 读取数据集配置 -------- #
        ds_cfg = _get(cfg, "datasets.CSL_Daily", {}) or {}

        # ---- 路径相关 ---- #
        paths_cfg = ds_cfg.get("paths", {}) if isinstance(ds_cfg, dict) else _get(ds_cfg, "paths", {})
        root = paths_cfg.get("root") if isinstance(paths_cfg, dict) else _get(paths_cfg, "root")
        if not root:
            raise ValueError("[CSL_Daily] 缺少 datasets.CSL_Daily.paths.root")
        self.root = os.path.abspath(root)

        rgb_dir_conf = paths_cfg.get("rgb", "sentence") if isinstance(paths_cfg, dict) else _get(paths_cfg, "rgb", "sentence")
        self.rgb_dir = _abs_path(self.root, rgb_dir_conf)

        # 分割文件（train/val/test）
        splits_cfg = ds_cfg.get("splits", {}) if isinstance(ds_cfg, dict) else _get(ds_cfg, "splits", {})
        phase_norm = str(phase).lower()
        key_for_phase = {
            "train": "train",
            "tr": "train",
            "training": "train",
            "val": "val",
            "dev": "dev",
            "valid": "val",
            "validation": "val",
            "test": "test",
            "te": "test",
        }.get(phase_norm, phase_norm)
        split_file_conf = splits_cfg.get(key_for_phase) if isinstance(splits_cfg, dict) else _get(splits_cfg, key_for_phase, None)
        # 兼容 val→dev
        if key_for_phase == "val" and not split_file_conf:
            split_file_conf = splits_cfg.get("dev") if isinstance(splits_cfg, dict) else _get(splits_cfg, "dev", None)
        self.split_file = _abs_path(self.root, split_file_conf) if split_file_conf else None

        # ---- 文本映射（video_map）预加载 ---- #
        self._text_map: Dict[str, str] = {}
        text_rel = paths_cfg.get("text") if isinstance(paths_cfg, dict) else _get(paths_cfg, "text")
        if text_rel:
            self._load_text_map(_abs_path(self.root, text_rel), ds_cfg)

        # ---- 时序配置 ---- #
        tmp_cfg = ds_cfg.get("temporal", {}) if isinstance(ds_cfg, dict) else _get(ds_cfg, "temporal", {}) or {}
        self.jitter = bool(tmp_cfg.get("jitter", True))
        self.min_frames = int(tmp_cfg.get("min_frames", 1))
        desired_max_len = tmp_cfg.get("max_frames", tmp_cfg.get("T", None))

        # ---- 可视化/调试输出目录 ---- #
        self.save_dir = _get(cfg, "save_dir", "./checkpoints")
        self.vis_dir = os.path.join(self.save_dir, "debug_vis")
        os.makedirs(self.vis_dir, exist_ok=True)
        self._vis_first_n = 6
        self._vis_every_n = 200

        # ---- 模态配置 ---- #
        modalities_cfg = ds_cfg.get("modalities", {}) if isinstance(ds_cfg, dict) else _get(ds_cfg, "modalities", {}) or {}
        self.use_keypoints = bool(modalities_cfg.get("pose", False))
        pose_rel = paths_cfg.get("pose", "lmdb") if isinstance(paths_cfg, dict) else _get(paths_cfg, "pose", "lmdb")
        self.keypoints_dir = _abs_path(self.root, pose_rel)

        self.joints = int(ds_cfg.get("joints", 21) if isinstance(ds_cfg, dict) else _get(ds_cfg, "joints", 21))
        self.keypoint_dims = int(ds_cfg.get("keypoint_dims", 2) if isinstance(ds_cfg, dict) else _get(ds_cfg, "keypoint_dims", 2))

        # ---- 分段标签目录（sentence_label/<vid>.txt） ---- #
        seg_rel = paths_cfg.get("segments", "sentence_label/video_map.txt") if isinstance(paths_cfg, dict) else _get(paths_cfg, "segments", "sentence_label/video_map.txt")
        self.segment_label_dir = _abs_path(self.root, os.path.dirname(seg_rel) if seg_rel else None)

        # ---- 捏合/拼接配置 ---- #
        mcfg = ds_cfg.get("merge", {}) if isinstance(ds_cfg, dict) else _get(ds_cfg, "merge", {}) or {}
        self.merge_enabled = bool(mcfg.get("enabled", True))
        self.merge_ratio = float(mcfg.get("merge_ratio", 0.6))
        self.max_segments = int(mcfg.get("max_segments", 3))
        self.transition_frames = int(mcfg.get("transition_frames", 4))
        self.long_seg_threshold = int(mcfg.get("long_segment_threshold", 64))

        # ---- RGB 数据增强（clip 级） ---- #
        # 训练：时序一致增强；验证/测试：只做 resize+normalize（无抖动）
        self.train_augment = preset_medium(size=224)
        self.val_augment = SignAugment(
            size=224,
            degrees=0.0,
            translate=0.0,
            scale=(1.0, 1.0),
            shear=0.0,
            enable_flip=False,
            hue=0.0,
            saturation=0.0,
            brightness=0.0,
            contrast=0.0,
            cutout_p=0.0,
        )

        # 交给基类做一些通用初始化（max_length / seed / enable_aug 等）
        super().__init__(args, cfg, phase)

        # 若 temporal 指定了 max_frames/T，则覆盖 args.max_length
        if desired_max_len is not None:
            self.max_length = int(desired_max_len)

        # 决定是否用 RGB
        self.use_rgb_ds = bool(modalities_cfg.get("rgb", True))
        self.rgb_support = self.rgb_support and self.use_rgb_ds

        # 基本校验
        if not os.path.isdir(self.rgb_dir):
            raise FileNotFoundError(f"[CSL_Daily] rgb_dir 不存在: {self.rgb_dir}")
        if self.split_file and not os.path.exists(self.split_file):
            print(f"[CSL_Daily][WARN] split_file 不存在，改为扫描 {self.rgb_dir}: {self.split_file}")

        # 构建样本列表
        self.items: List[str] = self._collect_items()
        if len(self.items) == 0:
            raise RuntimeError("[CSL_Daily] 样本列表为空，请检查 rgb_dir / split_file / min_frames 配置。")

        if getattr(args, "debug", False):
            print("[CSL_Daily][DEBUG] root:", self.root)
            print("[CSL_Daily][DEBUG] rgb_dir:", self.rgb_dir)
            print("[CSL_Daily][DEBUG] split_file:", self.split_file)
            print("[CSL_Daily][DEBUG] num_items:", len(self.items))
            print("[CSL_Daily][DEBUG] max_length:", self.max_length, "jitter:", self.jitter, "min_frames:", self.min_frames)
            print("[CSL_Daily][DEBUG] rgb_support:", self.rgb_support, "use_keypoints:", self.use_keypoints)

    # =========================
    # 文本映射（video_map）
    # =========================
    def _load_text_map(self, text_map_path: str, ds_cfg: Dict[str, Any]):
        if not text_map_path or not os.path.exists(text_map_path):
            return
        # 可在 YAML 里加一个 text_column: "word"|"char"|"gloss"，默认 word
        text_column = ds_cfg.get("text_column", "word") if isinstance(ds_cfg, dict) else _get(ds_cfg, "text_column", "word")
        col_idx = {"index": 0, "name": 1, "length": 2, "gloss": 3, "char": 4, "word": 5}

        use_col = col_idx.get(text_column, 5)  # 默认用 word 列
        with open(text_map_path, "r", encoding="utf-8") as f:
            first = True
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                # 跳过表头
                if first and ("index" in line and "name" in line):
                    first = False
                    continue
                first = False
                parts = line.split("|")
                if len(parts) <= 1:
                    continue
                # 形如：index|name|length|gloss|char|word|postag
                if len(parts) <= use_col:
                    continue
                vid = parts[1].strip()
                text = parts[use_col].strip()
                self._text_map[vid] = text

    # =========================
    # 分段 / 采样 / 关键点
    # =========================
    def _parse_segment_file(self, vid: str):
        """
        sentence_label/<vid>.txt，每行: start end text
        若不存在，则后续退化为整句。
        """
        if not self.segment_label_dir:
            return []
        path = os.path.join(self.segment_label_dir, f"{vid}.txt")
        segs = []
        if not os.path.exists(path):
            return segs
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    parts = line.split("\t")
                if len(parts) < 3:
                    continue
                try:
                    s = int(parts[0])
                    e = int(parts[1])
                    txt = " ".join(parts[2:]).strip()
                    if e >= s:
                        segs.append((s, e, txt))
                except Exception:
                    continue
        segs.sort(key=lambda x: (x[0], x[1]))
        return segs

    def _sample_indices(self, duration: int) -> np.ndarray:
        """
        覆盖基类：对整句做均匀采样+抖动（备用，不直接用于分段版）。
        """
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

    def _sample_indices_segment(self, start: int, end: int, want: int) -> np.ndarray:
        """
        在 [start, end] 内均匀切成 want 段，每段 1 帧（带抖动）。
        """
        dur = max(0, end - start + 1)
        if dur <= 0:
            return np.zeros((0,), dtype=np.int64)
        if dur <= want:
            return np.arange(start, end + 1, dtype=np.int64)

        seg = np.linspace(start, end + 1, num=want + 1, dtype=np.int64)
        starts, ends = seg[:-1], np.clip(seg[1:], start + 1, end + 1)
        rng = self._make_rng()
        if self.jitter:
            idx = [int(rng.integers(low=s, high=e) if e > s else min(s, end))
                   for s, e in zip(starts, ends)]
        else:
            idx = [int((s + e - 1) // 2) for s, e in zip(starts, ends)]
        return np.asarray(idx, dtype=np.int64)

    def _load_keypoints_clip(self, vid: str, frame_indices: np.ndarray) -> torch.Tensor:
        """
        关键点序列 [T,J,C]；若未启用关键点则全 0。
        """
        if not self.use_keypoints:
            return torch.zeros((len(frame_indices), self.joints, self.keypoint_dims), dtype=torch.float32)

        vdir = os.path.join(self.keypoints_dir, vid)
        T = len(frame_indices)
        out = np.zeros((T, self.joints, self.keypoint_dims), dtype=np.float32)
        valid = np.zeros((T,), dtype=np.bool_)

        for t, fidx in enumerate(frame_indices):
            fpath = os.path.join(vdir, f"{int(fidx)}.npy")
            if os.path.exists(fpath):
                try:
                    arr = np.load(fpath)
                    if arr.ndim == 1 and arr.size == self.joints * self.keypoint_dims:
                        arr = arr.reshape(self.joints, self.keypoint_dims)
                    if arr.ndim >= 2:
                        out[t, :, :self.keypoint_dims] = arr[:self.joints, :self.keypoint_dims]
                        valid[t] = True
                except Exception:
                    pass

        # 最近邻补全
        last = None
        for t in range(T):
            if valid[t]:
                last = out[t].copy()
            elif last is not None:
                out[t] = last
        last = None
        for t in range(T - 1, -1, -1):
            if valid[t]:
                last = out[t].copy()
            elif last is not None:
                out[t] = last
        return torch.from_numpy(out)

    def _concat_with_transition_and_pad(
        self,
        per_seg_indices: List[np.ndarray],
        per_seg_texts: List[str],
        pad_to_multiple_of: int = 8,
    ) -> Dict[str, Any]:
        """
        把多个 segment 的索引拼成一个长序列，中间插入若干过渡帧，并 pad 到 length 的倍数。
        """
        glued: List[int] = []
        starts, ends = [], []
        cur = 0
        for i, arr in enumerate(per_seg_indices):
            arr = arr.astype(np.int64)
            if i > 0 and self.transition_frames > 0 and len(glued) > 0:
                glued.extend([glued[-1]] * self.transition_frames)
                cur += self.transition_frames
            starts.append(cur)
            glued.extend(arr.tolist())
            cur += len(arr)
            ends.append(cur - 1)

        concat_idx = np.asarray(glued, dtype=np.int64)
        T_total = len(concat_idx)

        if pad_to_multiple_of > 0 and T_total > 0:
            pad_to = int(math.ceil(T_total / pad_to_multiple_of) * pad_to_multiple_of)
        else:
            pad_to = T_total

        attn_mask = np.zeros((pad_to,), dtype=np.bool_)
        attn_mask[:T_total] = True
        if pad_to > T_total and T_total > 0:
            concat_idx = np.concatenate(
                [concat_idx, np.full((pad_to - T_total,), concat_idx[-1], dtype=np.int64)],
                axis=0,
            )

        return {
            "global_indices": concat_idx,
            "attn_mask": torch.from_numpy(attn_mask),
            "starts": torch.tensor(starts, dtype=torch.long),
            "ends": torch.tensor(ends, dtype=torch.long),
            "texts": per_seg_texts,
        }

    # =========================
    # 文本 / RGB / 可视化
    # =========================
    def _load_text(self, vid: str) -> str:
        """
        优先使用预加载的 video_map；若没有，则尝试 sentence/<vid>.txt。
        """
        if vid in self._text_map:
            return self._text_map[vid]

        txt_path = os.path.join(self.rgb_dir, f"{vid}.txt")
        if os.path.exists(txt_path):
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except Exception:
                return ""
        return ""

    def _load_rgb_clip(self, frame_dir: str, indices: np.ndarray) -> torch.Tensor:
        """
        载入一个 clip，并在 clip 级别做增广或仅做 resize+normalize。
        """
        if not self.rgb_support:
            return torch.zeros((len(indices), 3, 224, 224))

        paths = list_frame_paths(frame_dir)
        if len(paths) == 0:
            return torch.zeros((len(indices), 3, 224, 224))

        frames: List[Image.Image] = []
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
            frames.append(img)

        if self.phase == "train" and getattr(self, "enable_aug", False):
            clip = self.train_augment(frames)
        else:
            clip = self.val_augment(frames)
        # clip: [T, C, H, W]
        return clip

    def _maybe_visualize_sample(
        self,
        vid: str,
        rgb_seq: torch.Tensor,
        kpt_seq: Optional[torch.Tensor],
        meta: Dict[str, Any],
        counter: int,
    ):
        """
        抽样保存若干样本的帧、关键点、时间轴和 mask，方便检查。
        """
        if "starts" not in meta and "segments" in meta:
            seg = meta.get("segments", {})
            meta = {
                "attn_mask": meta.get("attn_mask"),
                "starts": seg.get("starts"),
                "ends": seg.get("ends"),
                "texts": seg.get("texts"),
            }

        save_stub = os.path.join(self.vis_dir, f"{vid}_{counter:06d}")

        frames_pil = tensor_clip_to_pil_list(
            rgb_seq,
            denorm=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )

        if kpt_seq is not None and kpt_seq.numel() > 0:
            try:
                save_keypoint_overlay_grid(
                    frames_pil,
                    kpts_seq=kpt_seq.cpu().numpy(),
                    kpt_coord_mode="pixel",
                    save_path=f"{save_stub}_kp_grid.jpg",
                    log_key=None,
                )
            except Exception:
                pass

        try:
            T_eff = int(
                meta["attn_mask"].sum().item()
                if isinstance(meta["attn_mask"], torch.Tensor)
                else np.asarray(meta["attn_mask"]).sum()
            )
            save_timeline(
                starts=meta["starts"].cpu().tolist(),
                ends=meta["ends"].cpu().tolist(),
                T_total=T_eff,
                labels=meta["texts"],
                save_path=f"{save_stub}_timeline.png",
                log_key=None,
            )
        except Exception:
            pass

        try:
            am = (
                meta["attn_mask"].cpu().numpy()
                if isinstance(meta["attn_mask"], torch.Tensor)
                else np.asarray(meta["attn_mask"])
            )
            save_attn_mask(am, save_path=f"{save_stub}_attn_mask.png", log_key=None)
        except Exception:
            pass

        try:
            with open(f"{save_stub}_meta.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "vid": vid,
                        "starts": meta["starts"].cpu().tolist(),
                        "ends": meta["ends"].cpu().tolist(),
                        "texts": meta["texts"],
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception:
            pass

    # =========================
    # 样本列表 & Dataset 接口
    # =========================
    def _collect_items(self) -> List[str]:
        """
        若有 split_file，则按文件列出的 id 为准；否则扫描 rgb_dir 下所有子目录。
        """
        ids: Optional[List[str]] = self._read_split_ids(self.split_file) if self.split_file else None
        all_dirs = [
            d for d in os.listdir(self.rgb_dir)
            if os.path.isdir(os.path.join(self.rgb_dir, d))
        ]
        candidates = ids if ids is not None else all_dirs

        items: List[str] = []
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
        ids: List[str] = []
        with open(split_txt, "r", encoding="utf-8") as f:
            for line in f:
                vid = line.strip()
                if vid and not vid.startswith("#"):
                    ids.append(vid)
        return ids

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        # 可视化计数器
        if not hasattr(self, "_index_counter"):
            self._index_counter = 0
        self._index_counter += 1

        vid = self.items[idx]
        frame_dir = os.path.join(self.rgb_dir, vid)
        paths = list_frame_paths(frame_dir)
        F = len(paths)

        # 分段标签；若不存在则退化为整句
        segs = self._parse_segment_file(vid)
        if len(segs) == 0:
            segs = [(0, max(0, F - 1), self._load_text(vid))]

        # 是否捏合多句
        do_merge = self.merge_enabled
        if self.phase == "train" and self.merge_ratio < 1.0:
            rng = self._make_rng()
            do_merge = do_merge and (float(rng.random()) < self.merge_ratio)

        K_max = max(1, self.max_segments)
        want_len_per_seg = int(getattr(self, "max_length", 16))

        # 若存在超长段，则强制单句
        long_only = any((e - s + 1) >= self.long_seg_threshold for (s, e, _) in segs)

        if (not do_merge) or long_only or len(segs) == 1:
            segs_pick = [segs[0]]
        else:
            if self.phase == "train":
                rng = self._make_rng()
                K = int(rng.integers(low=2, high=min(K_max, len(segs)) + 1)) if len(segs) >= 2 else 1
                pick = sorted(list(rng.choice(len(segs), size=K, replace=False)))
                segs_pick = [segs[i] for i in pick]
            else:
                K = min(K_max, len(segs))
                segs_pick = segs[:K]

        # 每段内部采样
        per_seg_indices, per_seg_texts = [], []
        for (s, e, txt) in segs_pick:
            s = max(0, s)
            e = min(e, max(0, F - 1))
            if e < s:
                s, e = 0, max(0, F - 1)
            arr = self._sample_indices_segment(s, e, want_len_per_seg)
            per_seg_indices.append(arr)
            per_seg_texts.append(txt)

        # 拼接 + 过渡 + pad
        meta = self._concat_with_transition_and_pad(per_seg_indices, per_seg_texts, pad_to_multiple_of=8)
        global_idx = meta["global_indices"]

        # RGB clip
        rgb_seq = self._load_rgb_clip(frame_dir, global_idx)

        # 关键点
        kpt_seq = self._load_keypoints_clip(vid, global_idx) if self.use_keypoints else None

        # 文本：多段文本直接拼成一个句子（也保留在 meta["texts"] 里）
        text_concat = " ".join([t for t in per_seg_texts if t])

        # pose_sample：关键点格式统一为 {'keypoints': [T,J,C]}
        pose_sample = {"keypoints": kpt_seq} if (self.use_keypoints and kpt_seq is not None) else {}

        # support：RGB + 时间掩码 + 分段信息
        support = {
            "rgb_img": rgb_seq,
            "rgb_img_indices": global_idx,
            "attn_mask": meta["attn_mask"],
            "segments": {
                "starts": meta["starts"],
                "ends": meta["ends"],
                "texts": meta["texts"],
            },
        }

        # 可视化抽样
        if (self._index_counter <= self._vis_first_n) or (self._index_counter % self._vis_every_n == 0):
            try:
                self._maybe_visualize_sample(
                    vid=vid,
                    rgb_seq=rgb_seq,
                    kpt_seq=kpt_seq,
                    meta=meta,
                    counter=self._index_counter,
                )
            except Exception:
                pass

        # BaseDataset.collate_fn 期望格式: (name, pose_sample, text, support)
        return vid, pose_sample, text_concat, support
