# utils/CSLDaily.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os, glob
from typing import List, Dict, Any, Optional

import math, json
from typing import Tuple
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from pathlib import Path

from utils.plot import tensor_clip_to_pil_list, save_keypoint_overlay_grid, save_timeline, save_attn_mask
from utils.dataset import BaseDataset


# -------------------------
# tools
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

        # 关键：增强配置要在 super() 之前就赋值，供 Base.__init__ 里调用 build_*_transform 使用
        self.aug_train_cfg = _get(tmp_cfg, "augment", {}) or {}
        self.aug_val_cfg = _get(tmp_cfg, "augment_val", {}) or {}

        # === 新增：从 cfg 里拿 save_dir（永远开启可视化，不靠 yaml 开关）
        self.save_dir = _get(cfg, "save_dir", "./checkpoints")
        self.vis_dir  = os.path.join(self.save_dir, "debug_vis")
        os.makedirs(self.vis_dir, exist_ok=True)
        self._vis_first_n = 6       # 训练前若干个样本，强制保存可视化
        self._vis_every_n = 200     # 之后每隔 N 个样本抽检一次

        # === 新增：关键点配置（目录、关节数、维度）
        self.use_keypoints  = bool(_get(ds_cfg, "use_keypoints", False))
        self.keypoints_dir  = _abs_path(self.root, _get(ds_cfg, "keypoints_dir", "keypoints"))
        self.joints         = int(_get(ds_cfg, "joints", 21))
        self.keypoint_dims  = int(_get(ds_cfg, "keypoint_dims", 2))  # 2:(x,y) 或 3:(x,y,c)

        # === 新增：分段标签目录（sentence_label/vid.txt: start end text）
        seg_cfg = _get(ds_cfg, "segments", {}) or {}
        self.segment_label_dir = _abs_path(self.root, _get(seg_cfg, "label_dir", "sentence_label"))

        # === 新增：捏合/拼接强度 merge 参数（来自 config.merge）
        mcfg = _get(ds_cfg, "merge", {}) or {}
        self.merge_enabled        = bool(_get(mcfg, "enabled", True))
        self.merge_ratio          = float(_get(mcfg, "merge_ratio", 0.6))     # 有多大比例的样本使用多句捏合
        self.max_segments         = int(_get(mcfg, "max_segments", 3))        # 最多捏合几句
        self.transition_frames    = int(_get(mcfg, "transition_frames", 4))   # 句子之间的过渡帧（用上一帧复制）
        self.long_seg_threshold   = int(_get(mcfg, "long_segment_threshold", 64))  # 太长就不拼


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
                shear=shear or 0.0
            ))
        # 颜色增强
        if any([hue, saturation, brightness, contrast]):
            ops.append(transforms.ColorJitter(
                brightness=(1-brightness, 1+brightness) if brightness else None,
                contrast=(1-contrast, 1+contrast) if contrast else None,
                saturation=(1-saturation, 1+saturation) if saturation else None,
                hue=(-hue, hue) if hue else None
            ))

        ops += [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]
        return transforms.Compose(ops)
    # === 新增：读取 sentence_label/<vid>.txt（每行: start end text）
    def _parse_segment_file(self, vid: str):
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
                    s = int(parts[0]); e = int(parts[1])
                    txt = " ".join(parts[2:]).strip()
                    if e >= s:
                        segs.append((s, e, txt))
                except Exception:
                    continue
        segs.sort(key=lambda x: (x[0], x[1]))
        return segs

    # === 新增：在 [start,end] 内均匀采样 want 帧（带抖动）
    def _sample_indices_segment(self, start: int, end: int, want: int) -> np.ndarray:
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

    # === 新增：载入关键点序列 [T,J,C]；缺失做最近邻填补
    def _load_keypoints_clip(self, vid: str, frame_indices: np.ndarray) -> torch.Tensor:
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
        # 最近邻补
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

    # === 新增：把若干段 indices 拼接；段间插入 transition_frames；末尾 pad 到倍数
    def _concat_with_transition_and_pad(
        self,
        per_seg_indices: List[np.ndarray],
        per_seg_texts: List[str],
        pad_to_multiple_of: int = 8
    ) -> Dict[str, Any]:
        # 1) 段间过渡：在每两段之间插入若干“上一段最后一帧”的复制索引
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

        # 2) pad 到倍数
        if pad_to_multiple_of > 0 and T_total > 0:
            pad_to = int(math.ceil(T_total / pad_to_multiple_of) * pad_to_multiple_of)
        else:
            pad_to = T_total
        attn_mask = np.zeros((pad_to,), dtype=np.bool_)
        attn_mask[:T_total] = True
        if pad_to > T_total and T_total > 0:
            concat_idx = np.concatenate([concat_idx, np.full((pad_to - T_total,), concat_idx[-1], dtype=np.int64)], 0)

        return {
            "global_indices": concat_idx,                     # [T_pad]
            "attn_mask": torch.from_numpy(attn_mask),        # [T_pad] bool
            "starts": torch.tensor(starts, dtype=torch.long),
            "ends": torch.tensor(ends, dtype=torch.long),
            "texts": per_seg_texts,
        }

    # === 新增：可视化（前 N 个样本强制，之后每隔 M 个样本）
    def _maybe_visualize_sample(
        self,
        vid: str,
        rgb_seq: torch.Tensor,             # [T,3,H,W] (已normalize)
        kpt_seq: Optional[torch.Tensor],   # [T,J,C] 或 None
        meta: Dict[str, Any],              # starts/ends/texts/attn_mask
        counter: int
    ):
        if "starts" not in meta and "segments" in meta:
            seg = meta.get("segments", {})
            meta = {
                "attn_mask": meta.get("attn_mask"),
                "starts": seg.get("starts"),
                "ends": seg.get("ends"),
                "texts": seg.get("texts"),
            }

        save_stub = os.path.join(self.vis_dir, f"{vid}_{counter:06d}")
        # 反归一化到 PIL
        frames_pil = tensor_clip_to_pil_list(
            rgb_seq, denorm=((0.485,0.456,0.406),(0.229,0.224,0.225))
        )
        # 关键点叠加
        if kpt_seq is not None and kpt_seq.numel() > 0:
            try:
                save_keypoint_overlay_grid(
                    frames_pil,
                    kpts_seq=kpt_seq.cpu().numpy(),
                    kpt_coord_mode="pixel",                # 若你的关键点是[0,1]，这里改 "normalized"
                    save_path=f"{save_stub}_kp_grid.jpg",
                    log_key=None
                )
            except Exception:
                pass
        # 时间轴 + 文本
        try:
            T_eff = int(meta["attn_mask"].sum().item()) if isinstance(meta["attn_mask"], torch.Tensor) else int(np.asarray(meta["attn_mask"]).sum())
            save_timeline(
                starts=meta["starts"].cpu().tolist(),
                ends=meta["ends"].cpu().tolist(),
                T_total=T_eff,
                labels=meta["texts"],
                save_path=f"{save_stub}_timeline.png",
                log_key=None
            )
        except Exception:
            pass
        # 掩码
        try:
            am = meta["attn_mask"].cpu().numpy() if isinstance(meta["attn_mask"], torch.Tensor) else np.asarray(meta["attn_mask"])
            save_attn_mask(am, save_path=f"{save_stub}_attn_mask.png", log_key=None)
        except Exception:
            pass
        # 另存一份 json 元信息
        try:
            with open(f"{save_stub}_meta.json", "w", encoding="utf-8") as f:
                json.dump({
                    "vid": vid,
                    "starts": meta["starts"].cpu().tolist(),
                    "ends": meta["ends"].cpu().tolist(),
                    "texts": meta["texts"],
                }, f, ensure_ascii=False, indent=2)

        except Exception:
            pass


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
        # 计数器用于可视化抽检
        if not hasattr(self, "_index_counter"):
            self._index_counter = 0
        self._index_counter += 1

        vid = self.items[idx]
        frame_dir = os.path.join(self.rgb_dir, vid)
        paths = list_frame_paths(frame_dir)
        F = len(paths)

        # 读取分段标签；若无则退化为整段
        segs = self._parse_segment_file(vid)  # [(s,e,txt), ...]
        if len(segs) == 0:
            segs = [(0, max(0, F-1), self._load_text(vid))]

        # 这一次样本是否进行“多句捏合”
        do_merge = self.merge_enabled
        if self.phase == "train" and self.merge_ratio < 1.0:
            rng = self._make_rng()
            do_merge = do_merge and (float(rng.random()) < self.merge_ratio)
        # 选段数 K
        K_max = max(1, self.max_segments)
        want_len_per_seg = int(getattr(self, "max_length", 16))  # 每段目标帧数 T

        # 过滤掉“超长段”（> long_seg_threshold）的情况：该样本强制单句
        long_only = False
        for (s,e,txt) in segs:
            if (e - s + 1) >= self.long_seg_threshold:
                long_only = True
                break

        if (not do_merge) or long_only or len(segs) == 1:
            segs_pick = [segs[0]]  # 单句样本（验证/测试统一用前一句）
        else:
            # 训练：随机抽 2..K_max 段；验证/测试：取 min(K_max, len(segs))
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
        for (s,e,txt) in segs_pick:
            s = max(0, s); e = min(e, max(0, F-1))
            if e < s: s, e = 0, max(0, F-1)
            arr = self._sample_indices_segment(s, e, want_len_per_seg)
            per_seg_indices.append(arr)
            per_seg_texts.append(txt)

        # 拼接 + 过渡 + pad
        meta = self._concat_with_transition_and_pad(per_seg_indices, per_seg_texts, pad_to_multiple_of=8)
        global_idx = meta["global_indices"]

        # 载入 RGB（按 global_idx 对应的绝对帧号）
        rgb_seq = self._load_rgb_clip(frame_dir, global_idx)  # [T_pad,3,H,W]

        # 载入关键点
        kpt_seq = self._load_keypoints_clip(vid, global_idx) if self.use_keypoints else None

        # 文本（兼容：返回合并文本；原 single text 仍可从 support['segments']['texts'] 获取）
        text_concat = " ".join([t for t in per_seg_texts if t])

        # pose_sample：把关键点塞进去（若未启用则空 dict）
        pose_sample = {"keypoints": kpt_seq} if (self.use_keypoints and kpt_seq is not None) else {}

        # support：新增 attn_mask 与 segments（starts/ends/texts）
        support = {
            "rgb_img": rgb_seq,                   # [T_pad,3,H,W]
            "rgb_img_indices": global_idx,        # 绝对帧号
            "attn_mask": meta["attn_mask"],       # [T_pad] bool
            "segments": {
                "starts": meta["starts"],         # [K]
                "ends": meta["ends"],             # [K]
                "texts": meta["texts"],           # List[str]
            },
        }

        # 永久可视化：前 N 个样本必存，之后每隔 M 个样本抽检
        if (self._index_counter <= self._vis_first_n) or (self._index_counter % self._vis_every_n == 0):
            try:
                self._maybe_visualize_sample(
                    vid=vid,
                    rgb_seq=rgb_seq,
                    kpt_seq=kpt_seq,
                    meta=meta,
                    counter=self._index_counter
                )

            except Exception:
                pass

        return vid, pose_sample, text_concat, global_idx, support

