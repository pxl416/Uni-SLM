# utils/CSLNews.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import copy
import numpy as np
import pickle
import torch

from PIL import Image
from decord import VideoReader, cpu
from torchvision import transforms
from utils.dataset import BaseDataset


# ============ 小工具 ============
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

def _abs_path(base: str | None, p: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.join(base or "", p)

# ============ 视频 / 姿态 I/O ============
def load_rgb(video_path, frame_indices, size=(224, 224), transform=None):
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    if len(vr) == 0:
        return torch.zeros(len(frame_indices), 3, size[1], size[0])
    clamped = np.clip(frame_indices, 0, len(vr) - 1)
    buf = vr.get_batch(clamped).asnumpy()
    to_tensor = transforms.ToTensor()
    imgs = [
        (transform(Image.fromarray(im).convert('RGB').resize(size))
         if transform else to_tensor(Image.fromarray(im).convert('RGB').resize(size)))
        for im in buf
    ]
    return torch.stack(imgs, 0)  # [T, 3, H, W]

def load_kpt_from_obj(pose_obj, frame_indices, video_path=None):
    """
    pose_obj: pickle.load(...) 得到的 dict，包含 'keypoints', 'scores'
    返回:
      {'skeletons':[T,1,133,2], 'confs':[T,1,133], 'meta':{'space':'norm01','img_hw':(H,W)|None}}
    """
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    total = len(pose_obj.get('scores', []))
    if total == 0:
        T = len(frame_indices)
        return {
            'skeletons': np.zeros((T,1,133,2)),
            'confs':     np.zeros((T,1,133)),
            'meta':      {'space':'norm01','img_hw': None}
        }

    clamped = np.clip(frame_indices, 0, total-1)
    kpts = np.array([pose_obj['keypoints'][i] for i in clamped])  # [T,1,133,2]
    conf = np.array([pose_obj['scores'][i]    for i in clamped])  # [T,1,133]

    img_hw = None
    if video_path is not None:
        try:
            vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
            if len(vr) > 0:
                f0 = vr.get_batch([0]).asnumpy()[0]  # [H,W,C]
                img_hw = (f0.shape[0], f0.shape[1])
        except Exception:
            img_hw = None

    return {'skeletons': kpts, 'confs': conf, 'meta': {'space':'norm01','img_hw': img_hw}}

def crop_scale(motion, thr):
    """
    motion: [..., K, 3]，xy 为 0~1 归一化坐标，conf 在 motion[...,2]
    返回 (result[-1,1], scale, [xs,ys,dx,dy]) ，result 的 xy ∈ [-1,1]
    """
    try:
        result = copy.deepcopy(motion)
        valid = motion[..., 2] > thr
        valid_coords = motion[valid][..., :2]
        if valid_coords.size < 8:
            return np.zeros_like(motion), 0.0, None

        xmin = float(np.min(valid_coords[:, 0])); xmax = float(np.max(valid_coords[:, 0]))
        ymin = float(np.min(valid_coords[:, 1])); ymax = float(np.max(valid_coords[:, 1]))
        dx = xmax - xmin; dy = ymax - ymin
        if dx <= 0 or dy <= 0:
            return np.zeros_like(motion), 0.0, None

        s = max(dx, dy)
        cx = (xmin + xmax) * 0.5
        cy = (ymin + ymax) * 0.5
        xs = cx - s * 0.5
        ys = cy - s * 0.5

        result[..., :2] = (motion[..., :2] - [xs, ys]) / s
        result[..., :2] = (result[..., :2] - 0.5) * 2.0
        result = np.clip(result, -1, 1)
        result[result[..., 2] <= thr] = 0
        return result, s, [xs, ys, dx, dy]
    except Exception:
        return np.zeros_like(motion), 0.0, None

def load_part_kp(skeletons, confs, thr=0.3):
    """
    将 [T,1,133,2] / [T,1,133] 切出 body/left/right/face_all 四块并归一化到 [-1,1]
    返回: (dict{part: Tensor[T,K,3]}, crop_meta)
    """
    T = skeletons.shape[0]
    kps_with_scores, scale, xs, ys, dx, dy = {}, None, None, None, None, None
    try:
        for part in ['body','left','right','face_all']:
            kps, confidences = [], []
            for t in range(T):
                sk = skeletons[t, 0]  # [133,2]
                cf = confs[t, 0]      # [133]
                if part == 'body':
                    idx = [0] + list(range(3, 11))            # 9 关键点
                elif part == 'left':
                    idx = list(range(91, 112))                # 21
                elif part == 'right':
                    idx = list(range(112, 133))               # 21
                else:
                    idx = [i for i in list(range(23,40))[::2]] + [i for i in range(83,91)] + [53]  # 18
                kp = sk[idx, :]
                cf_sel = cf[idx]
                if part in ('left','right','face_all'):
                    # 局部平移归一（以第一个点或某参考点）
                    anchor = kp[0] if part != 'face_all' else kp[-1]
                    kp = kp - anchor
                kps.append(kp); confidences.append(cf_sel)

            kps = np.stack(kps,0)                    # [T,K,2]
            confidences = np.stack(confidences,0)    # [T,K]
            cat = np.concatenate([kps, confidences[...,None]], -1)  # [T,K,3]
            if part == 'body':
                res, s, xy = crop_scale(cat, thr)
                scale = s
                if xy is not None:
                    xs, ys, dx, dy = xy[0], xy[1], xy[2], xy[3]
                else:
                    xs = ys = dx = dy = 0.0
            else:
                if not scale:
                    res = np.zeros_like(cat)
                else:
                    res = cat.copy()
                    res[..., :2] = np.clip(res[..., :2] / scale, -1, 1)
                    res[res[...,2] <= thr] = 0
            kps_with_scores[part] = torch.tensor(res, dtype=torch.float32)

    except Exception:
        # 兜底：空姿态
        return (
            {
                'body': torch.zeros((T, 9, 3)),
                'left': torch.zeros((T, 21, 3)),
                'right': torch.zeros((T, 21, 3)),
                'face_all': torch.zeros((T, 18, 3))
            },
            {'scale':0.0,'xs':0.0,'ys':0.0,'scale_x':0.0,'scale_y':0.0}
        )

    return kps_with_scores, {
        'scale': float(scale or 0.0),
        'xs': float(xs or 0.0), 'ys': float(ys or 0.0),
        'scale_x': float(dx or 0.0), 'scale_y': float(dy or 0.0),
    }

# ============ 数据集实现 ============
class CSLNewsDataset(BaseDataset):
    """
    从 cfg['datasets']['CSL_News'] 读取：
      - rgb_dir, pose_dir
      - train_labels / val_labels / test_labels
      - temporal.{T/jitter/...}（可选）
      - augment_train / augment_val（可选）
    复用 BaseDataset.collate_fn；__getitem__ 返回：
      (name, pose_sample(dict), text(str), indices(np.ndarray), support(dict))
    """
    def __init__(self, args, cfg, phase: str):
        super().__init__(args, cfg, phase)

        ds_cfg = _get(cfg, "datasets.CSL_News", {}) or {}
        root = _get(ds_cfg, "root", None)  # 可选

        # 路径
        self.rgb_dir  = _abs_path(root, _get(ds_cfg, "rgb_dir"))
        self.pose_dir = _abs_path(root, _get(ds_cfg, "pose_dir"))

        if not os.path.isdir(self.rgb_dir):
            raise FileNotFoundError(f"[CSL_News] rgb_dir 不存在: {self.rgb_dir}")
        if not os.path.isdir(self.pose_dir):
            raise FileNotFoundError(f"[CSL_News] pose_dir 不存在: {self.pose_dir}")

        # 标注路径：随 phase 选择
        if self.phase == 'train':
            label_path = _abs_path(root, _get(ds_cfg, "train_labels"))
        elif self.phase == 'val':
            label_path = _abs_path(root, _get(ds_cfg, "val_labels"))
        else:
            label_path = _abs_path(root, _get(ds_cfg, "test_labels"))
        if not label_path or not os.path.exists(label_path):
            raise FileNotFoundError(f"[CSL_News] {self.phase} 标注不存在: {label_path}")

        with open(label_path, "r", encoding="utf-8") as f:
            self.annotation = json.load(f)
        if not isinstance(self.annotation, list):
            raise ValueError(f"[CSL_News] 标注应为 list，实际是 {type(self.annotation)}")

        # temporal
        tmp_cfg = _get(ds_cfg, "temporal", {}) or {}
        # 用 T 作为时间采样目标长度（与 Base 的 max_length 对齐）
        self.max_length = int(_get(tmp_cfg, "T", self.max_length))
        self.jitter     = bool(_get(tmp_cfg, "jitter", True))

        # transform（这里沿用 Base 的默认；如需额外增强，可覆盖 build_train_transform/build_val_transform）

        # 调试
        if getattr(args, "debug", False):
            print(f"[CSL_News][DEBUG] phase={self.phase} | rgb_dir={self.rgb_dir} | pose_dir={self.pose_dir}")
            print(f"[CSL_News][DEBUG] labels={label_path} | num samples={len(self.annotation)}")
            print(f"[CSL_News][DEBUG] max_length={self.max_length} | jitter={self.jitter}")

    # 覆盖为“分段均匀 + 抖动/中点”，与 Daily 保持一致
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

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        try:
            rec = self.annotation[index]
            text  = rec.get('text', '')
            vname = rec.get('video', '')
            pkl   = rec.get('pose', '')

            if not vname or not pkl:
                raise ValueError(f"记录缺少 video/pose 字段: {rec}")

            video_path = os.path.join(self.rgb_dir, vname)
            pose_path  = os.path.join(self.pose_dir, pkl)
            if not os.path.exists(pose_path):
                raise FileNotFoundError(f"姿态文件不存在: {pose_path}")

            # 读一次 pkl（避免重复 I/O）
            pose_all = pickle.load(open(pose_path, 'rb'))
            duration = len(pose_all.get('scores', []))
            indices  = self._sample_indices(duration)

            # 姿态按 indices 切片
            kpt_raw = load_kpt_from_obj(pose_all, indices, video_path=video_path)
            pose_sample, crop_meta_norm01 = load_part_kp(kpt_raw['skeletons'], kpt_raw['confs'])

            # RGB（可关）
            support = {
                'video_name': vname,
                'rgb_img_indices': np.asarray(indices, dtype=np.int64),
            }
            if self.rgb_support and os.path.exists(video_path):
                rgb_seq = load_rgb(video_path, indices, size=(224, 224), transform=self.data_transform)
                support['rgb_img'] = rgb_seq
            else:
                support['rgb_img'] = torch.zeros(len(indices), 3, 224, 224)

            return vname, pose_sample, text, indices, support

        except Exception as e:
            print(f"[CSL_News][WARN] 样本 index={index} 读取失败: {e}")
            # 占位样本（尺寸与 Base.collate 对齐）
            placeholder_pose = {
                'body':     torch.zeros((self.max_length, 9, 3)),
                'left':     torch.zeros((self.max_length, 21, 3)),
                'right':    torch.zeros((self.max_length, 21, 3)),
                'face_all': torch.zeros((self.max_length, 18, 3)),
            }
            indices = np.arange(self.max_length, dtype=np.int64)
            support = {
                'rgb_img': torch.zeros(self.max_length, 3, 224, 224),
                'rgb_img_indices': indices,
                'video_name': f"error_video_{index}",
            }
            return f"error_{index}", placeholder_pose, "", indices, support
