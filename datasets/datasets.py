# utils/datasets.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, get_worker_info
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict

# 注册表 —— 以后加入 BOBSL, Phoenix 只需加一行
DATASET_REGISTRY = {
    "CSL_Daily":  "datasets.CSLDaily:CSLDailyDataset",
    "CSL_News":   "datasets.CSLNews:CSLNewsDataset",
    # "BOBSL":   "datasets.BOBSL:BOBSLDataset",
    # "Phoenix": "datasets.Phoenix:PhoenixDataset",
}


# ============================================================================
# 1) BaseDataset：保持和你原来一致（只做微调）
# ============================================================================
class BaseDataset(Dataset):
    """
    统一数据集基类：
      - 可复现 RNG（worker 粒度）
      - train/val transform 钩子
      - 通用 collate_fn（统一 RGB/pose/text 输出）
    """
    def __init__(self, args, cfg, phase: str):
        super().__init__()
        self.args = args
        self.cfg = cfg
        self.phase = phase.lower()

        self.max_length = int(getattr(args, "max_length", 128))
        self.rgb_support = bool(getattr(args, "rgb_support", True))
        self.seed = int(getattr(args, "seed", cfg.get("seed", 3407)))

        # transform 由子类覆盖
        self.data_transform = None
        if self.phase == "train":
            self.data_transform = self.build_train_transform()
        else:
            self.data_transform = self.build_val_transform()

    # 由子类覆盖
    def build_train_transform(self):
        return None

    def build_val_transform(self):
        return None

    # worker-level RNG
    def _make_rng(self):
        wi = get_worker_info()
        wid = wi.id if wi is not None else 0
        return np.random.default_rng(self.seed + wid)

    # 默认采样（子类可覆盖）
    def _sample_indices(self, duration: int):
        L = self.max_length
        if duration <= L:
            return np.arange(duration, dtype=np.int64)
        rng = self._make_rng()
        idx = rng.choice(duration, size=L, replace=False)
        idx.sort()
        return idx

    # --------------------------------------------------------------
    # collate_fn：**必须统一输出格式（RGB, pose, mask, text）**
    # --------------------------------------------------------------
    def collate_fn(self, batch):
        """
        期望 dataset 返回：
        (name, pose_sample{keypoints}, text, support{rgb_img, attn_mask, segments})
        """
        name_batch = []
        pose_list = []
        text_batch = []
        support_list = []

        for item in batch:
            name, pose_sample, text, support = item
            name_batch.append(name)
            pose_list.append(pose_sample)
            text_batch.append(text)
            support_list.append(support)

        B = len(batch)
        src_input = {}

        # ========= pose (keypoints) =========
        has_pose = any(p and "keypoints" in p for p in pose_list)
        if has_pose:
            lengths = []
            padded = []
            for p in pose_list:
                if p and "keypoints" in p:
                    x = p["keypoints"]
                    lengths.append(x.shape[0])
                else:
                    x = torch.zeros((1, 21, 3))
                    lengths.append(1)

                padded.append(x)

            max_len = max(lengths)
            pad_out = []

            for x, L in zip(padded, lengths):
                if L < max_len:
                    pad = x[-1:].expand(max_len - L, *x.shape[1:])
                    x = torch.cat([x, pad], 0)
                pad_out.append(x)

            src_input["keypoints"] = torch.stack(pad_out, 0)
            src_input["kp_lengths"] = torch.tensor(lengths)

            src_input["attention_mask"] = pad_sequence(
                [torch.ones(L, dtype=torch.long) for L in lengths],
                batch_first=True, padding_value=0
            )
        else:
            src_input["keypoints"] = torch.zeros((B, 1, 21, 3))
            src_input["kp_lengths"] = torch.ones(B, dtype=torch.long)
            src_input["attention_mask"] = torch.ones((B, 1), dtype=torch.long)

        # ========= RGB =========
        if self.rgb_support:
            rgb_clips = []
            rgb_lengths = []

            for s in support_list:
                clip = s["rgb_img"]
                rgb_clips.append(clip)
                rgb_lengths.append(clip.shape[0])

            max_rgb = max(rgb_lengths)
            rgb_padded = []

            for clip, L in zip(rgb_clips, rgb_lengths):
                if L < max_rgb:
                    pad = clip[-1:].expand(max_rgb - L, *clip.shape[1:])
                    clip = torch.cat([clip, pad], 0)
                rgb_padded.append(clip)

            src_input["rgb_img"] = torch.stack(rgb_padded, 0)
            src_input["rgb_len"] = torch.tensor(rgb_lengths)
            src_input["rgb_attn_mask"] = pad_sequence(
                [torch.ones(L, dtype=torch.bool) for L in rgb_lengths],
                batch_first=True, padding_value=0
            )

        # ========= segments =========
        src_input["segments"] = {
            "starts": [s["segments"]["starts"] for s in support_list],
            "ends":   [s["segments"]["ends"] for s in support_list],
            "texts":  [s["segments"]["texts"] for s in support_list],
        }

        # ========= text =========
        tgt_input = {
            "gt_sentence": text_batch
        }

        src_input["name_batch"] = name_batch

        return src_input, tgt_input


# ============================================================================
# 2) MultiDataset wrapper —— 支持多个 dataset 混合训练
# ============================================================================
class MultiDataset(Dataset):
    """
    多数据集混合：
      datasets: List[(dataset_instance, weight)]
      自动按权重采样
    """
    def __init__(self, datasets, weights=None):
        self.datasets = datasets
        N = len(datasets)
        if weights is None:
            weights = [1.0] * N
        self.weights = np.array(weights, dtype=np.float64)
        self.weights /= self.weights.sum()

        self.lengths = [len(ds) for ds in datasets]
        self.cum = np.cumsum(self.weights)

    def __len__(self):
        # 让 dataloader 无限重复，所以这里返回最大长度
        return max(self.lengths)

    def _select_dataset(self):
        r = random.random()
        idx = int(np.searchsorted(self.cum, r))
        return idx, self.datasets[idx]

    def __getitem__(self, _):
        ds_idx, ds = self._select_dataset()
        ridx = random.randint(0, len(ds) - 1)
        return ds[ridx]


# ============================================================================
# 3) 工厂函数 create_dataloader（支持 multi-dataset）
# ============================================================================
def create_dataloader(args, cfg, phase="train"):
    phase = phase.lower()

    # -----------------------------------------
    # 解析 active_datasets
    # 允许：
    # active_datasets: ["CSL_Daily", "CSL_News"]
    # active_weights:  [1.0, 0.5]
    # -----------------------------------------
    ds_names = cfg.get("active_datasets", ["CSL_Daily"])
    ds_weights = cfg.get("active_weights", [1.0] * len(ds_names))

    ds_list = []

    for name in ds_names:
        if name not in DATASET_REGISTRY:
            raise KeyError(f"未注册的数据集：{name}")

        module_name, class_name = DATASET_REGISTRY[name].split(":")
        module = __import__(module_name, fromlist=[class_name])
        ds_class = getattr(module, class_name)

        inst = ds_class(args=args, cfg=cfg, phase=phase)
        ds_list.append(inst)

    # 若只有一个数据集 → 直接 dataloader
    if len(ds_list) == 1:
        dataset = ds_list[0]
    else:
        dataset = MultiDataset(ds_list, weights=ds_weights)

    # -----------------------------------------
    # DataLoader
    # -----------------------------------------
    train_cfg = cfg.get("Training", {})
    batch_size = train_cfg.get("batch_size", 8)
    num_workers = train_cfg.get("num_workers", 4)

    # worker seed
    def _seed_worker(worker_id):
        seed = torch.initial_seed() % (2**32)
        random.seed(seed)
        np.random.seed(seed)

    g = torch.Generator()
    g.manual_seed(cfg.get("seed", 3407))

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(phase == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(phase == "train"),
        collate_fn=ds_list[0].collate_fn,  # 用第一个 dataset 的 collate_fn 即可
        worker_init_fn=_seed_worker,
        generator=g,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
