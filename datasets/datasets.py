# datasets/datasets.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, get_worker_info
from types import SimpleNamespace


# ===========================================================
# 注册表：根据 datasets/xxx.py 里的类名来加载
# ===========================================================
DATASET_REGISTRY = {
    "CSL_Daily": "datasets.CSLDaily:CSLDailyDataset",
    "CSL_News": "datasets.CSLNews:CSLNewsDataset",
}
print("[DEBUG] BaseDataset.collate_fn loaded.")

# ===========================================================
# 基础工具：统一 dummy tensor（防止 missing）
# ===========================================================
def dummy_rgb_tensor():
    """返回一个 [1, 3, 224, 224] 的 dummy RGB clip"""
    return torch.zeros((1, 3, 224, 224), dtype=torch.float32)

def dummy_pose_tensor():
    """返回一个 [1, 21, 3] 的 dummy pose"""
    return torch.zeros((1, 21, 3), dtype=torch.float32)


# ===========================================================
# BaseDataset —— Uni-SLM 所有数据集都必须继承
# ===========================================================
class BaseDataset(Dataset):
    """
    所有派生数据集必须实现 get_item_data(idx)，并返回：
        name: str
        pose_sample: dict     { "keypoints": Tensor or None }
        text: str
        support: dict         { "rgb_img": Tensor or None, "segments": dict }

    collate_fn 将强制对所有模态进行统一 padding（raw tensor）
    mask 不在这里创建（避免 shape mismatch），由下游模块构造。
    """

    def __init__(self, args, cfg, phase: str):
        super().__init__()
        self.args = args
        self.cfg = cfg
        self.phase = phase.lower()

        # 全局 hyper-params
        self.max_length = int(getattr(args, "max_length", 128))
        self.seed = int(getattr(args, "seed", getattr(cfg, "seed", 3407)))

        # feature 开关（RGB / Pose / Text）
        # 但即便关闭，也会创建 dummy entries，避免 KeyError
        self.use_rgb = getattr(args, "rgb_support", True)
        self.use_pose = getattr(args, "pose_support", True)

        # 默认 transform，可以被子类覆盖
        self.transform_train = None
        self.transform_val = None

    # -----------------------------------------------------
    # 子类必须实现：返回四元组
    # -----------------------------------------------------
    def get_item_data(self, idx):
        raise NotImplementedError

    def __getitem__(self, idx):
        """
        把派生类的 get_item_data 返回的四元组包装成统一格式的字典
        """
        name, pose_sample, text, support = self.get_item_data(idx)

        return {
            "name": name,
            "keypoints": pose_sample.get("keypoints", None),
            "text": text,
            "gloss": support.get("gloss", []),  # ★ 支持 CSLR
            "rgb_img": support.get("rgb_img", None),
            "segments": support.get("segments", {"starts": [], "ends": [], "texts": []}),
        }

    # -----------------------------------------------------
    # worker-specific RNG
    # -----------------------------------------------------
    def _rng(self):
        wi = get_worker_info()
        wid = wi.id if wi else 0
        return np.random.default_rng(self.seed + wid)

    # -----------------------------------------------------
    # collate_fn：真正的统一多模态 batch 组装器
    # -----------------------------------------------------
    def collate_fn(self, batch):
        """
        batch: List[dict]
        统一输出 src_input, tgt_input
        """

        name_list = []
        rgb_list = []
        rgb_len_list = []
        pose_list = []
        pose_len_list = []
        text_list = []
        gloss_list = []
        segments_list = []

        for item in batch:
            name_list.append(item["name"])
            text_list.append(item["text"])
            gloss_list.append(item["gloss"])
            segments_list.append(item["segments"])

            # pose
            kp = item["keypoints"]
            if kp is None:
                kp = dummy_pose_tensor()
            pose_list.append(kp)
            pose_len_list.append(kp.shape[0])

            # rgb
            rgb = item["rgb_img"]
            if rgb is None:
                rgb = dummy_rgb_tensor()
            rgb_list.append(rgb)
            rgb_len_list.append(rgb.shape[0])

        # pad pose
        max_pose = max(pose_len_list)
        pose_padded = [
            torch.cat([p, p[-1:].expand(max_pose - L, *p.shape[1:])], 0)
            if L < max_pose else p
            for p, L in zip(pose_list, pose_len_list)
        ]
        pose_tensor = torch.stack(pose_padded, 0)

        # pad rgb
        max_rgb = max(rgb_len_list)
        rgb_padded = [
            torch.cat([r, r[-1:].expand(max_rgb - L, *r.shape[1:])], 0)
            if L < max_rgb else r
            for r, L in zip(rgb_list, rgb_len_list)
        ]
        rgb_tensor = torch.stack(rgb_padded, 0)

        # src_input = 模态数据
        src_input = {
            "name": name_list,

            "keypoints": pose_tensor,
            "kp_len": torch.tensor(pose_len_list, dtype=torch.long),

            "rgb_img": rgb_tensor,
            "rgb_len": torch.tensor(rgb_len_list, dtype=torch.long),

            "segments": segments_list,
        }

        # tgt_input = label
        tgt_input = {
            "gt_sentence": text_list,
            "gt_gloss": gloss_list,  # ★ 加上 gloss 标签（识别任务必需）
        }

        return src_input, tgt_input


# ===========================================================
# MultiDataset —— 多数据集加权采样（已修复）
# ===========================================================
class MultiDataset(Dataset):
    def __init__(self, datasets, weights=None):
        self.datasets = datasets
        self.lengths = [len(ds) for ds in datasets]

        self.weights = np.array(weights if weights else [1.0] * len(datasets), dtype=np.float64)
        self.weights /= self.weights.sum()
        self.cum = np.cumsum(self.weights)

    def __len__(self):
        return max(self.lengths)

    def _select_dataset(self):
        r = random.random()
        idx = int(np.searchsorted(self.cum, r))
        return idx, self.datasets[idx]

    def __getitem__(self, _):
        idx, ds = self._select_dataset()
        ridx = random.randint(0, self.lengths[idx] - 1)
        return ds[ridx]


# ===========================================================
# create_dataloader —— 全项目统一的数据入口
# ===========================================================

def create_dataloader(args, cfg, phase="train"):
    phase = phase.lower()

    ds_names = getattr(cfg, "active_datasets", ["CSL_Daily"])
    ds_weights = getattr(cfg, "active_weights", [1.0] * len(ds_names))

    ds_list = []
    for name in ds_names:
        if name not in DATASET_REGISTRY:
            raise KeyError(f"未知数据集：{name}")

        module_name, class_name = DATASET_REGISTRY[name].split(":")
        module = __import__(module_name, fromlist=[class_name])
        ds_class = getattr(module, class_name)

        inst = ds_class(args=args, cfg=cfg, phase=phase)
        ds_list.append(inst)

    # 单 dataset
    if len(ds_list) == 1:
        dataset = ds_list[0]
        collate_fn = ds_list[0].collate_fn
    else:
        dataset = MultiDataset(ds_list, ds_weights)
        collate_fn = ds_list[0].collate_fn  # 所有 dataset 都继承 BaseDataset → OK

    train_cfg = getattr(cfg, "Training", SimpleNamespace())
    batch_size = getattr(train_cfg, "batch_size", 8)
    num_workers = getattr(train_cfg, "num_workers", 4)

    # worker seed
    def _seed_worker(worker_id):
        seed = torch.initial_seed() % (2**32)
        random.seed(seed)
        np.random.seed(seed)

    generator = torch.Generator()
    generator.manual_seed(getattr(cfg, "seed", 3407))

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(phase == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(phase == "train"),
        collate_fn=collate_fn,
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
