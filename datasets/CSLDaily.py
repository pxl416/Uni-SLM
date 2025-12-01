# datasets/CSLDaily.py
# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from utils.config import _normalize_str
from datasets.datasets import BaseDataset, dummy_rgb_tensor


def resolve_path(root, p):
    """
    通用路径解析函数：
    - p 为 None → 返回 None
    - p 为绝对路径 → 不做修改
    - p 为相对路径 → 拼接 root/p
    """
    if p is None:
        return None

    p = str(p)
    if os.path.isabs(p):
        return p
    return os.path.join(root, p)


class CSLDailyDataset(BaseDataset):
    """
    CSL-Daily 单数据集实现：
    - RGB（帧序列）
    - Text（中文句子）
    - Gloss（CSLR）
    - 无 pose、无 segments
    """

    def __init__(self, args, cfg, phase: str = "train"):
        super().__init__(args, cfg, phase)

        self.ds_cfg = cfg.datasets.CSL_Daily

        # ---------------------------------------------
        # 1) 初始化路径
        # ---------------------------------------------
        self.root = _normalize_str(self.ds_cfg.paths.root)

        self.rgb_dir        = resolve_path(self.root, self.ds_cfg.paths.rgb)
        self.text_map_path  = resolve_path(self.root, self.ds_cfg.paths.text)
        self.seg_path       = resolve_path(self.root, self.ds_cfg.paths.segments)
        self.gloss_map_path = resolve_path(self.root, getattr(self.ds_cfg.paths, "gloss_map", None))

        # ---------------------------------------------
        # 2) split 文件
        # ---------------------------------------------
        splits_cfg = self.ds_cfg.splits
        if isinstance(splits_cfg, dict):
            split_file = splits_cfg[self.phase]
        else:
            split_file = getattr(splits_cfg, self.phase)
        self.split_file = resolve_path(self.root, split_file)

        self.sample_ids = self._load_split_ids()

        # ---------------------------------------------
        # 3) Annotation（text + gloss）
        # ---------------------------------------------
        if (self.text_map_path is None) or (not os.path.exists(self.text_map_path)):
            raise FileNotFoundError(f"[CSL Daily] annotation pkl not found: {self.text_map_path}")

        with open(self.text_map_path, "rb") as f:
            self.anno = pickle.load(f)

        self.anno_dict = {item["name"]: item for item in self.anno["info"]}

        # ---------------------------------------------
        # 4) 模态开关
        # ---------------------------------------------
        self.mod_rgb  = getattr(self.ds_cfg.modalities, "rgb", True)
        self.mod_pose = getattr(self.ds_cfg.modalities, "pose", False)
        self.mod_text = getattr(self.ds_cfg.modalities, "text", True)

    # -----------------------------
    # load split ids
    # -----------------------------
    def _load_split_ids(self):
        with open(self.split_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    # -----------------------------
    # load RGB frames
    # -----------------------------
    def _load_rgb_clip(self, sample_id: str):
        if not self.mod_rgb:
            return None

        folder = os.path.join(self.rgb_dir, sample_id)
        if not os.path.exists(folder):
            return None

        frame_files = sorted(
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".png"))
        )
        if not frame_files:
            return None

        frames = []
        for fname in frame_files:
            img_path = os.path.join(folder, fname)
            img = Image.open(img_path).convert("RGB")
            img = img.resize((112, 112))  # px. 251130 augment
            arr = np.array(img)
            tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
            frames.append(tensor)

        return torch.stack(frames, dim=0)

    # -----------------------------
    # CSL-Daily 无 pose
    # -----------------------------
    def _load_pose(self, sample_id: str):
        return None

    # -----------------------------
    # BaseDataset: required method
    # 返回：
    #   name, pose_sample, text, support
    # -----------------------------
    def get_item_data(self, idx):
        sample_id = self.sample_ids[idx]
        name = sample_id

        # pose 为空
        pose_sample = {"keypoints": None}

        # -------------------------
        # text & gloss
        # -------------------------
        if self.mod_text and name in self.anno_dict:
            anno  = self.anno_dict[name]
            text  = "".join(anno["label_char"])
            gloss = anno["label_gloss"]
        else:
            text  = ""
            gloss = []

        # -------------------------
        # RGB frames
        # -------------------------
        rgb_clip = self._load_rgb_clip(sample_id)
        if rgb_clip is None:
            rgb_clip = dummy_rgb_tensor()

        # -------------------------
        # 无 segments
        # -------------------------
        seg = {"starts": [], "ends": [], "texts": []}

        support = {
            "rgb_img": rgb_clip,
            "segments": seg,
            "gloss": gloss,
        }

        return name, pose_sample, text, support

    def __len__(self):
        return len(self.sample_ids)


'''
/home/pxl416/miniconda/envs/uslm_py310/bin/python /home/pxl416/PeixiLiu/px_proj/Uni-SLM/tools/test_dataloader.py 
[DEBUG] BaseDataset.collate_fn loaded.
[Info] Dataset size: 18401
[Info] Batch size = 2
[Info] Num workers = 2

[Check] Fetching one batch...

==== Batch 0 ====

[Debug] type(batch) = <class 'list'>
[Mode] Detected (src_input, tgt_input) tuple
[Name] ['S002288_P0005_T00', 'S006585_P0000_T00'] ... (total 2)
[Pose] keypoints shape: torch.Size([2, 1, 21, 3])
[Pose] kp_len: [1, 1]
[RGB] rgb_img shape: torch.Size([2, 164, 3, 512, 512])
[RGB] rgb_len: [164, 50]
[Segments] batch size: 2
  Sample 0 segments (0): []
  Sample 1 segments (0): []
[Text] ['团队内只有分工合作才能提高工作效率。', '把花插在玻璃瓶里。']

[Status] Batch test completed successfully.

Process finished with exit code 0

'''