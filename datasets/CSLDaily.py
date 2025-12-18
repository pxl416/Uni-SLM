# datasets/CSLDaily.py
# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import torch
from PIL import Image
from datasets.datasets import BaseDataset, dummy_rgb_tensor
from utils.config import _normalize_str


def resolve_path(root, p):
    if p is None:
        return None
    p = str(p)
    return p if os.path.isabs(p) else os.path.join(root, p)


class CSLDailyDataset(BaseDataset):
    def __init__(self, args, cfg, phase="train"):
        super().__init__(args, cfg, phase)

        self.language = "zh"
        self.ds_cfg = cfg.dataset
        self.aug_cfg = getattr(cfg, "augment", None)
        self.global_data = getattr(cfg, "global_data", None)

        self.root = _normalize_str(self.ds_cfg.paths.root)
        self.rgb_dir = resolve_path(self.root, self.ds_cfg.paths.rgb)
        self.text_map_path = resolve_path(self.root, self.ds_cfg.paths.text)
        self.seg_path = resolve_path(self.root, self.ds_cfg.paths.segments)

        splits_cfg = self.ds_cfg.splits
        split_file = splits_cfg[self.phase] if isinstance(splits_cfg, dict) else getattr(splits_cfg, self.phase)
        self.split_file = resolve_path(self.root, split_file)
        self.sample_ids = self._load_split_ids()

        with open(self.text_map_path, "rb") as f:
            self.anno = pickle.load(f)

        self.anno_dict = {item["name"]: item for item in self.anno["info"]}

        gloss_vocab = self.anno["gloss_map"]
        self.gloss2id = {g: i for i, g in enumerate(gloss_vocab)}
        print(f"[Info] Loaded gloss vocab size = {len(self.gloss2id)}")

        self.mod_rgb = getattr(self.ds_cfg.modalities, "rgb", True)
        self.mod_text = getattr(self.ds_cfg.modalities, "text", True)
        self.mod_pose = getattr(self.ds_cfg.modalities, "pose", False)

        self.resize = getattr(self.global_data, "resize", [224, 224]) if self.global_data else [224, 224]

    def _load_split_ids(self):
        with open(self.split_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def _load_rgb_clip(self, sample_id):
        if not self.mod_rgb:
            return None

        folder = os.path.join(self.rgb_dir, sample_id)
        if not os.path.exists(folder):
            return None

        frame_files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png")))
        if not frame_files:
            return None

        frames = []
        for fname in frame_files:
            img = Image.open(os.path.join(folder, fname)).convert("RGB")
            img = img.resize(tuple(self.resize))
            arr = np.array(img)
            tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
            frames.append(tensor)

        return torch.stack(frames, 0)

    def _load_pose(self, sample_id):
        return None

    def get_item_data(self, idx):
        sample_id = self.sample_ids[idx]
        name = sample_id

        pose_sample = {"keypoints": None}

        if self.mod_text and name in self.anno_dict:
            a = self.anno_dict[name]
            text = "".join(a["label_char"])
            gloss = a["label_gloss"]
            gloss_ids = [self.gloss2id[g] for g in gloss if g in self.gloss2id]
        else:
            text = ""
            gloss = []
            gloss_ids = []

        rgb_clip = self._load_rgb_clip(sample_id)
        if rgb_clip is None:
            rgb_clip = dummy_rgb_tensor()

        seg = {"starts": [], "ends": [], "texts": []}

        support = {
            "rgb_img": rgb_clip,
            "segments": seg,
            "gloss": gloss,
            "gloss_ids": gloss_ids,
        }

        return name, pose_sample, text, support

    def __len__(self):
        return len(self.sample_ids)
