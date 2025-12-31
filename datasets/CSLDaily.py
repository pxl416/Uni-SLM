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

        # -------- NEW: temporal cfg --------
        self.temporal_cfg = None
        if self.aug_cfg is not None:
            self.temporal_cfg = getattr(self.aug_cfg, "temporal", None)

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
        # -------- NEW: pretrain task flag --------
        self.is_temporal_pretrain = (
            hasattr(cfg, "Pretrain")
            and getattr(cfg.Pretrain, "task", None) in ["temporal", "tem&spa"]
        )


    def _load_split_ids(self):
        with open(self.split_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    # -------- NEW: temporal sampling --------
    def _apply_temporal_sampling(self, rgb: torch.Tensor):
        if self.temporal_cfg is None:
            return rgb

        max_frames = getattr(self.temporal_cfg, "max_frames", None)
        if max_frames is None:
            return rgb

        T = rgb.shape[0]
        if T <= max_frames:
            return rgb

        idx = torch.linspace(0, T - 1, max_frames).long()
        return rgb[idx]

    # -------- NEW: rgb augment placeholder --------
    def _apply_rgb_augment(self, rgb: torch.Tensor):
        return rgb

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

        rgb = torch.stack(frames, 0)  # (T,C,H,W)

        # -------- NEW: apply temporal + rgb augment --------
        rgb = self._apply_temporal_sampling(rgb)
        rgb = self._apply_rgb_augment(rgb)

        return rgb

    def _load_pose(self, sample_id):
        return None

    # -------- NEW: temporal concat builder --------
    def _build_temporal_concat_sample(self, idx):
        """
        Build temporal concatenated sample for pretraining.

        return:
            rgb: Tensor (T,C,H,W)
            temporal_gt: Tensor (T,)  # 1 = sign, 0 = non-sign
        """
        import random

        # -------- positive sample --------
        pos_id = self.sample_ids[idx]
        rgb_pos = self._load_rgb_clip(pos_id)
        if rgb_pos is None:
            rgb_pos = dummy_rgb_tensor()

        rgb_pos = self._apply_temporal_sampling(rgb_pos)
        T_pos = rgb_pos.shape[0]

        # -------- negative sample (random) --------
        neg_id = random.choice(self.sample_ids)
        rgb_neg = self._load_rgb_clip(neg_id)
        if rgb_neg is None:
            rgb_neg = dummy_rgb_tensor()

        rgb_neg = self._apply_temporal_sampling(rgb_neg)
        T_neg = rgb_neg.shape[0]

        # -------- concat: [neg | pos | neg] --------
        rgb = torch.cat([rgb_neg, rgb_pos, rgb_neg], dim=0)

        temporal_gt = torch.cat([
            torch.zeros(T_neg),
            torch.ones(T_pos),
            torch.zeros(T_neg)
        ], dim=0)

        return rgb, temporal_gt



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

        # rgb_clip = self._load_rgb_clip(sample_id)
        # if rgb_clip is None:
        #     rgb_clip = dummy_rgb_tensor()
        # -------- NEW: temporal pretrain branch --------
        if self.is_temporal_pretrain:
            rgb_clip, temporal_gt = self._build_temporal_concat_sample(idx)
        else:
            rgb_clip = self._load_rgb_clip(sample_id)
            if rgb_clip is None:
                rgb_clip = dummy_rgb_tensor()
            temporal_gt = None


        seg = {"starts": [], "ends": [], "texts": []}

        support = {
            "rgb_img": rgb_clip,
            "segments": seg,
            "gloss": gloss,
            "gloss_ids": gloss_ids,
        }

        if temporal_gt is not None:
            support["temporal_gt"] = temporal_gt


        return name, pose_sample, text, support

    def __len__(self):
        return len(self.sample_ids)
