# datasets/datasets.py
from __future__ import annotations
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from types import SimpleNamespace
from utils.config import load_yaml, dict_to_ns

DATASET_REGISTRY = {
    "CSL_Daily": "datasets.CSLDaily:CSLDailyDataset",
}

print("[DEBUG] datasets.py loaded.")


# dummy tensors
def dummy_rgb_tensor(time_steps=1, channel=3):
    return torch.zeros((time_steps, channel, 224, 224), dtype=torch.float32)

def dummy_pose_tensor(time_steps=1):
    return torch.zeros((time_steps, 21, 3), dtype=torch.float32)


# padding
def pad_sequence(seq_list, lengths):
    if len(seq_list) == 0:
        raise ValueError("pad_sequence: empty sequence list")
    max_len = max(lengths)
    B = len(seq_list)
    example = seq_list[0]
    out = torch.zeros((B, max_len, *example.shape[1:]),
                      dtype=example.dtype, device=example.device)
    for i, (seq, L) in enumerate(zip(seq_list, lengths)):
        out[i, :L] = seq
    return out


def to_tensor(img):
    if isinstance(img, torch.Tensor):
        return img
    # PIL or numpy -> tensor
    if hasattr(img, "mode"):
        # PIL Image
        img = torch.from_numpy(np.array(img))
    else:
        img = torch.tensor(img)
    # shape (H,W,C) → (C,H,W)
    if img.dim() == 3 and img.shape[-1] in [1,3]:
        img = img.permute(2,0,1)
    return img.float() / 255.0


def resize_tensor(img, size):
    # img: (C,H,W)
    H, W = img.shape[1], img.shape[2]
    return F.interpolate(img.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)


def adjust_channel(img, out_c):
    c = img.shape[0]
    if c == out_c:
        return img
    if out_c == 1:
        return img.mean(dim=0, keepdim=True)
    if out_c == 3 and c == 1:
        return img.repeat(3, 1, 1)
    if out_c > c:
        return img.repeat(out_c // c, 1, 1)
    raise ValueError(f"Unsupported channel conversion: {c} → {out_c}")




# BaseDataset
class BaseDataset(Dataset):
    def __init__(self, args, cfg, phase: str):
        super().__init__()
        self.args = args
        self.cfg = cfg
        self.phase = phase.lower()
        self.transform_train = None
        self.transform_val = None

    def get_item_data(self, idx):
        raise NotImplementedError

    def __getitem__(self, idx):
        name, pose_sample, text, support = self.get_item_data(idx)

        # pose
        keypoints = pose_sample.get("keypoints", None)
        if keypoints is None:
            keypoints = dummy_pose_tensor()
            has_pose = False
        else:
            has_pose = True

        if keypoints.dim() == 2:
            keypoints = keypoints.unsqueeze(0)
        elif keypoints.dim() == 3 and keypoints.shape[-1] != 3:
            if keypoints.shape[1] == 3:
                keypoints = keypoints.permute(0, 2, 1)
            else:
                raise ValueError(f"Invalid keypoints shape: {keypoints.shape}")

        # ===== RGB =====
        rgb = support.get("rgb_img", None)
        channel = getattr(self.cfg.global_data, "channel", 3)
        resize_hw = getattr(self.cfg.global_data, "resize", [224, 224])

        if rgb is None:
            rgb = dummy_rgb_tensor(channel=channel)
            has_rgb = False
        else:
            has_rgb = True
            # rgb: (T,C,H,W)
            # resize + channel adjust + normalize
            frames = []
            for frame in rgb:  # frame: (C,H,W)
                C = frame.shape[0]
                # channel adjust
                if C != channel:
                    if channel == 1:
                        frame = frame.mean(dim=0, keepdim=True)
                    elif channel == 3 and C == 1:
                        frame = frame.repeat(3, 1, 1)
                    else:
                        raise ValueError(f"Unsupported channel mapping: {C}->{channel}")

                # resize
                frame = F.interpolate(
                    frame.unsqueeze(0),
                    size=resize_hw,
                    mode="bilinear",
                    align_corners=False
                ).squeeze(0)

                # normalize
                if channel == 3:
                    mean = torch.tensor([0.485, 0.456, 0.406], dtype=frame.dtype, device=frame.device)[:, None, None]
                    std = torch.tensor([0.229, 0.224, 0.225], dtype=frame.dtype, device=frame.device)[:, None, None]
                    frame = (frame - mean) / std
                else:  # channel == 1
                    mean = torch.tensor([0.5], dtype=frame.dtype, device=frame.device)[:, None, None]
                    std = torch.tensor([0.5], dtype=frame.dtype, device=frame.device)[:, None, None]
                    frame = (frame - mean) / std

                frames.append(frame)

            rgb = torch.stack(frames, dim=0)

        gloss = support.get("gloss", [])
        segments = support.get("segments", {"starts": [], "ends": [], "texts": []})

        return {
            "name": name,
            "text": text,
            "gloss": gloss,
            "keypoints": keypoints,
            "rgb_img": rgb,
            "segments": segments,
            "has_pose": has_pose,
            "has_rgb": has_rgb,
        }

    def collate_fn(self, batch):
        B = len(batch)

        names = [item["name"] for item in batch]
        texts = [item["text"] for item in batch]
        gloss = [item["gloss"] for item in batch]
        segments = [item["segments"] for item in batch]

        pose_list = [item["keypoints"] for item in batch]
        pose_len = [x.shape[0] for x in pose_list]

        rgb_list = [item["rgb_img"] for item in batch]
        rgb_len = [x.shape[0] for x in rgb_list]

        has_pose = torch.tensor([item["has_pose"] for item in batch], dtype=torch.bool)
        has_rgb = torch.tensor([item["has_rgb"] for item in batch], dtype=torch.bool)

        pose_tensor = pad_sequence(pose_list, pose_len)
        rgb_tensor = pad_sequence(rgb_list, rgb_len)

        pose_len_tensor = torch.tensor(pose_len, dtype=torch.long)
        rgb_len_tensor = torch.tensor(rgb_len, dtype=torch.long)

        max_pose = pose_tensor.shape[1]
        kp_mask = torch.arange(max_pose)[None, :].expand(B, -1) < pose_len_tensor[:, None]

        max_rgb = rgb_tensor.shape[1]
        rgb_mask = torch.arange(max_rgb)[None, :].expand(B, -1) < rgb_len_tensor[:, None]

        src_input = {
            "name": names,
            "keypoints": pose_tensor,
            "kp_len": pose_len_tensor,
            "kp_mask": kp_mask,
            "has_pose": has_pose,
            "rgb_img": rgb_tensor,
            "rgb_len": rgb_len_tensor,
            "rgb_mask": rgb_mask,
            "has_rgb": has_rgb,
            "segments": segments,
        }

        tgt_input = {
            "gt_sentence": texts,
            "gt_gloss": gloss,
        }

        return src_input, tgt_input


class MultiDataset(Dataset):
    pass


# create_dataloader
def create_dataloader(args, cfg, phase="train"):
    # phase = phase.lower()
    alias = {
        "val": "dev",      # HuggingFace-style
        "valid": "dev",
        "validation": "dev",
    }
    if phase in alias:
        print(f"[Info] Phase '{phase}' mapped to '{alias[phase]}' (dataset compatibility)")
        phase = alias[phase]
    ds_names = getattr(cfg, "active_datasets", ["CSL_Daily"])
    if len(ds_names) != 1:
        raise NotImplementedError("Only single dataset supported for now")

    name = ds_names[0]

    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset: {name}")

    module_name, class_name = DATASET_REGISTRY[name].split(":")
    try:
        module = __import__(module_name, fromlist=[class_name])
        ds_class = getattr(module, class_name)
    except Exception as e:
        raise ImportError(f"Dataset import failed: {module_name}:{class_name}\n{e}")

    ds_paths_cfg = getattr(cfg, "datasets", None)
    if ds_paths_cfg is None:
        raise KeyError("cfg.datasets is empty")

    ds_yaml_path = (
        ds_paths_cfg[name]
        if isinstance(ds_paths_cfg, dict)
        else getattr(ds_paths_cfg, name)
    )

    raw_ds_cfg = load_yaml(ds_yaml_path)
    ds_cfg = dict_to_ns(raw_ds_cfg)

    global_data = getattr(cfg, "data", SimpleNamespace())
    setattr(ds_cfg, "global_data", global_data)

    dataset = ds_class(args=args, cfg=ds_cfg, phase=phase)

    train_cfg = getattr(cfg, "Training", SimpleNamespace())
    batch_size = getattr(train_cfg, "batch_size", 8)
    num_workers = getattr(train_cfg, "num_workers", 4)

    def _seed_worker(worker_id):
        seed = torch.initial_seed() % (2**32)
        np.random.seed(seed)
        random.seed(seed)

    generator = torch.Generator()
    generator.manual_seed(getattr(cfg, "seed", 3407))

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(phase == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(phase == "train"),
        collate_fn=dataset.collate_fn,
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=(num_workers > 0 and torch.cuda.is_available()),
        prefetch_factor=2 if num_workers > 0 else None,
    )
