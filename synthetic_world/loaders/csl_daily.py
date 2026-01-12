import os
import pickle
import numpy as np
import torch
from PIL import Image
from typing import List, Optional

from synthetic_world.assets import SignAsset


# Helpers
def _load_frames_from_folder(folder: str, resize=None) -> torch.Tensor:
    """
    Load a folder of RGB frames into (T,C,H,W) float32 tensor in [0,1]
    This matches CSLDailyDataset._load_rgb_clip semantics (before normalize).
    """
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png")))
    if not files:
        raise RuntimeError(f"No frames in {folder}")

    frames = []
    for fname in files:
        img = Image.open(os.path.join(folder, fname)).convert("RGB")
        if resize is not None:
            img = img.resize(resize)

        arr = np.array(img)                      # (H,W,3) uint8
        tensor = torch.from_numpy(arr).permute(2,0,1).float() / 255.0  # (3,H,W)
        frames.append(tensor)

    return torch.stack(frames, dim=0)  # (T,C,H,W)


def _load_text_and_gloss(pkl_path: str):
    with open(pkl_path, "rb") as f:
        anno = pickle.load(f)

    info = anno["info"]
    gloss_map = anno["gloss_map"]

    anno_dict = {item["name"]: item for item in info}
    return anno_dict, gloss_map


# Main Loader
def load_csl_daily_as_assets(
    root: str,
    rgb_dir: str,
    anno_pkl: str,
    split_file: Optional[str] = None,
    resize=(224,224),
    fps: int = 25,
    max_samples: Optional[int] = None,
) -> List[SignAsset]:
    """
    Load CSL-Daily into a list of SignAsset.
    This mirrors CSLDailyDataset._load_rgb_clip + annotation semantics,
    but without Dataset / torch.utils.data dependency.
    """

    rgb_root = os.path.join(root, rgb_dir)
    anno_path = os.path.join(root, anno_pkl)

    anno_dict, _ = _load_text_and_gloss(anno_path)

    # which samples to load
    if split_file:
        with open(os.path.join(root, split_file), "r") as f:
            names = [l.strip() for l in f if l.strip()]
    else:
        names = sorted(os.listdir(rgb_root))

    assets = []

    for name in names:
        if name not in anno_dict:
            continue

        clip_dir = os.path.join(rgb_root, name)
        if not os.path.isdir(clip_dir):
            continue

        try:
            rgb = _load_frames_from_folder(clip_dir, resize=resize)   # (T,C,H,W)
        except Exception as e:
            print(f"[Skip] {name}: {e}")
            continue

        a = anno_dict[name]
        text = "".join(a["label_char"])
        gloss = a["label_gloss"]

        asset = SignAsset(
            asset_id=name,
            frames=rgb,          # now torch Tensor (T,C,H,W) in [0,1]
            keypoints=None,
            text=text,
            gloss=gloss,
            fps=fps,
        )

        assets.append(asset)

        if max_samples and len(assets) >= max_samples:
            break

    print(f"[CSL-Daily] Loaded {len(assets)} SignAssets")

    return assets


# Test
if __name__ == "__main__":
    ROOT = "/home/pxl416/PeixiLiu/px_proj/px_data/csl-daily-frames-512x512"
    RGB = "sentence"
    ANNO = "sentence_label/csl2020ct_v2.pkl"
    SPLIT = "sentence_label/split_1_train.txt"

    assets = load_csl_daily_as_assets(
        root=ROOT,
        rgb_dir=RGB,
        anno_pkl=ANNO,
        split_file=SPLIT,
        resize=(224,224),
        max_samples=3,
    )

    a = assets[0]
    print("\nSample asset:")
    print("ID:", a.asset_id)
    print("Text:", a.text)
    print("Gloss:", a.gloss[:5])
    print("RGB tensor:", a.frames.shape, a.frames.dtype)
    print("Min/Max:", float(a.frames.min()), float(a.frames.max()))
    print("Duration:", a.duration, "sec")

