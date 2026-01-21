import os
import pickle
from typing import List, Optional
import cv2

from synthetic_world.assets import SignAsset


# ---------- Helpers (pure I/O only) ----------

def _load_annotations(pkl_path: str):
    """Load CSL-Daily annotation pkl."""
    with open(pkl_path, "rb") as f:
        anno = pickle.load(f)
    info = anno["info"]
    anno_dict = {item["name"]: item for item in info}
    return anno_dict


def _count_frames_in_folder(folder: str) -> int:
    extensions = (".jpg", ".jpeg", ".png", ".bmp")
    return sum(1 for f in os.listdir(folder) if f.lower().endswith(extensions))


def _get_resolution_from_folder(folder: str) -> tuple:
    extensions = (".jpg", ".jpeg", ".png", ".bmp")
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(extensions):
            img = cv2.imread(os.path.join(folder, f))
            if img is not None:
                return (img.shape[1], img.shape[0])  # (W, H)
    return (0, 0)


# ---------- Main Loader ----------

def load_csl_daily_as_assets_v1(
    root: str,
    rgb_dir: str,
    anno_pkl: str,
    split_file: Optional[str] = None,
    fps: int = 25,
    max_samples: Optional[int] = None,
    verbose: bool = True,
) -> List[SignAsset]:
    """
    v1 CSL-Daily loader:
    - ONLY indexing
    - NO semantic inference
    - Lazy SignAsset only
    """

    rgb_root = os.path.join(root, rgb_dir)
    anno_path = os.path.join(root, anno_pkl)

    if verbose:
        print(f"[CSL-Daily v1] Loading annotations from {anno_path}")

    anno_dict = _load_annotations(anno_path)

    if split_file:
        split_path = os.path.join(root, split_file)
        if verbose:
            print(f"[CSL-Daily v1] Using split file: {split_path}")
        with open(split_path, "r", encoding="utf-8") as f:
            names = [l.strip() for l in f if l.strip()]
    else:
        names = sorted(os.listdir(rgb_root))

    assets: List[SignAsset] = []
    skipped = 0

    for idx, name in enumerate(names):
        if max_samples and len(assets) >= max_samples:
            break

        if name not in anno_dict:
            skipped += 1
            continue

        clip_dir = os.path.join(rgb_root, name)
        if not os.path.isdir(clip_dir):
            skipped += 1
            continue

        num_frames = _count_frames_in_folder(clip_dir)
        if num_frames == 0:
            skipped += 1
            continue

        anno = anno_dict[name]
        text = "".join(anno["label_char"])
        gloss = anno["label_gloss"]

        asset = SignAsset(
            asset_id=name,
            video_path=clip_dir,
            text=text,
            gloss=gloss,
            fps=fps,
            num_frames=num_frames,
        )

        assets.append(asset)

        if verbose and (idx + 1) % 500 == 0:
            print(f"[CSL-Daily v1] Indexed {idx + 1}/{len(names)}")

    if verbose:
        print(f"[CSL-Daily v1] Indexed {len(assets)} SignAssets (skipped {skipped})")

    return assets
