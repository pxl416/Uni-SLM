import os
import cv2
from typing import List, Optional

from synthetic_world.assets import BackgroundAsset


# ---------- Helpers (pure I/O only) ----------

def _count_frames_and_fps(path: str) -> tuple:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(round(fps)) if fps and fps > 0 else 25

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if num_frames <= 0:
        raise RuntimeError(f"Invalid video length: {path}")

    return num_frames, fps


# ---------- Main Loader ----------

def load_ucf101_as_assets_v1(
    root: str,
    classes: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    verbose: bool = True,
) -> List[BackgroundAsset]:
    """
    v1 UCF101 loader:
    - ONLY background indexing
    - NO motion / brightness / people inference
    """

    if verbose:
        print(f"[UCF101 v1] Loading from {root}")

    if not os.path.exists(root):
        raise FileNotFoundError(root)

    all_classes = sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    )

    if classes is not None:
        selected_classes = [c for c in all_classes if c in classes]
    else:
        selected_classes = all_classes

    assets: List[BackgroundAsset] = []
    skipped = 0

    for class_idx, cls in enumerate(selected_classes):
        if max_samples and len(assets) >= max_samples:
            break

        class_dir = os.path.join(root, cls)
        video_files = sorted(f for f in os.listdir(class_dir) if f.endswith(".avi"))

        if verbose and class_idx % 10 == 0:
            print(f"[UCF101 v1] Class {class_idx+1}/{len(selected_classes)}: {cls}")

        for fname in video_files:
            if max_samples and len(assets) >= max_samples:
                break

            path = os.path.join(class_dir, fname)
            asset_id = f"{cls}/{fname}"

            try:
                num_frames, fps = _count_frames_and_fps(path)

                asset = BackgroundAsset(
                    asset_id=asset_id,
                    video_path=path,
                    fps=fps,
                    num_frames=num_frames,
                    scene_type=cls,
                )

                assets.append(asset)

            except Exception:
                skipped += 1
                continue

    if verbose:
        print(f"[UCF101 v1] Indexed {len(assets)} BackgroundAssets (skipped {skipped})")

    return assets
