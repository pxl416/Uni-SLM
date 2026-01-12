# synthetic_world/loaders/ucf101.py
import os
import cv2
import numpy as np
from typing import List, Optional

from synthetic_world.assets import BackgroundAsset


# -------------------------------
# Helpers
# -------------------------------

def _load_video_avi(path: str, resize=None, max_frames=None):
    """
    Load an .avi video into (T,C,H,W) torch-like float32 tensor in [0,1]
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(round(fps)) if fps > 0 else 25

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize is not None:
            frame = cv2.resize(frame, resize)

        frame = frame.astype(np.float32) / 255.0
        frame = np.transpose(frame, (2,0,1))  # (3,H,W)
        frames.append(frame)

        if max_frames and len(frames) >= max_frames:
            break

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"Empty video: {path}")

    frames = np.stack(frames, axis=0)  # (T,C,H,W)
    return frames, fps


def _estimate_motion_level(frames: np.ndarray) -> float:
    """
    Rough motion estimator: mean absolute frame difference.
    frames: (T,C,H,W)
    """
    if frames.shape[0] < 2:
        return 0.0
    diff = np.abs(frames[1:] - frames[:-1])
    return float(diff.mean())


# -------------------------------
# Main Loader
# -------------------------------

def load_ucf101_as_assets(
    root: str,
    classes: Optional[List[str]] = None,
    resize=(224,224),
    max_frames: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> List[BackgroundAsset]:
    """
    Load UCF101 videos into BackgroundAsset list.

    Expected structure:
        root/
            BaseballPitch/
                v_BaseballPitch_g01_c01.avi
                ...
            Basketball/
                ...
    """

    assets = []

    for cls in sorted(os.listdir(root)):
        class_dir = os.path.join(root, cls)
        if not os.path.isdir(class_dir):
            continue

        if classes is not None and cls not in classes:
            continue

        for fname in sorted(os.listdir(class_dir)):
            if not fname.endswith(".avi"):
                continue

            path = os.path.join(class_dir, fname)

            try:
                frames, fps = _load_video_avi(path, resize=resize, max_frames=max_frames)
            except Exception as e:
                print(f"[Skip] {path}: {e}")
                continue

            motion = _estimate_motion_level(frames)

            asset = BackgroundAsset(
                asset_id=f"{cls}/{fname}",
                frames=frames,
                fps=fps,
                motion_level=motion,
                scene_type=cls,
            )

            assets.append(asset)

            if max_samples and len(assets) >= max_samples:
                break

        if max_samples and len(assets) >= max_samples:
            break

    print(f"[UCF101] Loaded {len(assets)} BackgroundAssets")
    return assets


# Test
if __name__ == "__main__":
    ROOT = "/home/pxl416/PeixiLiu/px_proj/px_data/UCF-101"

    assets = load_ucf101_as_assets(
        root=ROOT,
        resize=(224,224),
        max_frames=80,
        max_samples=3,
    )

    a = assets[0]
    print("\nSample background:")
    print("ID:", a.asset_id)
    print("Scene:", a.scene_type)
    print("Frames:", a.frames.shape)
    print("FPS:", a.fps)
    print("Motion level:", a.motion_level)
    print("Duration:", a.duration, "sec")
