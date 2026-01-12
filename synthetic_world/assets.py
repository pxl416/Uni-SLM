# synthetic_world
# 从 CSL / UCF 里加载干净的视频
import os
import random
import numpy as np
import torch
from PIL import Image
from typing import List, Dict, Optional


# Core asset types
class SignAsset:
    """
    A sign-language clip with semantic meaning.
    """
    def __init__(
        self,
        asset_id: str,
        frames: np.ndarray,        # (T,H,W,3) uint8
        keypoints: Optional[np.ndarray],  # (T,21,3) or None
        text: str,
        gloss: List[str],
        fps: int = 25,
    ):
        self.asset_id = asset_id
        self.frames = frames
        self.keypoints = keypoints
        self.text = text
        self.gloss = gloss
        self.fps = fps

        self.T = frames.shape[0]
        self.H = frames.shape[1]
        self.W = frames.shape[2]

    @property
    def duration(self):
        return self.T / self.fps


class BackgroundAsset:
    """
    A background video with no sign semantics.
    """
    def __init__(
        self,
        asset_id: str,
        frames: np.ndarray,   # (T,H,W,3)
        fps: int = 25,
        motion_level: float = 0.0,
        scene_type: str = "unknown",
    ):
        self.asset_id = asset_id
        self.frames = frames
        self.fps = fps
        self.motion_level = motion_level
        self.scene_type = scene_type

        self.T = frames.shape[0]
        self.H = frames.shape[1]
        self.W = frames.shape[2]

    @property
    def duration(self):
        return self.T / self.fps


# Asset pools
class AssetPool:
    """
    Holds all available assets and allows sampling.
    """
    def __init__(self):
        self.sign_assets: List[SignAsset] = []
        self.bg_assets: List[BackgroundAsset] = []

    def add_sign(self, asset: SignAsset):
        self.sign_assets.append(asset)

    def add_background(self, asset: BackgroundAsset):
        self.bg_assets.append(asset)

    def sample_sign(self) -> SignAsset:
        return random.choice(self.sign_assets)

    def sample_background(self) -> BackgroundAsset:
        return random.choice(self.bg_assets)

    def summary(self):
        return {
            "num_signs": len(self.sign_assets),
            "num_backgrounds": len(self.bg_assets),
        }

if __name__ == "__main__":
    print("=== Synthetic World Asset Test ===")

    # ---- create fake sign clip ----
    T, H, W = 40, 128, 128
    sign_frames = np.random.randint(0, 255, (T, H, W, 3), dtype=np.uint8)
    keypoints = np.random.rand(T, 21, 3).astype(np.float32)

    sign = SignAsset(
        asset_id="sign_001",
        frames=sign_frames,
        keypoints=keypoints,
        text="hello",
        gloss=["HELLO"],
        fps=25,
    )

    # ---- create fake background clip ----
    T2 = 100
    bg_frames = np.random.randint(0, 255, (T2, H, W, 3), dtype=np.uint8)

    bg = BackgroundAsset(
        asset_id="bg_001",
        frames=bg_frames,
        fps=25,
        motion_level=0.3,
        scene_type="office"
    )

    # ---- put into pool ----
    pool = AssetPool()
    pool.add_sign(sign)
    pool.add_background(bg)

    print("Pool summary:", pool.summary())

    # ---- sample ----
    s = pool.sample_sign()
    b = pool.sample_background()

    print("\nSampled Sign:")
    print("  id:", s.asset_id)
    print("  text:", s.text)
    print("  duration:", s.duration, "sec")

    print("\nSampled Background:")
    print("  id:", b.asset_id)
    print("  scene:", b.scene_type)
    print("  duration:", b.duration, "sec")

    print("\nTest passed ✔")

