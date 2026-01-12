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


