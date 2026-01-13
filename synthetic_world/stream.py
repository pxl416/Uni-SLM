# synthetic_world/stream.py
import torch
from torch.utils.data import Dataset
import numpy as np

from world_sampler import WorldSampler
from renderer import WorldRenderer


class SignWorldStream(Dataset):
    """
    Synthetic dataset for temporal sign detection.
    Each item is a freshly sampled synthetic video.
    """

    def __init__(self, asset_pool,
                 min_signs=1,
                 max_signs=3,
                 target_duration=8.0,
                 output_size=(224,224),
                 fps=25,
                 max_worlds=10_000_000):

        self.sampler = WorldSampler(
            asset_pool,
            min_signs=min_signs,
            max_signs=max_signs,
            target_duration=target_duration
        )

        self.renderer = WorldRenderer(
            output_size=output_size,
            fps=fps
        )

        self.max_worlds = max_worlds

    def __len__(self):
        return self.max_worlds

    def __getitem__(self, idx):
        # 1. Sample world
        world = self.sampler.sample_world()

        # 2. Render video
        result = self.renderer.render(world)

        rgb = result.rgb            # (T,H,W,3) uint8
        temporal_gt = result.temporal_gt   # (T,) float32

        # 3. To torch
        rgb = torch.from_numpy(rgb).permute(0,3,1,2).float() / 255.0
        temporal_gt = torch.from_numpy(temporal_gt).float()

        return {
            "rgb": rgb,               # (T,3,H,W)
            "temporal_gt": temporal_gt,
            "segments": result.segments
        }

if __name__ == "__main__":
    print("=== Testing SignWorldStream ===")

    from synthetic_world.loaders.csl_daily import load_csl_daily_as_assets
    from synthetic_world.loaders.ucf101 import load_ucf101_as_assets
    from synthetic_world.assets import AssetPool

    print("Loading assets...")

    signs = load_csl_daily_as_assets(
        root="/home/pxl416/PeixiLiu/px_proj/px_data/csl-daily-frames-512x512",
        rgb_dir="sentence",
        anno_pkl="sentence_label/csl2020ct_v2.pkl",
        split_file="sentence_label/split_1_train.txt",
        max_samples=20,
        verbose=False,
    )

    bgs = load_ucf101_as_assets(
        root="/home/pxl416/PeixiLiu/px_proj/px_data/UCF-101",
        max_samples=5,
        verbose=False,
    )

    pool = AssetPool()
    for s in signs:
        pool.add_sign(s)
    for b in bgs:
        pool.add_background(b)

    print("Pool summary:", pool.summary())

    dataset = SignWorldStream(
        pool,
        min_signs=1,
        max_signs=2,
        target_duration=5.0,
        output_size=(224,224),
        fps=25,
        max_worlds=1000
    )

    print("\nSampling one synthetic world...")
    sample = dataset[0]

    print("\nOutput tensors:")
    print("RGB shape:", sample["rgb"].shape)
    print("RGB dtype:", sample["rgb"].dtype)
    print("RGB range:", float(sample["rgb"].min()), float(sample["rgb"].max()))
    print("Temporal GT shape:", sample["temporal_gt"].shape)
    print("Temporal GT sum (should >0):", sample["temporal_gt"].sum().item())

    print("\nSegments:")
    for seg in sample["segments"]:
        print(f"  {seg['sign'].asset_id}: "
              f"{seg['start_sec']:.2f}s → {seg['end_sec']:.2f}s")

    print("\nSignWorldStream test passed ✔")


