from __future__ import annotations

import torch
from torch.utils.data import Dataset
from typing import Dict, Any

from synthetic_world.world_sampler import WorldSampler
from synthetic_world.renderer import WorldRenderer


class SignWorldStream(Dataset):
    """
    Synthetic dataset for sign-language pretraining (v1).

    Each item contains:
      - rgb video tensor
      - temporal_gt
      - timeline (semantic / structural ground truth)
    """

    def __init__(
        self,
        asset_pool,
        *,
        min_signs: int = 1,
        max_signs: int = 3,
        output_size=(224, 224),
        fps: int = 25,
        max_worlds: int = 10_000_000,
        renderer_cfg: Dict[str, Any] | None = None,
    ):
        self.sampler = WorldSampler(
            asset_pool,
            min_signs=min_signs,
            max_signs=max_signs,
        )

        self.renderer = WorldRenderer(
            output_size=output_size,
            fps=fps,
            **(renderer_cfg or {}),
        )

        self.max_worlds = max_worlds

    def __len__(self):
        return self.max_worlds

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 1. Sample a world plan
        timeline = self.sampler.sample_world()

        # 2. Render
        result = self.renderer.render(timeline)

        # 3. Convert to torch
        rgb = (
            torch.from_numpy(result.rgb)
            .permute(0, 3, 1, 2)
            .float()
            / 255.0
        )  # (T,3,H,W)

        temporal_gt = torch.from_numpy(result.temporal_gt).float()  # (T,)

        return {
            "rgb": rgb,
            "temporal_gt": temporal_gt,
            "timeline": result.timeline,               # 高层 GT（秒级、语义）
            "frame_instructions": result.frame_instructions,  # 逐帧 GT（高级任务可用）
        }


# ------------------------------------------------------------------
# Test
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Testing SignWorldStream v1 ===")

    from synthetic_world.loaders.csl_daily import load_csl_daily_as_assets_v1
    from synthetic_world.loaders.ucf101 import load_ucf101_as_assets_v1
    from synthetic_world.assets import AssetPool

    print("Loading assets...")

    signs = load_csl_daily_as_assets_v1(
        root="/home/pxl416/PeixiLiu/px_proj/px_data/csl-daily-frames-512x512",
        rgb_dir="sentence",
        anno_pkl="sentence_label/csl2020ct_v2.pkl",
        split_file="sentence_label/split_1_train.txt",
        max_samples=20,
        verbose=False,
    )

    bgs = load_ucf101_as_assets_v1(
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
        output_size=(224, 224),
        fps=25,
        max_worlds=1000,
    )

    sample = dataset[0]

    print("\nOutput:")
    print("RGB:", sample["rgb"].shape, sample["rgb"].dtype)
    print("Temporal GT sum:", sample["temporal_gt"].sum().item())

    timeline = sample["timeline"]
    segments = getattr(timeline, "segments", getattr(timeline, "sign_segments", []))
    print("Segments:")
    for seg in segments:
        print(
            f"  {seg['sign'].asset_id}: "
            f"{seg['start_sec']:.2f}s → {seg['end_sec']:.2f}s"
        )

    print("\nSignWorldStream v1 test passed ✔")
