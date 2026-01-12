# synthetic_world/world_sampler.py
# 输入sign和bg视频，输出timeline

import random
from typing import List, Dict

from synthetic_world.assets import SignAsset, BackgroundAsset
from synthetic_world.loaders.csl_daily import load_csl_daily_as_assets
from synthetic_world.loaders.ucf101 import load_ucf101_as_assets


class WorldSampler:
    """
    Sample semantic-consistent synthetic worlds from asset pools.
    This does NOT render pixels – it only creates a timeline.
    """

    def __init__(
        self,
        sign_assets: List[SignAsset],
        background_assets: List[BackgroundAsset],
        max_signs_per_bg: int = 3,
        min_gap: float = 0.3,
    ):
        self.sign_assets = sign_assets
        self.background_assets = background_assets
        self.max_signs_per_bg = max_signs_per_bg
        self.min_gap = min_gap

    def sample_world(self) -> Dict:
        """
        Sample one synthetic world.

        Returns:
            {
                "background": BackgroundAsset,
                "timeline": [
                    {"sign": SignAsset, "start": float, "end": float},
                    ...
                ]
            }
        """
        bg = random.choice(self.background_assets)
        T = bg.duration

        # number of signs in this world
        k = random.randint(0, self.max_signs_per_bg)

        signs = random.sample(self.sign_assets, k) if k > 0 else []

        timeline = []

        cursor = 0.0

        for sign in signs:
            dur = sign.duration

            if cursor + dur >= T:
                break

            # random jitter inside remaining time
            start = random.uniform(cursor, max(cursor, T - dur))
            end = start + dur

            timeline.append(
                {
                    "sign": sign,
                    "start": start,
                    "end": end,
                }
            )

            cursor = end + self.min_gap

        timeline = sorted(timeline, key=lambda x: x["start"])

        return {
            "background": bg,
            "timeline": timeline,
        }

    def sample_batch(self, n: int):
        return [self.sample_world() for _ in range(n)]


if __name__ == "__main__":


    print("=== World Sampler Test ===")

    signs = load_csl_daily_as_assets(
        root="/home/pxl416/PeixiLiu/px_proj/px_data/csl-daily-frames-512x512",
        rgb_dir="sentence",
        anno_pkl="sentence_label/csl2020ct_v2.pkl",
        split_file="sentence_label/split_1_train.txt",
        max_samples=10,
    )

    bgs = load_ucf101_as_assets(
        root="/home/pxl416/PeixiLiu/px_proj/px_data/UCF-101",
        max_samples=5,
    )

    sampler = WorldSampler(signs, bgs, max_signs_per_bg=3)

    world = sampler.sample_world()

    bg = world["background"]
    print("\nBackground:", bg.asset_id, "duration:", bg.duration)

    print("Timeline:")
    for e in world["timeline"]:
        s = e["sign"]
        print(
            f"  {s.asset_id} ({s.text})  {e['start']:.2f}s → {e['end']:.2f}s"
        )

    print("\nTest passed ✔")

