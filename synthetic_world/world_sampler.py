# synthetic_world/world_sampler.py
import random
from dataclasses import dataclass
from typing import List

from synthetic_world.assets import AssetPool, SignAsset, BackgroundAsset


@dataclass
class WorldTimeline:
    background: BackgroundAsset
    segments: List[dict]   # each: {sign, start_sec, end_sec}


class WorldSampler:
    """
    Minimal world sampler:
      - pick 1 background
      - pick N sign clips
      - place them randomly on timeline (no overlap)
    """

    def __init__(self, pool: AssetPool,
                 min_signs=1,
                 max_signs=3,
                 target_duration=8.0):
        self.pool = pool
        self.min_signs = min_signs
        self.max_signs = max_signs
        self.target_duration = target_duration

    def sample_world(self) -> WorldTimeline:
        bg = self.pool.sample_background()

        max_dur = min(bg.duration, self.target_duration)

        num_signs = random.randint(self.min_signs, self.max_signs)

        signs = [self.pool.sample_sign() for _ in range(num_signs)]

        segments = []
        used = []

        for sign in signs:
            dur = sign.duration

            for _ in range(20):  # try 20 times to find free slot
                start = random.uniform(0, max_dur - dur)
                end = start + dur

                if not self._overlaps(start, end, used):
                    used.append((start, end))
                    segments.append({
                        "sign": sign,
                        "start_sec": start,
                        "end_sec": end,
                    })
                    break

        segments.sort(key=lambda x: x["start_sec"])

        return WorldTimeline(
            background=bg,
            segments=segments
        )

    def _overlaps(self, s, e, used):
        for us, ue in used:
            if not (e <= us or s >= ue):
                return True
        return False

if __name__ == "__main__":
    print("=== Testing WorldSampler ===")

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
        max_samples=10,
        verbose=False,
    )

    pool = AssetPool()
    for s in signs:
        pool.add_sign(s)
    for b in bgs:
        pool.add_background(b)

    print("Pool summary:", pool.summary())

    sampler = WorldSampler(pool, min_signs=1, max_signs=3, target_duration=6.0)

    world = sampler.sample_world()

    print("\nSampled world:")
    print("Background:", world.background.asset_id)
    print("Background duration:", world.background.duration)

    for i, seg in enumerate(world.segments):
        sign = seg["sign"]
        print(f"  Sign {i+1}: {sign.asset_id} "
              f"[{seg['start_sec']:.2f}s → {seg['end_sec']:.2f}s] "
              f"text='{sign.text}'")

    print("\nWorldSampler test passed ✔")

