# synthetic_world/world_sampler.py
from dataclasses import dataclass
from typing import List, Dict
import random

from synthetic_world.assets import AssetPool, SignAsset, BackgroundAsset


@dataclass
class WorldPlan:
    background: BackgroundAsset
    signs: List[SignAsset]
    meta: Dict


class WorldSampler:
    """
    v1 WorldSampler:
      - decide WHICH assets to use
      - NO timeline / NO temporal placement
    """

    def __init__(
        self,
        pool: AssetPool,
        min_signs: int = 1,
        max_signs: int = 3,
    ):
        self.pool = pool
        self.min_signs = min_signs
        self.max_signs = max_signs

    def sample_world(self) -> WorldPlan:
        # 1. sample background
        bg = self.pool.sample_background()

        # 2. sample signs
        num_signs = random.randint(self.min_signs, self.max_signs)
        signs = [self.pool.sample_sign() for _ in range(num_signs)]

        # 3. return abstract plan
        return WorldPlan(
            background=bg,
            signs=signs,
            meta={
                "num_signs": num_signs,
                "bg_id": bg.asset_id,
                "sign_ids": [s.asset_id for s in signs],
            }
        )
