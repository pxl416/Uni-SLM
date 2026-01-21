# synthetic_world/timeline.py

import random
from dataclasses import dataclass
from typing import List, Optional

from synthetic_world.assets import SignAsset, BackgroundAsset


# ============================================================
# Data structures
# ============================================================

@dataclass
class TimelineSegment:
    """
    One sign segment placed on the background timeline.
    """
    sign: SignAsset
    start_sec: float
    end_sec: float

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


@dataclass
class Timeline:
    """
    A concrete temporal plan for one synthetic world.
    """
    background: BackgroundAsset
    segments: List[TimelineSegment]

    @property
    def total_duration(self) -> float:
        return self.background.duration


# ============================================================
# Timeline planner
# ============================================================

class TimelinePlanner:
    """
    v1 TimelinePlanner

    Responsibilities:
      - decide WHEN each sign appears
      - avoid overlaps
      - respect background duration

    Non-responsibilities:
      - video decoding
      - frame-level operations
      - spatial logic
    """

    def __init__(
        self,
        target_duration: Optional[float] = None,
        max_placement_trials: int = 20,
        allow_overlap: bool = False,
    ):
        """
        Args:
            target_duration:
                Optional cap on usable background duration (seconds).
                If None, use full background duration.
            max_placement_trials:
                How many random attempts to place one sign.
            allow_overlap:
                v1 default False. If True, overlaps are allowed.
        """
        self.target_duration = target_duration
        self.max_placement_trials = max_placement_trials
        self.allow_overlap = allow_overlap

    # --------------------------------------------------------

    def plan(
        self,
        background: BackgroundAsset,
        signs: List[SignAsset],
    ) -> Timeline:
        """
        Generate a Timeline for given background and signs.
        """

        # Decide usable duration
        bg_duration = background.duration
        usable_duration = (
            min(bg_duration, self.target_duration)
            if self.target_duration is not None
            else bg_duration
        )

        segments: List[TimelineSegment] = []
        used_intervals = []  # list of (start, end)

        for sign in signs:
            sign_dur = sign.duration

            # Skip sign if it cannot possibly fit
            if sign_dur > usable_duration:
                continue

            placed = False

            for _ in range(self.max_placement_trials):
                start = random.uniform(0.0, usable_duration - sign_dur)
                end = start + sign_dur

                if self.allow_overlap or not self._overlaps(start, end, used_intervals):
                    segments.append(
                        TimelineSegment(
                            sign=sign,
                            start_sec=start,
                            end_sec=end,
                        )
                    )
                    used_intervals.append((start, end))
                    placed = True
                    break

            # If placement failed, silently drop this sign (v1 policy)
            if not placed:
                continue

        # Sort segments by time
        segments.sort(key=lambda seg: seg.start_sec)

        return Timeline(
            background=background,
            segments=segments
        )

    # --------------------------------------------------------

    @staticmethod
    def _overlaps(start: float, end: float, used: List[tuple]) -> bool:
        """
        Check whether [start, end) overlaps with any used interval.
        """
        for us, ue in used:
            if not (end <= us or start >= ue):
                return True
        return False


# ============================================================
# Minimal test
# ============================================================

if __name__ == "__main__":
    print("=== Testing TimelinePlanner ===")

    # NOTE: this test assumes assets already exist;
    # it does NOT load or decode video frames.

    from synthetic_world.loaders.csl_daily import load_csl_daily_as_assets_v1
    from synthetic_world.loaders.ucf101 import load_ucf101_as_assets_v1

    print("Loading assets (small subset)...")

    signs = load_csl_daily_as_assets_v1(
        root="/home/pxl416/PeixiLiu/px_proj/px_data/csl-daily-frames-512x512",
        rgb_dir="sentence",
        anno_pkl="sentence_label/csl2020ct_v2.pkl",
        split_file="sentence_label/split_1_train.txt",
        max_samples=5,
        verbose=False,
    )

    bgs = load_ucf101_as_assets_v1(
        root="/home/pxl416/PeixiLiu/px_proj/px_data/UCF-101",
        max_samples=2,
        verbose=False,
    )

    bg = bgs[0]
    planner = TimelinePlanner(target_duration=6.0)

    timeline = planner.plan(bg, signs)

    print("\nBackground:", bg.asset_id)
    print("Background duration:", bg.duration)

    print("\nTimeline segments:")
    for i, seg in enumerate(timeline.segments):
        print(
            f"  [{i}] {seg.sign.asset_id} "
            f"{seg.start_sec:.2f}s → {seg.end_sec:.2f}s "
            f"(dur={seg.duration:.2f}s)"
        )

    print("\nTimelinePlanner test passed ✔")
