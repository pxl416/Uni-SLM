# synthetic_world/temporal_composer.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
import random


class TemporalComposer:
    """
    Convert a WorldTimeline into per-frame temporal instructions.
    Temporal only. No pixel ops.

    Responsibilities:
      - timeline duration resolution
      - sign / background rate jitter
      - world-time -> asset-frame mapping
      - temporal GT & spans

    This module is intentionally stateless and side-effect free.
    """

    def __init__(
        self,
        fps: int = 25,
        loop_background: bool = False,
        duration_mode: str = "min",  # "min" | "background" | "target"
        temporal_cfg: Optional[dict] = None,
        debug: bool = False,
    ):
        self.fps = int(fps)
        self.loop_background = bool(loop_background)
        assert duration_mode in ("min", "background", "target")
        self.duration_mode = duration_mode

        self.temporal_cfg = temporal_cfg or {}
        self.debug = debug

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def iter_frames(self, timeline: Any) -> Iterable[Dict[str, Any]]:
        """
        Yield per-frame temporal instructions.

        Each yielded dict:
            {
              "frame_idx": int,
              "timestamp": float,
              "bg_asset": BackgroundAsset,
              "bg_frame_idx": int,
              "active_signs": [
                  {
                    "sign": SignAsset,
                    "asset_frame_idx": int,
                    "start_sec": float,
                    "end_sec": float,
                    "text": str,
                    "gloss": list,
                    "category": str,
                  }, ...
              ],
            }
        """
        self._validate_timeline(timeline)

        bg = timeline.background
        segs = self._get_segments(timeline)
        total_frames = self._get_total_frames(timeline)

        # ---- sample rate jitters once per timeline ----
        sign_rate = self._sample_rate(
            enabled=self.temporal_cfg.get("sign_rate_jitter", False),
            rate_range=self.temporal_cfg.get("sign_rate_range", (1.0, 1.0)),
        )
        bg_rate = self._sample_rate(
            enabled=self.temporal_cfg.get("bg_rate_jitter", False),
            rate_range=self.temporal_cfg.get("bg_rate_range", (1.0, 1.0)),
        )

        if self.debug:
            print(f"[TemporalComposer] sign_rate={sign_rate:.3f}, bg_rate={bg_rate:.3f}")

        # ---- normalize segments ----
        norm = []
        for seg in segs:
            start_sec, end_sec = self._get_seg_time(seg)
            if end_sec <= start_sec:
                continue
            sign = self._get_seg_sign(seg)
            if sign is None:
                continue
            norm.append((float(start_sec), float(end_sec), sign))

        norm.sort(key=lambda x: x[0])

        bg_num_frames = int(getattr(bg, "num_frames", getattr(bg, "T", 0)) or 1)

        for frame_idx in range(total_frames):
            timestamp = frame_idx / self.fps

            # ---- background frame index ----
            bg_t = timestamp * bg_rate
            bg_frame_idx = int(np.floor(bg_t * self.fps))

            if self.loop_background:
                bg_frame_idx = bg_frame_idx % bg_num_frames
            else:
                bg_frame_idx = min(bg_frame_idx, bg_num_frames - 1)

            active_signs = []

            for (start_sec, end_sec, sign) in norm:
                if start_sec <= timestamp < end_sec:
                    asset_frame_idx = self._map_world_time_to_sign_frame(
                        timestamp=timestamp,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        sign=sign,
                        rate=sign_rate,
                    )

                    active_signs.append({
                        "sign": sign,
                        "asset_frame_idx": asset_frame_idx,
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "text": getattr(sign, "text", ""),
                        "gloss": getattr(sign, "gloss", []),
                        "category": getattr(sign, "semantic_category", "unknown"),
                    })

            yield {
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "bg_asset": bg,
                "bg_frame_idx": bg_frame_idx,
                "active_signs": active_signs,
            }

    def compose(self, timeline: Any) -> List[Dict[str, Any]]:
        """
        Materialize all frame instructions.

        v1 SAFETY:
          - If timeline has segments but no active_signs are produced,
            fallback to naive second-to-frame mapping.
        """
        frames = list(self.iter_frames(timeline))

        # ------------------ v1 SAFETY CHECK ------------------
        segments = self._get_segments(timeline)
        has_segments = len(segments) > 0
        has_active = any(len(f.get("active_signs", [])) > 0 for f in frames)

        if has_segments and not has_active:
            print(
                "[TemporalComposer WARNING] segments exist but no active_signs produced. "
                "Falling back to v1 naive temporal mapping."
            )

            fps = self.fps

            # recompute active_signs naively
            for f in frames:
                t = f.get("timestamp", None)
                if t is None:
                    continue

                active = []
                for seg in segments:
                    start_sec, end_sec = self._get_seg_time(seg)
                    if start_sec <= t < end_sec:
                        sign = self._get_seg_sign(seg)
                        if sign is None:
                            continue

                        asset_frame_idx = int((t - start_sec) * fps)

                        active.append({
                            "sign": sign,
                            "asset_frame_idx": asset_frame_idx,
                            "start_sec": start_sec,
                            "end_sec": end_sec,
                            "text": getattr(sign, "text", ""),
                            "gloss": getattr(sign, "gloss", []),
                            "category": getattr(sign, "semantic_category", "unknown"),
                        })

                f["active_signs"] = active

        return frames

    def temporal_gt(self, timeline: Any, total_frames: Optional[int] = None) -> np.ndarray:
        """Binary temporal GT: 1 if any sign active, else 0."""
        if total_frames is None:
            total_frames = self._get_total_frames(timeline)

        gt = np.zeros((total_frames,), dtype=np.float32)

        for seg in self._get_segments(timeline):
            start_sec, end_sec = self._get_seg_time(seg)
            if end_sec <= start_sec:
                continue
            s = int(np.floor(start_sec * self.fps))
            e = int(np.ceil(end_sec * self.fps))
            s = max(0, min(s, total_frames - 1))
            e = max(0, min(e, total_frames))
            if e > s:
                gt[s:e] = 1.0

        return gt

    def spans(self, timeline: Any, total_frames: Optional[int] = None) -> np.ndarray:
        """Return spans (N,2): [start_frame, end_frame)."""
        if total_frames is None:
            total_frames = self._get_total_frames(timeline)

        spans = []
        for seg in self._get_segments(timeline):
            start_sec, end_sec = self._get_seg_time(seg)
            if end_sec <= start_sec:
                continue
            s = int(np.floor(start_sec * self.fps))
            e = int(np.ceil(end_sec * self.fps))
            s = max(0, min(s, total_frames - 1))
            e = max(0, min(e, total_frames))
            if e > s:
                spans.append((s, e))

        return np.array(spans, dtype=np.int64) if spans else np.zeros((0, 2), dtype=np.int64)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _validate_timeline(self, timeline: Any):
        if not hasattr(timeline, "background"):
            raise ValueError("Timeline must have attribute `background`")

    def _get_total_frames(self, timeline: Any) -> int:
        bg = timeline.background
        bg_dur = float(getattr(bg, "duration", 0.0) or 0.0)

        target = None
        cfg = getattr(timeline, "config", None)
        if isinstance(cfg, dict):
            target = cfg.get("target_duration", None)

        if self.duration_mode == "background":
            dur = bg_dur
        elif self.duration_mode == "target":
            dur = float(target) if target is not None else bg_dur
        else:  # "min"
            dur = min(bg_dur, float(target)) if target is not None else bg_dur

        dur = max(dur, 0.0)
        total = int(np.round(dur * self.fps))
        return max(total, 1)

    def _get_segments(self, timeline: Any) -> List[Dict[str, Any]]:
        if hasattr(timeline, "sign_segments"):
            return list(timeline.sign_segments)
        if hasattr(timeline, "segments"):
            return list(timeline.segments)
        return []

    def _get_seg_time(self, seg) -> Tuple[float, float]:
        # --- dataclass TimelineSegment ---
        if hasattr(seg, "start_sec") and hasattr(seg, "end_sec"):
            return float(seg.start_sec), float(seg.end_sec)

        # --- dict (legacy / loader style) ---
        if isinstance(seg, dict):
            if "start_sec" in seg and "end_sec" in seg:
                return float(seg["start_sec"]), float(seg["end_sec"])
            if "start" in seg and "end" in seg:
                return float(seg["start"]), float(seg["end"])

        raise KeyError(f"Segment missing time fields: {seg}")

    def _get_seg_sign(self, seg):
        if hasattr(seg, "sign"):
            return seg.sign
        if isinstance(seg, dict):
            return seg.get("sign", None)
        return None

    def _sample_rate(self, enabled: bool, rate_range: Tuple[float, float]) -> float:
        if not enabled:
            return 1.0
        lo, hi = rate_range
        return float(random.uniform(lo, hi))

    def _map_world_time_to_sign_frame(
        self,
        timestamp: float,
        start_sec: float,
        end_sec: float,
        sign: Any,
        rate: float,
    ) -> int:
        safe_rate = max(abs(rate), 1e-3)
        seg_dur = max(end_sec - start_sec, 1e-6)
        eff_dur = seg_dur / safe_rate

        ratio = (timestamp - start_sec) / eff_dur
        ratio = np.clip(ratio, 0.0, 1.0)

        num_frames = int(getattr(sign, "num_frames", getattr(sign, "T", 0)) or 0)
        if num_frames <= 0:
            return 0

        idx = int(np.floor(ratio * (num_frames - 1e-6)))
        return min(max(idx, 0), num_frames - 1)


# Test
if __name__ == "__main__":
    print("=== Testing TemporalComposer (v1) ===")

    from dataclasses import dataclass

    @dataclass
    class MockSign:
        asset_id: str
        text: str
        gloss: list
        num_frames: int
        duration: float
        semantic_category: str = "general"

    @dataclass
    class MockBG:
        asset_id: str
        num_frames: int
        duration: float

    @dataclass
    class MockTimeline:
        background: MockBG
        sign_segments: list
        config: dict

    bg = MockBG("bg", num_frames=100, duration=4.0)
    s1 = MockSign("s1", "hello", ["HELLO"], num_frames=40, duration=2.0)
    s2 = MockSign("s2", "thanks", ["THANKS"], num_frames=30, duration=1.5)

    tl = MockTimeline(
        background=bg,
        sign_segments=[
            {"sign": s1, "start_sec": 0.5, "end_sec": 2.5},
            {"sign": s2, "start_sec": 2.0, "end_sec": 3.5},
        ],
        config={"target_duration": 4.0},
    )

    composer = TemporalComposer(
        fps=25,
        duration_mode="min",
        temporal_cfg={
            "sign_rate_jitter": True,
            "sign_rate_range": (0.9, 1.1),
            "bg_rate_jitter": True,
            "bg_rate_range": (0.95, 1.05),
        },
        debug=True,
    )

    frames = composer.compose(tl)
    print(f"Total frames: {len(frames)}")

    # ---- sanity checks ----
    assert len(frames) > 0

    last_idx = {}
    for f in frames:
        for a in f["active_signs"]:
            sid = a["sign"].asset_id
            idx = a["asset_frame_idx"]
            if sid in last_idx:
                assert idx >= last_idx[sid] - 1
            last_idx[sid] = idx

    gt = composer.temporal_gt(tl)
    spans = composer.spans(tl)

    print("temporal_gt sum:", int(gt.sum()))
    print("spans:", spans)

    print("TemporalComposer v1 test passed ✔")
