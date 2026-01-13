# synthetic_world/temporal_composer.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np


class TemporalComposer:
    """
    Convert a WorldTimeline into per-frame instructions.
    Temporal only. No pixel ops.

    Recommended API:
        for ins in composer.iter_frames(timeline):
            ...

    Also provides:
        compose(timeline) -> List[Dict]  (materializes all frames, use only for debug)
        temporal_gt(timeline) -> np.ndarray (T,) float32
        spans(timeline) -> np.ndarray (N,2) int64 (start_frame, end_frame)
    """

    def __init__(
        self,
        fps: int = 25,
        loop_background: bool = False,
        duration_mode: str = "min",  # "min" | "background" | "target"
    ):
        """
        Args:
            fps: world fps
            loop_background: if True, background frame idx wraps around.
                             if False, clamp to last frame.
            duration_mode:
                - "min":    total duration = min(background.duration, config.target_duration if exists else bg.duration)
                - "background": total duration = background.duration
                - "target": total duration = timeline.config['target_duration'] (fallback to background.duration)
        """
        self.fps = int(fps)
        self.loop_background = bool(loop_background)
        assert duration_mode in ("min", "background", "target")
        self.duration_mode = duration_mode

    # ---------- public ----------

    def iter_frames(self, timeline: Any) -> Iterable[Dict[str, Any]]:
        """
        Yield per-frame instructions.

        Each yielded dict:
            {
              "frame_idx": int,
              "timestamp": float,
              "background": {
                  "asset": BackgroundAsset,
                  "frame_idx": int,
              },
              "active_signs": [
                  {
                    "sign": SignAsset,
                    "asset_frame_idx": int,   # sign's own frame index
                    "segment": dict,          # raw segment dict
                    "start_sec": float,
                    "end_sec": float,
                    "text": str,
                    "gloss": list,
                    "category": str,
                  }, ...
              ],
            }
        """
        bg = timeline.background
        segs = self._get_segments(timeline)

        total_frames = self._get_total_frames(timeline)

        # Pre-normalize segments: (start_sec, end_sec, sign, payload)
        norm = []
        for seg in segs:
            start_sec, end_sec = self._get_seg_time(seg)
            if end_sec <= start_sec:
                continue
            sign = self._get_seg_sign(seg)
            # allow segments without sign (should not happen, but keep robust)
            if sign is None:
                continue
            norm.append((float(start_sec), float(end_sec), sign, seg))

        # Optional: sort by start time
        norm.sort(key=lambda x: x[0])

        for frame_idx in range(total_frames):
            timestamp = frame_idx / self.fps

            # background frame index
            if self.loop_background:
                bg_frame_idx = frame_idx % int(getattr(bg, "num_frames", getattr(bg, "T", 0)) or 1)
            else:
                bg_nf = int(getattr(bg, "num_frames", getattr(bg, "T", 0)) or 1)
                bg_frame_idx = min(frame_idx, max(bg_nf - 1, 0))

            active_signs = []
            for (start_sec, end_sec, sign, seg_raw) in norm:
                if start_sec <= timestamp < end_sec:
                    asset_frame_idx = self._map_world_time_to_sign_frame(
                        timestamp=timestamp,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        sign=sign,
                    )

                    active_signs.append({
                        "sign": sign,
                        "asset_frame_idx": asset_frame_idx,
                        "segment": seg_raw,
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "text": getattr(sign, "text", seg_raw.get("text", "")),
                        "gloss": getattr(sign, "gloss", seg_raw.get("gloss", [])),
                        "category": getattr(sign, "semantic_category", seg_raw.get("category", "unknown")),
                    })

            yield {
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "bg_frame_idx": bg_frame_idx,
                "bg_asset": timeline.background,  # ← 加
                "active_signs": active_signs,
            }

    def compose(self, timeline: Any) -> List[Dict[str, Any]]:
        """Materialize all frame instructions (debug only)."""
        return list(self.iter_frames(timeline))

    def temporal_gt(self, timeline: Any, total_frames: Optional[int] = None) -> np.ndarray:
        """Binary temporal GT: 1 if any sign active at frame, else 0."""
        if total_frames is None:
            total_frames = self._get_total_frames(timeline)

        gt = np.zeros((total_frames,), dtype=np.float32)

        segs = self._get_segments(timeline)
        for seg in segs:
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
        """
        Return spans (N,2): [start_frame, end_frame) for each sign segment.
        Useful for label_emitter / detection style losses.
        """
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

        if not spans:
            return np.zeros((0, 2), dtype=np.int64)
        return np.array(spans, dtype=np.int64)

    # ---------- internals ----------

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
            if target is None:
                dur = bg_dur
            else:
                dur = min(bg_dur, float(target))

        dur = max(dur, 0.0)
        total = int(np.round(dur * self.fps))
        return max(total, 1)

    def _get_segments(self, timeline: Any) -> List[Dict[str, Any]]:
        # compatible fields: segments / sign_segments
        if hasattr(timeline, "sign_segments"):
            segs = getattr(timeline, "sign_segments")
        else:
            segs = getattr(timeline, "segments", [])

        # Some samplers store segments under timeline.timeline
        if segs is None and hasattr(timeline, "timeline"):
            segs = getattr(timeline, "timeline")

        return list(segs) if segs is not None else []

    def _get_seg_time(self, seg: Dict[str, Any]) -> Tuple[float, float]:
        # compatible keys: start_sec/end_sec or start_time/end_time or start/end
        if "start_sec" in seg and "end_sec" in seg:
            return float(seg["start_sec"]), float(seg["end_sec"])
        if "start_time" in seg and "end_time" in seg:
            return float(seg["start_time"]), float(seg["end_time"])
        if "start" in seg and "end" in seg:
            return float(seg["start"]), float(seg["end"])
        raise KeyError(f"Segment missing time keys: {list(seg.keys())}")

    def _get_seg_sign(self, seg: Dict[str, Any]):
        # compatible keys: sign / asset
        if "sign" in seg:
            return seg["sign"]
        if "asset" in seg:
            return seg["asset"]
        return None

    def _map_world_time_to_sign_frame(
        self,
        timestamp: float,
        start_sec: float,
        end_sec: float,
        sign: Any,
    ) -> int:
        """
        Map current world timestamp to sign's internal frame index.

        Strategy (robust):
            ratio = (t - start) / (end - start)
            idx = floor(ratio * sign.num_frames)
            clamp to [0, num_frames-1]
        """
        seg_dur = max(end_sec - start_sec, 1e-6)
        ratio = (timestamp - start_sec) / seg_dur
        ratio = max(0.0, min(1.0, ratio))

        num_frames = int(getattr(sign, "num_frames", getattr(sign, "T", 0)) or 0)
        if num_frames <= 0:
            return 0

        idx = int(np.floor(ratio * num_frames))
        if idx >= num_frames:
            idx = num_frames - 1
        if idx < 0:
            idx = 0
        return idx


# ----------------- Test -----------------
if __name__ == "__main__":
    print("=== Testing TemporalComposer ===")

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

    s1 = MockSign("s1", "hello", ["HELLO"], num_frames=50, duration=2.0)
    s2 = MockSign("s2", "thanks", ["THANKS"], num_frames=30, duration=1.5)

    tl = MockTimeline(
        background=bg,
        sign_segments=[
            {"sign": s1, "start_sec": 0.5, "end_sec": 2.5},
            {"sign": s2, "start_sec": 2.0, "end_sec": 3.5},
        ],
        config={"target_duration": 4.0},
    )

    composer = TemporalComposer(fps=25, loop_background=False, duration_mode="min")

    # generator test
    frames = list(composer.iter_frames(tl))
    print("Total frames:", len(frames))

    # check key frames
    for k in [0, 12, 25, 50, 75, 90]:
        if k >= len(frames):
            continue
        ins = frames[k]
        act = ins["active_signs"]
        print(
            f"Frame {k:3d} t={ins['timestamp']:.2f}s  "
            f"bg={ins['bg_frame_idx']:3d}  active={len(act)}"
        )

        for a in act:
            print("   ", a["sign"].asset_id, a["asset_frame_idx"], a["text"])

    gt = composer.temporal_gt(tl)
    sp = composer.spans(tl)

    print("temporal_gt:", gt.shape, "pos_frames=", float(gt.sum()))
    print("spans:", sp.shape, sp)

    print("Test passed ✔")
