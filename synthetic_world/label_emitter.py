# synthetic_world/label_emitter.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import numpy as np


# ============================================================
# Unified Label Container
# ============================================================

@dataclass
class Labels:
    """
    Unified supervision container for synthetic pretraining.
    All fields are model-facing or dataset-facing.
    """

    # -------- Temporal --------
    temporal_binary: np.ndarray          # (T,) float32 ∈ {0,1}
    temporal_soft: Optional[np.ndarray]  # (T,) float32 ∈ [0,1]
    segment_spans: np.ndarray            # (N,2) int64 [start, end)

    # -------- Spatial (per frame) --------
    frame_bboxes: List[List[Tuple[int, int, int, int]]]   # len=T
    frame_masks: Optional[np.ndarray]     # (T, K, H, W) uint8 or None

    # -------- Text / semantics --------
    text_alignments: List[Dict[str, Any]]
    vocabulary: List[str]

    # -------- Meta --------
    meta: Dict[str, Any]


# ============================================================
# LabelEmitter
# ============================================================

class LabelEmitter:
    """
    Convert (SpatialPipeline output + Timeline) into training labels.
    """

    def __init__(self, include_masks: bool = True):
        self.include_masks = include_masks

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def emit(
        self,
        *,
        spatial_output: Dict[str, Any],
        timeline: Any,
        fps: int,
        total_frames: int,
    ) -> Labels:
        """
        Args:
            spatial_output:
                Output dict from SpatialPipeline.run()
            timeline:
                Timeline object (with segments / sign_segments)
            fps:
                Frames per second
            total_frames:
                T
        """

        # ---------------- Temporal GT ----------------
        temporal_binary, temporal_soft = self._build_temporal_gt(
            timeline, fps, total_frames
        )

        segment_spans = self._extract_segment_spans(
            timeline, fps, total_frames
        )

        # ---------------- Spatial ----------------
        frame_bboxes = spatial_output["bboxes"]

        frame_masks = None
        if self.include_masks and "spatial_masks" in spatial_output:
            frame_masks = self._stack_frame_masks(
                spatial_output["spatial_masks"]
            )

        # ---------------- Text ----------------
        text_alignments = self._extract_text_alignments(
            timeline, fps, total_frames
        )

        vocabulary = self._extract_vocabulary(timeline)

        # ---------------- Meta ----------------
        meta = {
            "fps": fps,
            "total_frames": total_frames,
            "num_segments": len(segment_spans),
            "has_masks": frame_masks is not None,
            **spatial_output.get("meta", {}),
        }

        return Labels(
            temporal_binary=temporal_binary,
            temporal_soft=temporal_soft,
            segment_spans=segment_spans,
            frame_bboxes=frame_bboxes,
            frame_masks=frame_masks,
            text_alignments=text_alignments,
            vocabulary=vocabulary,
            meta=meta,
        )

    # ============================================================
    # Temporal
    # ============================================================

    def _build_temporal_gt(
        self,
        timeline: Any,
        fps: int,
        T: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Produce:
          - binary temporal label
          - soft temporal proximity label
        """
        binary = np.zeros(T, dtype=np.float32)
        soft = np.zeros(T, dtype=np.float32)

        segments = getattr(timeline, "segments",
                           getattr(timeline, "sign_segments", []))

        for seg in segments:
            start_sec, end_sec = self._get_segment_time(seg)
            if start_sec is None:
                continue

            s = int(start_sec * fps)
            e = int(end_sec * fps)
            s = max(0, min(s, T - 1))
            e = max(s + 1, min(e, T))

            binary[s:e] = 1.0

            # soft: triangular decay
            for t in range(T):
                if t < s:
                    soft[t] = max(soft[t], 1 - (s - t) / fps)
                elif t >= e:
                    soft[t] = max(soft[t], 1 - (t - e) / fps)
                else:
                    soft[t] = 1.0

        soft = np.clip(soft, 0.0, 1.0)
        return binary, soft

    def _extract_segment_spans(
        self,
        timeline: Any,
        fps: int,
        T: int,
    ) -> np.ndarray:
        spans = []

        segments = getattr(timeline, "segments",
                           getattr(timeline, "sign_segments", []))

        for seg in segments:
            start_sec, end_sec = self._get_segment_time(seg)
            if start_sec is None:
                continue

            s = int(start_sec * fps)
            e = int(end_sec * fps)

            s = max(0, min(s, T - 1))
            e = max(s + 1, min(e, T))
            spans.append([s, e])

        if not spans:
            return np.zeros((0, 2), dtype=np.int64)

        return np.asarray(spans, dtype=np.int64)

    # ============================================================
    # Spatial
    # ============================================================

    def _stack_frame_masks(
        self,
        frame_masks: List[List[np.ndarray]],
    ) -> np.ndarray:
        """
        Convert:
            List[T][K][H,W] → (T,K,H,W)
        """
        T = len(frame_masks)
        max_k = max((len(m) for m in frame_masks), default=0)

        if max_k == 0:
            return None

        H, W = frame_masks[0][0].shape
        out = np.zeros((T, max_k, H, W), dtype=np.uint8)

        for t, masks in enumerate(frame_masks):
            for k, m in enumerate(masks):
                out[t, k] = m

        return out

    # ============================================================
    # Text
    # ============================================================

    def _extract_text_alignments(
        self,
        timeline: Any,
        fps: int,
        T: int,
    ) -> List[Dict[str, Any]]:
        alignments = []

        segments = getattr(timeline, "segments",
                           getattr(timeline, "sign_segments", []))

        for idx, seg in enumerate(segments):
            sign = seg.get("sign")
            if sign is None:
                continue

            start_sec, end_sec = self._get_segment_time(seg)
            if start_sec is None:
                continue

            s = int(start_sec * fps)
            e = int(end_sec * fps)

            alignments.append({
                "segment_id": idx,
                "sign_id": getattr(sign, "asset_id", f"seg_{idx}"),
                "text": getattr(sign, "text", ""),
                "gloss": getattr(sign, "gloss", []),
                "start_frame": s,
                "end_frame": e,
            })

        return alignments

    def _extract_vocabulary(self, timeline: Any) -> List[str]:
        vocab = set()

        segments = getattr(timeline, "segments",
                           getattr(timeline, "sign_segments", []))

        for seg in segments:
            sign = seg.get("sign")
            if sign is None:
                continue

            text = getattr(sign, "text", "")
            if not text:
                continue

            if any(ord(c) > 127 for c in text):  # Chinese
                for c in text:
                    if c.strip():
                        vocab.add(c)
            else:
                for w in text.lower().split():
                    vocab.add(w)

        return sorted(vocab)

    # ============================================================
    # Utils
    # ============================================================

    def _get_segment_time(
        self,
        seg: Dict[str, Any],
    ) -> Tuple[Optional[float], Optional[float]]:
        for a, b in [
            ("start_sec", "end_sec"),
            ("start_time", "end_time"),
            ("start", "end"),
        ]:
            if a in seg and b in seg:
                return seg[a], seg[b]
        return None, None
