from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from synthetic_world.temporal_composer import TemporalComposer


# ============================================================
# Unified Label Container
# ============================================================

@dataclass
class Labels:
    """
    Unified supervision container for synthetic pretraining.
    """

    # -------- Temporal --------
    temporal_binary: np.ndarray          # (T,) float32 ∈ {0,1}
    temporal_soft: Optional[np.ndarray]  # (T,) float32 ∈ [0,1] or None
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
# LabelEmitter (v1 compatible)
# ============================================================

class LabelEmitter:
    """
    Convert RenderResult (+ Timeline) into unified training labels.

    Design:
      - v1: trust RenderResult.temporal_gt
      - v2+: allow rebuilding from SpatialPipeline outputs
    """

    def __init__(self, include_masks: bool = True):
        self.include_masks = include_masks

    # ------------------------------------------------------------
    # v1 Entry Point (RECOMMENDED)
    # ------------------------------------------------------------

    def emit_from_render_result(
        self,
        *,
        render_result,
        fps: int,
    ) -> Labels:
        """
        Build Labels directly from WorldRenderer.RenderResult.

        This is the ONLY recommended entry in v1.
        """

        temporal_binary = render_result.temporal_gt
        T = len(temporal_binary)

        # v1: do NOT fabricate soft labels
        temporal_soft = None

        # -------- Segment spans (reuse TemporalComposer logic) --------
        # composer = TemporalComposer(fps=fps)
        # segment_spans = composer.spans(
        #     render_result.timeline,
        #     total_frames=T,
        # )
        segment_spans = self._spans_from_temporal_binary(
            temporal_binary
        )

        # -------- Spatial --------
        frame_bboxes = render_result.bboxes_per_frame

        frame_masks = None
        if self.include_masks and hasattr(render_result, "spatial_masks"):
            frame_masks = self._stack_frame_masks(
                render_result.spatial_masks
            )

        # -------- Text --------
        text_alignments = self._extract_text_alignments(
            render_result.timeline,
            fps,
            T,
        )

        vocabulary = self._extract_vocabulary(
            render_result.timeline
        )

        # -------- Meta --------
        meta = {
            "fps": fps,
            "total_frames": T,
            "num_segments": len(segment_spans),
            "has_masks": frame_masks is not None,
            "source": "render_result_v1",
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

    # ------------------------------------------------------------
    # Spatial utils
    # ------------------------------------------------------------

    def _stack_frame_masks(self, frame_masks):
        """
        Stack per-frame masks into a tensor.
        v1 SAFE:
          - If no masks exist at all, return None
          - If some frames have no masks, fill with zeros
        """

        # -----------------------------------------
        # v1 safety: no masks at all
        # -----------------------------------------
        if not frame_masks:
            return None

        # find first valid mask to get H, W
        ref_mask = None
        for masks in frame_masks:
            if masks:
                ref_mask = masks[0]
                break

        if ref_mask is None:
            # all frames have empty masks
            return None

        H, W = ref_mask.shape
        T = len(frame_masks)

        stacked = np.zeros((T, H, W), dtype=np.uint8)

        for t, masks in enumerate(frame_masks):
            if not masks:
                continue
            for m in masks:
                stacked[t] |= (m > 0).astype(np.uint8)

        return stacked

    # ------------------------------------------------------------
    # Text / semantics
    # ------------------------------------------------------------

    def _extract_text_alignments(
            self,
            timeline: Any,
            fps: int,
            T: int,
    ) -> List[Dict[str, Any]]:

        alignments = []

        segments = getattr(
            timeline,
            "segments",
            getattr(timeline, "sign_segments", [])
        )

        for idx, seg in enumerate(segments):
            sign = getattr(seg, "sign", None)
            if sign is None:
                continue

            start_sec = getattr(seg, "start_sec", None)
            end_sec = getattr(seg, "end_sec", None)
            if start_sec is None or end_sec is None:
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


    def _extract_vocabulary(
            self,
            timeline: Any,
    ) -> List[str]:

        vocab = set()

        segments = getattr(
            timeline,
            "segments",
            getattr(timeline, "sign_segments", [])
        )

        for seg in segments:
            sign = getattr(seg, "sign", None)
            if sign is None:
                continue

            text = getattr(sign, "text", "")
            if not text:
                continue

            if any(ord(c) > 127 for c in text):
                for c in text:
                    if c.strip():
                        vocab.add(c)
            else:
                for w in text.lower().split():
                    vocab.add(w)

        return sorted(vocab)

    def _spans_from_temporal_binary(
            self,
            temporal: np.ndarray,
    ) -> np.ndarray:
        """
        Convert binary temporal GT into segment spans.
        """
        spans = []
        T = len(temporal)

        in_seg = False
        start = 0

        for t in range(T):
            if temporal[t] > 0 and not in_seg:
                in_seg = True
                start = t
            elif temporal[t] == 0 and in_seg:
                spans.append((start, t))
                in_seg = False

        if in_seg:
            spans.append((start, T))

        return np.asarray(spans, dtype=np.int64)



