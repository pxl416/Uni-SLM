from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import cv2

from synthetic_world.temporal_composer import TemporalComposer
from synthetic_world.spatial_composer import SpatialComposer


# Render Result
@dataclass
class RenderResult:
    """
    Rendering output container (v1 stable contract)
    """
    rgb: np.ndarray                                # (T,H,W,3) uint8
    temporal_gt: np.ndarray                        # (T,) float32
    spatial_masks: List[List[np.ndarray]]          # per-frame, per-sign masks (uint8 mask canvas coords)
    bboxes_per_frame: List[List[Tuple[int, int, int, int]]]
    frame_instructions: List[Dict[str, Any]]       # temporal composer output (patched)
    timeline: Any                                  # original world timeline


# World Renderer
class WorldRenderer:
    """
    WorldRenderer = Temporal → Spatial → Video + Labels

    V1 Design Principles:
      - TemporalComposer defines WHAT happens per frame (active_signs, bg_frame_idx, timestamp...)
      - SpatialComposer defines HOW it is rendered (compose_frame)
      - Renderer enforces the interface contract and prevents "silent all-zero GT" failure.
    """

    def __init__(
        self,
        output_size: Tuple[int, int] = (224, 224),  # (W,H)
        fps: int = 25,
        spatial_config: Optional[Dict[str, Any]] = None,
        temporal_config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        enable_cache: bool = True,
        sign_mask_provider=None,
    ):
        self.output_size = output_size
        self.fps = int(fps)
        self.enable_cache = bool(enable_cache)
        self.rng = np.random.default_rng(seed)

        # --- Temporal ---
        temporal_config = temporal_config or {}
        # v1: TemporalComposer currently only needs fps (keep clean)
        self.temporal_composer = TemporalComposer(fps=self.fps)

        # --- Spatial ---
        spatial_config = spatial_config or {}
        # IMPORTANT: SpatialComposer signature is (output_size, position_mode, spatial_cfg, debug)
        # so DO NOT pass arbitrary keys that belong to YAML root.
        self.spatial_composer = SpatialComposer(
            output_size=output_size,
            spatial_cfg=spatial_config,
            sign_mask_provider=sign_mask_provider,
        )

        # --- Caches ---
        self._bg_frame_cache: Dict[str, np.ndarray] = {}
        self._sign_full_cache: Dict[str, np.ndarray] = {}

        self.sign_mask_provider = sign_mask_provider

    # Public API
    def render(self, timeline, clear_cache: bool = True) -> RenderResult:
        """
        Render a full synthetic video from a timeline.
        """
        frame_instructions = self.temporal_composer.compose(timeline)

        # [V1 CONTRACT ENFORCEMENT + SAFETY FALLBACK]
        self._validate_and_patch_instructions(frame_instructions, timeline)

        T = len(frame_instructions)
        if T == 0:
            return self._create_empty_result(timeline)

        H, W = self.output_size[1], self.output_size[0]

        rgb = np.zeros((T, H, W, 3), dtype=np.uint8)
        temporal_gt = np.zeros((T,), dtype=np.float32)
        spatial_masks: List[List[np.ndarray]] = []
        bboxes_per_frame: List[List[Tuple[int, int, int, int]]] = []

        for t, inst in enumerate(frame_instructions):
            comp, masks, bboxes = self._render_single_frame(inst)
            rgb[t] = comp

            active = inst.get("active_signs", [])
            temporal_gt[t] = 1.0 if len(active) > 0 else 0.0

            spatial_masks.append(masks)
            bboxes_per_frame.append(bboxes)

        if clear_cache:
            self.clear_caches()

        return RenderResult(
            rgb=rgb,
            temporal_gt=temporal_gt,
            spatial_masks=spatial_masks,
            bboxes_per_frame=bboxes_per_frame,
            frame_instructions=frame_instructions,
            timeline=timeline,
        )

    def clear_caches(self):
        self._bg_frame_cache.clear()
        self._sign_full_cache.clear()

    # Contract enforcement + safety fallback
    def _get_timeline_segments(self, timeline) -> List[Dict[str, Any]]:
        # v1 compatibility: timeline may expose `segments` or `sign_segments`
        segs = getattr(timeline, "segments", None)
        if segs is None:
            segs = getattr(timeline, "sign_segments", [])
        return list(segs or [])

    def _get_seg_time(self, seg: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        # tolerate different field names
        if "start_sec" in seg and "end_sec" in seg:
            return float(seg["start_sec"]), float(seg["end_sec"])
        if "start_time" in seg and "end_time" in seg:
            return float(seg["start_time"]), float(seg["end_time"])
        if "start" in seg and "end" in seg:
            return float(seg["start"]), float(seg["end"])
        return None, None

    def _validate_and_patch_instructions(self, insts: List[Dict[str, Any]], timeline):
        """
        Enforce TemporalComposer → Renderer v1 contract.

        Required per frame:
          - bg_asset
          - bg_frame_idx
          - timestamp (seconds)
          - active_signs: list of dicts, each at least has:
              - sign
              - asset_frame_idx
        """
        fps = float(self.fps) if self.fps > 0 else 25.0

        # 1) normalize minimal keys
        for i, inst in enumerate(insts):
            if "bg_asset" not in inst:
                inst["bg_asset"] = timeline.background

            if "bg_frame_idx" not in inst:
                inst["bg_frame_idx"] = int(i)  # fallback: sequential

            if "timestamp" not in inst or inst["timestamp"] is None:
                inst["timestamp"] = float(i) / fps

            if "active_signs" not in inst or inst["active_signs"] is None:
                inst["active_signs"] = []

            # normalize active sign items (if any)
            for s in inst["active_signs"]:
                if "category" not in s:
                    s["category"] = getattr(s.get("sign", None), "semantic_category", "general")
                if "text" not in s:
                    s["text"] = getattr(s.get("sign", None), "text", "")

        # 2) SAFETY CHECK (v1): timeline has segments but composer produced no active_signs anywhere
        segments = self._get_timeline_segments(timeline)
        has_segments = len(segments) > 0
        has_active_anywhere = any(len(inst.get("active_signs", [])) > 0 for inst in insts)

        if has_segments and not has_active_anywhere:
            print(
                "[WorldRenderer WARNING] Timeline has segments but TemporalComposer produced no active signs. "
                "Falling back to naive temporal activation from timeline segments."
            )

            for i, inst in enumerate(insts):
                t = inst.get("timestamp", None)
                if t is None:
                    t = float(i) / fps
                    inst["timestamp"] = t

                active: List[Dict[str, Any]] = []
                for seg in segments:
                    sign = seg.get("sign", None)
                    if sign is None:
                        continue

                    start_sec, end_sec = self._get_seg_time(seg)
                    if start_sec is None or end_sec is None:
                        continue

                    if start_sec <= t < end_sec:
                        # frame index inside sign clip
                        asset_frame_idx = int(round((t - start_sec) * fps))
                        active.append({
                            "sign": sign,
                            "asset_frame_idx": asset_frame_idx,
                            "category": getattr(sign, "semantic_category", "general"),
                            "text": getattr(sign, "text", ""),
                        })

                inst["active_signs"] = active

    # Rendering internals
    def _render_single_frame(
        self,
        instruction: Dict[str, Any],
    ) -> Tuple[np.ndarray, List[np.ndarray], List[Tuple[int, int, int, int]]]:

        bg_asset = instruction["bg_asset"]
        bg_frame_idx = int(instruction["bg_frame_idx"])

        bg_frame = self._load_background_frame(bg_asset, bg_frame_idx)

        sign_frames_info: List[Dict[str, Any]] = []
        for s in instruction.get("active_signs", []):
            sign_asset = s["sign"]
            frame_idx = int(s.get("asset_frame_idx", 0))

            sign_frame = self._load_sign_frame(sign_asset, frame_idx)

            sign_frames_info.append({
                "frame": sign_frame,
                "category": s.get("category", getattr(sign_asset, "semantic_category", "general")),
                "sign_id": getattr(sign_asset, "asset_id", None),
            })

        composite, masks, bboxes = self.spatial_composer.compose_frame(
            bg_frame=bg_frame,
            sign_frames_info=sign_frames_info,
            rng=self.rng,
        )
        return composite, masks, bboxes

    # Asset loading
    def _load_background_frame(self, bg_asset, frame_idx: int) -> np.ndarray:
        H, W = self.output_size[1], self.output_size[0]
        key = f"{getattr(bg_asset, 'asset_id', 'bg')}:{frame_idx}"

        if self.enable_cache and key in self._bg_frame_cache:
            return self._bg_frame_cache[key].copy()

        try:
            frames = bg_asset.load_frames(
                start_frame=frame_idx,
                end_frame=frame_idx + 1,
                target_size=self.output_size,
            )
            if frames is None or len(frames) == 0:
                raise RuntimeError("empty frames returned")
            frame = frames[0]
        except Exception as e:
            print(f"[WorldRenderer] BG load failed: {getattr(bg_asset, 'asset_id', 'bg')} frame={frame_idx} err={e}")
            frame = np.zeros((H, W, 3), dtype=np.uint8)

        frame = self._ensure_rgb(frame, W=W, H=H)

        if self.enable_cache:
            self._bg_frame_cache[key] = frame.copy()

        return frame

    def _load_sign_frame(self, sign_asset, frame_idx: int) -> np.ndarray:
        asset_id = getattr(sign_asset, "asset_id", "sign")

        if self.enable_cache and asset_id in self._sign_full_cache:
            frames = self._sign_full_cache[asset_id]
        else:
            try:
                frames = sign_asset.load_frames()
                if frames is None or len(frames) == 0:
                    raise RuntimeError("empty sign frames")
            except Exception as e:
                print(f"[WorldRenderer] SIGN load failed: {asset_id} err={e}")
                frames = np.zeros((1, 64, 64, 3), dtype=np.uint8)
                frames[..., 0] = 255  # red error tile

            if self.enable_cache:
                self._sign_full_cache[asset_id] = frames

        frame_idx = int(np.clip(frame_idx, 0, len(frames) - 1))
        frame = frames[frame_idx]
        return self._ensure_rgb(frame)

    # Utils
    def _ensure_rgb(self, frame: np.ndarray, W: Optional[int] = None, H: Optional[int] = None) -> np.ndarray:
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        if frame.ndim == 3 and frame.shape[-1] == 4:
            frame = frame[..., :3]
        if (W is not None) and (H is not None) and frame.shape[:2] != (H, W):
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    def _create_empty_result(self, timeline) -> RenderResult:
        H, W = self.output_size[1], self.output_size[0]
        empty = np.zeros((1, H, W, 3), dtype=np.uint8)
        return RenderResult(
            rgb=empty,
            temporal_gt=np.zeros((1,), dtype=np.float32),
            spatial_masks=[[]],
            bboxes_per_frame=[[]],
            frame_instructions=[],
            timeline=timeline,
        )
