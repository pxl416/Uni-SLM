from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import cv2

from temporal_composer import TemporalComposer
from spatial_composer import SpatialComposer


@dataclass
class RenderResult:
    """渲染结果容器"""
    rgb: np.ndarray  # (T, H, W, 3) uint8
    temporal_gt: np.ndarray  # (T,) float32 (简单版：有sign=1，否则0)
    spatial_masks: List[List[np.ndarray]]  # 每帧一个list，每个sign一个(H,W) uint8 mask
    bboxes_per_frame: List[List[Tuple[int, int, int, int]]]  # 每帧一个list，每个sign一个bbox
    frame_instructions: List[Dict[str, Any]]  # TemporalComposer输出
    timeline: Any  # 原始timeline


class WorldRenderer:
    """
    合成世界渲染器：整合 temporal + spatial composer
    - temporal: 负责“这一帧有哪些sign、各自取第几帧”
    - spatial: 负责“把sign帧放到背景哪里、如何混合、输出mask/bbox”
    """

    def __init__(
        self,
        output_size: Tuple[int, int] = (224, 224),  # (W,H)
        fps: int = 25,
        spatial_config: Optional[Dict[str, Any]] = None,
        temporal_config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        enable_cache: bool = True,
    ):
        self.output_size = output_size
        self.fps = fps
        self.seed = seed
        self.enable_cache = enable_cache

        self.rng = np.random.default_rng(seed)

        temporal_config = temporal_config or {}
        # 你当前TemporalComposer只吃fps，所以这里不要乱传kwargs
        self.temporal_composer = TemporalComposer(fps=fps)

        spatial_config = spatial_config or {}
        self.spatial_composer = SpatialComposer(
            output_size=output_size,
            **spatial_config
        )

        # caches
        # 背景：按 (asset_id, frame_idx) 缓存单帧（可选）
        self._bg_frame_cache: Dict[str, np.ndarray] = {}
        # 手语：按 asset_id 缓存整段 frames（强烈建议）
        self._sign_full_cache: Dict[str, np.ndarray] = {}

    # ---------------- Public API ----------------

    def render(self, timeline, clear_cache: bool = True) -> RenderResult:
        """
        渲染完整timeline，输出视频与基础标注（temporal_gt + masks + bboxes）
        """
        frame_instructions = self.temporal_composer.compose(timeline)
        total_frames = len(frame_instructions)

        if total_frames == 0:
            return self._create_empty_result(timeline)

        H, W = self.output_size[1], self.output_size[0]
        rgb = np.zeros((total_frames, H, W, 3), dtype=np.uint8)
        temporal_gt = np.zeros((total_frames,), dtype=np.float32)
        spatial_masks: List[List[np.ndarray]] = []
        bboxes_per_frame: List[List[Tuple[int, int, int, int]]] = []

        for t, inst in enumerate(frame_instructions):
            comp, masks, bboxes = self._render_single_frame(inst)
            rgb[t] = comp
            temporal_gt[t] = 1.0 if len(inst.get("active_signs", [])) > 0 else 0.0
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
            timeline=timeline
        )

    def render_debug(self, timeline, max_frames: int = 10) -> Dict[str, Any]:
        """
        调试：只渲染前max_frames帧，返回更详细的中间信息
        """
        frame_instructions = self.temporal_composer.compose(timeline)
        total_frames = min(len(frame_instructions), max_frames)

        rendered_frames = []
        frame_details = []

        for t in range(total_frames):
            inst = frame_instructions[t]
            comp, masks, bboxes = self._render_single_frame(inst)

            rendered_frames.append(comp)
            frame_details.append({
                "t": t,
                "timestamp": inst.get("timestamp"),
                "bg_frame_idx": inst.get("bg_frame_idx"),
                "num_active_signs": len(inst.get("active_signs", [])),
                "sign_ids": [s["sign"].asset_id for s in inst.get("active_signs", [])],
                "bboxes": bboxes,
                "mask_nonzero": [int((m > 0).sum()) for m in masks],
            })

        return {
            "timeline": {
                "background": getattr(timeline.background, "asset_id", "unknown"),
                "num_segments": len(getattr(timeline, "segments", [])),
                "duration": getattr(timeline.background, "duration", None),
            },
            "frames": rendered_frames,
            "details": frame_details,
        }

    def clear_caches(self):
        self._bg_frame_cache.clear()
        self._sign_full_cache.clear()

    # ---------------- Internal ----------------

    def _render_single_frame(self, instruction: Dict[str, Any]) -> Tuple[np.ndarray, List[np.ndarray], List[Tuple[int, int, int, int]]]:
        """
        instruction 来自 TemporalComposer.compose
        """
        # 背景帧
        bg_asset = instruction.get("background_asset", None)  # 兼容字段（如果你未来改结构）
        bg_frame_idx = instruction.get("bg_frame_idx", 0)
        bg_asset = instruction.get("bg_asset", bg_asset)
        if bg_asset is None:
            raise ValueError("TemporalComposer.compose must include bg_asset in each instruction. "
                             "Please add: {'bg_asset': timeline.background} in compose().")

        bg_frame = self._load_background_frame(bg_asset, bg_frame_idx)

        # sign frames info
        sign_frames_info: List[Dict[str, Any]] = []
        for s in instruction.get("active_signs", []):
            sign_asset = s["sign"]
            sign_frame_idx = int(s["asset_frame_idx"])

            sign_frame = self._load_sign_frame(sign_asset, sign_frame_idx)

            sign_frames_info.append({
                "frame": sign_frame,  # (h,w,3) or (h,w,4) uint8
                "sign_id": sign_asset.asset_id,
                "category": s.get("category", getattr(sign_asset, "semantic_category", "general")),
                "text": s.get("text", getattr(sign_asset, "text", "")),
            })

        composite, masks, bboxes = self.spatial_composer.compose_frame(
            bg_frame=bg_frame,
            sign_frames_info=sign_frames_info,
            rng=self.rng
        )
        return composite, masks, bboxes

    def _load_background_frame(self, bg_asset, frame_idx: int) -> np.ndarray:
        """
        背景按需slice加载：BackgroundAsset.load_frames(start_frame, end_frame, target_size)
        """
        H, W = self.output_size[1], self.output_size[0]
        cache_key = f"{bg_asset.asset_id}::{frame_idx}"

        if self.enable_cache and cache_key in self._bg_frame_cache:
            return self._bg_frame_cache[cache_key].copy()

        try:
            frames = bg_asset.load_frames(
                start_frame=frame_idx,
                end_frame=frame_idx + 1,
                target_size=self.output_size
            )
            if frames is None or len(frames) == 0:
                raise RuntimeError("empty frames returned")
            frame = frames[0]
        except Exception as e:
            print(f"[WorldRenderer] bg load failed: {bg_asset.asset_id} frame={frame_idx} err={e}")
            frame = np.zeros((H, W, 3), dtype=np.uint8)

        # 保证尺寸与类型
        if frame.shape[:2] != (H, W):
            frame = cv2.resize(frame, (W, H))
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        if self.enable_cache:
            self._bg_frame_cache[cache_key] = frame.copy()
        return frame

    def _load_sign_frame(self, sign_asset, frame_idx: int) -> np.ndarray:
        """
        SignAsset 目前不支持 slice 加载，所以采取：
        - 首次加载：sign_asset.load_frames()，缓存整段
        - 后续直接取frame_idx
        """
        asset_id = sign_asset.asset_id

        if self.enable_cache and asset_id in self._sign_full_cache:
            frames = self._sign_full_cache[asset_id]
        else:
            try:
                frames = sign_asset.load_frames()  # (T,H,W,3) uint8
                if frames is None or len(frames) == 0:
                    raise RuntimeError("empty sign frames")
            except Exception as e:
                print(f"[WorldRenderer] sign load failed: {asset_id} err={e}")
                # 红块作为错误指示
                frames = np.zeros((1, 64, 64, 3), dtype=np.uint8)
                frames[..., 0] = 255

            if self.enable_cache:
                self._sign_full_cache[asset_id] = frames

        # clamp idx
        frame_idx = int(np.clip(frame_idx, 0, len(frames) - 1))
        frame = frames[frame_idx]

        # ensure RGB
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        if frame.shape[-1] not in (3, 4):
            # fallback: force to 3
            frame = frame[..., :3]

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
            timeline=timeline
        )


# ----------------- Test -----------------
if __name__ == "__main__":
    print("=== Testing WorldRenderer ===")

    from dataclasses import dataclass
    from typing import List

    @dataclass
    class MockBackground:
        asset_id: str
        num_frames: int
        duration: float

        def load_frames(self, start_frame=0, end_frame=None, target_size=None):
            if end_frame is None:
                end_frame = start_frame + 1
            W, H = target_size if target_size is not None else (160, 120)
            frames = []
            for i in range(start_frame, min(end_frame, self.num_frames)):
                img = np.zeros((H, W, 3), dtype=np.uint8)
                img[:] = (20, 80, 20)  # green-ish
                cv2.putText(img, f"BG {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                frames.append(img)
            return np.stack(frames) if frames else np.zeros((0, H, W, 3), dtype=np.uint8)

    @dataclass
    class MockSign:
        asset_id: str
        text: str
        num_frames: int
        duration: float
        semantic_category: str = "general"
        gloss: List[str] = None

        def load_frames(self, max_frames=None, target_size=None):
            # 返回整段 (T,H,W,3)，模拟SignAsset folder模式
            T = self.num_frames if max_frames is None else min(self.num_frames, max_frames)
            W, H = target_size if target_size is not None else (80, 80)
            frames = []
            for i in range(T):
                img = np.zeros((H, W, 4), dtype=np.uint8)
                # 一块带alpha的彩色块
                img[..., 3] = 0
                cv2.rectangle(img, (10, 10), (W - 10, H - 10), (255, 0, 0, 255), -1)
                cv2.putText(img, f"S{i}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255, 255), 2)
                frames.append(img)
            return np.stack(frames, axis=0)

    @dataclass
    class MockTimeline:
        background: MockBackground
        segments: List[dict]

    # 1) timeline
    bg = MockBackground("bg_office", num_frames=60, duration=6.0)
    s1 = MockSign("sign_hello", "hello", num_frames=30, duration=3.0, semantic_category="greeting", gloss=["HELLO"])
    s2 = MockSign("sign_thanks", "thanks", num_frames=20, duration=2.0, semantic_category="general", gloss=["THANKS"])

    timeline = MockTimeline(
        background=bg,
        segments=[
            {"sign": s1, "start_sec": 0.5, "end_sec": 2.5},
            {"sign": s2, "start_sec": 2.0, "end_sec": 4.0},
        ]
    )

    # 2) IMPORTANT: 你当前 TemporalComposer.compose() 需要加 bg_asset 到 instruction
    #    所以这里我们临时 monkey-patch 一下，模拟你会在 TemporalComposer 里加：
    #    inst["bg_asset"] = timeline.background
    class TemporalComposerPatched(TemporalComposer):
        def compose(self, tl):
            insts = super().compose(tl)
            for it in insts:
                it["bg_asset"] = tl.background
            return insts

    renderer = WorldRenderer(
        output_size=(320, 240),
        fps=10,
        seed=42,
        spatial_config={"position_mode": "random", "blend_mode": "alpha"},
    )
    renderer.temporal_composer = TemporalComposerPatched(fps=10)

    result = renderer.render(timeline, clear_cache=True)

    print("RGB:", result.rgb.shape, result.rgb.dtype)
    print("temporal_gt:", result.temporal_gt.shape, "pos_frames=", float(result.temporal_gt.sum()))
    print("frames masks:", len(result.spatial_masks), "frames bboxes:", len(result.bboxes_per_frame))

    # sanity check: bbox count should match active sign count for those frames
    checked = 0
    for t, inst in enumerate(result.frame_instructions[:20]):
        n = len(inst["active_signs"])
        if n > 0:
            assert len(result.bboxes_per_frame[t]) == n
            assert len(result.spatial_masks[t]) == n
            checked += 1
    print("checked active frames:", checked)

    print("Test passed ✔")
