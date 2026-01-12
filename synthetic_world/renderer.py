# synthetic_world/renderer.py
# 合成器，输入bg_frame, list of (sign_frame, mask)，输出final_frame

# synthetic_world/renderer.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2
import torch

from synthetic_world.assets import SignAsset, BackgroundAsset


# -----------------------------
# Utilities
# -----------------------------

def _to_torch_clip(x) -> torch.Tensor:
    """
    Accept np.ndarray or torch.Tensor clip.
    Return torch float32 (T,C,H,W) in [0,1].
    """
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.from_numpy(x)

    # (T,H,W,C) -> (T,C,H,W)
    if t.ndim == 4 and t.shape[-1] in (1, 3):
        t = t.permute(0, 3, 1, 2)

    if t.dtype != torch.float32:
        t = t.float()

    # If looks like uint8 [0,255], normalize
    if t.max() > 1.5:
        t = t / 255.0

    t = t.clamp(0.0, 1.0)
    return t


def _resample_to_fps(clip: torch.Tensor, fps_src: int, fps_tgt: int) -> torch.Tensor:
    """
    Resample by index mapping. clip: (T,C,H,W)
    """
    if fps_src == fps_tgt:
        return clip

    T = clip.shape[0]
    ratio = float(fps_tgt) / float(fps_src)
    new_T = max(1, int(round(T * ratio)))

    idx = torch.linspace(0, T - 1, new_T).long()
    return clip[idx]


def _ensure_same_hw(src: torch.Tensor, tgt_hw: Tuple[int, int]) -> torch.Tensor:
    """
    Resize clip to target H,W using bilinear.
    """
    _, C, H, W = src.shape
    tgt_h, tgt_w = tgt_hw
    if (H, W) == (tgt_h, tgt_w):
        return src
    src = torch.nn.functional.interpolate(src, size=(tgt_h, tgt_w), mode="bilinear", align_corners=False)
    return src


def save_video_mp4(
    clip: torch.Tensor,
    save_path: str,
    fps: int = 25
):
    """
    clip: (T,C,H,W) float in [0,1]
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    clip = clip.detach().cpu().clamp(0, 1)
    T, C, H, W = clip.shape
    assert C == 3, "save_video_mp4 expects RGB 3 channels"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(save_path, fourcc, fps, (W, H))

    for i in range(T):
        rgb = (clip[i].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)  # (H,W,3) RGB
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()
    print(f"[Saved video] {save_path}")


# -----------------------------
# Render result container
# -----------------------------

@dataclass
class RenderResult:
    rgb: torch.Tensor          # (T,C,H,W) float in [0,1]
    temporal_gt: torch.Tensor  # (T,) float32 in {0,1}
    segments: List[Dict]       # list of dict segments for audit/debug


# -----------------------------
# Renderer (MVP)
# -----------------------------

class WorldRenderer:
    """
    MVP renderer:
    - timeline is in seconds
    - render by blending sign frames onto background frames (full-frame blend)
    - output temporal_gt (frame-level)
    """

    def __init__(
        self,
        blend_alpha: float = 0.85,     # 1.0 = replace, 0.0 = keep bg
        force_fps: Optional[int] = None
    ):
        self.blend_alpha = float(blend_alpha)
        self.force_fps = force_fps

    def render(self, world: Dict) -> RenderResult:
        bg: BackgroundAsset = world["background"]
        timeline: List[Dict] = world.get("timeline", [])

        # --- background clip to torch ---
        bg_clip = _to_torch_clip(bg.frames)  # (T,C,H,W)
        fps_bg = int(self.force_fps or bg.fps)

        if fps_bg != bg.fps:
            bg_clip = _resample_to_fps(bg_clip, bg.fps, fps_bg)

        T_bg, C, H, W = bg_clip.shape

        out = bg_clip.clone()
        temporal_gt = torch.zeros((T_bg,), dtype=torch.float32)
        segments_out: List[Dict] = []

        # --- render each sign event ---
        for ev in timeline:
            sign: SignAsset = ev["sign"]
            start_sec = float(ev["start"])
            end_sec = float(ev["end"])

            # convert seconds -> frame indices in bg timeline
            start_f = int(round(start_sec * fps_bg))
            end_f = int(round(end_sec * fps_bg))
            start_f = max(0, min(T_bg, start_f))
            end_f = max(0, min(T_bg, end_f))

            if end_f <= start_f:
                continue

            # sign clip to torch, resample to bg fps, resize to bg hw
            sign_clip = _to_torch_clip(sign.frames)
            fps_sign = int(getattr(sign, "fps", fps_bg))

            sign_clip = _resample_to_fps(sign_clip, fps_sign, fps_bg)
            sign_clip = _ensure_same_hw(sign_clip, (H, W))

            # fit sign length into [start_f, end_f)
            L_slot = end_f - start_f
            if sign_clip.shape[0] >= L_slot:
                sign_use = sign_clip[:L_slot]
            else:
                # loop pad (simple + stable)
                reps = int(np.ceil(L_slot / sign_clip.shape[0]))
                sign_use = sign_clip.repeat((reps, 1, 1, 1))[:L_slot]

            # full-frame blend
            a = self.blend_alpha
            out[start_f:end_f] = (1 - a) * out[start_f:end_f] + a * sign_use

            temporal_gt[start_f:end_f] = 1.0

            segments_out.append({
                "sign_id": getattr(sign, "asset_id", "unknown_sign"),
                "text": getattr(sign, "text", ""),
                "start_sec": start_f / fps_bg,
                "end_sec": end_f / fps_bg,
                "start_frame": start_f,
                "end_frame": end_f,
            })

        return RenderResult(rgb=out.clamp(0, 1), temporal_gt=temporal_gt, segments=segments_out)


# -----------------------------
# Test
# -----------------------------

if __name__ == "__main__":
    import random
    from synthetic_world.loaders.csl_daily import load_csl_daily_as_assets
    from synthetic_world.loaders.ucf101 import load_ucf101_as_assets
    from synthetic_world.world_sampler import WorldSampler

    print("=== Renderer Test ===")

    # Load small pools
    signs = load_csl_daily_as_assets(
        root="/home/pxl416/PeixiLiu/px_proj/px_data/csl-daily-frames-512x512",
        rgb_dir="sentence",
        anno_pkl="sentence_label/csl2020ct_v2.pkl",
        split_file="sentence_label/split_1_train.txt",
        resize=(224, 224),
        max_samples=20,
    )

    bgs = load_ucf101_as_assets(
        root="/home/pxl416/PeixiLiu/px_proj/px_data/UCF-101",
        resize=(224, 224),
        max_frames=120,
        max_samples=10,
    )

    sampler = WorldSampler(signs, bgs, max_signs_per_bg=3, min_gap=0.3)
    world = sampler.sample_world()

    renderer = WorldRenderer(blend_alpha=0.85)
    result = renderer.render(world)

    print("\n[Render] rgb:", tuple(result.rgb.shape))
    print("[Render] temporal_gt:", tuple(result.temporal_gt.shape), "pos_frames:", int(result.temporal_gt.sum().item()))
    print("[Render] segments:", len(result.segments))
    for s in result.segments[:3]:
        print(" ", s)

    # Save audit video
    out_dir = "./synthetic_world_debug"
    os.makedirs(out_dir, exist_ok=True)
    save_video_mp4(result.rgb, os.path.join(out_dir, "render_demo.mp4"), fps=int(world["background"].fps))

    # Save temporal gt
    np.save(os.path.join(out_dir, "temporal_gt.npy"), result.temporal_gt.cpu().numpy())
    print(f"[Saved] {out_dir}/temporal_gt.npy")

    print("\nTest passed ✔")



