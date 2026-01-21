# synthetic_world/audit.py
"""
可视化调试工具：保存和检查渲染结果
(v1 frozen, interface-aligned)
"""

from __future__ import annotations

import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib

# matplotlib 设置（兼容中文）
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# ============================================================
# VideoAuditor
# ============================================================

class VideoAuditor:
    """
    视频审计器：保存和可视化合成结果

    v1 assumptions:
      - render_result.rgb: (T,H,W,3) uint8
      - render_result.temporal_gt: (T,)
      - render_result.bboxes_per_frame: List[List[bbox]]
      - render_result.timeline: Timeline object
      - render_result.labels (optional): Labels
    """

    def __init__(self, output_dir: str = "./audit_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Video with overlay
    # ------------------------------------------------------------

    def save_video_with_overlay(
        self,
        rgb_frames: np.ndarray,
        bboxes_per_frame: List[List[Tuple[int, int, int, int]]],
        masks_per_frame: Optional[List[List[np.ndarray]]] = None,
        temporal_gt: Optional[np.ndarray] = None,
        fps: int = 25,
        filename: str = "overlay.mp4",
        show_frame_idx: bool = True,
        show_active_signs: bool = True,
    ) -> str:
        if len(rgb_frames) == 0:
            print("[Auditor] No frames to save")
            return ""

        T, H, W, _ = rgb_frames.shape
        output_path = self.output_dir / filename

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))

        print(f"[Auditor] Saving overlay video → {output_path}")
        print(f"          {T} frames | {W}x{H} | {fps} fps")

        for t in range(T):
            frame = rgb_frames[t].copy()

            # ---- mask overlay ----
            if masks_per_frame and t < len(masks_per_frame):
                for mask in masks_per_frame[t]:
                    if mask is not None and mask.sum() > 0:
                        overlay = np.zeros_like(frame)
                        overlay[:, :, 1] = mask  # green
                        frame = cv2.addWeighted(frame, 1.0, overlay, 0.3, 0)

            # ---- bbox ----
            if t < len(bboxes_per_frame):
                for x1, y1, x2, y2 in bboxes_per_frame[t]:
                    if x2 > x1 and y2 > y1:
                        cv2.rectangle(frame, (x1, y1), (x2, y2),
                                      (0, 0, 255), 2)

            # ---- text ----
            if show_frame_idx:
                cv2.putText(frame, f"Frame {t:04d}", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"{t / fps:.2f}s", (10, 56),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if show_active_signs:
                n = len(bboxes_per_frame[t]) if t < len(bboxes_per_frame) else 0
                cv2.putText(frame, f"Active: {n}",
                            (W - 180, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0) if n > 0 else (200, 200, 200), 2)

            if temporal_gt is not None and t < len(temporal_gt):
                has_sign = temporal_gt[t] > 0.5
                color = (0, 255, 0) if has_sign else (120, 120, 120)
                cv2.rectangle(frame, (0, 0), (W, 6), color, -1)

            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        out.release()
        print(f"[Auditor] Saved: {output_path}")
        return str(output_path)

    # ------------------------------------------------------------
    # Analysis plots
    # ------------------------------------------------------------

    def save_temporal_analysis(
        self,
        temporal_gt: np.ndarray,
        segment_spans: np.ndarray,
        filename: str = "temporal.png",
    ) -> str:
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))

        T = len(temporal_gt)
        x = np.arange(T)

        ax[0].step(x, temporal_gt, where="post")
        ax[0].set_ylim(-0.1, 1.1)
        ax[0].set_title("Temporal GT")
        ax[0].grid(True, alpha=0.3)

        for s, e in segment_spans:
            ax[0].axvspan(s, e, alpha=0.2, color="green")

        if len(segment_spans) > 0:
            dur = segment_spans[:, 1] - segment_spans[:, 0]
            ax[1].bar(np.arange(len(dur)), dur)
            ax[1].set_title(f"Segment Durations (avg={dur.mean():.1f})")
        else:
            ax[1].text(0.5, 0.5, "No segments",
                       ha="center", va="center")

        plt.tight_layout()
        out = self.output_dir / filename
        plt.savefig(out, dpi=150)
        plt.close()
        return str(out)

    def save_spatial_analysis(
        self,
        bboxes_per_frame: List[List[Tuple[int, int, int, int]]],
        image_size: Tuple[int, int],
        filename: str = "spatial.png",
    ) -> str:
        W, H = image_size
        all_bboxes = [b for f in bboxes_per_frame for b in f]
        if not all_bboxes:
            return ""

        heat = np.zeros((H, W))
        for x1, y1, x2, y2 in all_bboxes:
            heat[y1:y2, x1:x2] += 1

        plt.figure(figsize=(6, 6))
        plt.imshow(heat, cmap="hot")
        plt.title("BBox Heatmap")
        plt.colorbar()
        out = self.output_dir / filename
        plt.savefig(out, dpi=150)
        plt.close()
        return str(out)

    # ------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------

    def save_metadata_report(
        self,
        render_result: Any,
        labels: Optional[Any] = None,
        filename: str = "report.json",
    ) -> str:
        report = {
            "frames": int(render_result.rgb.shape[0]),
            "resolution": list(render_result.rgb.shape[1:3]),
            "frames_with_sign": int(render_result.temporal_gt.sum()),
        }

        if labels is not None:
            report["labels"] = {
                "num_segments": len(labels.segment_spans),
                "vocab_size": len(labels.vocabulary),
            }

        if hasattr(render_result, "timeline"):
            tl = render_result.timeline
            report["timeline"] = {
                "background": getattr(tl.background, "asset_id", "unknown"),
                "num_segments": len(getattr(tl, "segments", [])),
            }

        out = self.output_dir / filename
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return str(out)

    # ------------------------------------------------------------
    # Full audit
    # ------------------------------------------------------------

    def audit_render_result(
        self,
        render_result: Any,
        base_name: str = "sample",
        fps: int = 25,
    ) -> Dict[str, str]:

        outputs = {}

        outputs["overlay"] = self.save_video_with_overlay(
            render_result.rgb,
            render_result.bboxes_per_frame,
            getattr(render_result, "spatial_masks", None),
            render_result.temporal_gt,
            fps=fps,
            filename=f"{base_name}_overlay.mp4",
        )

        labels = getattr(render_result, "labels", None)
        spans = labels.segment_spans if labels else np.zeros((0, 2))

        outputs["temporal"] = self.save_temporal_analysis(
            render_result.temporal_gt,
            spans,
            filename=f"{base_name}_temporal.png",
        )

        outputs["spatial"] = self.save_spatial_analysis(
            render_result.bboxes_per_frame,
            (render_result.rgb.shape[2], render_result.rgb.shape[1]),
            filename=f"{base_name}_spatial.png",
        )

        outputs["report"] = self.save_metadata_report(
            render_result,
            labels,
            filename=f"{base_name}_report.json",
        )

        return outputs


# ============================================================
# Deprecated helper
# ============================================================

def audit_sample(*args, **kwargs):
    """
    DEPRECATED (v1):

    WorldSampler no longer produces Timeline.
    Please explicitly use:

      plan = sampler.sample_world()
      timeline = planner.plan(plan.background, plan.signs)
      render_result = renderer.render(timeline)
      auditor.audit_render_result(render_result)

    This helper is intentionally disabled to avoid misuse.
    """
    raise RuntimeError(
        "audit_sample() is deprecated in v1. "
        "Please construct Timeline explicitly via TimelinePlanner."
    )
