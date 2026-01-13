# synthetic_world/audit.py
"""
可视化调试工具：保存和检查渲染结果
"""

from __future__ import annotations

import os
import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib

# 设置matplotlib中文字体
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


class VideoAuditor:
    """
    视频审计器：保存和可视化合成结果
    """

    def __init__(self, output_dir: str = "./audit_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_video_with_overlay(
            self,
            rgb_frames: np.ndarray,  # (T, H, W, 3) uint8
            bboxes_per_frame: List[List[Tuple[int, int, int, int]]],
            masks_per_frame: Optional[List[List[np.ndarray]]] = None,
            temporal_gt: Optional[np.ndarray] = None,
            fps: int = 25,
            filename: str = "composite_overlay.mp4",
            show_frame_idx: bool = True,
            show_active_signs: bool = True,
    ) -> str:
        """
        保存带覆盖层的视频（bbox + mask + 时间标注）
        """
        if len(rgb_frames) == 0:
            print("[Auditor] No frames to save")
            return ""

        T, H, W, _ = rgb_frames.shape
        output_path = self.output_dir / filename

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))

        print(f"[Auditor] Saving overlay video to {output_path}")
        print(f"         Frames: {T}, Resolution: {W}x{H}, FPS: {fps}")

        for t in range(T):
            # 复制原始帧
            frame = rgb_frames[t].copy()

            # 如果有mask，半透明覆盖
            if masks_per_frame and t < len(masks_per_frame):
                for mask in masks_per_frame[t]:
                    if mask is not None and mask.sum() > 0:
                        # 创建彩色覆盖层
                        overlay = np.zeros_like(frame)
                        overlay[:, :, 1] = mask  # 绿色覆盖

                        # 半透明混合
                        alpha = 0.3
                        frame = cv2.addWeighted(frame, 1.0, overlay, alpha, 0)

            # 绘制bbox
            if t < len(bboxes_per_frame):
                for bbox in bboxes_per_frame[t]:
                    if bbox[2] > bbox[0] and bbox[3] > bbox[1]:  # 有效bbox
                        x1, y1, x2, y2 = bbox
                        # 画矩形
                        cv2.rectangle(frame, (x1, y1), (x2, y2),
                                      (0, 0, 255), 2)  # 红色边框

                        # 画角点
                        corner_size = 4
                        cv2.rectangle(frame, (x1, y1),
                                      (x1 + corner_size, y1 + corner_size),
                                      (0, 255, 255), -1)  # 黄色角点
                        cv2.rectangle(frame, (x2 - corner_size, y1),
                                      (x2, y1 + corner_size),
                                      (0, 255, 255), -1)
                        cv2.rectangle(frame, (x1, y2 - corner_size),
                                      (x1 + corner_size, y2),
                                      (0, 255, 255), -1)
                        cv2.rectangle(frame, (x2 - corner_size, y2 - corner_size),
                                      (x2, y2), (0, 255, 255), -1)

            # 添加帧信息
            if show_frame_idx:
                cv2.putText(frame, f"Frame: {t:04d}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Time: {t / fps:.2f}s", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 添加活动sign数量
            if show_active_signs:
                num_signs = len(bboxes_per_frame[t]) if t < len(bboxes_per_frame) else 0
                sign_text = f"Active Signs: {num_signs}"
                color = (0, 255, 0) if num_signs > 0 else (255, 255, 255)
                cv2.putText(frame, sign_text, (W - 200, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 添加时间标注指示器
            if temporal_gt is not None and t < len(temporal_gt):
                has_sign = temporal_gt[t] > 0.5
                indicator_color = (0, 255, 0) if has_sign else (255, 0, 0)
                indicator_text = "SIGN" if has_sign else "NO SIGN"
                cv2.putText(frame, indicator_text, (W - 150, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, indicator_color, 2)

                # 顶部状态条
                bar_height = 5
                bar_color = indicator_color if has_sign else (100, 100, 100)
                cv2.rectangle(frame, (0, 0), (W, bar_height), bar_color, -1)

            # RGB -> BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

            # 进度显示
            if t % (max(1, T // 10)) == 0:
                print(f"  Writing frame {t}/{T} ({t / T * 100:.0f}%)")

        out.release()
        print(f"[Auditor] Video saved: {output_path}")
        return str(output_path)

    def save_separate_views(
            self,
            rgb_frames: np.ndarray,
            bboxes_per_frame: List[List[Tuple[int, int, int, int]]],
            masks_per_frame: Optional[List[List[np.ndarray]]] = None,
            fps: int = 25,
            base_name: str = "debug",
    ) -> Dict[str, str]:
        """
        保存多个视角的视频用于调试
        """
        T, H, W, _ = rgb_frames.shape

        outputs = {}

        # 1. 原始视频
        orig_path = self.output_dir / f"{base_name}_original.mp4"
        self._save_simple_video(rgb_frames, orig_path, fps)
        outputs['original'] = str(orig_path)

        # 2. 带bbox的视频
        bbox_path = self.output_dir / f"{base_name}_bbox_only.mp4"
        self._save_bbox_video(rgb_frames, bboxes_per_frame, bbox_path, fps)
        outputs['bbox'] = str(bbox_path)

        # 3. 带mask的视频
        if masks_per_frame:
            mask_path = self.output_dir / f"{base_name}_mask_only.mp4"
            self._save_mask_video(rgb_frames, masks_per_frame, mask_path, fps)
            outputs['mask'] = str(mask_path)

        # 4. 网格视图（如果帧数不多）
        if T <= 100:
            grid_path = self.output_dir / f"{base_name}_grid.png"
            self._save_frame_grid(rgb_frames, bboxes_per_frame, grid_path)
            outputs['grid'] = str(grid_path)

        return outputs

    def _save_simple_video(self, frames: np.ndarray, path: Path, fps: int):
        """保存简单视频"""
        if len(frames) == 0:
            return

        H, W = frames.shape[1:3]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(path), fourcc, fps, (W, H))

        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()
        print(f"  Saved simple video: {path.name}")

    def _save_bbox_video(self, frames, bboxes_per_frame, path, fps):
        """保存仅带bbox的视频"""
        if len(frames) == 0:
            return

        H, W = frames.shape[1:3]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(path), fourcc, fps, (W, H))

        for t, frame in enumerate(frames):
            frame_copy = frame.copy()
            if t < len(bboxes_per_frame):
                for bbox in bboxes_per_frame[t]:
                    if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(frame_copy, (x1, y1), (x2, y2),
                                      (0, 0, 255), 3)

            frame_bgr = cv2.cvtColor(frame_copy, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()
        print(f"  Saved bbox video: {path.name}")

    def _save_mask_video(self, frames, masks_per_frame, path, fps):
        """保存仅带mask的视频"""
        if len(frames) == 0:
            return

        H, W = frames.shape[1:3]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(path), fourcc, fps, (W, H))

        for t, frame in enumerate(frames):
            frame_copy = frame.copy()
            if t < len(masks_per_frame):
                for mask in masks_per_frame[t]:
                    if mask is not None and mask.sum() > 0:
                        # 红色mask覆盖
                        mask_bool = mask > 0
                        frame_copy[mask_bool, 0] = 255  # 红色通道
                        frame_copy[mask_bool, 1] = 0
                        frame_copy[mask_bool, 2] = 0

            frame_bgr = cv2.cvtColor(frame_copy, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()
        print(f"  Saved mask video: {path.name}")

    def _save_frame_grid(self, frames, bboxes_per_frame, path, cols: int = 10):
        """保存帧网格图像"""
        T = len(frames)
        if T == 0:
            return

        rows = (T + cols - 1) // cols
        H, W = frames.shape[1:3]

        # 创建大画布
        grid_h = rows * H
        grid_w = cols * W
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        for idx in range(min(T, rows * cols)):
            r = idx // cols
            c = idx % cols

            frame = frames[idx].copy()

            # 在网格帧上画bbox
            if idx < len(bboxes_per_frame):
                for bbox in bboxes_per_frame[idx]:
                    if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # 添加帧编号
            cv2.putText(frame, f"{idx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 放置到网格
            grid[r * H:(r + 1) * H, c * W:(c + 1) * W] = frame

        cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"  Saved frame grid: {path.name} ({rows}x{cols}, {T} frames)")

    def save_temporal_analysis(
            self,
            temporal_gt: np.ndarray,
            segment_spans: np.ndarray,
            labels: Optional[Any] = None,
            filename: str = "temporal_analysis.png",
    ) -> str:
        """
        保存时间标注分析图
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        T = len(temporal_gt)
        time_axis = np.arange(T)

        # 1. 时间标注信号
        ax = axes[0]
        ax.step(time_axis, temporal_gt, where='post', linewidth=2)
        ax.set_xlim(0, T)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Has Sign (0/1)')
        ax.set_title('Temporal Ground Truth Signal')
        ax.grid(True, alpha=0.3)

        # 高亮segment区域
        for span in segment_spans:
            start, end = span
            if end > start:
                ax.axvspan(start, end, alpha=0.2, color='green')

        # 2. Segment持续时间分布
        ax = axes[1]
        if len(segment_spans) > 0:
            durations = segment_spans[:, 1] - segment_spans[:, 0]
            ax.bar(range(len(durations)), durations, alpha=0.7)
            ax.set_xlabel('Segment Index')
            ax.set_ylabel('Duration (frames)')
            ax.set_title(f'Segment Durations (mean={durations.mean():.1f} frames)')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No segments', ha='center', va='center')
            ax.set_title('Segment Durations')

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[Auditor] Temporal analysis saved: {output_path}")
        return str(output_path)

    def save_spatial_analysis(
            self,
            bboxes_per_frame: List[List[Tuple[int, int, int, int]]],
            image_size: Tuple[int, int],
            filename: str = "spatial_analysis.png",
    ) -> str:
        """
        保存空间标注分析图
        """
        W, H = image_size

        # 收集所有bbox
        all_bboxes = []
        for bboxes in bboxes_per_frame:
            all_bboxes.extend(bboxes)

        if not all_bboxes:
            print("[Auditor] No bboxes for spatial analysis")
            return ""

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. BBox位置热力图
        ax = axes[0, 0]
        heatmap = np.zeros((H, W))

        for bbox in all_bboxes:
            x1, y1, x2, y2 = bbox
            if x2 > x1 and y2 > y1:
                # 将bbox区域加1
                heatmap[y1:y2, x1:x2] += 1

        im = ax.imshow(heatmap, cmap='hot', aspect='auto')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_title('BBox Location Heatmap')
        plt.colorbar(im, ax=ax, label='Frequency')

        # 2. BBox大小分布
        ax = axes[0, 1]
        widths = []
        heights = []
        areas = []

        for bbox in all_bboxes:
            x1, y1, x2, y2 = bbox
            if x2 > x1 and y2 > y1:
                w = x2 - x1
                h = y2 - y1
                widths.append(w)
                heights.append(h)
                areas.append(w * h)

        if widths:
            ax.hist(widths, bins=20, alpha=0.7, label=f'Width (mean={np.mean(widths):.1f})')
            ax.hist(heights, bins=20, alpha=0.7, label=f'Height (mean={np.mean(heights):.1f})')
            ax.set_xlabel('Size (pixels)')
            ax.set_ylabel('Count')
            ax.set_title('BBox Size Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No valid bboxes', ha='center', va='center')

        # 3. BBox中心点分布
        ax = axes[1, 0]
        centers_x = []
        centers_y = []

        for bbox in all_bboxes:
            x1, y1, x2, y2 = bbox
            if x2 > x1 and y2 > y1:
                centers_x.append((x1 + x2) / 2)
                centers_y.append((y1 + y2) / 2)

        if centers_x:
            ax.scatter(centers_x, centers_y, alpha=0.5, s=10)
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)  # 反转Y轴以匹配图像坐标
            ax.set_xlabel('Center X')
            ax.set_ylabel('Center Y')
            ax.set_title(f'BBox Center Distribution (n={len(centers_x)})')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No valid bboxes', ha='center', va='center')

        # 4. 每帧BBox数量
        ax = axes[1, 1]
        bbox_counts = [len(bboxes) for bboxes in bboxes_per_frame]
        ax.plot(bbox_counts, linewidth=2)
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Number of BBoxes')
        ax.set_title(f'BBoxes per Frame (max={max(bbox_counts)})')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(bbox_counts) + 1)

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[Auditor] Spatial analysis saved: {output_path}")
        return str(output_path)

    def save_metadata_report(
            self,
            render_result: Any,
            labels: Optional[Any] = None,
            filename: str = "metadata_report.json",
    ) -> str:
        """
        保存元数据报告
        """
        report = {
            'timestamp': np.datetime64('now').astype(str),
        }

        # 从render_result提取信息
        if hasattr(render_result, 'rgb'):
            rgb = render_result.rgb
            report['video'] = {
                'num_frames': rgb.shape[0],
                'height': rgb.shape[1],
                'width': rgb.shape[2],
                'channels': rgb.shape[3],
                'dtype': str(rgb.dtype),
            }

        if hasattr(render_result, 'temporal_gt'):
            temporal_gt = render_result.temporal_gt
            report['temporal'] = {
                'total_frames': len(temporal_gt),
                'frames_with_sign': int(temporal_gt.sum()),
                'percentage_with_sign': float(temporal_gt.mean() * 100),
            }

        if hasattr(render_result, 'bboxes_per_frame'):
            bboxes = render_result.bboxes_per_frame
            total_bboxes = sum(len(b) for b in bboxes)
            report['spatial'] = {
                'total_bboxes': total_bboxes,
                'frames_with_bboxes': sum(1 for b in bboxes if len(b) > 0),
                'max_bboxes_per_frame': max(len(b) for b in bboxes),
            }

        # 从labels提取信息
        if labels is not None:
            report['labels'] = {
                'num_segments': len(labels.segment_spans),
                'vocabulary_size': len(labels.vocabulary),
                'vocabulary': labels.vocabulary,
            }

        # 从timeline提取信息
        if hasattr(render_result, 'timeline'):
            timeline = render_result.timeline
            report['timeline'] = {
                'background': getattr(timeline.background, 'asset_id', 'unknown'),
                'num_signs': len(getattr(timeline, 'segments', [])),
            }

        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"[Auditor] Metadata report saved: {output_path}")
        return str(output_path)

    def audit_render_result(
            self,
            render_result: Any,
            base_name: str = "sample",
            fps: int = 25,
            save_all: bool = True,
    ) -> Dict[str, str]:
        """
        完整审计渲染结果：保存所有可视化和分析
        """
        print(f"\n{'=' * 60}")
        print(f"Auditing render result: {base_name}")
        print(f"{'=' * 60}")

        outputs = {}

        # 1. 保存带覆盖层的视频
        overlay_video = self.save_video_with_overlay(
            rgb_frames=render_result.rgb,
            bboxes_per_frame=render_result.bboxes_per_frame,
            masks_per_frame=getattr(render_result, 'spatial_masks', None),
            temporal_gt=render_result.temporal_gt,
            fps=fps,
            filename=f"{base_name}_overlay.mp4",
        )
        outputs['overlay_video'] = overlay_video

        # 2. 保存多个视角
        if save_all:
            separate_outputs = self.save_separate_views(
                rgb_frames=render_result.rgb,
                bboxes_per_frame=render_result.bboxes_per_frame,
                masks_per_frame=getattr(render_result, 'spatial_masks', None),
                fps=fps,
                base_name=base_name,
            )
            outputs.update(separate_outputs)

        # 3. 时间分析
        labels = getattr(render_result, 'labels', None)
        segment_spans = getattr(labels, 'segment_spans', np.zeros((0, 2))) if labels else np.zeros((0, 2))

        temporal_plot = self.save_temporal_analysis(
            temporal_gt=render_result.temporal_gt,
            segment_spans=segment_spans,
            labels=labels,
            filename=f"{base_name}_temporal.png",
        )
        outputs['temporal_analysis'] = temporal_plot

        # 4. 空间分析
        if len(render_result.rgb) > 0:
            image_size = (render_result.rgb.shape[2], render_result.rgb.shape[1])
            spatial_plot = self.save_spatial_analysis(
                bboxes_per_frame=render_result.bboxes_per_frame,
                image_size=image_size,
                filename=f"{base_name}_spatial.png",
            )
            outputs['spatial_analysis'] = spatial_plot

        # 5. 元数据报告
        report = self.save_metadata_report(
            render_result=render_result,
            labels=labels,
            filename=f"{base_name}_report.json",
        )
        outputs['metadata_report'] = report

        # 6. 生成摘要
        self._print_summary(render_result, labels)

        print(f"\nAudit complete. Outputs saved to: {self.output_dir}")
        for key, path in outputs.items():
            print(f"  {key}: {Path(path).name}")

        return outputs

    def _print_summary(self, render_result, labels):
        """打印审计摘要"""
        print(f"\n{'=' * 40}")
        print(f"AUDIT SUMMARY")
        print(f"{'=' * 40}")

        # 视频信息
        rgb = render_result.rgb
        print(f"Video: {rgb.shape[0]} frames, {rgb.shape[1]}x{rgb.shape[2]}")

        # 时间信息
        temporal_gt = render_result.temporal_gt
        sign_frames = int(temporal_gt.sum())
        sign_percentage = temporal_gt.mean() * 100
        print(f"Temporal: {sign_frames}/{len(temporal_gt)} frames with sign ({sign_percentage:.1f}%)")

        # 空间信息
        bboxes = render_result.bboxes_per_frame
        total_bboxes = sum(len(b) for b in bboxes)
        frames_with_bbox = sum(1 for b in bboxes if len(b) > 0)
        print(f"Spatial: {total_bboxes} bboxes in {frames_with_bbox}/{len(bboxes)} frames")

        # Segment信息
        if labels and hasattr(labels, 'segment_spans'):
            segments = labels.segment_spans
            if len(segments) > 0:
                durations = segments[:, 1] - segments[:, 0]
                print(f"Segments: {len(segments)}, "
                      f"Duration: {durations.min()}-{durations.max()} frames "
                      f"(avg {durations.mean():.1f})")

        # 背景信息
        if hasattr(render_result, 'timeline'):
            timeline = render_result.timeline
            bg = timeline.background
            print(f"Background: {getattr(bg, 'asset_id', 'unknown')}, "
                  f"Duration: {getattr(bg, 'duration', 0):.1f}s")

        print(f"{'=' * 40}")


# 便捷函数：一键审计
def audit_sample(
        asset_pool,
        sample_idx: int = 0,
        output_dir: str = "./audit_results",
        sampler_config: Optional[Dict] = None,
        renderer_config: Optional[Dict] = None,
        save_all: bool = True,
) -> Dict[str, str]:
    """
    一键审计单个样本
    """
    # 导入必要的模块
    from .world_sampler import WorldSampler
    from .renderer import WorldRenderer

    # 初始化组件
    sampler_config = sampler_config or {}
    sampler = WorldSampler(asset_pool, **sampler_config)

    renderer_config = renderer_config or {}
    renderer = WorldRenderer(**renderer_config)

    # 创建审计器
    auditor = VideoAuditor(output_dir)

    # 生成样本
    print(f"\nGenerating sample {sample_idx} for audit...")
    timeline = sampler.sample_world()
    render_result = renderer.render(timeline, clear_cache=True)

    # 审计结果
    outputs = auditor.audit_render_result(
        render_result,
        base_name=f"sample_{sample_idx:04d}",
        fps=renderer.fps,
        save_all=save_all,
    )

    return outputs


# 测试代码
if __name__ == "__main__":
    print("=== Testing VideoAuditor ===")

    # 创建模拟数据
    np.random.seed(42)

    # 生成模拟视频 (50帧, 240x320)
    T, H, W = 50, 240, 320
    rgb_frames = np.random.randint(0, 255, (T, H, W, 3), dtype=np.uint8)

    # 生成模拟bbox（一些帧有，一些没有）
    bboxes_per_frame = []
    for t in range(T):
        bboxes = []
        if t % 7 < 3:  # 约43%的帧有bbox
            num_bboxes = np.random.randint(1, 3)
            for _ in range(num_bboxes):
                x1 = np.random.randint(0, W - 50)
                y1 = np.random.randint(0, H - 50)
                x2 = x1 + np.random.randint(30, 100)
                y2 = y1 + np.random.randint(30, 100)
                bboxes.append((x1, y1, x2, y2))
        bboxes_per_frame.append(bboxes)

    # 生成模拟mask
    masks_per_frame = []
    for bboxes in bboxes_per_frame:
        masks = []
        for bbox in bboxes:
            mask = np.zeros((H, W), dtype=np.uint8)
            x1, y1, x2, y2 = bbox
            mask[y1:y2, x1:x2] = 255
            masks.append(mask)
        masks_per_frame.append(masks)

    # 生成模拟时间标注
    temporal_gt = np.zeros(T, dtype=np.float32)
    for t in range(10, 30):  # 帧10-29有sign
        temporal_gt[t] = 1.0
    for t in range(35, 45):  # 帧35-44有sign
        temporal_gt[t] = 1.0

    # 模拟segment spans
    segment_spans = np.array([[10, 30], [35, 45]], dtype=np.int64)


    # 创建模拟render_result对象
    class MockRenderResult:
        def __init__(self):
            self.rgb = rgb_frames
            self.temporal_gt = temporal_gt
            self.bboxes_per_frame = bboxes_per_frame
            self.spatial_masks = masks_per_frame
            self.timeline = type('MockTimeline', (), {
                'background': type('MockBG', (), {'asset_id': 'test_bg', 'duration': 2.0})()
            })()


    mock_result = MockRenderResult()

    # 测试审计器
    auditor = VideoAuditor(output_dir="./test_audit")

    print("\n1. Testing overlay video...")
    overlay_path = auditor.save_video_with_overlay(
        rgb_frames=mock_result.rgb,
        bboxes_per_frame=mock_result.bboxes_per_frame,
        masks_per_frame=mock_result.spatial_masks,
        temporal_gt=mock_result.temporal_gt,
        fps=10,
        filename="test_overlay.mp4",
    )

    print("\n2. Testing separate views...")
    separate_outputs = auditor.save_separate_views(
        rgb_frames=mock_result.rgb,
        bboxes_per_frame=mock_result.bboxes_per_frame,
        masks_per_frame=mock_result.spatial_masks,
        fps=10,
        base_name="test",
    )

    print("\n3. Testing temporal analysis...")
    temporal_path = auditor.save_temporal_analysis(
        temporal_gt=mock_result.temporal_gt,
        segment_spans=segment_spans,
        filename="test_temporal.png",
    )

    print("\n4. Testing spatial analysis...")
    spatial_path = auditor.save_spatial_analysis(
        bboxes_per_frame=mock_result.bboxes_per_frame,
        image_size=(W, H),
        filename="test_spatial.png",
    )

    print("\n5. Testing complete audit...")
    audit_outputs = auditor.audit_render_result(
        render_result=mock_result,
        base_name="test_audit",
        fps=10,
        save_all=True,
    )

    print("\n" + "=" * 50)
    print("Test complete! Check the audit_results folder for outputs.")
    print("=" * 50)