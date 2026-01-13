# synthetic_world/label_emitter.py
from __future__ import annotations

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Labels:
    """统一标注格式"""
    # 时间标注
    temporal_gt: np.ndarray  # (T,) float32，有手语=1
    segment_spans: np.ndarray  # (N, 2) int64，[start_frame, end_frame)

    # 空间标注（每帧）
    frame_bboxes: List[List[Tuple[int, int, int, int]]]  # 每帧的bbox列表

    # 文本标注
    text_alignments: List[Dict[str, Any]]  # 文本对齐信息
    vocabulary: List[str]  # 词汇表

    # 元数据
    metadata: Dict[str, Any]

    # 可选参数（放在最后）
    frame_masks: Optional[List[List[np.ndarray]]] = None  # 每帧的mask列表（可选）


class LabelEmitter:
    """
    从RenderResult生成标准化标注

    支持多种输出格式：
    1. 内部格式（用于训练）
    2. COCO格式（检测任务）
    3. 自定义JSON格式
    """

    def __init__(self, include_masks: bool = True):
        """
        Args:
            include_masks: 是否包含mask数据（增加文件大小）
        """
        self.include_masks = include_masks

    def emit(self, render_result) -> Labels:
        """
        从渲染结果生成标注

        Args:
            render_result: RenderResult对象

        Returns:
            Labels: 统一标注对象
        """
        # 提取基本信息
        temporal_gt = render_result.temporal_gt
        bboxes_per_frame = render_result.bboxes_per_frame
        frame_instructions = render_result.frame_instructions
        timeline = render_result.timeline

        # 1. 生成segment spans（从时间线）
        segment_spans = self._extract_segment_spans(timeline, len(temporal_gt))

        # 2. 生成文本对齐信息
        text_alignments = self._extract_text_alignments(timeline, frame_instructions)

        # 3. 提取词汇表
        vocabulary = self._extract_vocabulary(timeline)

        # 4. 准备mask数据（可选）
        frame_masks = None
        if self.include_masks and hasattr(render_result, 'spatial_masks'):
            frame_masks = render_result.spatial_masks

        # 5. 构建元数据
        metadata = self._build_metadata(render_result)

        return Labels(
            temporal_gt=temporal_gt,
            segment_spans=segment_spans,
            frame_bboxes=bboxes_per_frame,
            text_alignments=text_alignments,
            vocabulary=vocabulary,
            metadata=metadata,
            frame_masks=frame_masks
        )

    def _extract_segment_spans(self, timeline, total_frames: int) -> np.ndarray:
        """从时间线提取segment帧范围"""
        segments = getattr(timeline, 'segments', getattr(timeline, 'sign_segments', []))

        if not segments:
            return np.zeros((0, 2), dtype=np.int64)

        spans = []
        fps = getattr(timeline, 'config', {}).get('fps', 25)
        if not fps:
            # 尝试从background获取
            bg = timeline.background
            fps = getattr(bg, 'fps', 25)

        for seg in segments:
            # 获取时间（兼容不同字段名）
            start_sec, end_sec = self._get_segment_time(seg)
            if start_sec is None or end_sec is None:
                continue

            start_frame = int(start_sec * fps)
            end_frame = int(end_sec * fps)

            # 确保在有效范围内
            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(start_frame + 1, min(end_frame, total_frames))

            spans.append([start_frame, end_frame])

        return np.array(spans, dtype=np.int64) if spans else np.zeros((0, 2), dtype=np.int64)

    def _get_segment_time(self, seg: Dict) -> Tuple[Optional[float], Optional[float]]:
        """安全获取segment时间"""
        # 兼容不同字段名
        if 'start_sec' in seg and 'end_sec' in seg:
            return seg['start_sec'], seg['end_sec']
        if 'start_time' in seg and 'end_time' in seg:
            return seg['start_time'], seg['end_time']
        if 'start' in seg and 'end' in seg:
            return seg['start'], seg['end']
        return None, None

    def _extract_text_alignments(self, timeline, frame_instructions) -> List[Dict[str, Any]]:
        """生成文本-帧对齐信息"""
        alignments = []
        segments = getattr(timeline, 'segments', getattr(timeline, 'sign_segments', []))

        for seg_idx, seg in enumerate(segments):
            sign = seg.get('sign')
            if not sign:
                continue

            # 获取时间
            start_sec, end_sec = self._get_segment_time(seg)
            if start_sec is None or end_sec is None:
                continue

            # 收集该segment中所有帧
            segment_frames = []
            for frame_idx, inst in enumerate(frame_instructions):
                timestamp = inst.get('timestamp', frame_idx / 25.0)
                if start_sec <= timestamp < end_sec:
                    # 检查这个手语是否在当前帧活跃
                    for active in inst.get('active_signs', []):
                        if active.get('sign') is sign:
                            segment_frames.append(frame_idx)
                            break

            alignments.append({
                'segment_id': seg_idx,
                'sign_id': getattr(sign, 'asset_id', f'segment_{seg_idx}'),
                'text': getattr(sign, 'text', ''),
                'gloss': getattr(sign, 'gloss', []),
                'start_sec': start_sec,
                'end_sec': end_sec,
                'frames': segment_frames,
                'num_frames': len(segment_frames),
            })

        return alignments

    def _extract_vocabulary(self, timeline) -> List[str]:
        """提取本视频中的词汇表"""
        vocabulary = set()
        segments = getattr(timeline, 'segments', getattr(timeline, 'sign_segments', []))

        for seg in segments:
            sign = seg.get('sign')
            if sign:
                text = getattr(sign, 'text', '')
                if text:
                    # 简单分词（中文按字符，英文按单词）
                    if any(ord(c) > 127 for c in text):  # 包含非ASCII，可能是中文
                        for char in text:
                            if char.strip():
                                vocabulary.add(char)
                    else:  # 英文
                        for word in text.split():
                            vocabulary.add(word.lower())

        return sorted(list(vocabulary))

    def _build_metadata(self, render_result) -> Dict[str, Any]:
        """构建元数据"""
        timeline = render_result.timeline

        return {
            'background_id': getattr(timeline.background, 'asset_id', 'unknown'),
            'background_duration': getattr(timeline.background, 'duration', 0),
            'num_segments': len(getattr(timeline, 'segments', [])),
            'total_frames': len(render_result.temporal_gt),
            'resolution': {
                'width': render_result.rgb.shape[2],
                'height': render_result.rgb.shape[1],
            },
            'has_masks': self.include_masks and hasattr(render_result, 'spatial_masks'),
            'timestamp': np.datetime64('now').astype(str),
        }

    def save_to_json(self, labels: Labels, output_path: Path, compact: bool = False):
        """保存为JSON格式"""
        # 转换numpy数组为列表
        data = {
            'temporal_gt': labels.temporal_gt.tolist(),
            'segment_spans': labels.segment_spans.tolist(),
            'frame_bboxes': [
                [list(bbox) for bbox in frame_bboxes]
                for frame_bboxes in labels.frame_bboxes
            ],
            'text_alignments': labels.text_alignments,
            'vocabulary': labels.vocabulary,
            'metadata': labels.metadata,
        }

        # 可选：保存mask（通常分开保存为二进制文件）
        if labels.frame_masks and self.include_masks:
            # 注意：mask数据很大，通常保存为.npy文件
            mask_path = output_path.with_suffix('.masks.npy')
            all_masks = []
            for frame_masks in labels.frame_masks:
                if frame_masks:
                    all_masks.append(np.stack(frame_masks))
                else:
                    all_masks.append(np.zeros((0, labels.metadata['resolution']['height'],
                                               labels.metadata['resolution']['width']), dtype=np.uint8))
            np.save(mask_path, all_masks)
            data['mask_path'] = str(mask_path)

        with open(output_path, 'w', encoding='utf-8') as f:
            if compact:
                json.dump(data, f, separators=(',', ':'))
            else:
                json.dump(data, f, indent=2, ensure_ascii=False)

    def save_to_coco(self, labels: Labels, output_path: Path):
        """保存为COCO格式（用于目标检测）"""
        # 简化的COCO格式
        coco_data = {
            'info': {
                'description': 'Synthetic Sign Language Dataset',
                'version': '1.0',
                'year': 2024,
            },
            'licenses': [],
            'categories': [
                {'id': 1, 'name': 'sign', 'supercategory': 'action'}
            ],
            'images': [],
            'annotations': [],
        }

        # 添加图像信息
        for frame_idx in range(len(labels.frame_bboxes)):
            coco_data['images'].append({
                'id': frame_idx,
                'width': labels.metadata['resolution']['width'],
                'height': labels.metadata['resolution']['height'],
                'file_name': f'frame_{frame_idx:06d}.jpg',
            })

            # 添加标注
            for bbox_idx, bbox in enumerate(labels.frame_bboxes[frame_idx]):
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1

                # COCO格式： [x, y, width, height]
                coco_bbox = [float(x1), float(y1), float(width), float(height)]

                annotation_id = frame_idx * 1000 + bbox_idx
                coco_data['annotations'].append({
                    'id': annotation_id,
                    'image_id': frame_idx,
                    'category_id': 1,  # sign类别
                    'bbox': coco_bbox,
                    'area': width * height,
                    'segmentation': [],  # 可以填充多边形
                    'iscrowd': 0,
                })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2)


# 便捷函数
def emit_and_save(render_result, output_dir: Path, format: str = 'json'):
    """一键生成并保存标注"""
    emitter = LabelEmitter(include_masks=True)
    labels = emitter.emit(render_result)

    output_dir.mkdir(parents=True, exist_ok=True)

    if format == 'json':
        emitter.save_to_json(labels, output_dir / 'labels.json')
    elif format == 'coco':
        emitter.save_to_coco(labels, output_dir / 'coco.json')
    else:
        raise ValueError(f"Unsupported format: {format}")

    return labels


# 测试代码
if __name__ == "__main__":
    print("=== Testing LabelEmitter ===")

    from dataclasses import dataclass
    from typing import List


    # 模拟RenderResult
    @dataclass
    class MockRenderResult:
        rgb: np.ndarray
        temporal_gt: np.ndarray
        bboxes_per_frame: List[List[Tuple[int, int, int, int]]]
        spatial_masks: List[List[np.ndarray]]
        frame_instructions: List[Dict]
        timeline: Any


    @dataclass
    class MockTimeline:
        background: Any
        segments: List[Dict]
        config: Dict

        @property
        def sign_segments(self):
            return self.segments


    # 创建测试数据
    bg = type('MockBG', (), {'asset_id': 'bg_test', 'duration': 5.0, 'fps': 10})()

    sign1 = type('MockSign', (), {
        'asset_id': 'sign_hello',
        'text': '你好',
        'gloss': ['NI', 'HAO']
    })()

    sign2 = type('MockSign', (), {
        'asset_id': 'sign_thanks',
        'text': 'thank you',
        'gloss': ['THANK', 'YOU']
    })()

    timeline = MockTimeline(
        background=bg,
        segments=[
            {'sign': sign1, 'start_sec': 0.5, 'end_sec': 2.5},
            {'sign': sign2, 'start_sec': 2.0, 'end_sec': 4.0},
        ],
        config={'fps': 10}
    )

    # 模拟frame_instructions（简化）
    frame_instructions = []
    for frame_idx in range(50):  # 5秒 * 10fps
        timestamp = frame_idx / 10.0
        active_signs = []

        if 0.5 <= timestamp < 2.5:
            active_signs.append({'sign': sign1})
        if 2.0 <= timestamp < 4.0:
            active_signs.append({'sign': sign2})

        frame_instructions.append({
            'timestamp': timestamp,
            'active_signs': active_signs
        })

    # 创建RenderResult
    render_result = MockRenderResult(
        rgb=np.zeros((50, 240, 320, 3), dtype=np.uint8),
        temporal_gt=np.zeros(50, dtype=np.float32),
        bboxes_per_frame=[[] for _ in range(50)],
        spatial_masks=[[] for _ in range(50)],
        frame_instructions=frame_instructions,
        timeline=timeline
    )

    # 填充一些测试bbox
    for frame_idx in range(10, 25):  # 手语1活跃的帧
        render_result.bboxes_per_frame[frame_idx] = [(50, 50, 150, 150)]
        render_result.temporal_gt[frame_idx] = 1.0

    for frame_idx in range(20, 40):  # 手语2活跃的帧
        render_result.bboxes_per_frame[frame_idx].append((200, 100, 300, 200))
        render_result.temporal_gt[frame_idx] = 1.0

    # 测试LabelEmitter
    emitter = LabelEmitter(include_masks=False)
    labels = emitter.emit(render_result)

    print(f"\nLabels generated:")
    print(f"  Temporal GT shape: {labels.temporal_gt.shape}")
    print(f"  Segment spans: {labels.segment_spans}")
    print(f"  Frame bboxes: {len(labels.frame_bboxes)} frames")
    print(f"  Text alignments: {len(labels.text_alignments)} segments")
    print(f"  Vocabulary: {labels.vocabulary}")
    print(f"  Metadata: {labels.metadata}")

    # 测试保存
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # 保存为JSON
        json_path = tmp_path / 'test_labels.json'
        emitter.save_to_json(labels, json_path)
        print(f"\nSaved JSON to: {json_path}")
        print(f"File size: {json_path.stat().st_size / 1024:.1f} KB")

        # 验证可读性
        with open(json_path, 'r') as f:
            loaded = json.load(f)
            print(f"Loaded back: {len(loaded['temporal_gt'])} frames")

        # 保存为COCO
        coco_path = tmp_path / 'test_coco.json'
        emitter.save_to_coco(labels, coco_path)
        print(f"\nSaved COCO to: {coco_path}")

    print("\nTest passed! LabelEmitter is ready. ✔")