# synthetic_world/loaders/ucf101.py
import os
import cv2
import numpy as np
from typing import List, Optional
import random

from synthetic_world.assets import BackgroundAsset


# -------- Helpers --------

def _count_frames_and_fps(path: str) -> tuple:
    """获取视频的帧数和帧率"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(round(fps)) if fps > 0 else 25

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if length <= 0:
        raise RuntimeError(f"Invalid video length: {path}")

    return length, fps


def _estimate_motion_level_quick(path: str, num_samples: int = 10) -> float:
    """
    快速估计视频运动程度（通过采样几帧计算帧间差异）
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0.0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        cap.release()
        return 0.0

    # 均匀采样帧
    if num_samples > total:
        num_samples = total

    if num_samples < 2:
        cap.release()
        return 0.0

    idxs = np.linspace(0, total - 1, num_samples).astype(int)

    prev = None
    diffs = []

    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue

        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if prev is not None:
            # 计算绝对差异
            diff = np.mean(np.abs(gray - prev))
            diffs.append(diff)

        prev = gray

    cap.release()

    if not diffs:
        return 0.0

    # 归一化到[0, 1]
    mean_diff = np.mean(diffs)
    # 经验阈值：差异大于40认为是高运动
    normalized = min(mean_diff / 40.0, 1.0)
    return float(normalized)


def _estimate_brightness_quick(path: str, num_samples: int = 5) -> float:
    """快速估计视频亮度"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0.5

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return 0.5

    # 随机采样几帧
    idxs = random.sample(range(total), min(num_samples, total))

    brightness_values = []

    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue

        # 转换为HSV，取V通道的均值作为亮度
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        brightness = np.mean(hsv[:, :, 2]) / 255.0  # 归一化到[0,1]
        brightness_values.append(brightness)

    cap.release()

    if not brightness_values:
        return 0.5

    return float(np.mean(brightness_values))


def _contains_human_keyword(scene_type: str) -> bool:
    """根据场景类型判断是否可能包含人物"""
    scene_lower = scene_lower = scene_type.lower()

    # 可能包含人物的场景
    human_scenes = [
        'basketball', 'soccer', 'tennis', 'volleyball', 'baseball', 'golf',
        'walking', 'running', 'jumping', 'diving', 'swimming',
        'boxing', 'fencing', 'skating', 'skiing',
        'handstand', 'pushup', 'pullup', 'somersault',
        'hairdress', 'shaving', 'brushing', 'makeup',
        'pizza', 'baby', 'horse', 'dog', 'cat'
    ]

    for keyword in human_scenes:
        if keyword in scene_lower:
            return True

    return False


def _is_indoor_scene(scene_type: str) -> bool:
    """判断是否为室内场景"""
    scene_lower = scene_type.lower()

    # 室内场景关键词
    indoor_keywords = [
        'indoor', 'room', 'kitchen', 'bathroom', 'office', 'home', 'house',
        'gym', 'classroom', 'lab', 'studio', 'mall', 'store', 'shop',
        'restaurant', 'bar', 'cafe', 'library'
    ]

    # 室外场景关键词
    outdoor_keywords = [
        'outdoor', 'beach', 'park', 'street', 'mountain', 'field', 'forest',
        'river', 'lake', 'ocean', 'sea', 'sky', 'snow', 'desert'
    ]

    for keyword in indoor_keywords:
        if keyword in scene_lower:
            return True

    for keyword in outdoor_keywords:
        if keyword in scene_lower:
            return False

    # 默认室内（UCF101中大部分是室内）
    return True

    # -------- Main Loader --------


def load_ucf101_as_assets(
        root: str,
        classes: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        verbose: bool = True,
        motion_samples: int = 10,
        brightness_samples: int = 5,
) -> List[BackgroundAsset]:
    """
    Load UCF101 as BackgroundAsset list (lazy loading).

    Args:
        root: UCF101数据集根目录
        classes: 要加载的类别列表（None表示全部）
        max_samples: 最大加载样本数（用于测试）
        verbose: 是否打印进度
        motion_samples: 运动估计采样帧数
        brightness_samples: 亮度估计采样帧数

    Returns:
        List[BackgroundAsset]: 背景资产列表
    """

    if verbose:
        print(f"[UCF101] Loading from {root}")

    if not os.path.exists(root):
        raise FileNotFoundError(f"Directory not found: {root}")

    assets = []
    skipped = 0

    # 获取所有类别
    all_classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

    # 过滤类别
    if classes is not None:
        selected_classes = [c for c in all_classes if c in classes]
        if verbose:
            print(f"[UCF101] Loading {len(selected_classes)}/{len(all_classes)} classes")
    else:
        selected_classes = all_classes
        if verbose:
            print(f"[UCF101] Loading all {len(all_classes)} classes")

    for class_idx, cls in enumerate(selected_classes):
        if max_samples and len(assets) >= max_samples:
            break

        class_dir = os.path.join(root, cls)
        if not os.path.isdir(class_dir):
            if verbose:
                print(f"[UCF101] Skip {cls}: not a directory")
            continue

        # 获取该类别的所有视频文件
        video_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.avi')])

        if verbose and class_idx % 10 == 0:
            print(
                f"[UCF101] Processing class {class_idx + 1}/{len(selected_classes)}: {cls} ({len(video_files)} videos)")

        for vid_idx, fname in enumerate(video_files):
            if max_samples and len(assets) >= max_samples:
                break

            path = os.path.join(class_dir, fname)
            asset_id = f"{cls}/{fname}"

            try:
                # 获取基础信息
                num_frames, fps = _count_frames_and_fps(path)

                # 估计运动程度
                motion = _estimate_motion_level_quick(path, motion_samples)

                # 估计亮度
                brightness = _estimate_brightness_quick(path, brightness_samples)

                # 判断是否包含人物
                has_people = _contains_human_keyword(cls)

                # 判断室内外
                is_indoor = _is_indoor_scene(cls)

                # 创建资产（懒加载模式）
                asset = BackgroundAsset(
                    asset_id=asset_id,
                    video_path=path,
                    fps=fps,
                    num_frames=num_frames,
                    scene_type=cls,
                    motion_level=motion,
                    brightness=brightness,
                    has_people=has_people,
                    is_indoor=is_indoor,
                )

                assets.append(asset)

                if verbose and len(assets) % 100 == 0:
                    print(f"[UCF101] Indexed {len(assets)} videos...")

            except Exception as e:
                if verbose and skipped < 5:
                    print(f"[UCF101] Skip {asset_id}: {e}")
                skipped += 1
                continue

    if verbose:
        print(f"[UCF101] Indexed {len(assets)} BackgroundAssets (skipped {skipped})")

    return assets

    # -------- Test --------


if __name__ == "__main__":
    # 注意：需要修改为你的实际路径
    ROOT = "/home/pxl416/PeixiLiu/px_proj/px_data/UCF-101"

    print("=== Testing UCF101 Loader ===\n")

    try:
        # 测试加载少量样本
        assets = load_ucf101_as_assets(
            root=ROOT,
            max_samples=3,
            verbose=True,
        )

        if assets:
            print("\n" + "=" * 50)
            print("Sample background details:")
            print("=" * 50)

            a = assets[0]
            print(f"Asset ID: {a.asset_id}")
            print(f"Scene type: {a.scene_type}")
            print(f"Video path: {a.video_path}")
            print(f"Num frames: {a.num_frames}")
            print(f"FPS: {a.fps}")
            print(f"Duration: {a.duration:.2f} sec")
            print(f"Resolution: {a.resolution}")
            print(f"Motion level: {a.motion_level:.3f}")
            print(f"Brightness: {a.brightness:.3f}")
            print(f"Has people: {a.has_people}")
            print(f"Is indoor: {a.is_indoor}")

            # 测试懒加载
            print("\nTesting lazy loading...")
            frames = a.load_frames(start_frame=0, end_frame=10)  # 只加载前10帧
            print(f"Loaded frames shape: {frames.shape}")
            print(f"Frames dtype: {frames.dtype}")
            print(f"Frames range: [{frames.min()}, {frames.max()}]")

            print("\nTest passed! ✔")
        else:
            print("No assets loaded. Please check paths.")

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please update the paths in the test code.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()