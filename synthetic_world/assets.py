# # synthetic_world
# # 从 CSL / UCF 里加载干净的视频
# import os
# import random
# import numpy as np
# import torch
# from PIL import Image
# from typing import List, Dict, Optional
#
#
# # Core asset types
# class SignAsset:
#     """
#     A sign-language clip with semantic meaning.
#     """
#     def __init__(
#         self,
#         asset_id: str,
#         frames: np.ndarray,        # (T,H,W,3) uint8
#         keypoints: Optional[np.ndarray],  # (T,21,3) or None
#         text: str,
#         gloss: List[str],
#         fps: int = 25,
#     ):
#         self.asset_id = asset_id
#         self.frames = frames
#         self.keypoints = keypoints
#         self.text = text
#         self.gloss = gloss
#         self.fps = fps
#
#         self.T = frames.shape[0]
#         self.H = frames.shape[1]
#         self.W = frames.shape[2]
#
#     @property
#     def duration(self):
#         return self.T / self.fps
#
#
# class BackgroundAsset:
#     """
#     A background video with no sign semantics.
#     """
#     def __init__(
#         self,
#         asset_id: str,
#         frames: np.ndarray,   # (T,H,W,3)
#         fps: int = 25,
#         motion_level: float = 0.0,
#         scene_type: str = "unknown",
#     ):
#         self.asset_id = asset_id
#         self.frames = frames
#         self.fps = fps
#         self.motion_level = motion_level
#         self.scene_type = scene_type
#
#         self.T = frames.shape[0]
#         self.H = frames.shape[1]
#         self.W = frames.shape[2]
#
#     @property
#     def duration(self):
#         return self.T / self.fps
#
#
# # Asset pools
# class AssetPool:
#     """
#     Holds all available assets and allows sampling.
#     """
#     def __init__(self):
#         self.sign_assets: List[SignAsset] = []
#         self.bg_assets: List[BackgroundAsset] = []
#
#     def add_sign(self, asset: SignAsset):
#         self.sign_assets.append(asset)
#
#     def add_background(self, asset: BackgroundAsset):
#         self.bg_assets.append(asset)
#
#     def sample_sign(self) -> SignAsset:
#         return random.choice(self.sign_assets)
#
#     def sample_background(self) -> BackgroundAsset:
#         return random.choice(self.bg_assets)
#
#     def summary(self):
#         return {
#             "num_signs": len(self.sign_assets),
#             "num_backgrounds": len(self.bg_assets),
#         }
#
# if __name__ == "__main__":
#     print("=== Synthetic World Asset Test ===")
#
#     # ---- create fake sign clip ----
#     T, H, W = 40, 128, 128
#     sign_frames = np.random.randint(0, 255, (T, H, W, 3), dtype=np.uint8)
#     keypoints = np.random.rand(T, 21, 3).astype(np.float32)
#
#     sign = SignAsset(
#         asset_id="sign_001",
#         frames=sign_frames,
#         keypoints=keypoints,
#         text="hello",
#         gloss=["HELLO"],
#         fps=25,
#     )
#
#     # ---- create fake background clip ----
#     T2 = 100
#     bg_frames = np.random.randint(0, 255, (T2, H, W, 3), dtype=np.uint8)
#
#     bg = BackgroundAsset(
#         asset_id="bg_001",
#         frames=bg_frames,
#         fps=25,
#         motion_level=0.3,
#         scene_type="office"
#     )
#
#     # ---- put into pool ----
#     pool = AssetPool()
#     pool.add_sign(sign)
#     pool.add_background(bg)
#
#     print("Pool summary:", pool.summary())
#
#     # ---- sample ----
#     s = pool.sample_sign()
#     b = pool.sample_background()
#
#     print("\nSampled Sign:")
#     print("  id:", s.asset_id)
#     print("  text:", s.text)
#     print("  duration:", s.duration, "sec")
#
#     print("\nSampled Background:")
#     print("  id:", b.asset_id)
#     print("  scene:", b.scene_type)
#     print("  duration:", b.duration, "sec")
#
#     print("\nTest passed ✔")


# synthetic_world/assets.py
import os
import json
import random
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any
import logging

logger = logging.getLogger(__name__)


# =============== Data Structures ===============

class SignAsset:
    """A sign-language clip with semantic meaning. Supports lazy loading."""

    def __init__(
            self,
            asset_id: str,
            video_path: str,  # 可以是视频文件路径或图片文件夹路径
            text: str,
            gloss: List[str],
            fps: int = 25,
            num_frames: Optional[int] = None,  # 可选的帧数（避免重复统计）
            # 语义信息
            semantic_category: str = "unknown",
            complexity: float = 0.5,
            two_handed: bool = False,
            # 关键点（可选）
            keypoints_path: Optional[str] = None,
    ):
        self.asset_id = asset_id
        self.video_path = video_path
        self.text = text
        self.gloss = gloss
        self.fps = fps
        self.semantic_category = semantic_category
        self.complexity = complexity
        self.two_handed = two_handed
        self.keypoints_path = keypoints_path

        # 动态加载的属性
        self._frames: Optional[np.ndarray] = None
        self._keypoints: Optional[np.ndarray] = None

        # 元数据
        self._is_folder = os.path.isdir(video_path)
        self._is_video = os.path.isfile(video_path) and video_path.lower().endswith(
            ('.mp4', '.avi', '.mov', '.mkv')
        )

        if not (self._is_folder or self._is_video):
            raise ValueError(f"video_path must be a folder or video file: {video_path}")

        # 初始化元数据
        self._init_metadata(num_frames)

    def _init_metadata(self, num_frames: Optional[int] = None):
        """初始化元数据，不加载帧"""
        if self._is_folder:
            # 文件夹模式
            if num_frames is not None:
                self.num_frames = num_frames
            else:
                self.num_frames = self._count_frames_in_folder()

            # 获取分辨率（从第一张图片）
            self.resolution = self._get_resolution_from_folder()

        else:  # 视频文件模式
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {self.video_path}")

            if num_frames is not None:
                self.num_frames = num_frames
            else:
                self.num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            self.fps = int(cap.get(cv2.CAP_PROP_FPS)) or self.fps
            self.resolution = (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
            cap.release()

        self.duration = self.num_frames / self.fps if self.fps > 0 else 0

    def _count_frames_in_folder(self) -> int:
        """统计文件夹中的图片数量"""
        extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        count = 0
        for f in os.listdir(self.video_path):
            if f.lower().endswith(extensions):
                count += 1
        return count

    def _get_resolution_from_folder(self) -> Tuple[int, int]:
        """从文件夹第一张图片获取分辨率"""
        extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        for f in sorted(os.listdir(self.video_path)):
            if f.lower().endswith(extensions):
                img_path = os.path.join(self.video_path, f)
                img = cv2.imread(img_path)
                if img is not None:
                    return (img.shape[1], img.shape[0])  # (W, H)
        return (0, 0)  # 默认值

    def load_frames(
            self,
            max_frames: Optional[int] = None,
            target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """懒加载视频帧，支持resize"""
        if self._frames is not None:
            if max_frames:
                return self._frames[:max_frames]
            return self._frames

        logger.info(f"Loading frames for {self.asset_id}...")

        if self._is_folder:
            frames = self._load_frames_from_folder(max_frames, target_size)
        else:
            frames = self._load_frames_from_video(max_frames, target_size)

        self._frames = frames
        return frames

    def _load_frames_from_folder(
            self,
            max_frames: Optional[int] = None,
            target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """从文件夹加载图片序列"""
        extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = sorted([
            f for f in os.listdir(self.video_path)
            if f.lower().endswith(extensions)
        ])

        if not image_files:
            raise ValueError(f"No image files found in {self.video_path}")

        if max_frames:
            image_files = image_files[:max_frames]

        frames = []
        for fname in image_files:
            img_path = os.path.join(self.video_path, fname)
            frame = cv2.imread(img_path)
            if frame is None:
                logger.warning(f"Failed to load image: {img_path}")
                continue

            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if target_size:
                frame_rgb = cv2.resize(frame_rgb, target_size)

            frames.append(frame_rgb)

        if not frames:
            raise ValueError(f"Failed to load any frames from {self.video_path}")

        return np.stack(frames)  # (T, H, W, 3)

    def _load_frames_from_video(
            self,
            max_frames: Optional[int] = None,
            target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """从视频文件加载"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and frame_count >= max_frames:
                break

            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if target_size:
                frame_rgb = cv2.resize(frame_rgb, target_size)

            frames.append(frame_rgb)
            frame_count += 1

        cap.release()

        if not frames:
            raise ValueError(f"No frames loaded from {self.video_path}")

        return np.stack(frames)  # (T, H, W, 3)

    def load_keypoints(self) -> Optional[np.ndarray]:
        """加载关键点数据"""
        if self._keypoints is not None:
            return self._keypoints

        if self.keypoints_path and os.path.exists(self.keypoints_path):
            try:
                if self.keypoints_path.endswith('.npy'):
                    self._keypoints = np.load(self.keypoints_path)
                elif self.keypoints_path.endswith('.json'):
                    with open(self.keypoints_path, 'r') as f:
                        data = json.load(f)
                        self._keypoints = np.array(data)
                logger.info(f"Loaded keypoints for {self.asset_id}")
            except Exception as e:
                logger.error(f"Failed to load keypoints from {self.keypoints_path}: {e}")
                self._keypoints = None

        return self._keypoints

    def clear_cache(self):
        """清除缓存以释放内存"""
        self._frames = None
        self._keypoints = None

    @property
    def frames(self) -> np.ndarray:
        """属性访问器，自动加载完整视频（小心内存！）"""
        if self._frames is None:
            self.load_frames()
        return self._frames

    @property
    def T(self) -> int:
        return self.num_frames

    @property
    def H(self) -> int:
        return self.resolution[1] if self.resolution else 0

    @property
    def W(self) -> int:
        return self.resolution[0] if self.resolution else 0

    def __repr__(self):
        return (f"SignAsset(id={self.asset_id}, text={self.text}, "
                f"frames={self.num_frames}, duration={self.duration:.2f}s, "
                f"category={self.semantic_category})")


class BackgroundAsset:
    """A background video with no sign semantics. Supports lazy loading."""

    def __init__(
            self,
            asset_id: str,
            video_path: str,
            fps: int = 25,
            num_frames: Optional[int] = None,
            # 场景信息
            scene_type: str = "unknown",
            motion_level: float = 0.0,
            brightness: float = 0.5,
            has_people: bool = False,
            is_indoor: bool = True,
    ):
        self.asset_id = asset_id
        self.video_path = video_path
        self.fps = fps
        self.scene_type = scene_type
        self.motion_level = motion_level
        self.brightness = brightness
        self.has_people = has_people
        self.is_indoor = is_indoor

        # 动态加载的属性
        self._frames: Optional[np.ndarray] = None

        # 元数据
        if not os.path.isfile(video_path):
            raise ValueError(f"video_path must be a video file: {video_path}")

        # 初始化元数据
        self._init_metadata(num_frames)

    def _init_metadata(self, num_frames: Optional[int] = None):
        """初始化元数据"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        if num_frames is not None:
            self.num_frames = num_frames
        else:
            self.num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.fps = int(cap.get(cv2.CAP_PROP_FPS)) or self.fps
        self.resolution = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        cap.release()

        self.duration = self.num_frames / self.fps if self.fps > 0 else 0

    def load_frames(
            self,
            start_frame: int = 0,
            end_frame: Optional[int] = None,
            target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """懒加载视频帧（支持切片和resize）"""
        if self._frames is not None:
            # 已加载，返回切片
            frames = self._frames[start_frame:end_frame]
            if target_size and frames.shape[1:3] != target_size[::-1]:
                # 需要resize
                resized = []
                for frame in frames:
                    resized.append(cv2.resize(frame, target_size))
                return np.stack(resized)
            return frames

        logger.info(f"Loading background frames for {self.asset_id}...")
        frames = self._load_frames_from_video(start_frame, end_frame, target_size)

        # 可选：缓存完整视频（如果内存允许）
        # self._frames = frames

        return frames

    def _load_frames_from_video(
            self,
            start_frame: int = 0,
            end_frame: Optional[int] = None,
            target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """从视频文件加载指定范围的帧"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        if end_frame is None:
            end_frame = self.num_frames

        # 跳到起始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        for i in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break

            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if target_size:
                frame_rgb = cv2.resize(frame_rgb, target_size)

            frames.append(frame_rgb)

        cap.release()

        if not frames:
            raise ValueError(f"No frames loaded from {self.video_path}")

        return np.stack(frames)  # (T, H, W, 3)

    def clear_cache(self):
        """清除缓存"""
        self._frames = None

    @property
    def frames(self) -> np.ndarray:
        """属性访问器，自动加载完整视频（小心内存！）"""
        if self._frames is None:
            self.load_frames()
        return self._frames

    @property
    def T(self) -> int:
        return self.num_frames

    @property
    def H(self) -> int:
        return self.resolution[1] if self.resolution else 0

    @property
    def W(self) -> int:
        return self.resolution[0] if self.resolution else 0

    def __repr__(self):
        return (f"BackgroundAsset(id={self.asset_id}, scene={self.scene_type}, "
                f"frames={self.num_frames}, duration={self.duration:.2f}s)")


# =============== Asset Pool ===============

class AssetPool:
    """Holds all available assets and allows sampling with filters."""

    def __init__(self, cache_size_mb: int = 1024):
        self.sign_assets: Dict[str, SignAsset] = {}
        self.bg_assets: Dict[str, BackgroundAsset] = {}

        # 按类别索引
        self.sign_by_category: Dict[str, List[str]] = {}
        self.bg_by_scene: Dict[str, List[str]] = {}

        # 内存管理
        self.cache_size_mb = cache_size_mb
        self._current_cache_size = 0

    def add_sign(self, asset: SignAsset, category: Optional[str] = None):
        """添加手语资产"""
        self.sign_assets[asset.asset_id] = asset

        # 按类别索引
        cat = category or asset.semantic_category
        if cat not in self.sign_by_category:
            self.sign_by_category[cat] = []
        self.sign_by_category[cat].append(asset.asset_id)

    def add_background(self, asset: BackgroundAsset):
        """添加背景资产"""
        self.bg_assets[asset.asset_id] = asset

        # 按场景索引
        scene = asset.scene_type
        if scene not in self.bg_by_scene:
            self.bg_by_scene[scene] = []
        self.bg_by_scene[scene].append(asset.asset_id)

    def sample_sign(
            self,
            category: Optional[str] = None,
            min_duration: float = 0,
            max_duration: float = float('inf'),
            exclude_ids: List[str] = None
    ) -> SignAsset:
        """带条件采样手语资产"""
        candidates = []
        for asset_id, asset in self.sign_assets.items():
            if exclude_ids and asset_id in exclude_ids:
                continue
            if category and asset.semantic_category != category:
                continue
            if not (min_duration <= asset.duration <= max_duration):
                continue
            candidates.append(asset)

        if not candidates:
            logger.warning(f"No sign asset matches criteria, sampling randomly")
            return random.choice(list(self.sign_assets.values()))

        return random.choice(candidates)

    def sample_background(
            self,
            scene_type: Optional[str] = None,
            max_motion: float = 1.0,
            min_brightness: float = 0,
            exclude_ids: List[str] = None
    ) -> BackgroundAsset:
        """带条件采样背景资产"""
        candidates = []
        for asset_id, asset in self.bg_assets.items():
            if exclude_ids and asset_id in exclude_ids:
                continue
            if scene_type and asset.scene_type != scene_type:
                continue
            if asset.motion_level > max_motion:
                continue
            if asset.brightness < min_brightness:
                continue
            candidates.append(asset)

        if not candidates:
            logger.warning(f"No background asset matches criteria, sampling randomly")
            return random.choice(list(self.bg_assets.values()))

        return random.choice(candidates)

    def get_sign_by_id(self, asset_id: str) -> SignAsset:
        """通过ID获取手语资产"""
        if asset_id not in self.sign_assets:
            raise KeyError(f"Sign asset {asset_id} not found")
        return self.sign_assets[asset_id]

    def get_background_by_id(self, asset_id: str) -> BackgroundAsset:
        """通过ID获取背景资产"""
        if asset_id not in self.bg_assets:
            raise KeyError(f"Background asset {asset_id} not found")
        return self.bg_assets[asset_id]

    def clear_all_caches(self):
        """清除所有资产的缓存"""
        for asset in self.sign_assets.values():
            asset.clear_cache()
        for asset in self.bg_assets.values():
            asset.clear_cache()
        self._current_cache_size = 0

    def summary(self) -> Dict[str, Any]:
        """返回资产池统计信息"""
        sign_categories = {}
        for cat, ids in self.sign_by_category.items():
            sign_categories[cat] = len(ids)

        bg_scenes = {}
        for scene, ids in self.bg_by_scene.items():
            bg_scenes[scene] = len(ids)

        # 计算已缓存的大小（简化估算）
        cached_frames = 0
        for asset in self.sign_assets.values():
            if asset._frames is not None:
                cached_frames += asset._frames.nbytes

        for asset in self.bg_assets.values():
            if asset._frames is not None:
                cached_frames += asset._frames.nbytes

        self._current_cache_size = cached_frames / (1024 * 1024)  # MB

        return {
            "num_signs": len(self.sign_assets),
            "num_backgrounds": len(self.bg_assets),
            "sign_categories": sign_categories,
            "background_scenes": bg_scenes,
            "cache_size_mb": f"{self._current_cache_size:.1f}/{self.cache_size_mb}",
        }


# =============== Test ===============

if __name__ == "__main__":
    print("=== Synthetic World Asset Test ===")

    # 设置日志
    logging.basicConfig(level=logging.INFO)

    # 创建资产池
    pool = AssetPool()

    # ---- 测试文件夹模式（CSL-Daily） ----
    print("\n1. Testing folder mode (CSL-Daily style)...")

    # 创建测试文件夹
    test_sign_dir = Path("test_data/signs")
    test_sign_dir.mkdir(parents=True, exist_ok=True)

    # 创建几个测试图片（模拟CSL-Daily）
    import numpy as np

    for i in range(5):
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        cv2.imwrite(str(test_sign_dir / f"frame_{i:04d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    sign = SignAsset(
        asset_id="test_sign_folder",
        video_path=str(test_sign_dir),
        text="hello",
        gloss=["HELLO"],
        num_frames=5,  # 已知帧数，避免重复统计
        fps=25,
        semantic_category="greeting",
        complexity=0.3,
    )

    print(f"  Created SignAsset from folder:")
    print(f"    ID: {sign.asset_id}")
    print(f"    Text: {sign.text}")
    print(f"    Frames: {sign.num_frames}")
    print(f"    Resolution: {sign.resolution}")

    # 测试懒加载
    frames = sign.load_frames(max_frames=3)
    print(f"    Loaded {len(frames)} frames, shape: {frames.shape}")

    # ---- 测试视频文件模式（UCF101） ----
    print("\n2. Testing video file mode (UCF101 style)...")

    # 创建测试视频（实际使用中应该用真实UCF101视频）
    test_video_path = "test_data/test_bg.avi"
    Path("test_data").mkdir(exist_ok=True)

    # 如果存在测试视频则使用，否则跳过
    if os.path.exists(test_video_path):
        bg = BackgroundAsset(
            asset_id="test_bg_video",
            video_path=test_video_path,
            scene_type="office",
            motion_level=0.2,
            brightness=0.6,
        )

        print(f"  Created BackgroundAsset from video:")
        print(f"    ID: {bg.asset_id}")
        print(f"    Scene: {bg.scene_type}")
        print(f"    Frames: {bg.num_frames}")
        print(f"    Resolution: {bg.resolution}")

        # 测试懒加载切片
        frames = bg.load_frames(start_frame=0, end_frame=10)
        print(f"    Loaded {len(frames)} frames, shape: {frames.shape}")
    else:
        print(f"  Skipping video test (test video not found: {test_video_path})")

    # ---- 测试资产池 ----
    print("\n3. Testing AssetPool...")

    pool.add_sign(sign, "greeting")
    if 'bg' in locals():
        pool.add_background(bg)

    summary = pool.summary()
    print(f"  Pool summary: {summary}")

    # 测试采样
    sampled_sign = pool.sample_sign()
    print(f"  Sampled sign: {sampled_sign.asset_id}")

    print("\nTest completed! ✔")
