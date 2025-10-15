# -*- coding: utf-8 -*-
from typing import List, Tuple, Optional, Dict, Union
import math
import random
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageDraw, UnidentifiedImageError

TensorOrPIL = Union[torch.Tensor, Image.Image]

def sample_temporal_indices(
    n: int,
    ratio: float,
    *,
    jitter: bool = True,
    min_frames: int = 1,
    max_frames: Optional[int] = None,
    seed: Optional[int] = None,   # ✅ 可选：保证可复现（不污染全局随机状态）
) -> List[int]:
    """
    按比例采样时间索引：将 [0,n) 均匀切成 k 段，每段取 1 帧（抖动 or 中点）
    - n: 原始帧数
    - ratio: 采样比例 (0~1]；会被夹到 [0,1]
    - jitter: True 则在段内随机取，False 取段中点
    - min_frames: 至少取多少帧（默认 1）
    - max_frames: 最多取多少帧（None 表示不限制）
    - seed: 若给定，使用独立 RNG，保证可复现
    """
    if n <= 0:
        return []

    r = max(0.0, min(1.0, float(ratio)))  # 夹到 [0,1]
    if r == 0.0 and min_frames <= 0:
        return []

    # 目标帧数
    k = max(min_frames, int(round(n * r)))
    if max_frames is not None:
        k = min(k, max_frames)
    k = max(1, min(k, n))  # 最终保证 1 <= k <= n

    # 不需要真正采样（全取）
    if k == n:
        return list(range(n))

    # 独立随机源（不污染全局）
    rng = random.Random(seed)

    # 将 [0,n) 切成 k 段，每段内取 1 帧
    edges = np.linspace(0, n, num=k + 1, dtype=np.int32)
    idxs: List[int] = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a + 1:
            # 段内容量 0 或 1：直接取 a（或 n-1）
            idxs.append(min(a, n - 1))
        else:
            if jitter:
                idxs.append(rng.randrange(a, b))       # [a, b-1]
            else:
                idxs.append((a + b - 1) // 2)          # 中点（整数）
    return idxs


# ---------------------------
# 工具
# ---------------------------
def set_clip_seed(seed: Optional[int]):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pil_list_from_tensor_clip(t: torch.Tensor) -> List[Image.Image]:
    """
    t: [T,C,H,W]，假设为 0-1 空间（若已 normalize，需先反归一化再可视化）
    """
    out = []
    T, C, H, W = t.shape
    for f in t:
        x = f
        if x.min() < 0.0 or x.max() > 1.0:
            x = x.clamp(0, 1)
        im = (x * 255).byte().permute(1, 2, 0).cpu().numpy()
        if C == 1:
            im = im[..., 0]
            out.append(Image.fromarray(im, mode="L"))
        else:
            out.append(Image.fromarray(im, mode="RGB"))
    return out


# ---------------------------
# 主增强器
# ---------------------------
class SignAugment:
    """
    时序一致的手语视频数据增强：
      - 仿射(缩放/平移/旋转/错切)
      - 颜色(亮度/对比度/饱和度/色相)
      - 小面积 Cutout (tube)
      - Resize + Normalize
    支持灰度/彩色；灰度可选择输出 1 通道或重复为 3 通道。
    """

    def __init__(
        self,
        size: int = 224,
        # channel
        channel: str = "rgb",           # "rgb" | "gray"
        gray_as_rgb: bool = False,      # gray 模式下输出 3 通道以兼容 Imagenet 预训练
        # geometric
        degrees: float = 5.0,
        translate: float = 0.05,
        scale: Tuple[float, float] = (0.90, 1.10),
        shear: float = 2.0,
        enable_flip: bool = False,      # 手语默认 False
        # photometric
        hue: float = 0.015,             # [-0.5, 0.5]
        saturation: float = 0.40,
        brightness: float = 0.40,
        contrast: float = 0.20,
        # cutout
        cutout_p: float = 0.0,
        cutout_max_area: float = 0.05,  # 单块面积比例
        cutout_num: Tuple[int, int] = (1, 2),
        # normalize
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ):
        assert channel in ("rgb", "gray")
        self.size = int(size)
        self.channel = channel
        self.gray_as_rgb = bool(gray_as_rgb)

        self.degrees = float(degrees)
        self.translate = float(translate)
        self.scale = scale
        self.shear = float(shear)
        self.enable_flip = bool(enable_flip)

        # clamp hue to [-0.5, 0.5]
        self.hue = float(max(-0.5, min(0.5, hue)))
        self.saturation = float(max(0.0, saturation))
        self.brightness = float(max(0.0, brightness))
        self.contrast = float(max(0.0, contrast))

        self.cutout_p = float(cutout_p)
        self.cutout_max_area = float(max(0.0, cutout_max_area))
        self.cutout_num = cutout_num

        # mean/std 根据通道数自动适配
        if self.channel == "gray" and not self.gray_as_rgb:
            if len(mean) != 1 or len(std) != 1:
                # 默认灰度均值方差
                mean, std = (0.5,), (0.5,)
        else:
            if len(mean) != 3 or len(std) != 3:
                mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self.mean = mean
        self.std = std

    # ---------- sampling ----------
    def _sample_affine_params(self) -> Dict:
        deg = random.uniform(-self.degrees, self.degrees)
        sx = sy = random.uniform(self.scale[0], self.scale[1])
        tx = random.uniform(-self.translate, self.translate)
        ty = random.uniform(-self.translate, self.translate)
        shear = random.uniform(-self.shear, self.shear)
        do_flip = self.enable_flip and (random.random() < 0.5)
        return dict(degrees=deg, translate=(tx, ty), scale=(sx, sy), shear=shear, hflip=do_flip)

    def _sample_color_params(self) -> Dict:
        return dict(
            brightness=random.uniform(1 - self.brightness, 1 + self.brightness),
            contrast=random.uniform(1 - self.contrast, 1 + self.contrast),
            saturation=random.uniform(1 - self.saturation, 1 + self.saturation),
            hue=random.uniform(-self.hue, self.hue),
        )

    def _sample_cutout_params(self, H: int, W: int) -> List[Tuple[int, int, int, int]]:
        if random.random() > self.cutout_p:
            return []
        n = random.randint(self.cutout_num[0], self.cutout_num[1])
        boxes = []
        max_area = self.cutout_max_area * H * W
        for _ in range(n):
            area = random.uniform(0.2, 1.0) * max_area
            ratio = random.uniform(0.3, 3.0)
            h = int(round(math.sqrt(area / ratio)))
            w = int(round(h * ratio))
            if h < 1 or w < 1:
                continue
            x = random.randint(0, max(0, W - w))
            y = random.randint(0, max(0, H - h))
            boxes.append((x, y, w, h))
        return boxes

    # ---------- apply ----------
    def _ensure_mode(self, img: Image.Image) -> Image.Image:
        if self.channel == "rgb":
            return img.convert("RGB")
        else:
            return img.convert("L")

    def _apply_affine(self, img: Image.Image, p: Dict) -> Image.Image:
        im = self._ensure_mode(img)
        W, H = im.size
        tx_px = int(round(p["translate"][0] * W))
        ty_px = int(round(p["translate"][1] * H))

        # PIL fill 不同模式不同类型
        if im.mode == "L":
            fill = int(self.mean[0] * 255)
        else:
            fill = tuple(int(m * 255) for m in self.mean)

        out = F.affine(
            im,
            angle=p["degrees"],
            translate=(tx_px, ty_px),
            scale=p["scale"][0],  # sx=sy
            shear=(p["shear"], 0.0),
            interpolation=InterpolationMode.BILINEAR,
            fill=fill,
        )
        if p["hflip"]:
            out = F.hflip(out)
        return out

    def _apply_color(self, img: Image.Image, p: Dict) -> Image.Image:
        # 灰度图像：跳过 hue/saturation，仅适度亮度/对比度
        if img.mode == "L":
            im = F.adjust_brightness(img, p["brightness"])
            im = F.adjust_contrast(im, p["contrast"])
            return im
        # RGB
        im = F.adjust_brightness(img, p["brightness"])
        im = F.adjust_contrast(im, p["contrast"])
        im = F.adjust_saturation(im, p["saturation"])
        # hue ∈ [-0.5,0.5]
        h = max(-0.5, min(0.5, p["hue"]))
        im = F.adjust_hue(im, h)
        return im

    def _apply_cutout(self, img: Image.Image, boxes: List[Tuple[int, int, int, int]]) -> Image.Image:
        if not boxes:
            return img
        draw = ImageDraw.Draw(img)
        if img.mode == "L":
            fill = int(self.mean[0] * 255)
        else:
            fill = tuple(int(m * 255) for m in self.mean)
        for (x, y, w, h) in boxes:
            draw.rectangle([x, y, x + w, y + h], fill=fill)
        return img

    def _to_tensor_norm(self, img: Image.Image) -> torch.Tensor:
        im = F.resize(img, [self.size, self.size], interpolation=InterpolationMode.BILINEAR)
        t = F.pil_to_tensor(im).float() / 255.0  # [C,H,W]
        # 灰度 1 通道→3 通道（可选）
        if im.mode == "L" and self.gray_as_rgb:
            t = t.repeat(3, 1, 1)  # [3,H,W]
        # normalize
        t = F.normalize(t, mean=self.mean, std=self.std)
        return t

    # ---------- main ----------
    def __call__(self, frames: List[Image.Image], *, seed: Optional[int] = None) -> torch.Tensor:
        """
        frames: List[PIL.Image]，同一 clip 的帧
        return: torch.Tensor [T, C, size, size]
        """
        T = len(frames)
        if T == 0:
            C = 3 if (self.channel == "rgb" or self.gray_as_rgb) else 1
            return torch.zeros(0, C, self.size, self.size)

        set_clip_seed(seed)  # 确保 clip 内一致、可复现

        # 一次采样 → 全帧复用
        p_geo = self._sample_affine_params()
        p_col = self._sample_color_params()
        H, W = frames[0].height, frames[0].width
        boxes = self._sample_cutout_params(H, W)

        out: List[torch.Tensor] = []
        for img in frames:
            # 确保模式正确（RGB 或 L）
            im = self._ensure_mode(img)
            # 几何→颜色→遮挡
            im = self._apply_affine(im, p_geo)
            im = self._apply_color(im, p_col)
            im = self._apply_cutout(im, boxes)
            # 尺寸与归一化
            t = self._to_tensor_norm(im)
            out.append(t)

        return torch.stack(out, dim=0)  # [T,C,H,W]


# 便捷预设
def preset_light(size=224, channel: str = "rgb", gray_as_rgb: bool = False) -> SignAugment:
    return SignAugment(
        size=size,
        channel=channel,
        gray_as_rgb=gray_as_rgb,
        degrees=5,
        translate=0.05,
        scale=(0.9, 1.1),
        shear=2,
        enable_flip=False,
        hue=0.015,
        saturation=0.4,
        brightness=0.4,
        contrast=0.2,
        cutout_p=0.10,
        cutout_max_area=0.03,
        cutout_num=(1, 2),
    )


def preset_medium(size=224, channel: str = "rgb", gray_as_rgb: bool = False) -> SignAugment:
    return SignAugment(
        size=size,
        channel=channel,
        gray_as_rgb=gray_as_rgb,
        degrees=8,
        translate=0.06,
        scale=(0.85, 1.15),
        shear=3,
        enable_flip=False,
        hue=0.02,
        saturation=0.5,
        brightness=0.5,
        contrast=0.25,
        cutout_p=0.15,
        cutout_max_area=0.05,
        cutout_num=(1, 2),
    )

