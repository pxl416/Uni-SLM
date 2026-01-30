# synthetic_world/sign_mask.py
from __future__ import annotations
import cv2

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np
import hashlib
import os
from pathlib import Path

# ----------------------------
# Utils
# ----------------------------

def _ensure_rgb_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img


def _frame_hash(img: np.ndarray) -> str:
    h, w = img.shape[:2]
    ds = img[:: max(1, h // 64), :: max(1, w // 64)]
    return hashlib.md5(ds.tobytes()).hexdigest()


def _largest_cc(mask: np.ndarray) -> np.ndarray:
    import cv2
    num, labels = cv2.connectedComponents(mask.astype(np.uint8))
    if num <= 1:
        return mask
    areas = []
    for i in range(1, num):
        areas.append((labels == i).sum())
    idx = np.argmax(areas) + 1
    return (labels == idx).astype(np.uint8)


# ----------------------------
# Config
# ----------------------------

# project root = Uni-SLM/
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "pretrained-model"


@dataclass
class MaskConfig:
    method: str = "yolo"
    # 自动定位模型
    yolo_model: str = str(MODEL_DIR / "yolov8n-seg.pt")
    device: str = "cuda"
    conf_thres: float = 0.25
    enable_cache: bool = True
    cache_size: int = 512



# ----------------------------
# YOLO Backend
# ----------------------------

class YoloSegBackend:

    def __init__(
        self,
        model_path: str,
        device: str,
        conf: float,
    ):
        from ultralytics import YOLO
        import torch
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        # ---------------- FORCE CPU ----------------
        os.environ["CUDA_VISIBLE_DEVICES"] = ""   # 禁用所有GPU
        torch.cuda.is_available = lambda: False   # 双保险
        self.model = YOLO(model_path)
        self.model.to("cpu")   # 强制CPU
        self.conf = conf

    def predict(self, frame):

        import cv2

        res = self.model.predict(
            frame,
            conf=self.conf,
            verbose=False,
        )[0]

        if res.masks is None:
            return np.zeros(frame.shape[:2], np.uint8), {"num_instances": 0}

        masks = res.masks.data.cpu().numpy()  # (N,h,w)

        H, W = frame.shape[:2]

        resized = []

        for m in masks:
            m = cv2.resize(
                m.astype(np.float32),
                (W, H),
                interpolation=cv2.INTER_LINEAR
            )
            resized.append(m)

        masks = np.stack(resized, axis=0)

        areas = masks.sum(axis=(1, 2))

        idx = np.argmax(areas)

        mask = (masks[idx] > 0.5).astype(np.uint8)

        mask = _largest_cc(mask)

        info = {
            "num_instances": int(len(masks)),
            "selected": int(idx),
            "area": int(mask.sum()),
        }

        return mask, info


# ----------------------------
# Main Provider
# ----------------------------

class SignMaskProvider:


    def __init__(self, cfg: MaskConfig):

        assert cfg.method == "yolo", "This version supports only YOLO"

        self.cfg = cfg

        self._cache: Dict[str, Tuple[np.ndarray, Dict]] = {}

        self._backend = YoloSegBackend(
            cfg.yolo_model,
            cfg.device,
            cfg.conf_thres,
        )


    def get_mask(
        self,
        frame_rgb: np.ndarray,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:


        frame = _ensure_rgb_uint8(frame_rgb)


        # ---------------- cache ----------------

        key = None

        if self.cfg.enable_cache:

            key = _frame_hash(frame)

            if key in self._cache:

                m, info = self._cache[key]

                return m.copy(), dict(info)


        # ---------------- infer ----------------

        mask01, info = self._backend.predict(frame)


        # ---------------- cache write ----------------

        if key is not None:

            self._cache[key] = (mask01.copy(), dict(info))

            if len(self._cache) > self.cfg.cache_size:

                self._cache.pop(next(iter(self._cache)))


        info["method"] = "yolo"

        return mask01, info


# ----------------------------
# Factory
# ----------------------------

def build_sign_mask_provider(cfg_dict: Dict[str, Any]) -> SignMaskProvider:

    cfg = MaskConfig(**cfg_dict)

    return SignMaskProvider(cfg)
