# synthetic_world/sign_mask.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List, Union
import os
import hashlib
import numpy as np

# Optional deps (only required when you enable SAM)
try:
    import torch
except Exception:
    torch = None


# ----------------------------
# Utilities
# ----------------------------

def _ensure_rgb_uint8(frame_rgb: np.ndarray) -> np.ndarray:
    """Ensure (H,W,3) uint8 RGB."""
    if frame_rgb is None:
        raise ValueError("frame_rgb is None")

    if frame_rgb.ndim != 3 or frame_rgb.shape[2] not in (3, 4):
        raise ValueError(f"Expected (H,W,3/4), got {frame_rgb.shape}")

    if frame_rgb.dtype != np.uint8:
        # best-effort conversion
        frame = frame_rgb.astype(np.float32)
        if frame.max() <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    else:
        frame = frame_rgb

    if frame.shape[2] == 4:
        frame = frame[:, :, :3]  # drop alpha

    return frame


def _frame_hash(frame_rgb: np.ndarray) -> str:
    """Small hash for caching. Uses downsample + md5."""
    h, w = frame_rgb.shape[:2]
    # downsample for speed
    ds = frame_rgb[:: max(1, h // 64), :: max(1, w // 64), :3]
    m = hashlib.md5(ds.tobytes()).hexdigest()
    return m


def _bbox_from_mask(mask01: np.ndarray) -> Tuple[int, int, int, int]:
    """Return (x1,y1,x2,y2) in pixel coords from {0,1} mask."""
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return (x1, y1, x2, y2)


def _largest_connected_component(mask01: np.ndarray) -> np.ndarray:
    """
    Keep largest connected component for stability.
    Uses simple BFS via scipy-free implementation (slow-ish but ok for masks).
    For speed, you can remove this if needed.
    """
    mask = (mask01 > 0).astype(np.uint8)
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=np.uint8)

    best_count = 0
    best_coords: List[Tuple[int, int]] = []

    # 4-neighborhood
    for y in range(H):
        row = mask[y]
        for x in range(W):
            if row[x] == 0 or visited[y, x] == 1:
                continue
            # BFS
            q = [(y, x)]
            visited[y, x] = 1
            coords = [(y, x)]
            while q:
                cy, cx = q.pop()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < H and 0 <= nx < W:
                        if mask[ny, nx] == 1 and visited[ny, nx] == 0:
                            visited[ny, nx] = 1
                            q.append((ny, nx))
                            coords.append((ny, nx))
            if len(coords) > best_count:
                best_count = len(coords)
                best_coords = coords

    out = np.zeros_like(mask, dtype=np.uint8)
    for (yy, xx) in best_coords:
        out[yy, xx] = 1
    return out


# ----------------------------
# Config
# ----------------------------

@dataclass
class MaskConfig:
    # method: "rgba" | "threshold" | "sam_auto" | "sam_prompt"
    method: str = "sam_auto"

    # RGBA / threshold fallback
    threshold_gray: int = 15  # for simple threshold mask (0-255)
    invert_threshold: bool = False

    # SAM: model
    sam_checkpoint: Optional[str] = None
    sam_model_type: str = "vit_h"  # vit_h | vit_l | vit_b
    device: str = "cuda"  # "cuda" | "cpu"
    half: bool = False

    # SAM AutoMaskGenerator params (tune if needed)
    points_per_side: int = 24
    pred_iou_thresh: float = 0.86
    stability_score_thresh: float = 0.92
    min_mask_region_area: int = 2000

    # Postprocess
    keep_largest_cc: bool = True

    # Cache
    enable_cache: bool = True
    cache_size: int = 512  # number of frames cached


# ----------------------------
# Main provider
# ----------------------------

class SignMaskProvider:
    """
    Produce a sign/foreground mask for a sign frame.

    Output:
        mask01: (H,W) uint8 in {0,1}
        info: dict (method-specific diagnostics)
    """

    def __init__(self, cfg: MaskConfig):
        self.cfg = cfg
        self._cache: Dict[str, Tuple[np.ndarray, Dict[str, Any]]] = {}

        # SAM objects
        self._sam = None
        self._sam_predictor = None
        self._sam_auto = None

        if self.cfg.method.startswith("sam"):
            self._init_sam()

    # -------- public --------

    def get_mask(
        self,
        frame_rgb: np.ndarray,
        *,
        # only for sam_prompt
        box_xyxy: Optional[Tuple[int, int, int, int]] = None,
        point_xy: Optional[Tuple[int, int]] = None,
        point_label: int = 1,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Returns:
            mask01: (H,W) uint8 in {0,1}
            info: dict
        """
        method = self.cfg.method

        frame = _ensure_rgb_uint8(frame_rgb)

        # cache
        key = None
        if self.cfg.enable_cache:
            key = f"{method}:{_frame_hash(frame)}"
            if key in self._cache:
                return self._cache[key][0].copy(), dict(self._cache[key][1])

        if method == "rgba":
            mask01, info = self._mask_from_rgba(frame_rgb)
        elif method == "threshold":
            mask01, info = self._mask_from_threshold(frame)
        elif method == "sam_auto":
            mask01, info = self._mask_from_sam_auto(frame)
        elif method == "sam_prompt":
            mask01, info = self._mask_from_sam_prompt(
                frame, box_xyxy=box_xyxy, point_xy=point_xy, point_label=point_label
            )
        else:
            raise ValueError(f"Unknown mask method: {method}")

        # postprocess
        if self.cfg.keep_largest_cc and mask01.any():
            mask01 = _largest_connected_component(mask01)

        # cache write
        if key is not None:
            self._cache[key] = (mask01.copy(), dict(info))
            if len(self._cache) > int(self.cfg.cache_size):
                # pop one (FIFO-ish)
                self._cache.pop(next(iter(self._cache)))

        return mask01, info

    # -------- implementations --------

    def _mask_from_rgba(self, frame_rgba_or_rgb: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        if frame_rgba_or_rgb.ndim != 3 or frame_rgba_or_rgb.shape[2] != 4:
            # if no alpha, fallback to threshold
            frame = _ensure_rgb_uint8(frame_rgba_or_rgb)
            return self._mask_from_threshold(frame)

        alpha = frame_rgba_or_rgb[:, :, 3].astype(np.uint8)
        mask01 = (alpha > 0).astype(np.uint8)
        info = {"method": "rgba", "note": "mask from alpha>0"}
        return mask01, info

    def _mask_from_threshold(self, frame_rgb: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        # simple grayscale threshold
        gray = (0.299 * frame_rgb[:, :, 0] + 0.587 * frame_rgb[:, :, 1] + 0.114 * frame_rgb[:, :, 2]).astype(np.uint8)
        mask01 = (gray > int(self.cfg.threshold_gray)).astype(np.uint8)
        if self.cfg.invert_threshold:
            mask01 = (1 - mask01).astype(np.uint8)
        info = {"method": "threshold", "threshold_gray": int(self.cfg.threshold_gray), "invert": bool(self.cfg.invert_threshold)}
        return mask01, info

    # -------- SAM init + usage --------

    def _init_sam(self):
        if torch is None:
            raise RuntimeError(
                "PyTorch is required for SAM. Please install torch first."
            )

        try:
            from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
        except Exception as e:
            raise RuntimeError(
                "segment-anything is not installed.\n"
                "Install (example):\n"
                "  pip install git+https://github.com/facebookresearch/segment-anything.git\n"
                "and ensure you have opencv-python, numpy.\n"
                f"Original error: {e}"
            )

        ckpt = self.cfg.sam_checkpoint
        if ckpt is None or not os.path.exists(ckpt):
            raise RuntimeError(
                f"SAM checkpoint not found: {ckpt}\n"
                "Set cfg.sam_checkpoint to the .pth checkpoint path."
            )

        device = self.cfg.device
        if device == "cuda":
            if not torch.cuda.is_available():
                device = "cpu"

        sam = sam_model_registry[self.cfg.sam_model_type](checkpoint=ckpt)
        sam.to(device=device)
        sam.eval()

        if self.cfg.half and device == "cuda":
            sam = sam.half()

        self._sam = sam
        self._sam_predictor = SamPredictor(sam)

        # Auto generator (for 'sam_auto')
        self._sam_auto = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=int(self.cfg.points_per_side),
            pred_iou_thresh=float(self.cfg.pred_iou_thresh),
            stability_score_thresh=float(self.cfg.stability_score_thresh),
            min_mask_region_area=int(self.cfg.min_mask_region_area),
        )

    def _mask_from_sam_auto(self, frame_rgb: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Auto mode:
          - generate a set of masks
          - choose the "best" one (largest area by default)
        """
        if self._sam_auto is None:
            self._init_sam()

        # SAM expects RGB uint8
        masks = self._sam_auto.generate(frame_rgb)
        if not masks:
            return np.zeros(frame_rgb.shape[:2], dtype=np.uint8), {"method": "sam_auto", "num_masks": 0, "selected": None}

        # Choose best mask:
        # default: largest area (often human / main actor)
        best = max(masks, key=lambda m: int(m.get("area", 0)))
        seg = best["segmentation"]  # bool (H,W)
        mask01 = seg.astype(np.uint8)

        info = {
            "method": "sam_auto",
            "num_masks": len(masks),
            "selected": {
                "area": int(best.get("area", 0)),
                "pred_iou": float(best.get("predicted_iou", 0.0)),
                "stability": float(best.get("stability_score", 0.0)),
                "bbox_xywh": tuple(best.get("bbox", (0, 0, 0, 0))),
            },
        }
        return mask01, info

    def _mask_from_sam_prompt(
        self,
        frame_rgb: np.ndarray,
        *,
        box_xyxy: Optional[Tuple[int, int, int, int]],
        point_xy: Optional[Tuple[int, int]],
        point_label: int = 1,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Prompted mode:
          - if you can provide a box (x1,y1,x2,y2), it's usually most stable
          - point prompt also supported
        """
        if self._sam_predictor is None:
            self._init_sam()

        predictor = self._sam_predictor
        predictor.set_image(frame_rgb)

        input_box = None
        if box_xyxy is not None:
            x1, y1, x2, y2 = box_xyxy
            input_box = np.array([x1, y1, x2, y2], dtype=np.float32)

        input_point = None
        input_label = None
        if point_xy is not None:
            px, py = point_xy
            input_point = np.array([[px, py]], dtype=np.float32)
            input_label = np.array([int(point_label)], dtype=np.int32)

        if input_box is None and input_point is None:
            raise ValueError("sam_prompt requires box_xyxy or point_xy")

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box[None, :] if input_box is not None else None,
            multimask_output=True,
        )

        # pick highest-score mask
        best_i = int(np.argmax(scores))
        mask01 = masks[best_i].astype(np.uint8)

        info = {
            "method": "sam_prompt",
            "scores": [float(s) for s in scores],
            "selected": best_i,
            "bbox_from_mask": _bbox_from_mask(mask01),
            "used_box": tuple(box_xyxy) if box_xyxy is not None else None,
            "used_point": tuple(point_xy) if point_xy is not None else None,
        }
        return mask01, info


# ----------------------------
# Simple factory (optional)
# ----------------------------

def build_sign_mask_provider(cfg_dict: Dict[str, Any]) -> SignMaskProvider:
    """
    Build provider from YAML-like dict.
    Example:
        sign_mask:
          method: sam_auto
          sam_checkpoint: /path/to/sam_vit_h_4b8939.pth
          sam_model_type: vit_h
          device: cuda
    """
    cfg = MaskConfig(**cfg_dict)
    return SignMaskProvider(cfg)


# ----------------------------
# Test
# ----------------------------

if __name__ == "__main__":
    print("=== Testing SignMaskProvider ===")

    # 1) threshold test (no SAM required)
    H, W = 240, 320
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[:] = 10
    frame[60:200, 120:240, :] = 200  # bright "person block"

    prov = SignMaskProvider(MaskConfig(method="threshold", threshold_gray=50))
    mask01, info = prov.get_mask(frame)
    bbox = _bbox_from_mask(mask01)
    print("[threshold] info:", info, "bbox:", bbox, "mask_sum:", int(mask01.sum()))
    assert mask01.sum() > 0
    assert bbox[2] > bbox[0] and bbox[3] > bbox[1]

    # 2) SAM auto test (requires checkpoint + segment-anything)
    # Uncomment and set your checkpoint path:

    sam_ckpt = "/home/pxl416/PeixiLiu/px_proj/Uni-SLM/pretrained-model/sam_vit_h_4b8939.pth"
    prov_sam = SignMaskProvider(MaskConfig(
        method="sam_auto",
        sam_checkpoint=sam_ckpt,
        sam_model_type="vit_h",
        device="cpu",
        points_per_side=24,
    ))
    mask01, info = prov_sam.get_mask(frame)
    print("[sam_auto] info:", info, "bbox:", _bbox_from_mask(mask01), "mask_sum:", int(mask01.sum()))

    print("SignMaskProvider test passed ✔")
