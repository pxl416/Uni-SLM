# synthetic_world/spatial_composer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2


@dataclass
class SpatialConfig:
    """Parsed v1 spatial config (with safe defaults)."""
    # pipeline stages
    occlusion_enabled: bool = False
    occlusion_mode: str = "patch"          # patch | block | random_pixels
    occlusion_patch_size: int = 16
    occlusion_ratio: float = 0.25
    occlusion_fill: str = "zero"           # zero | mean | random

    depatch_enabled: bool = False
    depatch_grid: Tuple[int, int] = (3, 4)  # (rows, cols)
    depatch_scatter_mode: str = "random"    # random | permute
    depatch_keep_tile_size: bool = True

    scale_range: Tuple[float, float] = (0.8, 1.2)
    allow_horizontal_flip: bool = False
    blend_mode: str = "alpha"              # alpha | replace | screen

    # v1 photometric toggles (kept but default off)
    photometric_on_sign: Dict[str, bool] = None
    photometric_on_background: Dict[str, bool] = None


def _parse_spatial_cfg(cfg: Optional[dict]) -> SpatialConfig:
    """Parse YAML-like dict into SpatialConfig with defaults."""
    sc = SpatialConfig()
    if not cfg:
        sc.photometric_on_sign = {}
        sc.photometric_on_background = {}
        return sc

    sign_ops = cfg.get("sign_ops", {}) if isinstance(cfg, dict) else {}
    occ = sign_ops.get("occlusion", {}) if isinstance(sign_ops, dict) else {}
    dep = sign_ops.get("depatch", {}) if isinstance(sign_ops, dict) else {}

    transform = cfg.get("transform", {}) if isinstance(cfg, dict) else {}
    photo = cfg.get("photometric", {}) if isinstance(cfg, dict) else {}

    sc.occlusion_enabled = bool(occ.get("enabled", sc.occlusion_enabled))
    sc.occlusion_mode = str(occ.get("mode", sc.occlusion_mode))
    sc.occlusion_patch_size = int(occ.get("patch_size", sc.occlusion_patch_size))
    sc.occlusion_ratio = float(occ.get("ratio", sc.occlusion_ratio))
    sc.occlusion_fill = str(occ.get("fill", sc.occlusion_fill))

    sc.depatch_enabled = bool(dep.get("enabled", sc.depatch_enabled))
    grid = dep.get("grid", list(sc.depatch_grid))
    if isinstance(grid, (list, tuple)) and len(grid) == 2:
        sc.depatch_grid = (int(grid[0]), int(grid[1]))
    sc.depatch_scatter_mode = str(dep.get("scatter_mode", sc.depatch_scatter_mode))
    sc.depatch_keep_tile_size = bool(dep.get("keep_tile_size", sc.depatch_keep_tile_size))

    sr = transform.get("scale_range", list(sc.scale_range))
    if isinstance(sr, (list, tuple)) and len(sr) == 2:
        sc.scale_range = (float(sr[0]), float(sr[1]))
    sc.allow_horizontal_flip = bool(transform.get("allow_horizontal_flip", sc.allow_horizontal_flip))
    sc.blend_mode = str(transform.get("blend_mode", sc.blend_mode))

    sc.photometric_on_sign = dict(photo.get("on_sign", {}) or {})
    sc.photometric_on_background = dict(photo.get("on_background", {}) or {})
    return sc


class SpatialComposer:
    """
    SpatialComposer v1 (refactored pipeline).
    Goals:
      - Geometry-safe: alpha/mask/bbox consistency
      - v1 pipeline stages:
          sign_ops (occlusion/depatch) -> transform (flip/scale/pos/blend) -> photometric (disabled by default)
      - By default (all disabled), behavior matches a simple alpha blend compositor.
      - All sign_ops are applied ONLY within sign alpha mask region.

    Notes:
      - output_size is the final canvas size (W,H), usually aligned with your bg input size.
      - This module does not do temporal logic; it composes one frame at a time.
    """

    def __init__(
        self,
        output_size: Optional[Tuple[int, int]] = None,
        position_mode: str = "random",
        spatial_cfg: Optional[dict] = None,
        debug: bool = False,
        sign_mask_provider=None,
    ):
        self.output_size = output_size
        self.position_mode = position_mode
        self.cfg = _parse_spatial_cfg(spatial_cfg)
        self.debug = debug

        # 🔒 clip-level cache
        self._clip_param_cache: Dict[str, Dict[str, Any]] = {}
        self._clip_occlusion_cache: Dict[str, List[Tuple[int, int, int, int]]] = {}

        self.mask_provider = sign_mask_provider

        self.layout_policy = {
            "greeting": "center_bottom",
            "question": "right_center",
            "number": "center",
            "family": "left_center",
            "general": "center",
        }

    def _sample_clip_params(self, rng: np.random.Generator) -> Dict[str, Any]:
        lo, hi = self.cfg.scale_range
        return {
            "scale": float(rng.uniform(lo, hi)),
            "hflip": self.cfg.allow_horizontal_flip and (rng.random() > 0.5),
            # v1: position jitter 也在这里固定
            "pos_jitter": (
                int(rng.integers(-3, 4)),
                int(rng.integers(-3, 4)),
            ),
        }
    # Public API
    def compose_frame(
        self,
        bg_frame: np.ndarray,
        sign_frames_info: List[Dict[str, Any]],
        rng: Optional[np.random.Generator] = None,
    ):
        if rng is None:
            rng = np.random.default_rng()

        canvas = self._prepare_bg(bg_frame)
        H, W = canvas.shape[:2]

        masks, bboxes = [], []

        for info in sign_frames_info:
            sign_frame = info.get("frame")
            if sign_frame is None:
                masks.append(np.zeros((H, W), np.uint8))
                bboxes.append((0, 0, 0, 0))
                continue

            sign_id = info.get("sign_id", "default")
            category = info.get("category", "general")

            # 🔒 get or sample clip params
            if sign_id not in self._clip_param_cache:
                self._clip_param_cache[sign_id] = self._sample_clip_params(rng)
            params = self._clip_param_cache[sign_id]

            rgb, alpha = self._prepare_sign(sign_frame)

            # sign_ops: still frame-level (OK)
            rgb, alpha = self._apply_sign_ops(rgb, alpha, rng, sign_id)

            # ✅ transform: clip-level
            rgb, alpha = self._apply_transform(rgb, alpha, params)

            # ✅ position: clip-level
            pos_mode = self.layout_policy.get(category, self.position_mode)
            x, y = self._compute_position(
                rgb.shape[1], rgb.shape[0], W, H, pos_mode
            )
            dx, dy = params["pos_jitter"]
            x, y = x + dx, y + dy

            canvas, mask_i, bbox_i = self._composite_one(canvas, rgb, alpha, x, y)
            masks.append(mask_i)
            bboxes.append(bbox_i)

        return canvas, masks, bboxes
    def compose_single(
        self,
        bg_frame: np.ndarray,
        sign_frame: np.ndarray,
        category: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
        comp, masks, bboxes = self.compose_frame(
            bg_frame,
            [{"frame": sign_frame, "category": category or "general"}],
            rng=rng,
        )
        return comp, masks[0], bboxes[0]

    # Stage 0: background prep
    def _prepare_bg(self, bg_frame: np.ndarray) -> np.ndarray:
        if bg_frame.ndim != 3 or bg_frame.shape[2] != 3:
            raise ValueError(f"bg_frame must be (H,W,3) RGB uint8, got {bg_frame.shape}")

        canvas = bg_frame
        if self.output_size is not None:
            # W_out, H_out = self.output_size
            W_out, H_out = int(self.output_size[0]), int(self.output_size[1])
            if canvas.shape[:2] != (H_out, W_out):
                canvas = cv2.resize(canvas, (W_out, H_out), interpolation=cv2.INTER_LINEAR)
        return canvas.copy()

    # Stage 1: sign prep
    def _prepare_sign(self, sign_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            rgb:   (h,w,3) uint8
            alpha: (h,w)   float32 in [0,1]
        """
        if sign_frame.ndim != 3 or sign_frame.shape[2] not in (3, 4):
            raise ValueError(f"sign frame must be RGB/RGBA, got shape={sign_frame.shape}")

        # ---- Case 1: RGBA provided ----
        if sign_frame.shape[2] == 4:
            rgb = sign_frame[:, :, :3].copy()
            alpha = sign_frame[:, :, 3].astype(np.float32) / 255.0
            alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
            return rgb, alpha

        # ---- Case 2: RGB only: try provider ----
        rgb = sign_frame[:, :, :3].copy()

        if self.mask_provider is not None:
            try:
                mask01, info = self.mask_provider.get_mask(rgb)  # (h,w) {0,1} or maybe other size

                # Safety: resize if mismatch (YOLO can return model-native size)
                if mask01.shape != rgb.shape[:2]:
                    mask01 = cv2.resize(
                        mask01.astype(np.uint8),
                        (rgb.shape[1], rgb.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )

                alpha = mask01.astype(np.float32)  # {0,1}
                alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
                return rgb, alpha

            except Exception as e:
                # Provider failed -> fallback full alpha
                if self.debug:
                    print(f"[SpatialComposer] mask_provider failed, fallback full alpha. err={e}")

        # ---- Case 3: fallback ----
        h, w = rgb.shape[:2]
        alpha = np.ones((h, w), dtype=np.float32)
        return rgb, alpha

    # Stage 2: sign_ops
    def _apply_sign_ops(
            self,
            rgb: np.ndarray,
            alpha: np.ndarray,
            rng: np.random.Generator,
            sign_id: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:

        """
        Apply occlusion/depatch ONLY within alpha>0 region.
        Both rgb and alpha may be modified.
        """
        if self.cfg.occlusion_enabled:
            # rgb, alpha = self._apply_occlusion(rgb, alpha, rng)
            rgb, alpha = self._apply_occlusion(rgb, alpha, rng, sign_id)
        if self.cfg.depatch_enabled:
            rgb, alpha = self._apply_depatch(rgb, alpha, rng)
        return rgb, alpha

    def _apply_occlusion(
            self,
            rgb: np.ndarray,
            alpha: np.ndarray,
            rng: np.random.Generator,
            sign_id: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:

        mode = self.cfg.occlusion_mode
        patch = max(int(self.cfg.occlusion_patch_size), 1)
        ratio = float(self.cfg.occlusion_ratio)
        fill = self.cfg.occlusion_fill

        h, w = alpha.shape
        mask = (alpha > 0.0).astype(np.uint8)

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return rgb, alpha

        x1, x2 = xs.min(), xs.max() + 1
        y1, y2 = ys.min(), ys.max() + 1

        bbox_w = x2 - x1
        bbox_h = y2 - y1

        area = bbox_w * bbox_h
        patch_area = patch * patch

        K = int(np.ceil((ratio * area) / max(1, patch_area)))
        K = max(K, 1)

        # -------------------------
        # 🔒 clip-level cache
        # -------------------------
        if sign_id is not None:

            if sign_id not in self._clip_occlusion_cache:

                patches = []

                for _ in range(K):
                    px = int(rng.integers(x1, x2))
                    py = int(rng.integers(y1, y2))

                    x0 = int(np.clip(px - patch // 2, 0, w - 1))
                    y0 = int(np.clip(py - patch // 2, 0, h - 1))

                    x1p = min(w, x0 + patch)
                    y1p = min(h, y0 + patch)

                    patches.append((x0, y0, x1p, y1p))

                self._clip_occlusion_cache[sign_id] = patches

            patches = self._clip_occlusion_cache[sign_id]

        else:
            patches = []

        rgb2 = rgb.copy()
        alpha2 = alpha.copy()

        for (x0, y0, x1p, y1p) in patches:

            local = alpha2[y0:y1p, x0:x1p] > 0

            if not np.any(local):
                continue

            if fill == "zero":
                rgb2[y0:y1p, x0:x1p][local] = 0

            elif fill == "mean":
                sign_pix = rgb2[mask.astype(bool)]
                mean_rgb = sign_pix.mean(axis=0).astype(np.float32)
                rgb2[y0:y1p, x0:x1p][local] = mean_rgb

            elif fill == "random":
                noise = rng.integers(
                    0, 256,
                    size=(y1p - y0, x1p - x0, 3),
                    dtype=np.uint8
                )
                rgb2[y0:y1p, x0:x1p][local] = noise[local]

            alpha2[y0:y1p, x0:x1p][local] = 0.0

        return rgb2, alpha2

    def _apply_depatch(
        self,
        rgb: np.ndarray,
        alpha: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Depatch v1:
          - split sign bbox into grid tiles
          - scatter tiles (random permutation) within the same bbox
          - apply to BOTH rgb and alpha
          - operate only on sign bbox (derived from alpha>0)

        This is a "hard" spatial scrambling but keeps global bbox location.
        """
        rows, cols = self.cfg.depatch_grid
        rows = max(1, int(rows))
        cols = max(1, int(cols))

        mask = (alpha > 0.0).astype(np.uint8)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return rgb, alpha

        x1, x2 = xs.min(), xs.max() + 1
        y1, y2 = ys.min(), ys.max() + 1

        bbox_w = x2 - x1
        bbox_h = y2 - y1
        if bbox_w < cols or bbox_h < rows:
            # too small to grid
            return rgb, alpha

        # compute tile sizes (floor), last tile takes remainder
        xsplits = [x1]
        for c in range(1, cols):
            xsplits.append(x1 + (bbox_w * c) // cols)
        xsplits.append(x2)

        ysplits = [y1]
        for r in range(1, rows):
            ysplits.append(y1 + (bbox_h * r) // rows)
        ysplits.append(y2)

        # extract tiles
        tiles_rgb = []
        tiles_a = []
        coords = []
        for r in range(rows):
            for c in range(cols):
                xa0, xa1 = xsplits[c], xsplits[c + 1]
                ya0, ya1 = ysplits[r], ysplits[r + 1]
                coords.append((xa0, ya0, xa1, ya1))
                tiles_rgb.append(rgb[ya0:ya1, xa0:xa1].copy())
                tiles_a.append(alpha[ya0:ya1, xa0:xa1].copy())

        n = len(coords)
        if n <= 1:
            return rgb, alpha

        idxs = np.arange(n)
        if self.cfg.depatch_scatter_mode in ("random", "permute"):
            rng.shuffle(idxs)
        else:
            rng.shuffle(idxs)

        rgb2 = rgb.copy()
        alpha2 = alpha.copy()

        # paste tiles back in permuted order
        for dst_i, src_i in enumerate(idxs):
            xa0, ya0, xa1, ya1 = coords[dst_i]
            t_rgb = tiles_rgb[src_i]
            t_a = tiles_a[src_i]

            # keep_tile_size: tiles already match dst size by construction
            # Still, guard shape mismatches (due to rounding)
            dh, dw = (ya1 - ya0), (xa1 - xa0)
            if t_rgb.shape[0] != dh or t_rgb.shape[1] != dw:
                t_rgb = cv2.resize(t_rgb, (dw, dh), interpolation=cv2.INTER_LINEAR)
                t_a = cv2.resize(t_a, (dw, dh), interpolation=cv2.INTER_NEAREST)

            rgb2[ya0:ya1, xa0:xa1] = t_rgb
            alpha2[ya0:ya1, xa0:xa1] = t_a

        # v1: outside sign bbox unchanged; bbox stays same.
        # Note: depatch may move alpha holes around; that's intended.
        alpha2 = np.clip(alpha2, 0.0, 1.0).astype(np.float32)
        return rgb2, alpha2

    # Stage 3: transform (flip/scale)
    def _apply_transform(
        self,
        rgb: np.ndarray,
        alpha: np.ndarray,
        params: Dict[str, Any],
    ):
        if params["hflip"]:
            rgb = cv2.flip(rgb, 1)
            alpha = cv2.flip(alpha, 1)

        scale = params["scale"]
        h0, w0 = rgb.shape[:2]
        h1, w1 = int(round(h0 * scale)), int(round(w0 * scale))

        if h1 > 0 and w1 > 0 and (h1, w1) != (h0, w0):
            rgb = cv2.resize(rgb, (w1, h1), interpolation=cv2.INTER_LINEAR)
            alpha = cv2.resize(alpha, (w1, h1), interpolation=cv2.INTER_NEAREST)

        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
        return rgb, alpha

    # Stage 4: compositing
    def _composite_one(
        self,
        canvas: np.ndarray,
        rgb: np.ndarray,
        alpha: np.ndarray,
        x: int,
        y: int,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
        """
        Composite sign rgb/alpha onto canvas at top-left (x,y),
        including crop-to-canvas logic. Returns updated canvas,
        per-sign mask in canvas coords, and bbox from mask>0.
        """
        H, W = canvas.shape[:2]
        h1, w1 = rgb.shape[:2]

        # crop to canvas
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(W, x + w1)
        y1 = min(H, y + h1)

        sx0 = x0 - x
        sy0 = y0 - y
        sx1 = sx0 + (x1 - x0)
        sy1 = sy0 + (y1 - y0)

        if x1 <= x0 or y1 <= y0:
            mask = np.zeros((H, W), dtype=np.uint8)
            return canvas, mask, (0, 0, 0, 0)

        rgb_crop = rgb[sy0:sy1, sx0:sx1].astype(np.float32)
        alpha_crop = alpha[sy0:sy1, sx0:sx1].astype(np.float32)
        a = alpha_crop[:, :, None]

        bg_patch = canvas[y0:y1, x0:x1].astype(np.float32)

        if self.cfg.blend_mode == "alpha":
            blended = rgb_crop * a + bg_patch * (1.0 - a)
        elif self.cfg.blend_mode == "replace":
            blended = np.where(a > 0.5, rgb_crop, bg_patch)
        elif self.cfg.blend_mode == "screen":
            blended = 1.0 - (1.0 - rgb_crop / 255.0) * (1.0 - bg_patch / 255.0)
            blended *= 255.0
        else:
            blended = rgb_crop * a + bg_patch * (1.0 - a)

        out = canvas.copy()
        out[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)

        # Build per-sign mask (canvas coords)
        mask = np.zeros((H, W), dtype=np.uint8)
        mask_patch = (alpha_crop * 255.0).astype(np.uint8)
        mask[y0:y1, x0:x1] = mask_patch

        bbox = self._bbox_from_mask(mask)
        return out, mask, bbox

    def _bbox_from_mask(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return (0, 0, 0, 0)
        x_min, x_max = int(xs.min()), int(xs.max()) + 1
        y_min, y_max = int(ys.min()), int(ys.max()) + 1
        return (x_min, y_min, x_max, y_max)


    # Positioning
    def _compute_position(self, w, h, W, H, mode):
        if mode == "center_bottom":
            return (W - w) // 2, H - h - 20
        if mode == "center":
            return (W - w) // 2, (H - h) // 2
        if mode == "right_center":
            return W - w - 20, (H - h) // 2
        if mode == "left_center":
            return 20, (H - h) // 2

        return (
            (W - w) // 2,
            (H - h) // 2,
        )

    def clear_caches(self):
        self._clip_param_cache.clear()
        self._clip_occlusion_cache.clear()


# TEST
if __name__ == "__main__":
    print("=== SpatialComposer v1 Pipeline Test ===")
    rng = np.random.default_rng(42)

    H, W = 240, 320
    bg = np.zeros((H, W, 3), dtype=np.uint8)
    bg[:] = 30

    # sign 1: red circle RGBA
    sign1 = np.zeros((80, 80, 4), dtype=np.uint8)
    cv2.circle(sign1, (40, 40), 30, (255, 0, 0, 255), -1)

    # sign 2: green rect RGBA
    sign2 = np.zeros((60, 60, 4), dtype=np.uint8)
    cv2.rectangle(sign2, (10, 10), (50, 50), (0, 255, 0, 255), -1)

    # ---------------- Baseline (all ops disabled) ----------------
    print("\n--- Baseline (no sign_ops) ---")
    composer = SpatialComposer(
        output_size=(W, H),
        position_mode="random",
        spatial_cfg={
            "sign_ops": {
                "occlusion": {"enabled": False},
                "depatch": {"enabled": False},
            },
            "transform": {
                "scale_range": [0.8, 1.2],
                "allow_horizontal_flip": False,
                "blend_mode": "alpha",
            },
            "photometric": {
                "on_sign": {},
                "on_background": {},
            }
        },
        debug=True,
    )

    comp, mask, bbox = composer.compose_single(bg, sign1, "greeting", rng)
    print("Composite:", comp.shape, "BBox:", bbox, "Mask unique:", np.unique(mask))

    # mask -> bbox consistency
    ys, xs = np.where(mask > 0)
    assert len(xs) > 0, "Mask is empty in baseline"
    mx1, mx2 = xs.min(), xs.max() + 1
    my1, my2 = ys.min(), ys.max() + 1
    assert bbox == (mx1, my1, mx2, my2), f"Baseline bbox mismatch: {bbox} vs {(mx1,my1,mx2,my2)}"
    print("✔ baseline mask-bbox consistency")

    # ---------------- Occlusion only ----------------
    print("\n--- Occlusion-only ---")
    composer_occ = SpatialComposer(
        output_size=(W, H),
        position_mode="center",
        spatial_cfg={
            "sign_ops": {
                "occlusion": {
                    "enabled": True,
                    "mode": "patch",
                    "patch_size": 16,
                    "ratio": 0.25,
                    "fill": "zero",
                },
                "depatch": {"enabled": False},
            },
            "transform": {
                "scale_range": [1.0, 1.0],
                "allow_horizontal_flip": False,
                "blend_mode": "alpha",
            },
            "photometric": {"on_sign": {}, "on_background": {}}
        },
    )

    comp2, mask2, bbox2 = composer_occ.compose_single(bg, sign1, "general", rng)
    ys2, xs2 = np.where(mask2 > 0)
    assert len(xs2) > 0, "Mask became empty after occlusion (too aggressive?)"
    mx1, mx2 = xs2.min(), xs2.max() + 1
    my1, my2 = ys2.min(), ys2.max() + 1
    assert bbox2 == (mx1, my1, mx2, my2), f"Occlusion bbox mismatch: {bbox2} vs {(mx1,my1,mx2,my2)}"
    print("✔ occlusion mask-bbox consistency")

    # ---------------- Depatch only ----------------
    print("\n--- Depatch-only ---")
    composer_dep = SpatialComposer(
        output_size=(W, H),
        position_mode="center",
        spatial_cfg={
            "sign_ops": {
                "occlusion": {"enabled": False},
                "depatch": {
                    "enabled": True,
                    "grid": [3, 4],
                    "scatter_mode": "random",
                    "keep_tile_size": True,
                },
            },
            "transform": {
                "scale_range": [1.0, 1.0],
                "allow_horizontal_flip": False,
                "blend_mode": "alpha",
            },
            "photometric": {"on_sign": {}, "on_background": {}}
        },
    )

    comp3, mask3, bbox3 = composer_dep.compose_single(bg, sign1, "general", rng)
    ys3, xs3 = np.where(mask3 > 0)
    assert len(xs3) > 0, "Mask became empty after depatch (unexpected)"
    mx1, mx2 = xs3.min(), xs3.max() + 1
    my1, my2 = ys3.min(), ys3.max() + 1
    assert bbox3 == (mx1, my1, mx2, my2), f"Depatch bbox mismatch: {bbox3} vs {(mx1,my1,mx2,my2)}"
    print("✔ depatch mask-bbox consistency")

    # ---------------- Multiple signs geometry ----------------
    print("\n--- Multiple signs ---")
    frames = [
        {"frame": sign1, "category": "greeting"},
        {"frame": sign2, "category": "question"},
    ]
    comp4, masks, bboxes = composer.compose_frame(bg, frames, rng)

    assert len(masks) == 2 and len(bboxes) == 2
    for i in range(2):
        m = masks[i]
        bb = bboxes[i]
        ys, xs = np.where(m > 0)
        if len(xs) == 0:
            print(f"Sign {i} fully offscreen (allowed).")
            continue
        mx1, mx2 = xs.min(), xs.max() + 1
        my1, my2 = ys.min(), ys.max() + 1
        assert bb == (mx1, my1, mx2, my2), f"Multi sign bbox mismatch at {i}"
    print("✔ multiple signs mask-bbox consistency")

    # ---------------- Semantic placement sanity ----------------
    print("\n--- Semantic placement sanity (greeting near bottom) ---")
    composer_place = SpatialComposer(
        output_size=(W, H),
        position_mode="random",
        spatial_cfg={"transform": {"scale_range": [1.0, 1.0]}},
    )
    comp5, mask5, bbox5 = composer_place.compose_single(bg, sign1, "greeting", rng)
    x1, y1, x2, y2 = bbox5
    assert y2 > H * 0.6, "Greeting should be placed near bottom"
    print("✔ placement policy ok")

    print("\nSpatialComposer v1 pipeline test passed ✔")
