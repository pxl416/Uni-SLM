# synthetic_world/spatial_composer.py
from __future__ import annotations
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import cv2


class SpatialComposer:
    """
    Geometry-safe spatial composer.
    Guarantees mask, bbox, alpha consistency.
    """

    def __init__(
        self,
        output_size: Tuple[int, int] = (224, 224),   # (W,H)
        position_mode: str = "random",             # random | center_bottom | grid
        scale_range: Tuple[float, float] = (0.8, 1.2),
        allow_flip: bool = False,
        blend_mode: str = "alpha",                  # alpha | replace | screen
    ):
        self.output_size = output_size
        self.position_mode = position_mode
        self.scale_range = scale_range
        self.allow_flip = allow_flip
        self.blend_mode = blend_mode

        self.layout_policy = {
            "greeting": "center_bottom",
            "question": "right_center",
            "number": "center",
            "family": "left_center",
            "general": "center",
        }

    # ----------------------------------------------------

    def compose_frame(
        self,
        bg_frame: np.ndarray,
        sign_frames_info: List[Dict[str, Any]],
        rng: Optional[np.random.Generator] = None,
    ):
        if rng is None:
            rng = np.random.default_rng()

        W, H = self.output_size
        if bg_frame.shape[:2] != (H, W):
            bg_frame = cv2.resize(bg_frame, (W, H))

        canvas = bg_frame.copy()
        masks = []
        bboxes = []

        for info in sign_frames_info:
            sign = info["frame"]
            category = info.get("category", "general")

            # -------------------------
            # 1. Flip
            if self.allow_flip and rng.random() > 0.5:
                sign = cv2.flip(sign, 1)

            # -------------------------
            # 2. Resize
            scale = rng.uniform(*self.scale_range)
            h0, w0 = sign.shape[:2]
            h1, w1 = int(h0 * scale), int(w0 * scale)
            if h1 <= 0 or w1 <= 0:
                masks.append(np.zeros((H, W), np.uint8))
                bboxes.append((0, 0, 0, 0))
                continue
            sign = cv2.resize(sign, (w1, h1))

            # -------------------------
            # 3. Alpha + RGB
            if sign.shape[2] == 4:
                rgb = sign[:, :, :3]
                alpha = sign[:, :, 3].astype(np.float32) / 255.0
            else:
                rgb = sign
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                _, a = cv2.threshold(gray, 8, 255, cv2.THRESH_BINARY)
                alpha = a.astype(np.float32) / 255.0

            # -------------------------
            # 4. Position
            pos_mode = self.layout_policy.get(category, self.position_mode)
            x, y = self._compute_position(w1, h1, W, H, pos_mode, rng)

            # -------------------------
            # 5. Crop to canvas
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(W, x + w1)
            y1 = min(H, y + h1)

            sx0 = x0 - x
            sy0 = y0 - y
            sx1 = sx0 + (x1 - x0)
            sy1 = sy0 + (y1 - y0)

            if x1 <= x0 or y1 <= y0:
                masks.append(np.zeros((H, W), np.uint8))
                bboxes.append((0, 0, 0, 0))
                continue

            rgb_crop = rgb[sy0:sy1, sx0:sx1]
            alpha_crop = alpha[sy0:sy1, sx0:sx1]

            # -------------------------
            # 6. Blend
            bg_patch = canvas[y0:y1, x0:x1].astype(np.float32)
            fg = rgb_crop.astype(np.float32)
            a = alpha_crop[:, :, None]

            if self.blend_mode == "alpha":
                blended = fg * a + bg_patch * (1 - a)
            elif self.blend_mode == "replace":
                blended = np.where(a > 0.5, fg, bg_patch)
            elif self.blend_mode == "screen":
                blended = 1 - (1 - fg/255) * (1 - bg_patch/255)
                blended *= 255
            else:
                blended = fg * a + bg_patch * (1 - a)

            canvas[y0:y1, x0:x1] = blended.astype(np.uint8)

            # -------------------------
            # 7. Build mask
            mask = np.zeros((H, W), dtype=np.uint8)
            mask[y0:y1, x0:x1] = (alpha_crop * 255).astype(np.uint8)

            # -------------------------
            # 8. Compute bbox from mask
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                bboxes.append((0, 0, 0, 0))
            else:
                x_min, x_max = xs.min(), xs.max() + 1
                y_min, y_max = ys.min(), ys.max() + 1
                bboxes.append((x_min, y_min, x_max, y_max))

            masks.append(mask)

        return canvas, masks, bboxes

    # ----------------------------------------------------

    def _compute_position(self, w, h, W, H, mode, rng):
        if mode == "center_bottom":
            return (W - w) // 2, H - h - 20
        if mode == "center":
            return (W - w) // 2, (H - h) // 2
        if mode == "right_center":
            return W - w - 20, (H - h) // 2
        if mode == "left_center":
            return 20, (H - h) // 2

        # random
        return (
            int(rng.integers(0, max(1, W - w))),
            int(rng.integers(0, max(1, H - h))),
        )

    # ----------------------------------------------------

    def compose_single(self, bg_frame, sign_frame, category=None, rng=None):
        comp, masks, bboxes = self.compose_frame(
            bg_frame,
            [{"frame": sign_frame, "category": category}],
            rng
        )
        return comp, masks[0], bboxes[0]


class SLRGBOcclusion:
    pass

class SLRGBDepatch:
    pass

class SLRGBPixel:
    pass

def figure_segment(mode='SAM'):
    pass


# ------------------- TEST -------------------
if __name__ == "__main__":
    print("=== SpatialComposer Geometry Test ===")

    rng = np.random.default_rng(42)

    H, W = 240, 320
    bg = np.zeros((H, W, 3), dtype=np.uint8)
    bg[:] = 30

    # 构造一个带 alpha 的 sign
    sign = np.zeros((80, 80, 4), dtype=np.uint8)
    cv2.circle(sign, (40, 40), 30, (255, 0, 0, 255), -1)

    composer = SpatialComposer(
        output_size=(W, H),
        position_mode="random",
        scale_range=(0.8, 1.2),
        allow_flip=False,
        blend_mode="alpha"
    )

    # ---------------- Single sign ----------------
    print("\n--- Single Sign Test ---")
    comp, mask, bbox = composer.compose_single(bg, sign, "greeting", rng)

    print("Composite:", comp.shape)
    print("Mask unique:", np.unique(mask))
    print("BBox:", bbox)

    x1, y1, x2, y2 = bbox

    # mask → bbox consistency
    ys, xs = np.where(mask > 0)
    assert len(xs) > 0, "Mask is empty"

    mx1, mx2 = xs.min(), xs.max() + 1
    my1, my2 = ys.min(), ys.max() + 1

    print("Mask bbox:", (mx1, my1, mx2, my2))

    assert (x1, y1, x2, y2) == (mx1, my1, mx2, my2), \
        f"BBox mismatch: bbox={bbox} mask={mx1,my1,mx2,my2}"

    print("✔ mask-bbox geometry correct")

    # ---------------- Multiple signs ----------------
    print("\n--- Multiple Sign Test ---")

    sign2 = np.zeros((60, 60, 4), dtype=np.uint8)
    cv2.rectangle(sign2, (10, 10), (50, 50), (0, 255, 0, 255), -1)

    frames = [
        {"frame": sign, "category": "greeting"},
        {"frame": sign2, "category": "question"},
    ]

    comp, masks, bboxes = composer.compose_frame(bg, frames, rng)

    print("Num masks:", len(masks))
    print("Num bboxes:", len(bboxes))

    for i in range(2):
        mask = masks[i]
        bbox = bboxes[i]

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            print(f"Sign {i} fully offscreen")
            continue

        mx1, mx2 = xs.min(), xs.max() + 1
        my1, my2 = ys.min(), ys.max() + 1

        print(f"Sign {i} bbox={bbox} mask_bbox={(mx1,my1,mx2,my2)}")

        assert bbox == (mx1, my1, mx2, my2), f"Sign {i} bbox mismatch"

    print("✔ multiple sign geometry correct")

    # ---------------- Semantic placement ----------------
    print("\n--- Semantic Placement Test ---")

    composer2 = SpatialComposer(
        output_size=(W, H),
        position_mode="grid"
    )

    comp, mask, bbox = composer2.compose_single(bg, sign, "greeting", rng)

    x1, y1, x2, y2 = bbox
    print("Greeting bbox:", bbox)

    # greeting 应该靠近底部
    assert y2 > H * 0.6, "Greeting should be placed near bottom"

    print("✔ semantic placement correct")

    print("\nSpatialComposer is geometry-safe ✔")

