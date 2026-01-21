# synthetic_world/spatial_pipeline.py
from __future__ import annotations

from typing import Dict, Any, List, Optional
import numpy as np

from synthetic_world.sign_mask import SignMaskProvider, MaskConfig
from synthetic_world.spatial_composer import SpatialComposer


class SpatialPipeline:
    """
    Assemble ONE spatial pretraining sample.

    Input:
        bg_frame: (H, W, 3) uint8
        sign_frames: list of (H, W, 3) uint8

    Output:
        dict with:
            composite_rgb: (H, W, 3)
            spatial_masks: list[(H, W)] uint8 {0..255}
            bboxes: list[(x1,y1,x2,y2)]
            meta: dict
    """

    def __init__(
        self,
        *,
        mask_cfg: MaskConfig,
        spatial_cfg: Dict[str, Any],
        output_size: Optional[tuple[int, int]] = None,
        position_mode: str = "random",
    ):
        # 1. sign mask provider (SAM / threshold / rgba)
        self.mask_provider = SignMaskProvider(mask_cfg)

        # 2. spatial composer
        self.spatial_composer = SpatialComposer(
            output_size=output_size,
            position_mode=position_mode,
            spatial_cfg=spatial_cfg,
        )

    def run(
        self,
        *,
        bg_frame: np.ndarray,
        sign_frames: List[np.ndarray],
        categories: Optional[List[str]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            bg_frame: (H,W,3) RGB uint8
            sign_frames: list of (H,W,3) RGB sign frames
            categories: optional semantic categories (len == len(sign_frames))
            rng: clip-level RNG
        """
        if rng is None:
            rng = np.random.default_rng()

        if categories is None:
            categories = ["general"] * len(sign_frames)

        assert len(categories) == len(sign_frames), \
            "categories must have same length as sign_frames"

        H, W = bg_frame.shape[:2]

        sign_frames_info = []
        mask_infos = []

        # -------------------------------------------------
        # 1) mask / segmentation stage (SAM or fallback)
        # -------------------------------------------------
        for frame, cat in zip(sign_frames, categories):
            mask01, info = self.mask_provider.get_mask(frame)

            # build RGBA for SpatialComposer
            rgba = np.zeros((H, W, 4), dtype=np.uint8)
            rgba[..., :3] = frame
            rgba[..., 3] = (mask01 * 255).astype(np.uint8)

            sign_frames_info.append({
                "frame": rgba,
                "category": cat,
            })
            mask_infos.append(info)

        # -------------------------------------------------
        # 2) spatial composition
        # -------------------------------------------------
        composite, masks, bboxes = self.spatial_composer.compose_frame(
            bg_frame=bg_frame,
            sign_frames_info=sign_frames_info,
            rng=rng,
        )

        return {
            "composite_rgb": composite,
            "spatial_masks": masks,
            "bboxes": bboxes,
            "meta": {
                "mask_method": self.mask_provider.cfg.method,
                "mask_infos": mask_infos,
            }
        }


# -------------------------------------------------
# Minimal test
# -------------------------------------------------
if __name__ == "__main__":
    print("=== SpatialPipeline test ===")

    H, W = 240, 320
    bg = np.full((H, W, 3), 120, np.uint8)

    sign = np.zeros((H, W, 3), np.uint8)
    sign[60:200, 120:240] = 220

    mask_cfg = MaskConfig(
        method="threshold",
        threshold_gray=50,
    )

    spatial_cfg = {
        "sign_ops": {
            "occlusion": {"enabled": True, "ratio": 0.25},
            "depatch": {"enabled": False},
        },
        "transform": {
            "scale_range": [1.0, 1.0],
            "allow_horizontal_flip": False,
            "blend_mode": "alpha",
        },
    }

    pipe = SpatialPipeline(
        mask_cfg=mask_cfg,
        spatial_cfg=spatial_cfg,
        output_size=(W, H),
        position_mode="center",
    )

    out = pipe.run(
        bg_frame=bg,
        sign_frames=[sign],
    )

    print(
        "composite:", out["composite_rgb"].shape,
        "mask_sum:", int(out["spatial_masks"][0].sum()),
        "bboxes:", out["bboxes"],
    )
    print("SpatialPipeline test passed ✔")
