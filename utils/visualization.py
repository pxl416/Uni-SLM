# utils/visualization.py
import numpy as np
import cv2
from typing import List
from PIL import Image


def frames_to_pil(frames: np.ndarray) -> List[Image.Image]:
    assert frames.ndim == 4
    return [Image.fromarray(f) for f in frames]


def show_video(frames: np.ndarray, fps: int = 25, win="video"):
    for f in frames:
        cv2.imshow(win, f)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


def overlay_heatmap(
    frame: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    color=(0, 255, 0)
) -> np.ndarray:
    h, w = frame.shape[:2]
    hm = cv2.resize(heatmap, (w, h))
    hm = np.clip(hm, 0, 1)

    overlay = np.zeros_like(frame)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]

    mask = (hm[..., None] * alpha).astype(np.float32)
    out = frame.astype(np.float32) * (1 - mask) + overlay.astype(np.float32) * mask
    return out.astype(np.uint8)


def preview_spatial_prior(
    video_frames: np.ndarray,
    spatial_prior: np.ndarray,
    picks=(0, -1)
):
    T = video_frames.shape[0]
    for idx in picks:
        t = idx if idx >= 0 else T + idx
        frame = video_frames[t]
        prior = spatial_prior[t]
        out = overlay_heatmap(frame, prior)
        cv2.imshow(f"t={t}", out)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_heatmap(
    heatmap: np.ndarray,
    cmap: str = "jet",
    title: str = "Spatial prior",
    save_path: str = None,
):
    """
    heatmap: (H, W), values in [0,1]
    """
    import matplotlib.pyplot as plt

    assert heatmap.ndim == 2

    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap=cmap, vmin=0.0, vmax=1.0)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()


def show_frame_with_prior(
    frame: np.ndarray,
    prior: np.ndarray,
    alpha: float = 0.5,
    title: str = "",
    save_path: str = None,
):
    """
    frame: (H,W,3) uint8
    prior: (H,W) float in [0,1]
    """
    import matplotlib.pyplot as plt

    assert frame.shape[:2] == prior.shape

    plt.figure(figsize=(6, 6))
    plt.imshow(frame)
    plt.imshow(prior, cmap="jet", alpha=alpha, vmin=0.0, vmax=1.0)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()
