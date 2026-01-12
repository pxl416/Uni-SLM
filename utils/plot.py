# utils/plot.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, math
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw

# Optional wandb
try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None
    _WANDB_AVAILABLE = False


# ------------------------- ASCII SAFE -------------------------
def ascii_safe(s: Any) -> str:
    """Remove all non-ASCII chars to avoid font missing glyph warnings."""
    if not isinstance(s, str):
        s = str(s)
    return s.encode("ascii", errors="ignore").decode("ascii")


def safe_title(t): return ascii_safe(t)
def safe_label(t): return ascii_safe(t)
def safe_text(t): return ascii_safe(t)
def safe_key(t): return ascii_safe(t)


# ------------------------- Skeleton Topology -------------------------
DEFAULT_TOPOLOGY_21 = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]

# utils/plots.py
import numpy as np
import matplotlib.pyplot as plt


def plot_temporal_label(
    label: np.ndarray,
    title: str = "Temporal Supervision",
    save_path: str = None
):
    label = np.asarray(label).astype(np.float32)

    plt.figure(figsize=(10, 2))
    plt.plot(label, linewidth=2)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Frame index")
    plt.ylabel("Sign (0/1)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=160)
    else:
        plt.show()

    plt.close()
def save_temporal_timeline(
    label: np.ndarray,
    title: str = "Temporal supervision",
    save_path: str = None,
):
    """
    label: (T,) array, values in {0,1} or [0,1]
    """
    import matplotlib.pyplot as plt

    label = label.astype(float)
    T = len(label)

    plt.figure(figsize=(max(6, T / 50), 2))
    plt.plot(label, linewidth=2)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Frame index")
    plt.ylabel("Sign probability")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=160)
        plt.close()
    else:
        plt.show()


# ------------------------- Skeleton Drawing -------------------------
def draw_skeleton_on_pil(
    img: Image.Image,
    kpts: np.ndarray,
    topology: Optional[List[Tuple[int,int]]] = None,
    kpt_coord_mode: str = "pixel",
    radius: int = 2,
    line_width: int = 2,
) -> Image.Image:
    if topology is None:
        topology = DEFAULT_TOPOLOGY_21

    img = img.copy()
    W, H = img.size
    draw = ImageDraw.Draw(img)

    if kpts is None or kpts.size == 0:
        return img
    kpts = np.asarray(kpts, dtype=np.float32)

    def _xy(j):
        x, y = float(kpts[j, 0]), float(kpts[j, 1])
        if kpt_coord_mode == "normalized":
            x, y = x * W, y * H
        return x, y

    for a, b in topology:
        if a < kpts.shape[0] and b < kpts.shape[0]:
            xa, ya = _xy(a); xb, yb = _xy(b)
            draw.line((xa, ya, xb, yb), width=line_width, fill=(0,255,0))

    for j in range(min(kpts.shape[0], 128)):
        x, y = _xy(j)
        draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                     outline=(255,0,0), width=line_width)
    return img


# ------------------------- Keypoint Grid -------------------------
def save_keypoint_overlay_grid(
    frames_pil: List[Image.Image],
    kpts_seq: np.ndarray,
    picks: Optional[List[int]] = None,
    kpt_coord_mode: str = "pixel",
    topology: Optional[List[Tuple[int,int]]] = None,
    max_cols: int = 8,
    save_path: Optional[str] = None,
    log_key: Optional[str] = None,
):
    T = len(frames_pil)
    if picks is None:
        picks = [0, T//3, 2*T//3, T-1] if T >= 4 else list(range(T))

    vis_frames = []
    for t in picks:
        t = int(max(0, min(T-1, t)))
        kp = kpts_seq[t] if kpts_seq is not None and len(kpts_seq) > t else None
        vis_frames.append(draw_skeleton_on_pil(frames_pil[t], kp,
                                               topology=topology,
                                               kpt_coord_mode=kpt_coord_mode))

    show_clip_grid(
        vis_frames,
        title="keypoints overlay",
        max_cols=max_cols,
        save_path=save_path,
        log_key=safe_key(log_key or "debug_keypoints_overlay")
    )


# ------------------------- Timeline -------------------------
def save_timeline(
    starts: List[int],
    ends: List[int],
    T_total: int,
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    log_key: Optional[str] = None,
):
    starts = list(map(int, starts))
    ends   = list(map(int, ends))
    K = len(starts)

    plt.figure(figsize=(max(6, T_total/80), 1.6 + 0.2*K))

    plt.plot([0, T_total-1], [0, 0], linewidth=6)

    for i, (s, e) in enumerate(zip(starts, ends)):
        s = max(0, min(T_total-1, s))
        e = max(0, min(T_total-1, e))
        if e < s: s, e = e, s

        plt.fill_between([s, e], [-0.15, -0.15], [0.15, 0.15],
                         alpha=0.4, step="pre", label=f"seg{i}")
        plt.vlines([s, e], -0.3, 0.3)

        if labels and i < len(labels) and labels[i]:
            txt = safe_text(labels[i])
            mid = (s + e) / 2.0
            plt.text(mid, 0.35 + 0.18*(i%3), txt, ha="center", fontsize=8)

    plt.yticks([])
    plt.xlim(-1, T_total)
    plt.xlabel("time (frame index)")
    plt.tight_layout()

    if save_path:
        _ensure_dir_for(save_path)
        plt.savefig(save_path, dpi=160)

    _maybe_wandb_log({
        safe_key(log_key or "debug_timeline"):
            wandb.Image(plt.gcf()) if _WANDB_AVAILABLE else (save_path or "timeline")
    })

    plt.close()


# ------------------------- Attention Mask -------------------------
def save_attn_mask(
    attn_mask: np.ndarray,
    save_path: Optional[str] = None,
    log_key: Optional[str] = None,
):
    T = int(attn_mask.shape[0])
    y = (attn_mask.astype(np.int32) * 1)

    plt.figure(figsize=(max(6, T/80), 1.6))
    plt.bar(np.arange(T), y)
    plt.yticks([0,1], ["pad", "valid"])
    plt.xlabel("time")
    plt.tight_layout()

    if save_path:
        _ensure_dir_for(save_path)
        plt.savefig(save_path, dpi=160)

    _maybe_wandb_log({
        safe_key(log_key or "debug_attn_mask"):
            wandb.Image(plt.gcf()) if _WANDB_AVAILABLE else (save_path or "attn_mask")
    })
    plt.close()


# ------------------------- WandB Helper -------------------------
def _maybe_wandb_log(data: Dict[str, Any]):
    if _WANDB_AVAILABLE and getattr(wandb, "run", None) is not None:
        # Ensure keys/values are ASCII-safe
        safe_data = {}
        for k, v in data.items():
            safe_data[safe_key(k)] = v
        wandb.log(safe_data)


# ------------------------- Utility -------------------------
def _ensure_dir_for(path: Optional[str]):
    if path:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)


# ------------------------- Tensor -> PIL -------------------------
def tensor_clip_to_pil_list(
    t: torch.Tensor,
    denorm=None
) -> List[Image.Image]:
    assert t.ndim == 4
    T, C, H, W = t.shape
    imgs = []

    for i in range(T):
        x = t[i]
        if denorm:
            mean, std = denorm
            x = x.clone()
            for c in range(x.shape[0]):
                x[c] = x[c] * std[c] + mean[c]
        x = x.clamp(0,1)
        arr = (x.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        imgs.append(Image.fromarray(arr if C>1 else arr[...,0], mode="RGB" if C>1 else "L"))
    return imgs


# ------------------------- Retrieval Heatmap -------------------------
def save_retrieval_vis(
    sim_matrix: np.ndarray,
    save_path: str = "retrieval_vis.png",
    log_key: str = "retrieval/sim_matrix"
):
    plt.figure(figsize=(8, 6))
    plt.imshow(sim_matrix, cmap="viridis")
    plt.colorbar()
    plt.title("Similarity Matrix")
    plt.tight_layout()

    _ensure_dir_for(save_path)
    plt.savefig(save_path, dpi=200)
    plt.close()

    _maybe_wandb_log({
        safe_key(log_key): wandb.Image(save_path) if _WANDB_AVAILABLE else save_path
    })


# ------------------------- Retrieval Curve -------------------------
def plot_retrieval_curves(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    log_key: str = "retrieval/recall_curve"
):
    xs = [1,5,10]
    plt.figure(figsize=(5,4))

    for direction in ["t2v", "v2t"]:
        if direction not in metrics_dict:
            continue
        m = metrics_dict[direction]
        ys = [m.get("R1",0), m.get("R5",0), m.get("R10",0)]
        plt.plot(xs, ys, marker="o", label=direction)

    plt.xlabel("K")
    plt.ylabel("Recall@K")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        _ensure_dir_for(save_path)
        plt.savefig(save_path, dpi=200)

    _maybe_wandb_log({
        safe_key(log_key): wandb.Image(plt.gcf()) if _WANDB_AVAILABLE else "recall_curve"
    })

    plt.close()


# ------------------------- Clip Grid -------------------------
def show_clip_grid(
    frames_pil: List[Image.Image],
    title: str = "",
    max_cols: int = 8,
    step: int = 1,
    save_path: Optional[str] = None,
    log_key: Optional[str] = None,
):
    frames = frames_pil[::max(1, step)]
    n = len(frames)
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)

    w, h = frames[0].size
    fig, axes = plt.subplots(rows, cols, figsize=(cols*(w/100), rows*(h/100)))

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r,c]
            ax.axis("off")
            if idx < n:
                ax.imshow(frames[idx])
                idx += 1

    if title:
        fig.suptitle(safe_title(title))

    plt.tight_layout(rect=[0,0,1,0.97])

    if save_path:
        _ensure_dir_for(save_path)
        fig.savefig(save_path, dpi=150)

    if log_key:
        _maybe_wandb_log({
            safe_key(log_key): wandb.Image(fig) if _WANDB_AVAILABLE else (save_path or "grid")
        })

    plt.close(fig)


# ------------------------- Compare Stages -------------------------
def compare_stages(
    original: List[Image.Image],
    stages: Dict[str, List[Image.Image]],
    picks: Tuple[int,...]=(0,-1),
    save_path: Optional[str]=None,
    log_key: Optional[str]=None,
):
    stage_names = ["original"] + list(stages.keys())
    columns = len(stage_names)

    T = len(original)
    pick_idxs = sorted({
        idx if idx>=0 else T+idx for idx in picks if 0 <= (idx if idx>=0 else T+idx) < T
    })
    rows = len(pick_idxs)

    fig, axes = plt.subplots(rows, columns, figsize=(3.5*columns, 3.5*rows))

    def _get(arr, i):
        return arr[i] if i < len(arr) else arr[-1]

    for r, t_idx in enumerate(pick_idxs):
        axes[r,0].imshow(original[t_idx])
        axes[r,0].set_title(f"t={t_idx}\noriginal")
        axes[r,0].axis("off")

        for c,name in enumerate(stage_names[1:], start=1):
            arr = stages.get(name, [])
            if not arr:
                axes[r,c].axis("off")
                continue
            img = _get(arr, t_idx)
            axes[r,c].imshow(img)
            axes[r,c].set_title(safe_title(name))
            axes[r,c].axis("off")

    plt.tight_layout()

    if save_path:
        _ensure_dir_for(save_path)
        fig.savefig(save_path, dpi=150)

    if log_key:
        _maybe_wandb_log({
            safe_key(log_key): wandb.Image(fig) if _WANDB_AVAILABLE else (save_path or "compare_stages")
        })

    plt.close(fig)


# ------------------------- Mask Overlay -------------------------
def overlay_mask(
    img: Image.Image,
    mask: np.ndarray,
    alpha: float = 0.4,
    color: Tuple[int,int,int]=(0,255,0)
) -> Image.Image:
    base = img.convert("RGBA")
    m = (mask.astype(np.float32)>0).astype(np.uint8)*255
    m_img = Image.fromarray(m, mode="L").resize(base.size, Image.NEAREST)

    overlay = Image.new("RGBA", base.size, color+(0,))
    color_layer = Image.new("RGBA", base.size, color+(int(255*alpha),))
    overlay = Image.composite(color_layer, overlay, m_img)

    return Image.alpha_composite(base, overlay).convert(img.mode)


# ------------------------- Aug Preview -------------------------
def preview_augment(
    frames_pil: List[Image.Image],
    aug_callable,
    seed: Optional[int]=123,
    max_cols: int=8,
    step: int=1,
    denorm=None,
    save_path: Optional[str]=None,
    log_key: Optional[str]=None,
):
    out_t = aug_callable(frames_pil, seed=seed)
    frames_aug = tensor_clip_to_pil_list(out_t, denorm=denorm)

    ori = frames_pil[::max(1,step)]
    aug = frames_aug[::max(1,step)]
    n = min(len(ori), len(aug))
    cols = min(max_cols, n)

    fig, axes = plt.subplots(2, cols, figsize=(cols*2.5, 5))

    for i in range(cols):
        axes[0,i].imshow(ori[i]); axes[0,i].axis("off")
        axes[1,i].imshow(aug[i]); axes[1,i].axis("off")

    fig.suptitle(f"Augmented seed={seed}")
    plt.tight_layout(rect=[0,0,1,0.95])

    if save_path:
        _ensure_dir_for(save_path)
        fig.savefig(save_path, dpi=150)

    if log_key:
        _maybe_wandb_log({
            safe_key(log_key): wandb.Image(fig) if _WANDB_AVAILABLE else (save_path or "augment")
        })

    plt.close(fig)
