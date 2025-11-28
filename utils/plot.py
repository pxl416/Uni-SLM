# utils/plot.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, math
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# 可选的 wandb：未安装/未 init 时不报错
try:
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None  # type: ignore
    _WANDB_AVAILABLE = False

# === 放到文件顶部 import 后任意位置追加 ===
from PIL import ImageDraw

# 常用 21 点手部骨架（按需改）
DEFAULT_TOPOLOGY_21 = [
    (0,1),(1,2),(2,3),(3,4),          # 拇指
    (0,5),(5,6),(6,7),(7,8),          # 食指
    (0,9),(9,10),(10,11),(11,12),     # 中指
    (0,13),(13,14),(14,15),(15,16),   # 无名指
    (0,17),(17,18),(18,19),(19,20),   # 小指
]

def draw_skeleton_on_pil(
    img: Image.Image,
    kpts: np.ndarray,                  # [J,C], C=2或3
    topology: Optional[List[Tuple[int,int]]] = None,
    kpt_coord_mode: str = "pixel",     # "pixel" 或 "normalized"
    radius: int = 2,
    line_width: int = 2,
) -> Image.Image:
    """
    将关键点/骨架画到 PIL 图像上（返回新图像）。
    - kpt_coord_mode="normalized" 表示 kpts 落在 [0,1]，会按图像尺寸放缩
    """
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

    # 先画骨架线
    for a, b in topology:
        if a < kpts.shape[0] and b < kpts.shape[0]:
            xa, ya = _xy(a); xb, yb = _xy(b)
            draw.line((xa, ya, xb, yb), width=line_width, fill=(0, 255, 0))
    # 再画关键点
    for j in range(min(kpts.shape[0], 128)):
        x, y = _xy(j)
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=(255, 0, 0), width=line_width)
    return img


def save_keypoint_overlay_grid(
    frames_pil: List[Image.Image],
    kpts_seq: np.ndarray,             # [T,J,C]
    picks: Optional[List[int]] = None,
    kpt_coord_mode: str = "pixel",
    topology: Optional[List[Tuple[int,int]]] = None,
    max_cols: int = 8,
    save_path: Optional[str] = None,
    log_key: Optional[str] = None,
):
    """
    把若干时刻的帧做成网格，叠加关键点/骨架后保存/记录。
    """
    T = len(frames_pil)
    if picks is None:
        # 默认选 4 帧：首、中、中、尾
        if T >= 4:
            picks = [0, T//3, 2*T//3, T-1]
        else:
            picks = list(range(T))

    vis_frames: List[Image.Image] = []
    for t in picks:
        t = int(max(0, min(T-1, t)))
        kp = kpts_seq[t] if kpts_seq is not None and len(kpts_seq) > t else None
        vis = draw_skeleton_on_pil(frames_pil[t], kp, topology=topology, kpt_coord_mode=kpt_coord_mode)
        vis_frames.append(vis)

    show_clip_grid(
        vis_frames,
        title="keypoints-overlay",
        max_cols=max_cols,
        step=1,
        save_path=save_path,
        log_key=log_key or "debug/keypoints_overlay"
    )


def save_timeline(
    starts: List[int],
    ends: List[int],
    T_total: int,
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    log_key: Optional[str] = None,
):
    """
    画一条长度为 T_total 的时间轴，并标出若干 [start, end] 段，附文本标签。
    """
    _ensure_dir_for(save_path)
    starts = list(map(int, starts))
    ends   = list(map(int, ends))
    K = len(starts)
    xs = np.arange(T_total, dtype=np.int32)

    plt.figure(figsize=(max(6, T_total/80), 1.6 + 0.2*K))
    # 底条
    plt.plot([0, T_total-1], [0, 0], linewidth=6)

    # 填段
    for i, (s, e) in enumerate(zip(starts, ends)):
        s = max(0, min(T_total-1, s))
        e = max(0, min(T_total-1, e))
        if e < s: s, e = e, s
        plt.fill_between([s, e], [-0.15, -0.15], [0.15, 0.15], alpha=0.4, step="pre", label=f"seg{i}")
        plt.vlines([s, e], -0.3, 0.3, linewidth=1)

        if labels and i < len(labels) and labels[i]:
            txt = labels[i]
            mid = (s + e) / 2.0
            plt.text(mid, 0.35 + 0.18*(i%3), txt, ha="center", va="bottom", fontsize=8, rotation=0)

    plt.yticks([])
    plt.xlim(-1, T_total)
    plt.xlabel("time (frame index)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
    _maybe_wandb_log({(log_key or "debug/timeline"): wandb.Image(plt.gcf()) if _WANDB_AVAILABLE else (save_path or "timeline")})
    plt.close()


def save_attn_mask(
    attn_mask: np.ndarray,  # [T] bool/0-1
    save_path: Optional[str] = None,
    log_key: Optional[str] = None,
):
    """
    显示时间维有效/填充的分布情况。
    """
    T = int(attn_mask.shape[0])
    y = (attn_mask.astype(np.int32) * 1)
    plt.figure(figsize=(max(6, T/80), 1.6))
    plt.bar(np.arange(T), y, width=1.0)
    plt.yticks([0,1], ["pad","valid"])
    plt.ylim(0, 1.2)
    plt.xlabel("time")
    plt.tight_layout()
    if save_path:
        _ensure_dir_for(save_path)
        plt.savefig(save_path, dpi=160)
    _maybe_wandb_log({(log_key or "debug/attn_mask"): wandb.Image(plt.gcf()) if _WANDB_AVAILABLE else (save_path or "attn_mask")})
    plt.close()


# ------------------------- 小工具 -------------------------
def _maybe_wandb_log(data: Dict[str, Any]):
    """若已 init wandb 则记录，否则静默跳过。"""
    if _WANDB_AVAILABLE and getattr(wandb, "run", None) is not None:
        wandb.log(data)

def _ensure_dir_for(path: Optional[str]):
    if path is None:
        return
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def tensor_clip_to_pil_list(
    t: torch.Tensor,
    denorm: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None
) -> List[Image.Image]:
    """
    将 [T,C,H,W] 的张量转为 PIL 列表，用于可视化。
    - 若传入 denorm=(mean,std) 会先反归一化。
    - 假设像素在 [0,1]（或反归一化后会被 clamp）。
    """
    assert t.ndim == 4, f"expect [T,C,H,W], got {list(t.shape)}"
    T, C, H, W = t.shape
    imgs: List[Image.Image] = []
    for i in range(T):
        x = t[i]
        if denorm is not None:
            mean, std = denorm
            x = x.clone()
            for c in range(x.shape[0]):
                x[c] = x[c] * std[c] + mean[c]
        x = x.clamp(0, 1)
        arr = (x.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        if C == 1:
            imgs.append(Image.fromarray(arr[..., 0], mode="L"))
        else:
            imgs.append(Image.fromarray(arr, mode="RGB"))
    return imgs


# ------------------------- 1) 检索可视化 -------------------------
def save_retrieval_vis(
    sim_matrix: np.ndarray,
    save_path: str = "retrieval_vis.png",
    log_key: str = "retrieval/sim_matrix"
):
    """保存相似度矩阵热力图（wandb 可选）。"""
    _ensure_dir_for(save_path)
    plt.figure(figsize=(8, 6))
    plt.imshow(sim_matrix, cmap="viridis")
    plt.colorbar()
    plt.title("Similarity Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    _maybe_wandb_log({log_key: wandb.Image(save_path) if _WANDB_AVAILABLE else save_path})


def plot_retrieval_curves(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    log_key: str = "retrieval/recall_curve"
):
    """
    绘制 Recall@1/5/10 曲线；metrics_dict 形如：
    {'t2v':{'R1':..,'R5':..,'R10':..}, 'v2t':{...}}
    """
    xs = [1, 5, 10]
    plt.figure(figsize=(5, 4))
    for direction in ["t2v", "v2t"]:
        if direction not in metrics_dict:
            continue
        m = metrics_dict[direction]
        ys = [m.get("R1", 0), m.get("R5", 0), m.get("R10", 0)]
        plt.plot(xs, ys, marker="o", label=direction)
    plt.xlabel("K")
    plt.ylabel("Recall@K")
    plt.xticks(xs)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        _ensure_dir_for(save_path)
        plt.savefig(save_path, dpi=200)
    _maybe_wandb_log({log_key: wandb.Image(plt.gcf()) if _WANDB_AVAILABLE else "recall_curve"})
    plt.close()


# ------------------------- 2) 增强可视化核心 -------------------------
def show_clip_grid(
    frames_pil: List[Image.Image],
    title: str = "",
    max_cols: int = 8,
    step: int = 1,
    save_path: Optional[str] = None,
    log_key: Optional[str] = None,
):
    """
    将一个 clip 的若干帧画成网格。
    - frames_pil: PIL 列表
    - step: 每 step 取一帧，控制展示密度
    """
    assert len(frames_pil) > 0, "frames_pil is empty"
    frames = frames_pil[::max(1, step)]
    n = len(frames)
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)

    w, h = frames[0].size
    fig_w = cols * (w / 100)
    fig_h = rows * (h / 100)

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(rows, cols)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            ax.axis("off")
            if idx < n:
                img = frames[idx]
                if img.mode == "L":
                    ax.imshow(np.array(img), cmap="gray")
                else:
                    ax.imshow(img)
                idx += 1

    if title:
        fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        _ensure_dir_for(save_path)
        fig.savefig(save_path, dpi=150)
    if log_key:
        _maybe_wandb_log({log_key: wandb.Image(fig) if _WANDB_AVAILABLE else (save_path or title)})
    plt.close(fig)


def compare_stages(
    original: List[Image.Image],
    stages: Dict[str, List[Image.Image]],
    picks: Tuple[int, ...] = (0, -1),
    save_path: Optional[str] = None,
    log_key: Optional[str] = None,
):
    """
    对比多个阶段在若干时间点的效果。
    - original: 原始帧列表
    - stages: {"affine":[...], "color":[...], "cutout":[...], "final":[...]}（键自定义）
    - picks: 选择展示的帧下标（支持负索引，如 -1 表示最后一帧）
    """
    assert len(original) > 0, "original empty"
    stage_names = ["original"] + list(stages.keys())
    columns = len(stage_names)

    T = len(original)
    pick_idxs = []
    for p in picks:
        i = p if p >= 0 else (T + p)
        if 0 <= i < T:
            pick_idxs.append(i)
    pick_idxs = sorted(set(pick_idxs))
    rows = len(pick_idxs)

    fig, axes = plt.subplots(rows, columns, figsize=(3.5 * columns, 3.5 * rows))
    axes = np.array(axes).reshape(rows, columns)

    def _get(arr: List[Image.Image], i: int) -> Image.Image:
        return arr[i if i < len(arr) else -1]

    for r, t_idx in enumerate(pick_idxs):
        # original
        img0 = _get(original, t_idx)
        axes[r, 0].imshow(np.array(img0), cmap="gray" if img0.mode == "L" else None)
        axes[r, 0].set_title(f"t={t_idx}\noriginal")
        axes[r, 0].axis("off")
        # stages
        for c, name in enumerate(stage_names[1:], start=1):
            arr = stages.get(name, [])
            if len(arr) == 0:
                axes[r, c].axis("off"); continue
            img = _get(arr, t_idx)
            axes[r, c].imshow(np.array(img), cmap="gray" if img.mode == "L" else None)
            axes[r, c].set_title(f"{name}")
            axes[r, c].axis("off")

    plt.tight_layout()
    if save_path:
        _ensure_dir_for(save_path)
        fig.savefig(save_path, dpi=150)
    if log_key:
        _maybe_wandb_log({log_key: wandb.Image(fig) if _WANDB_AVAILABLE else (save_path or "compare_stages")})
    plt.close(fig)


def overlay_mask(
    img: Image.Image,
    mask: np.ndarray,
    alpha: float = 0.4,
    color: Tuple[int, int, int] = (0, 255, 0)
) -> Image.Image:
    """
    在图像上叠加半透明掩码。
    - mask: HxW（bool/0-1/0-255）；非零视为遮挡区域
    """
    base = img.convert("RGBA")
    m = (mask.astype(np.float32) > 0).astype(np.float32)
    m = (m * 255).astype(np.uint8)
    m_img = Image.fromarray(m, mode="L").resize(base.size, resample=Image.NEAREST)

    overlay = Image.new("RGBA", base.size, color + (0,))
    color_layer = Image.new("RGBA", base.size, color + (int(255 * alpha),))
    overlay = Image.composite(color_layer, overlay, m_img)
    out = Image.alpha_composite(base, overlay).convert(img.mode)
    return out


# ------------------------- 3) 一键预览增强 -------------------------
def preview_augment(
    frames_pil: List[Image.Image],
    aug_callable,
    seed: Optional[int] = 123,
    max_cols: int = 8,
    step: int = 1,
    denorm: Optional[Tuple[Tuple[float,...],Tuple[float,...]]] = None,
    save_path: Optional[str] = None,
    log_key: Optional[str] = None,
):
    """
    将原始 clip 与增强后的 clip 并排展示（各一排）。
    - aug_callable: 如 SignAugment 实例；需支持 __call__(frames_pil, seed=seed) -> Tensor[T,C,H,W]
    - denorm: 若返回的是已 normalize 的 Tensor，则传 (mean,std) 以反归一化显示；否则传 None
    """
    # 生成增强结果
    out_t = aug_callable(frames_pil, seed=seed)  # [T,C,H,W]
    frames_aug = tensor_clip_to_pil_list(out_t, denorm=denorm)

    # 仅取部分帧展示
    ori = frames_pil[::max(1, step)]
    aug = frames_aug[::max(1, step)]
    n = min(len(ori), len(aug))
    cols = min(max_cols, n)
    rows = 2

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = np.array(axes).reshape(rows, cols)

    def _imshow(ax, im: Image.Image):
        ax.imshow(np.array(im), cmap="gray" if im.mode == "L" else None)
        ax.axis("off")

    for i in range(cols):
        _imshow(axes[0, i], ori[i])
        _imshow(axes[1, i], aug[i])

    fig.suptitle(f"Augmented (seed={seed})")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        _ensure_dir_for(save_path)
        fig.savefig(save_path, dpi=150)
    if log_key:
        _maybe_wandb_log({log_key: wandb.Image(fig) if _WANDB_AVAILABLE else (save_path or "augment_preview")})
    plt.close(fig)
