# tools/augment_export.py
# -*- coding: utf-8 -*-
"""
Export 7 kinds of visual augmentation results into folderA/subfolders:
1) rgb/gray
2) temporal downsample (1/4, 1/8, 1/16, 1/32)
3) geometric (rotate, crop, translate)
4) photometric (brightness, contrast, color)
5) temporal concat (clip1 + clip2)
6) spatial mosaic (2x2, 3x3)
7) background replace (paste to img-a.jpg)
"""
import os, glob, math, argparse, random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from PIL import ImageChops


# -------------------- 基础 I/O --------------------
def list_frame_paths(frame_dir: str) -> List[str]:
    exts = ("*.jpg", "*.jpeg", "*.png")
    files = []
    for ex in exts:
        files.extend(glob.glob(os.path.join(frame_dir, ex)))
    def key(p):
        stem = os.path.splitext(os.path.basename(p))[0]
        try: return int(stem)
        except: return stem
    return sorted(files, key=key)

def load_pil_frames(paths: List[str], limit: Optional[int] = None) -> List[Image.Image]:
    paths = paths if limit is None else paths[:limit]
    out = []
    for p in paths:
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            im = Image.new("RGB", (224, 224), (0, 0, 0))
        out.append(im)
    return out

def save_sequence(frames: List[Image.Image], out_dir: str, prefix: str = "", start_idx: int = 0):
    os.makedirs(out_dir, exist_ok=True)
    for i, im in enumerate(frames):
        fn = f"{prefix}{start_idx + i:06d}.jpg"
        im.save(os.path.join(out_dir, fn), quality=95)

def ensure_size(frames: List[Image.Image], size: int) -> List[Image.Image]:
    return [im.resize((size, size), Image.BILINEAR) for im in frames]

def to_gray(frames: List[Image.Image], keep_3ch: bool = True) -> List[Image.Image]:
    out = []
    for im in frames:
        g = ImageOps.grayscale(im)
        if keep_3ch:
            g = Image.merge("RGB", (g, g, g))
        out.append(g)
    return out


# -------------------- 1. RGB / Gray --------------------
def export_rgb_gray(frames: List[Image.Image], out_root: str, size: int):
    rgb_dir = os.path.join(out_root, "1_rgb_gray", "rgb")
    gray_dir = os.path.join(out_root, "1_rgb_gray", "gray")
    save_sequence(ensure_size(frames, size), rgb_dir)
    save_sequence(ensure_size(to_gray(frames, keep_3ch=True), size), gray_dir)


# -------------------- 2. Temporal Downsample --------------------
# def sample_temporal_indices(n: int, ratio: float) -> List[int]:
#     m = max(1, int(n * ratio))
#     idxs = np.linspace(0, n - 1, m).astype(int)
#     return idxs.tolist()

def sample_temporal_indices(n: int, ratio: float) -> List[int]:
    if n <= 0:
        return []
    m = max(1, int(n * ratio))
    idxs = np.linspace(0, n - 1, m).astype(int)
    idxs = np.clip(idxs, 0, n - 1)
    return idxs.tolist()


def export_temporal_downsample(frames: List[Image.Image], out_root: str, size: int):
    ratios = [1/4, 1/8, 1/16, 1/32]
    n = len(frames)
    for r in ratios:
        idxs = sample_temporal_indices(n, r)
        sub = [frames[i] for i in idxs]
        sub = ensure_size(sub, size)
        subdir = os.path.join(out_root, "2_temporal_downsample", f"r{int(1/r):02d}")
        save_sequence(sub, subdir, prefix=f"r{int(1/r):02d}_")


# -------------------- 3. Geometric --------------------
def rotate(frames, deg):
    return [im.rotate(deg) for im in frames]

def crop(frames, crop_ratio=0.8):
    out = []
    for im in frames:
        W, H = im.size
        dW, dH = int(W * (1 - crop_ratio)), int(H * (1 - crop_ratio))
        box = (dW//2, dH//2, W - dW//2, H - dH//2)
        out.append(im.crop(box).resize((W, H)))
    return out

def translate(frames, shift=20):
    out = []
    for im in frames:
        # out.append(ImageOps.offset(im, shift, shift))
        out.append(ImageChops.offset(im, shift, shift))

    return out

def export_geometric(frames, out_root, size):
    geom_dir = os.path.join(out_root, "3_geometric")
    for name, f in [("rotate", rotate(frames, 15)),
                    ("crop", crop(frames, 0.8)),
                    ("translate", translate(frames, 20))]:
        sub = ensure_size(f, size)
        save_sequence(sub, os.path.join(geom_dir, name))


# -------------------- 4. Photometric --------------------
def adjust_brightness(frames, factor):
    return [ImageEnhance.Brightness(im).enhance(factor) for im in frames]

def adjust_contrast(frames, factor):
    return [ImageEnhance.Contrast(im).enhance(factor) for im in frames]

def adjust_color(frames, factor):
    return [ImageEnhance.Color(im).enhance(factor) for im in frames]

def export_photometric(frames, out_root, size):
    photo_dir = os.path.join(out_root, "4_photometric")
    for name, f in [("brightness", adjust_brightness(frames, 1.4)),
                    ("contrast", adjust_contrast(frames, 1.4)),
                    ("color", adjust_color(frames, 1.5))]:
        sub = ensure_size(f, size)
        save_sequence(sub, os.path.join(photo_dir, name))


# -------------------- 5. Temporal Concat --------------------
def export_temporal_concat(frames1, frames2, out_root, size):
    all_frames = frames1 + frames2
    outdir = os.path.join(out_root, "5_temporal_concat")
    save_sequence(ensure_size(all_frames, size), outdir)


# -------------------- 6. Spatial Mosaic --------------------

def make_mosaic_grid(clips: List[List[Image.Image]], grid_size: int, size: int | Tuple[int, int]) -> Image.Image:
    # ---- Step 1: 解析 size ----
    if isinstance(size, (tuple, list)):
        if len(size) == 2:
            tile_w, tile_h = int(size[0]), int(size[1])
        else:
            tile_w = tile_h = int(size[0])
    else:
        tile_w = tile_h = int(size)

    # ---- Step 2: 计算整体尺寸 ----
    full_w = tile_w * grid_size
    full_h = tile_h * grid_size
    assert isinstance(full_w, int) and isinstance(full_h, int), f"Bad full_w/full_h: {full_w}, {full_h}"

    # ---- Step 3: 创建画布 ----
    canvas = Image.new("RGB", (full_w, full_h), (255, 255, 255))
    total_cells = grid_size * grid_size

    # ---- Step 4: 填补空格 ----
    filled_clips = []
    for i in range(total_cells):
        if i < len(clips) and len(clips[i]) > 0:
            filled_clips.append(clips[i])
        else:
            filled_clips.append([Image.new("RGB", (tile_w, tile_h), (220, 220, 220))])

    # ---- Step 5: 绘制拼图 ----
    for idx, clip in enumerate(filled_clips):
        row = idx // grid_size
        col = idx % grid_size
        im = clip[0].resize((tile_w, tile_h))
        canvas.paste(im, (col * tile_w, row * tile_h))
    return canvas




def export_spatial_mosaic(clips: List[List[Image.Image]], out_root: str, size: int):
    grids = [(1, 2), (2, 2), (3, 3)]
    outdir = os.path.join(out_root, "6_spatial_mosaic")
    os.makedirs(outdir, exist_ok=True)
    for g in grids:
        img = make_mosaic_grid(clips, g, size)
        img.save(os.path.join(outdir, f"mosaic_{g[0]}x{g[1]}.jpg"), quality=95)


# -------------------- 7. Background Replace --------------------
def simple_mask(im: Image.Image, threshold=180):
    arr = np.array(im.convert("L"))
    mask = (arr > threshold).astype(np.uint8) * 255
    return Image.fromarray(mask, mode="L")

from PIL import Image, ImageOps
import os

def export_background_replace(frames, bg_path, out_root, size):
    bg = Image.open(bg_path).convert("RGB").resize((size, size))
    outdir = os.path.join(out_root, "7_background_replace")
    os.makedirs(outdir, exist_ok=True)
    for i, im in enumerate(frames):
        im_r = im.resize((size, size))
        mask = simple_mask(im_r)
        mask = ImageOps.invert(mask)  # ✅ 关键：反转mask，让“人”为白（255），背景为黑（0）
        comp = Image.composite(im_r, bg, mask)
        comp.save(os.path.join(outdir, f"bg_{i:04d}.jpg"), quality=95)



# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip1_dir", required=True)
    parser.add_argument("--clip2_dir", required=False)
    parser.add_argument("--bg_image", required=False)
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--max_frames", type=int, default=32)
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    clip1_paths = list_frame_paths(args.clip1_dir)
    frames1 = load_pil_frames(clip1_paths, args.max_frames)
    frames1 = ensure_size(frames1, args.size)

    # 1
    export_rgb_gray(frames1, args.out, args.size)
    # 2
    export_temporal_downsample(frames1, args.out, args.size)
    # 3
    export_geometric(frames1, args.out, args.size)
    # 4
    export_photometric(frames1, args.out, args.size)

    # 5 temporal concat
    if args.clip2_dir:
        frames2 = load_pil_frames(list_frame_paths(args.clip2_dir), args.max_frames)
        export_temporal_concat(frames1, frames2, args.out, args.size)

        # 6 spatial mosaic
        # export_spatial_mosaic([frames1, frames2], args.out, args.size)

    # 7 background replace
    if args.bg_image:
        export_background_replace(frames1, args.bg_image, args.out, args.size)

    print(f"✅ All exports done! Saved under: {args.out}")


if __name__ == "__main__":
    main()



'''
python tools/augment_export.py \
  --clip1_dir /home/pxl416/PeixiLiu/px_proj/pxUni/data/mini_CSL_Daily/sentence/S000000_P0000_T00 \
  --clip2_dir /home/pxl416/PeixiLiu/px_proj/pxUni/data/mini_CSL_Daily/sentence/S000001_P0000_T00 \
  --bg_image /home/pxl416/PeixiLiu/px_proj/pxUni/tools/img-b.jpg \
  --out /home/pxl416/PeixiLiu/px_proj/pxUni/test_results_lpx251014-2 \
  --size 224 \
  --max_frames 32


'''