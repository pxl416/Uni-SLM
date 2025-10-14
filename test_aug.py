# test_aug.py
import argparse, yaml, json, random
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from utils.dataset2 import CSLNewsDataset


# 反标准化+转 PIL
_RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_RGB_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def rgb_tensor_to_pil(x: torch.Tensor, out_size=None) -> Image.Image:
    """
    x: [3,H,W] 标准化后的张量
    out_size: (W,H) 可选，输出尺寸
    """
    x = x.detach().cpu().float().numpy()  # [3,H,W]
    x = (x.transpose(1,2,0) * _RGB_STD + _RGB_MEAN)  # [H,W,3], 反标准化到 [0,1]
    x = np.clip(x, 0.0, 1.0)
    img = (x * 255.0).round().astype(np.uint8)
    pil = Image.fromarray(img)
    if out_size is not None:
        pil = pil.resize(out_size)
    return pil



def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def draw_keypoints_on_image(img: Image.Image, kpts_k3: np.ndarray, color=(255, 0, 0), conf_thr: float = 0.3,
                            r: int = 2):
    """绘制关键点到图像上，支持指定颜色"""
    W, H = img.size
    if kpts_k3.size == 0:
        return img

    pts = kpts_k3.copy().astype(float)

    # 自适应：若像素范围很小/在 [-2,2] 内，视为 [-1,1] 归一化坐标，先转像素
    xy = pts[..., :2]
    finite = np.isfinite(xy).all(axis=-1)
    if finite.any():
        mx = np.nanmax(np.abs(xy[finite]))
        if mx <= 2.0:  # 经验阈值
            xy = (xy + 1.0) * 0.5
            xy[..., 0] *= (W - 1)
            xy[..., 1] *= (H - 1)
            pts[..., :2] = xy

    draw = ImageDraw.Draw(img)
    for (x, y, c) in pts:
        if c > conf_thr and np.isfinite(x) and np.isfinite(y):
            x0, y0 = int(x - r), int(y - r)
            x1, y1 = int(x + r), int(y + r)
            draw.ellipse([x0, y0, x1, y1], outline=color, width=1)
    return img


def plot_hist(values, title, xlabel, save_path: Path, bins=30):
    if len(values) == 0:
        return
    plt.figure()
    plt.hist(values, bins=bins)
    plt.title(title);
    plt.xlabel(xlabel);
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(save_path);
    plt.close()


# -------------- main --------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="config/config.yaml")
    parser.add_argument("--phase", type=str, choices=["train", "val", "test"], default="train")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--outdir", type=str, default="reports")
    args_cli = parser.parse_args()

    # reproducibility（仅用于本脚本）
    random.seed(3407);
    np.random.seed(3407);
    torch.manual_seed(3407)

    # 读取你的原始 yaml 结构
    with open(args_cli.cfg, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    cfg = SimpleNamespace(**raw_cfg)

    # 取第一个 active 数据集
    dataset_name = cfg.active_datasets[0]
    ds_cfg = cfg.datasets[dataset_name]

    # 构造 shim，使之符合 CSLNewsDataset 期望的 cfg 结构
    cfg_shim = SimpleNamespace(
        data_path=SimpleNamespace(
            pose_dirs=ds_cfg["pose_dirs"],
            rgb_dirs=ds_cfg["rgb_dirs"],
            train_label_paths=ds_cfg["train_label_paths"],
            val_label_paths=ds_cfg["val_label_paths"],
            test_label_paths=ds_cfg["test_label_paths"],
        ),
        augmentation=SimpleNamespace(
            output=SimpleNamespace(size=[224, 224]),  # 如有更详细配置可在此填入
        ),
        seed=3407,
        left_kp_ids=[],
        right_kp_ids=[],
    )

    # 最小 args（Dataset 对齐）
    args = SimpleNamespace(
        dataset_name=dataset_name,
        max_length=128,
        rgb_support=True,  #
        use_aug=True,  # 训练期增强开；本脚本只可视化读取 OK
        batch_size=1,
        num_workers=0,
        seed=3407,
    )

    # 选 label 路径并实例化 Dataset —— 放在最前，后续都用 ds
    label_path = {
        "train": cfg_shim.data_path.train_label_paths,
        "val": cfg_shim.data_path.val_label_paths,
        "test": cfg_shim.data_path.test_label_paths
    }[args_cli.phase]

    ds = CSLNewsDataset(label_path=label_path, args=args, phase=args_cli.phase, cfg=cfg_shim)
    print(f"Using dataset {dataset_name}")
    print(f"Pose dir: {cfg_shim.data_path.pose_dirs}")
    print(f"RGB dir:  {cfg_shim.data_path.rgb_dirs}")
    print(f"Label:    {label_path}")
    print(f"[OK] Dataset built. Total={len(ds)}")

    # 输出目录
    out_root = ensure_dir(Path(args_cli.outdir))
    figs_dir = ensure_dir(out_root / "figs")
    samples_dir = ensure_dir(out_root / "samples")

    # 抽样 N 条
    N = min(args_cli.limit, len(ds))
    print(f"[Info] Sampling {N} examples from phase={args_cli.phase}")

    lens_T, text_lens, conf_values = [], [], []
    W, H = cfg_shim.augmentation.output.size

    # 预先选择要可视化的样本ID
    viz_ids = set(random.sample(range(N), k=min(8, N)))
    print(f"[Info] Will visualize samples: {sorted(viz_ids)}")

    for i in range(N):
        name, pose_sample, text, rgb_idx, supp = ds[i]

        if i == 0:
            print("[Sanity] keys:", list(pose_sample.keys()))
            for k, v in pose_sample.items():
                print(f"  {k}: {tuple(v.shape)}  dtype={v.dtype}")

        # 统计
        T_len = int(pose_sample["body"].shape[0])
        lens_T.append(T_len)
        text_lens.append(len(text))
        for part, tens in pose_sample.items():
            arr = tens.detach().cpu().numpy()
            conf_values.extend(arr[..., 2].ravel().tolist())

        # 可视化：RGB 叠加 + 白底对照
        if i in viz_ids:
            t0 = 0  # 取第 1 帧演示，你也可以改成其它帧或做多帧网格

            # 目标画布尺寸
            W, H = cfg_shim.augmentation.output.size
            target_size = (W, H)

            # 1) 取 RGB 第 t0 帧（需要 rgb_support=True 才会有）
            rgb_pil = None
            if isinstance(supp, dict) and 'rgb_img' in supp and supp['rgb_img'] is not None:
                rgb_seq = supp['rgb_img']  # [T,3,112,112]
                if isinstance(rgb_seq, torch.Tensor) and rgb_seq.ndim == 4 and rgb_seq.shape[0] > t0:
                    rgb_pil = rgb_tensor_to_pil(rgb_seq[t0], out_size=target_size)

            # 若没有 RGB，则用白底兜底
            if rgb_pil is None:
                print('[ERROR] RGB image not found:', i)
                rgb_pil = Image.new("RGB", target_size, (255, 255, 255))

            # 2) 复制一份作为白底对照
            canvas_white = Image.new("RGB", target_size, (255, 255, 255))
            canvas_rgb = rgb_pil.copy()

            # 3) 定义颜色
            part_colors = {
                "body": (255, 0, 0),        # red
                "left": (0, 128, 0),        # green
                "right": (0, 0, 255),       # blue
                "face_all": (255, 140, 0)   # orange
            }

            # 4) 叠加绘制四个部位的 t0 帧关键点
            for part, tens in pose_sample.items():
                kpts = tens[t0].detach().cpu().numpy()  # [K,3]，坐标是 [-1,1]，draw 函数会自适应
                color = part_colors.get(part, (255, 0, 0))
                canvas_rgb  = draw_keypoints_on_image(canvas_rgb,  kpts, color=color, conf_thr=0.05, r=2)
                canvas_white = draw_keypoints_on_image(canvas_white, kpts, color=color, conf_thr=0.05, r=2)

            # 5) 保存两份：RGB叠加版 + 白底版
            canvas_rgb.save(samples_dir / f"{i:04d}_{name}_rgb+kp_t{t0}.png")
            canvas_white.save(samples_dir / f"{i:04d}_{name}_kpts_white_t{t0}.png")

            with open(samples_dir / f"{i:04d}_{name}_meta.json", "w", encoding="utf-8") as f:
                json.dump({"name": name, "text": text, "T": T_len, "frame": int(t0)}, f, ensure_ascii=False, indent=2)


    # 画统计图 + summary
    plot_hist(lens_T, "Temporal length (T) distribution", "T", figs_dir / "len_T.png")
    plot_hist(text_lens, "Text length distribution", "len", figs_dir / "text_len.png")
    plot_hist(conf_values, "Keypoint confidence distribution", "conf", figs_dir / "kp_conf.png")

    summary = {
        "num_samples": N,
        "T_min": int(np.min(lens_T)) if lens_T else 0,
        "T_max": int(np.max(lens_T)) if lens_T else 0,
        "T_mean": float(np.mean(lens_T)) if lens_T else 0.0,
        "text_len_mean": float(np.mean(text_lens)) if text_lens else 0.0,
        "kp_conf_mean": float(np.mean(conf_values)) if conf_values else 0.0,
    }
    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[Done] Figures ->", figs_dir)
    print("[Done] Samples ->", samples_dir)
    print("[Done] Summary ->", out_root / "summary.json")


if __name__ == "__main__":
    main()