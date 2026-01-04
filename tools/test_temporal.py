# tools/test_temporal.py
import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
import matplotlib.pyplot as plt

from utils.config import load_yaml_as_ns, resolve_path
from datasets.datasets import create_dataloader
from models.build_model import build_model
from utils.loss import temporal_bce_loss


# =========================================================
# Project root (Uni-SLM/)
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =========================================================
# Args
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser("Temporal Pretrain Sanity Check")

    parser.add_argument(
        "--cfg",
        type=str,
        default="/home/pxl416/PeixiLiu/px_proj/Uni-SLM/config/pretrain.yaml",
        help="Path to pretrain yaml config",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "dev", "test"],
        help="Dataset split to test",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=2,
        help="Number of samples to visualize from one batch",
    )

    return parser.parse_args()


# =========================================================
# Main
# =========================================================
@torch.no_grad()
def main():
    args = parse_args()

    # -----------------------------------------------------
    # Load pretrain config
    # -----------------------------------------------------
    cfg_path = Path(args.cfg).resolve()
    cfg = load_yaml_as_ns(cfg_path)

    print(f"[Info] Config loaded from: {cfg_path}")

    # -----------------------------------------------------
    # Resolve model yaml (relative to PROJECT_ROOT)
    # -----------------------------------------------------
    cfg.model = load_yaml_as_ns(
        resolve_path(cfg.model, PROJECT_ROOT)
    )

    # -----------------------------------------------------
    # Resolve dataset yamls (relative to PROJECT_ROOT)
    # -----------------------------------------------------
    for k, v in vars(cfg.datasets).items():
        setattr(cfg.datasets, k, str(resolve_path(v, PROJECT_ROOT)))

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    # -----------------------------------------------------
    # Build dataloader
    # -----------------------------------------------------
    dummy_args = SimpleNamespace()  # datasets expect an args object

    loader = create_dataloader(
        args=dummy_args,
        cfg=cfg,
        phase=args.split,
    )

    print(f"[Info] Dataloader built | split={args.split}")
    print(f"[Info] Dataset size = {len(loader.dataset)}")

    # -----------------------------------------------------
    # Build model
    # -----------------------------------------------------
    model = build_model(cfg).to(device)
    model.eval()

    print("[Info] Model built successfully")

    # -----------------------------------------------------
    # Fetch ONE batch
    # -----------------------------------------------------
    src, tgt = next(iter(loader))

    # move src tensors to device
    src = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in src.items()
    }

    print("[Info] One batch fetched (src only):")
    for k, v in src.items():
        if torch.is_tensor(v):
            print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
        else:
            print(f"  {k}: type={type(v)}")

    # -----------------------------------------------------
    # Forward: temporal pretext task
    # -----------------------------------------------------
    out = model(batch=src, task="temporal")

    temporal_logits = out["temporal_logits"]  # (B, T)
    heatmap = torch.sigmoid(temporal_logits)

    B, T = heatmap.shape

    print(f"[Info] temporal_logits shape = {temporal_logits.shape}")
    print(
        f"[Info] heatmap stats | "
        f"min={heatmap.min().item():.4f} "
        f"max={heatmap.max().item():.4f} "
        f"mean={heatmap.mean().item():.4f}"
    )

    # -----------------------------------------------------
    # Temporary target (sanity only)
    # NOTE: this is NOT the final target design
    # -----------------------------------------------------
    if "rgb_mask" in src and src["rgb_mask"] is not None:
        target = src["rgb_mask"].float()
        print("[Info] Using rgb_mask as temporal target (sanity check)")
    else:
        target = torch.ones((B, T), device=device)
        print("[Info] Using all-ones temporal target (fallback)")

    # -----------------------------------------------------
    # Loss sanity check
    # -----------------------------------------------------
    print("[Info] Running temporal loss sanity check...")

    # logits: (B, T_enc)
    # gt_mask: (B, T_raw)
    loss = temporal_bce_loss(
        logits=temporal_logits,
        gt_mask=src["rgb_mask"],
        downsample_mode="avg",
    )

    print(
        f"[Info] Temporal BCE loss = {loss.item():.6f}"
    )

    # Basic safety checks
    assert torch.isfinite(loss), "Loss is NaN or Inf!"
    assert loss.item() >= 0.0, "Loss should be non-negative!"

    print("[Info] Temporal loss sanity check passed ✔")


    # -----------------------------------------------------
    # Visualization
    # -----------------------------------------------------
    num_vis = min(args.num_samples, B)

    for i in range(num_vis):
        plt.figure(figsize=(12, 3))

        plt.plot(
            heatmap[i].cpu().numpy(),
            label="Predicted heatmap (sigmoid)",
            linewidth=2,
        )
        plt.plot(
            target[i].cpu().numpy(),
            label="Target",
            linestyle="--",
        )

        plt.ylim(-0.05, 1.05)
        plt.xlabel("Time step")
        plt.ylabel("Value")
        plt.title(f"Temporal Heatmap | sample {i}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    print("[Done] Temporal pretrain sanity check finished.")


if __name__ == "__main__":
    main()
