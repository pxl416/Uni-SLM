# tools/test_features.py
import os
import sys
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from utils.config import load_yaml_as_ns
from models.build_model import build_model
from datasets.datasets import create_dataloader


def print_feat_stats(name, feat: torch.Tensor):
    print(f"\n[{name}]")
    print("  shape:", tuple(feat.shape))
    print("  mean :", feat.mean().item())
    print("  std  :", feat.std().item())
    print("  min  :", feat.min().item())
    print("  max  :", feat.max().item())


def main(device="cuda"):
    # ---------- 1. load ft config ----------
    ft_cfg_path = os.path.join(ROOT, "config", "ft.yaml")
    print(f"[Info] Loading ft config from: {ft_cfg_path}")
    cfg = load_yaml_as_ns(ft_cfg_path)
    if cfg is None:
        raise RuntimeError("Failed to load ft.yaml")

    # ---------- 2. resolve model config ----------
    if not isinstance(cfg.model, str):
        raise TypeError("cfg.model must be a path string")

    model_cfg_path = cfg.model
    if not os.path.isabs(model_cfg_path):
        model_cfg_path = os.path.join(ROOT, model_cfg_path)

    print(f"[Info] Loading model config from: {model_cfg_path}")
    model_cfg = load_yaml_as_ns(model_cfg_path)
    if model_cfg is None:
        raise RuntimeError("Failed to load model yaml")

    # ---------- 3. build dataloader ----------
    loader = create_dataloader(
        args=None,
        cfg=cfg,
        phase="train"
    )


    # ---------- 4. build model ----------
    model = build_model(model_cfg).to(device)
    model.eval()

    # ---------- 5. fetch one batch ----------
    src, tgt = next(iter(loader))
    for k, v in src.items():
        if torch.is_tensor(v):
            src[k] = v.to(device)
            print("\n[Src Lengths]")
            print("  rgb_len:", src["rgb_len"][:8].tolist())
            print("  kp_len :", src["kp_len"][:8].tolist())
            print("  rgb_mask shape:", tuple(src["rgb_mask"].shape))
            print("  kp_mask  shape:", tuple(src["kp_mask"].shape))

    # ---------- 6. forward encoders ----------
    with torch.no_grad():
        if "rgb_img" in src:
            rgb_feat = model.rgb_encoder(src["rgb_img"])
            print_feat_stats("RGB Encoder Output", rgb_feat)

        if "keypoints" in src:
            pose_feat = model.pose_encoder(src["keypoints"])
            print_feat_stats("Pose Encoder Output", pose_feat)

        if "gt_sentence" in tgt:
            text_feat = model.text_encoder(tgt["gt_sentence"])
            print_feat_stats("Text Encoder Output", text_feat)
        print("\n[Encoder Time Lengths]")
        print("  rgb_feat T':", rgb_feat.shape[1])
        print("  pose_feat T':", pose_feat.shape[1] if pose_feat.dim() == 3 else None)


if __name__ == "__main__":
    main()
