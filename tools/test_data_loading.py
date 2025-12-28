# tools/test_data_loading.py
import os
import sys
import yaml
import argparse
import torch
from types import SimpleNamespace
from datasets.datasets import create_dataloader


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def dict_to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
    return d


def inspect_src(src):
    print("\n[Src Keys]:", list(src.keys()))
    print("rgb_img:", src["rgb_img"].shape)
    print("keypoints:", src["keypoints"].shape)
    print("rgb_len:", src["rgb_len"])
    print("kp_len:", src["kp_len"])
    print("names:", src["name"][:2])


def inspect_tgt(tgt):
    print("\n[Tgt Keys]:", list(tgt.keys()))
    print("Sample text:", tgt["gt_sentence"][:2])
    print("Sample gloss:", tgt["gt_gloss"][:2])


def inspect_rgb_tensor(rgb):
    print("\n[Check RGB Normalize / Resize / Channel]")
    print("shape:", rgb.shape)

    flat = rgb.flatten()
    print("min =", float(flat.min()))
    print("max =", float(flat.max()))
    print("mean =", float(flat.mean()))
    print("std  =", float(flat.std()))


def test_gloss(dataloader):
    print("\n=== Gloss Test ===")

    ds = dataloader.dataset
    name, pose_sample, text, support = ds.get_item_data(0)

    gloss = support.get("gloss", None)
    gloss_ids = support.get("gloss_ids", None)

    print("Sample name:", name)
    print("Gloss:", gloss)
    print("Gloss IDs:", gloss_ids)

    if hasattr(ds, "gloss2id"):
        print("gloss vocab size:", len(ds.gloss2id))



def inspect_rgb_details(src):
    print("\n=== Inspecting RGB Details ===")

    rgb = src["rgb_img"]  # (B,T,C,H,W)
    B, T, C, H, W = rgb.shape

    print(f"[RGB Shape] B={B}, T={T}, C={C}, H={H}, W={W}")

    # --- 检查 resize 是否正确 ---
    print(f"[Resize Check] Expected size = 224x224, Actual = {H}x{W}")

    # --- 检查 channel 是否正确 ---
    print(f"[Channel Check] C = {C} (should match cfg.global_data.channel)")

    # --- 检查 normalize 是否成功 ---
    flat = rgb.flatten()
    mean = flat.mean().item()
    std = flat.std().item()

    print(f"[Normalize Check] mean={mean:.4f}, std={std:.4f}")
    print("  (mean≈0, std≈1 表示归一化成功)")


def inspect_pose_details(src):
    print("\n=== Inspecting Pose Details ===")
    kp = src["keypoints"]  # (B,T,21,3)
    print(f"[Keypoints Shape] {kp.shape}")

    if kp.shape[-2:] != (21, 3):
        print("❌ Pose shape incorrect! Expected (21,3)")
    else:
        print("✅ Pose shape correct.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="../config/ft.yaml")
    parser.add_argument("--phase", default="train")
    args = parser.parse_args()

    raw_cfg = load_yaml(args.cfg)
    cfg = dict_to_ns(raw_cfg)

    print("\nLoading dataloader...")
    dataloader = create_dataloader(args, cfg, phase=args.phase)

    print("[Info] Dataset size:", len(dataloader.dataset))
    print("[Info] batch_size:", dataloader.batch_size)

    for batch_idx, (src_input, tgt_input) in enumerate(dataloader):
        print(f"\n==== Batch {batch_idx} ====")

        inspect_src(src_input)
        inspect_tgt(tgt_input)

        inspect_rgb_details(src_input)
        inspect_pose_details(src_input)

        break

    test_gloss(dataloader)
    print("\nTest completed.\n")


if __name__ == "__main__":
    main()
