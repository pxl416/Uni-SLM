# tools/test_config_loading.py
# -*- coding: utf-8 -*-

import os
import sys
import yaml
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from utils.config import load_yaml_as_ns
from datasets.datasets import create_dataloader


def check_file(path: str):
    print(f"[Check] Path = {path}")
    print("❌ File not found." if not os.path.exists(path) else "✅ File exists.")


def test_yaml_loading(path: str, name: str):
    print(f"\n=== Testing YAML loading: {name} ===")
    cfg = load_yaml_as_ns(path)
    if cfg is None:
        print(f"❌ Failed to load {name}")
        return None
    print(f"✅ YAML loaded successfully: {name}")
    print(f"    Keys: {list(vars(cfg).keys())}")
    return cfg


def test_gloss_loading(finetune_cfg):
    print("\n=== Testing Gloss Loading ===")

    # 创建一个 SimpleNamespace 用来模拟 args
    args = SimpleNamespace()
    args.cfg = None
    args.phase = "train"

    # 创建 dataloader
    dl = create_dataloader(args, finetune_cfg, phase="train")
    ds = dl.dataset

    print(f"[Info] Dataset class = {ds.__class__.__name__}")
    print(f"[Info] Dataset length = {len(ds)}")

    # 取第一条样本
    sample = ds.get_item_data(0)
    name, pose_sample, text, support = sample

    gloss = support.get("gloss", None)
    gloss_ids = support.get("gloss_ids", None)

    print(f"\nSample name = {name}")
    print(f"Text = {text}")
    print(f"Gloss = {gloss}")
    print(f"Gloss IDs = {gloss_ids}")

    if gloss is None or gloss_ids is None:
        print("❌ Gloss or gloss_ids missing.")
        return

    # 检查 gloss_map 长度
    if hasattr(ds, "gloss2id"):
        print(f"[Info] gloss vocab size = {len(ds.gloss2id)}")

    # 检查映射是否一一对应
    oov = [g for g in gloss if g not in ds.gloss2id]
    if oov:
        print(f"⚠️ OOV gloss: {oov}")
    else:
        print("✅ All gloss tokens found in gloss_map.")

    # 检查 gloss_ids 是否单调有意义
    if len(gloss_ids) != len(gloss):
        print("⚠️ gloss_ids length mismatch (some gloss may be missing)")
    else:
        print("✅ gloss_ids length is correct.")

    print("\nGloss loading test completed.")


def main():
    print("========================================")
    print(" Test 1: Config Loading System")
    print("========================================\n")

    finetune_path = os.path.join(ROOT, "config/ft.yaml")
    check_file(finetune_path)
    finetune_cfg = test_yaml_loading(finetune_path, "finetuner.yaml")

    if finetune_cfg and hasattr(finetune_cfg, "model"):
        model_path = os.path.join(ROOT, finetune_cfg.model)
        check_file(model_path)
        test_yaml_loading(model_path, "model.yaml")

    if hasattr(finetune_cfg, "datasets"):
        for name, rel_path in vars(finetune_cfg.datasets).items():
            full_path = os.path.join(ROOT, rel_path)
            check_file(full_path)
            test_yaml_loading(full_path, f"{name}.yaml")

    print("\n========================================")
    print(" Config Loading Test Completed")
    print("========================================\n")

    # -------- 新增：Gloss 测试 --------
    test_gloss_loading(finetune_cfg)


if __name__ == "__main__":
    main()
