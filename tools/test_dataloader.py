# tools/test_dataloader.py
# -*- coding: utf-8 -*-
import sys
import torch
import argparse
import yaml
from types import SimpleNamespace
from datasets.datasets import create_dataloader


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def dict_to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
    return d


def inspect_batch(batch):
    print(f"\n[Debug] type(batch) = {type(batch)}")

    # 情况 1：batch 是 (src_input, tgt_input)
    if isinstance(batch, (tuple, list)) and len(batch) == 2 \
            and isinstance(batch[0], dict) and isinstance(batch[1], dict):
        src_input, tgt_input = batch
        print("[Mode] Detected (src_input, tgt_input) tuple")

        # ---- Name ----
        if "name" in src_input:
            print(f"[Name] {src_input['name'][:4]} ... (total {len(src_input['name'])})")

        # ---- Pose ----
        if "keypoints" in src_input and "kp_len" in src_input:
            kp = src_input["keypoints"]
            kp_len = src_input["kp_len"]
            print(f"[Pose] keypoints shape: {kp.shape}")
            print(f"[Pose] kp_len: {kp_len.tolist()}")

        # ---- RGB ----
        if "rgb_img" in src_input and "rgb_len" in src_input:
            rgb = src_input["rgb_img"]
            rgb_len = src_input["rgb_len"]
            print(f"[RGB] rgb_img shape: {rgb.shape}")
            print(f"[RGB] rgb_len: {rgb_len.tolist()}")

        # ---- Segments ----
        if "segments" in src_input:
            segs = src_input["segments"]
            print(f"[Segments] batch size: {len(segs)}")
            for i in range(min(2, len(segs))):
                s = segs[i]
                starts = s.get("starts", [])
                ends = s.get("ends", [])
                texts = s.get("texts", [])
                zipped = list(zip(starts, ends, texts))
                print(f"  Sample {i} segments ({len(zipped)}): {zipped[:3]}")

        # ---- Text ----
        if "gt_sentence" in tgt_input:
            print(f"[Text] {tgt_input['gt_sentence'][:3]}")

    # 情况 2：batch 是单个 dict（flatten 格式）
    elif isinstance(batch, dict):
        print("[Mode] Detected single batch dict")

        print("[Keys]:", list(batch.keys()))

        # ---- Name ----
        if "name" in batch:
            print(f"[Name] {batch['name'][:4]} ... (total {len(batch['name'])})")

        # ---- Pose ----
        if "keypoints" in batch:
            print(f"[Pose] keypoints shape: {batch['keypoints'].shape}")
        if "kp_len" in batch:
            print(f"[Pose] kp_len: {batch['kp_len'].tolist()}")

        # ---- RGB ----
        if "rgb_img" in batch:
            print(f"[RGB] rgb_img shape: {batch['rgb_img'].shape}")
        if "rgb_len" in batch:
            print(f"[RGB] rgb_len: {batch['rgb_len'].tolist()}")

        # ---- Text ----
        if "text" in batch:
            print(f"[Text] {batch['text'][:3]}")
        if "gt_sentence" in batch:
            print(f"[Text(gt_sentence)] {batch['gt_sentence'][:3]}")

        # ---- Segments ----
        if "segments" in batch:
            segs = batch["segments"]
            print(f"[Segments] batch size: {len(segs)}")
            # segs 可能是 list[dict] 或 dict of list，分别处理
            if isinstance(segs, list):
                for i in range(min(2, len(segs))):
                    s = segs[i]
                    starts = s.get("starts", [])
                    ends = s.get("ends", [])
                    texts = s.get("texts", [])
                    zipped = list(zip(starts, ends, texts))
                    print(f"  Sample {i} segments ({len(zipped)}): {zipped[:3]}")
            elif isinstance(segs, dict):
                print("  segments is a dict with keys:", segs.keys())

    else:
        print("[Mode] Unknown batch type, raw print:")
        print(batch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="/home/pxl416/PeixiLiu/px_proj/Uni-SLM/config/finetune1.yaml")
    parser.add_argument("--phase", type=str, default="train")
    args = parser.parse_args()

    try:
        raw_cfg = load_yaml(args.cfg)
        cfg = dict_to_ns(raw_cfg)

        from datasets.datasets import create_dataloader
        dataloader = create_dataloader(args, cfg, phase=args.phase)

        print(f"[Info] Dataset size: {len(dataloader.dataset)}")
        print(f"[Info] Batch size = {dataloader.batch_size}")
        print(f"[Info] Num workers = {dataloader.num_workers}")

        print("\n[Check] Fetching one batch...")

        for batch_idx, batch in enumerate(dataloader):
            print(f"\n==== Batch {batch_idx} ====")
            inspect_batch(batch)
            print("\n[Status] Batch test completed successfully.")
            break

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
/home/pxl416/miniconda/envs/uslm_py310/bin/python /home/pxl416/PeixiLiu/px_proj/Uni-SLM/tools/test_dataloader.py 
[DEBUG] BaseDataset.collate_fn loaded.
[Info] Dataset size: 18401
[Info] Batch size = 2
[Info] Num workers = 2

[Check] Fetching one batch...

==== Batch 0 ====

[Debug] type(batch) = <class 'list'>
[Mode] Detected (src_input, tgt_input) tuple
[Name] ['S002288_P0005_T00', 'S006585_P0000_T00'] ... (total 2)
[Pose] keypoints shape: torch.Size([2, 1, 21, 3])
[Pose] kp_len: [1, 1]
[RGB] rgb_img shape: torch.Size([2, 1, 3, 224, 224])
[RGB] rgb_len: [1, 1]
[Segments] batch size: 2
  Sample 0 segments (0): []
  Sample 1 segments (0): []
[Text] ['团队内只有分工合作才能提高工作效率。', '把花插在玻璃瓶里。']

[Status] Batch test completed successfully.

Process finished with exit code 0

"""

