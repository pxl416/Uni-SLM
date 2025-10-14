# tests/smoke_loader.py
import os
import types
import yaml
import torch

# === 修改这行，指向你的 dataset2.py 路径 ===
from utils.dataset2 import create_dataloader

# ---- 1) 加载 config 并做适配 ----
def load_cfg_with_adapter(yaml_path, dataset_name):
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg_raw = yaml.safe_load(f)

    # 把 cfg_raw 映射到一个对象，让你现有 dataset2.py 能用点号访问
    cfg = types.SimpleNamespace(**cfg_raw)

    # 适配：dataset2.py 期望 cfg.data_path.* 有路径
    ds_map = cfg_raw.get("datasets", {})
    if dataset_name not in ds_map:
        raise KeyError(f"datasets 里找不到 {dataset_name}，可选：{list(ds_map.keys())}")

    data_path = types.SimpleNamespace(**ds_map[dataset_name])
    cfg.data_path = data_path

    return cfg

# ---- 2) 构造 args（dataset2.py 里用到的字段）----
def build_args(dataset_name="CSL_News"):
    A = types.SimpleNamespace()
    A.dataset_name = dataset_name
    A.batch_size = 2
    A.num_workers = 0   # 先用 0，方便本地调试
    A.max_length = 64
    A.rgb_support = True
    A.use_aug = False
    return A

# ---- 3) 跑一个 batch 看看形状 ----
def main():
    yaml_path = "config/config.yaml"  # 修改成你的实际路径
    dataset_name = "CSL_News"

    cfg = load_cfg_with_adapter(yaml_path, dataset_name)
    args = build_args(dataset_name)

    # phase 可选 'train' | 'val' | 'test'
    for phase in ["train"]:
        print(f"\n>>> Building dataloader for {dataset_name} [{phase}] ...")
        dl = create_dataloader(args, cfg, phase=phase)

        # 取一个 batch
        it = iter(dl)
        src_input, tgt_input = next(it)

        print("batch keys in src_input:", list(src_input.keys()))
        print("name_batch len:", len(src_input["name_batch"]))
        print("attention_mask:", tuple(src_input["attention_mask"].shape))

        for part in ["body", "left", "right", "face_all"]:
            if part in src_input:
                print(f"{part}:", tuple(src_input[part].shape))  # [B, T, K, 3]

        if "rgb_img" in src_input:
            print("rgb_img:", tuple(src_input["rgb_img"].shape))  # [B, T, 3, 112, 112]

        print("gt_sentence len:", len(tgt_input["gt_sentence"]))

if __name__ == "__main__":
    main()
