# make_split2.py
import os
from pathlib import Path

root = "/home/pxl416/PeixiLiu/px_proj/Uni-SLM/data/mini_CSL_Daily"
sentence_dir = Path(root) / "sentence"
split_file = Path(root) / "sentence_label" / "split_2.txt"

dirs = sorted([p.name for p in sentence_dir.iterdir() if p.is_dir()])
print(f"Found {len(dirs)} sentence folders.")

# 简单规则：前 70% 训练，后 30% 验证
n = len(dirs)
split_point = int(n * 0.7)
train_ids, val_ids = dirs[:split_point], dirs[split_point:]

with open(split_file, "w", encoding="utf-8") as f:
    f.write("name|split\n")
    for vid in train_ids:
        f.write(f"{vid}|train\n")
    for vid in val_ids:
        f.write(f"{vid}|val\n")

print(f"Wrote {split_file}")
