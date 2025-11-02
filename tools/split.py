#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, random

VALID_SPLITS = {"train", "dev", "test"}

def parse_line(line: str):
    """
    支持格式：
    1) 'train sample' 或 'dev sample'（空格分隔，顺序任意）
    2) 'sample|train' / 'sample,train'（用 | 或 ,）
    3) 'sample'（无标签，则返回 (sample, None)）
    返回: (sample_id, split)；split 若未知返回 None
    """
    s = line.strip()
    if not s: return None, None
    # 去掉可能的注释
    if s.startswith("#"): return None, None

    # 先试竖线 / 逗号
    for sep in ("|", ","):
        if sep in s:
            left, right = s.split(sep, 1)
            left, right = left.strip(), right.strip()
            if right.lower() in VALID_SPLITS:
                return left, right.lower()
            if left.lower() in VALID_SPLITS:
                return right, left.lower()
            # 都不是合法 split，则视为未知标签
            return left, None

    # 再试空格
    parts = s.split()
    if len(parts) == 1:
        return parts[0], None
    if len(parts) >= 2:
        a, b = parts[0].strip(), parts[1].strip()
        if a.lower() in VALID_SPLITS:
            return b, a.lower()
        if b.lower() in VALID_SPLITS:
            return a, b.lower()
        # 其他情况：当作无标签
        return parts[0], None

    return None, None


def split_file(input_path, output_folder, ratio=(0.8, 0.1, 0.1), seed=3407):
    random.seed(seed)
    os.makedirs(output_folder, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        raw = [l.rstrip("\n") for l in f]

    train, dev, test, unlabeled, bad = [], [], [], [], []
    for idx, line in enumerate(raw):
        sid, split = parse_line(line)
        if sid is None:
            continue
        if split is None:
            unlabeled.append(sid)
        else:
            if split == "train":
                train.append(sid)
            elif split == "dev":
                dev.append(sid)
            elif split == "test":
                test.append(sid)
            else:
                bad.append((sid, split))

    # 对无标签的样本，按比例随机切分
    if unlabeled:
        random.shuffle(unlabeled)
        n = len(unlabeled)
        n_tr = int(ratio[0] * n)
        n_de = int(ratio[1] * n)
        train += unlabeled[:n_tr]
        dev   += unlabeled[n_tr:n_tr+n_de]
        test  += unlabeled[n_tr+n_de:]

    def save(name, li):
        path = os.path.join(output_folder, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(li))
        print(f"[Saved] {path} ({len(li)} samples)")

    save("split_1_train.txt", train)
    save("split_1_dev.txt",   dev)
    save("split_1_test.txt",  test)

    print(f"[Info] Parsed lines: {len(raw)} | train={len(train)} dev={len(dev)} test={len(test)} unlabeled={len(unlabeled)}")
    if bad:
        print(f"[Warn] Ignored {len(bad)} lines with unknown split labels (not in {VALID_SPLITS}). Example: {bad[:3]}")


if __name__ == "__main__":
    # 按需改成你的绝对路径
    input_path   = "/home/pxl416/PeixiLiu/px_proj/px_data/csl-daily-frames-512x512/sentence_label/split_1.txt"
    output_folder= "/home/pxl416/PeixiLiu/px_proj/px_data/csl-daily-frames-512x512/sentence_label"
    split_file(input_path, output_folder)
