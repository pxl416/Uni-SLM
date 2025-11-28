# tools/make_gloss_vocab.py
import argparse
import os
from collections import OrderedDict


def build_gloss_vocab(video_map_path, save_path, column="gloss"):
    """
    从 CSL-Daily 的 video_map.txt 生成 gloss_vocab.txt
    column 选择 gloss/word/char （默认 gloss）
    """
    col_idx = {"index": 0, "name": 1, "length": 2, "gloss": 3, "char": 4, "word": 5}
    idx = col_idx[column]

    vocab = OrderedDict()

    with open(video_map_path, "r", encoding="utf-8") as f:
        first = True
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # 跳过表头
            if first and ("index" in line and "name" in line):
                first = False
                continue

            parts = line.split("|")
            if len(parts) <= idx:
                continue

            text = parts[idx].strip()
            for t in text.split():
                if t not in vocab:
                    vocab[t] = len(vocab)

    # 手动加入 [blank] 和 [pad]
    vocab_list = ["<pad>", "<blank>"] + list(vocab.keys())

    with open(save_path, "w", encoding="utf-8") as f:
        for w in vocab_list:
            f.write(w + "\n")

    print(f"[OK] Built gloss vocab: {len(vocab_list)} words → saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_map", default="../data/mini_CSL_Daily/sentence_label/video_map.txt")
    parser.add_argument("--save", default="../data/mini_CSL_Daily/sentence_label/gloss_vocab.txt")
    parser.add_argument("--column", default="gloss", choices=["gloss", "char", "word"])
    args = parser.parse_args()

    build_gloss_vocab(args.video_map, args.save, column=args.column)
