# -*- coding: utf-8 -*-
import os, glob, pickle, random
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from pathlib import Path

# ======================
# 工具函数
# ======================

def read_map_txt(path: str, allow_single_token: bool = True):
    """健壮的 map 解析：支持单列自动编号/空行/注释/多分隔符。"""
    stoi, itos, auto_id = {}, {}, 0

    def add_pair(tok, idx):
        nonlocal stoi, itos
        if tok in stoi:
            return
        stoi[tok] = idx
        if idx not in itos:
            itos[idx] = tok

    if not os.path.exists(path):
        return stoi, itos

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip().lstrip("\ufeff")
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            parts = None
            for sep in ["\t", " ", ":"]:
                if sep in line:
                    parts = line.split() if sep == " " else line.split(sep)
                    if len(parts) >= 2:
                        parts = [parts[0], parts[1]]
                        break
            if parts is None or len(parts) == 1:
                if allow_single_token:
                    add_pair(line, auto_id)
                    auto_id += 1
                    continue
                else:
                    raise ValueError(f"Expect two fields: {line}")
            a, b = parts[0], parts[1]
            if b.isdigit() and not a.isdigit():
                tok, idx = a, int(b)
            elif a.isdigit() and not b.isdigit():
                tok, idx = b, int(a)
            else:
                try:
                    tok, idx = a, int(b)
                except ValueError:
                    tok, idx = a, auto_id
                    auto_id += 1
            add_pair(tok, idx)
    return stoi, itos


def list_frame_paths(frame_dir: str) -> List[str]:
    """列出帧并按数值名排序，过滤 0 字节等明显异常文件。"""
    if not frame_dir or not os.path.isdir(frame_dir):
        return []
    pats = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        pats.extend(glob.glob(os.path.join(frame_dir, ext)))
    # 过滤 0 字节
    pats = [p for p in pats if os.path.getsize(p) > 0]
    if not pats:
        return []

    def key_fn(p):
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            return int(stem)
        except ValueError:
            return stem

    return sorted(pats, key=key_fn)


def uniform_sample_indices(num_frames: int, T: int, random_offset: bool = True) -> List[int]:
    """分段均匀采样索引。"""
    if num_frames <= 0:
        return [0] * T
    if T <= 1:
        return [min(num_frames - 1, 0)]
    seg = np.linspace(0, num_frames, num=T + 1, dtype=np.int32)
    starts, ends = seg[:-1], np.clip(seg[1:], 1, num_frames)
    if random_offset:
        return [random.randrange(s, e) if e > s else min(s, num_frames - 1) for s, e in zip(starts, ends)]
    else:
        return [(s + e - 1) // 2 for s, e in zip(starts, ends)]


def load_frames_clip(frame_paths: List[str], indices: List[int], transform=None, size=(224, 224), debug: bool = False) -> torch.Tensor:
    """
    载入指定 indices 的帧并变换，输出 [T,C,H,W]。
    遇到坏帧：向左右邻近(最多±5帧)寻找替代；若仍失败，用黑图占位。
    """
    T = len(indices)
    if not frame_paths:
        return torch.zeros(T, 3, size[1], size[0])

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    out = []
    n = len(frame_paths)

    def try_open(k: int):
        try:
            img = Image.open(frame_paths[k])
            img = img.convert("RGB")
            return img
        except (UnidentifiedImageError, OSError, ValueError):
            return None

    for i in indices:
        j = min(max(i, 0), n - 1)
        img = try_open(j)
        if img is None:
            # 就近搜索可读帧
            found = False
            for off in range(1, min(6, n)):  # ±1..±5
                for cand in (j - off, j + off):
                    if 0 <= cand < n:
                        img = try_open(cand)
                        if img is not None:
                            found = True
                            if debug:
                                print(f"[WARN] Bad frame '{frame_paths[j]}', fallback -> '{frame_paths[cand]}'")
                            break
                if found:
                    break
            if img is None:
                if debug:
                    print(f"[WARN] Bad frame '{frame_paths[j]}' and neighbors; use black image.")
                img = Image.new("RGB", size)

        out.append(transform(img))

    return torch.stack(out, dim=0)


def load_meta_pkl(pkl_path: str) -> Dict[str, Dict[str, Any]]:
    """
    统一标准化为：
        meta[id] = {
          "text_char":  "<无空格字符级>",
          "text_word":  "<空格分隔词级>",
          "text_gloss": "<空格分隔gloss>",
          "length": int|None,
          "signer": int|None,
        }
    兼容 CSL-Daily 官方结构：{"info": [{name, length, label_char, label_word, label_gloss, ...}], ...}
    并保留通用兜底逻辑。
    """
    meta_std: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(pkl_path):
        return meta_std

    obj = pickle.load(open(pkl_path, "rb"))

    # 优先：CSL-Daily 官方结构
    if isinstance(obj, dict) and "info" in obj and isinstance(obj["info"], list):
        for rec in obj["info"]:
            if not isinstance(rec, dict):
                continue
            rid = rec.get("name")
            if not rid:
                continue
            txt_char  = "".join(rec.get("label_char", []) or [])
            txt_word  = " ".join(rec.get("label_word", []) or [])
            txt_gloss = " ".join(rec.get("label_gloss", []) or [])
            meta_std[rid] = {
                "text_char":  txt_char,
                "text_word":  txt_word,
                "text_gloss": txt_gloss,
                "length": rec.get("length", None),
                "signer": rec.get("signer", None),
            }
        return meta_std

    # 兜底：其他结构（保持兼容）
    def norm_id(rec):
        for k in ["video_path", "video", "path"]:
            if isinstance(rec, dict) and rec.get(k):
                return Path(rec[k]).stem
        for k in ["name", "id", "uid"]:
            if isinstance(rec, dict) and rec.get(k):
                return Path(str(rec[k])).stem
        return None

    if isinstance(obj, dict):
        if all(isinstance(v, dict) for v in obj.values()):
            for k, rec in obj.items():
                rid = Path(str(k)).stem
                meta_std[rid] = {
                    "text_char":  rec.get("text_char",  ""),
                    "text_word":  rec.get("text_word",  ""),
                    "text_gloss": rec.get("text_gloss", ""),
                    "length": rec.get("length", None),
                    "signer": rec.get("signer", None),
                }
        elif "data" in obj and isinstance(obj["data"], list):
            for rec in obj["data"]:
                if not isinstance(rec, dict):
                    continue
                rid = norm_id(rec)
                if rid is None:
                    continue
                meta_std[rid] = {
                    "text_char":  rec.get("text_char",  ""),
                    "text_word":  rec.get("text_word",  ""),
                    "text_gloss": rec.get("text_gloss", ""),
                    "length": rec.get("length", None),
                    "signer": rec.get("signer", None),
                }
    elif isinstance(obj, list):
        for rec in obj:
            if not isinstance(rec, dict):
                continue
            rid = norm_id(rec)
            if rid is None:
                continue
            meta_std[rid] = {
                "text_char":  rec.get("text_char",  ""),
                "text_word":  rec.get("text_word",  ""),
                "text_gloss": rec.get("text_gloss", ""),
                "length": rec.get("length", None),
                "signer": rec.get("signer", None),
            }
    return meta_std


# ======================
# 数据集（帧目录模式）
# ======================
class CSLDailyDataset(Dataset):
    def __init__(self,
                 root: str,
                 split: str = "train",
                 token_level: str = "char",   # "char" | "word" | "gloss"
                 T: int = 48,
                 random_offset: bool = True,
                 max_text_len: int = 128,
                 use_rgb: bool = True,
                 frame_base: Optional[str] = None,
                 rgb_transform=None,
                 img_size: int = 224,
                 min_frames: int = 1,
                 skip_empty_text: bool = False,
                 min_text_len: int = 1,
                 debug: bool = False):
        super().__init__()
        self.root = str(root)
        self.split = split
        self.token_level = token_level
        self.T = T
        self.random_offset = (random_offset if split == "train" else False)
        self.max_text_len = max_text_len
        self.use_rgb = use_rgb
        self.img_size = int(img_size)
        self.min_frames = int(min_frames)
        self.skip_empty_text = bool(skip_empty_text)
        self.min_text_len = int(min_text_len)
        self.debug = bool(debug)

        # 目录
        sl = os.path.join(self.root, "sentence_label")
        self.frame_base = frame_base if frame_base else os.path.join(self.root, "sentence")

        # 词表（可选）
        self.char_map, _ = read_map_txt(os.path.join(sl, "char_map.txt"))
        self.word_map, _ = read_map_txt(os.path.join(sl, "word_map.txt"))
        self.gloss_map, _ = read_map_txt(os.path.join(sl, "gloss_map.txt"))

        # 元数据（从 pkl.info 读取）
        self.meta = load_meta_pkl(os.path.join(sl, "csl2020ct_v2.pkl"))

        # split → id 列表
        raw_ids = self._read_split_ids(os.path.join(sl, "split_1.txt"), split)

        # 绑定 id → 帧目录（存在且帧数≥min_frames）
        self.id2dir: Dict[str, str] = {}
        fb = Path(self.frame_base)
        missing, too_few = [], []
        for vid in raw_ids:
            d = fb / vid
            if not d.is_dir():
                missing.append((vid, str(d)))
                continue
            cnt = len(list_frame_paths(str(d)))
            if cnt < self.min_frames:
                too_few.append((vid, str(d), cnt))
                continue
            self.id2dir[vid] = str(d)

        self.items = [vid for vid in raw_ids if vid in self.id2dir]

        # 若为空，回退使用 frame_base 下所有子目录（同样按 min_frames 过滤）
        if len(self.items) == 0 and os.path.isdir(self.frame_base):
            fallback = []
            for p in sorted(os.listdir(self.frame_base)):
                d = fb / p
                if d.is_dir():
                    cnt = len(list_frame_paths(str(d)))
                    if cnt >= self.min_frames:
                        fallback.append(p)
                        self.id2dir[p] = str(d)
            self.items = fallback

        # 过滤空文本（可选）
        if self.skip_empty_text and len(self.items) > 0:
            kept, dropped = [], []
            for vid in self.items:
                raw_text, _ = self._get_texts(vid)
                if len(self.encode_text(raw_text)) >= self.min_text_len:
                    kept.append(vid)
                else:
                    dropped.append(vid)
            if self.debug:
                print(f"[DEBUG] text filter: kept={len(kept)} dropped={len(dropped)} (min_text_len={self.min_text_len})")
            self.items = kept

        # 变换
        self.rgb_transform = rgb_transform or transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # debug 输出
        if self.debug:
            if missing:
                print(f"[DEBUG] missing dirs: {len(missing)} (show ≤10)")
                for vid, d in missing[:10]:
                    print("  -", vid, "->", d)
            if too_few:
                print(f"[DEBUG] too few frames: {len(too_few)} (show ≤10)")
                for vid, d, c in too_few[:10]:
                    print("  -", vid, "->", d, "frames=", c)

    def _read_split_ids(self, split_txt: str, split: str) -> List[str]:
        """
        读取 split_1.txt，兼容：
          - 竖线：name|split（含表头）
          - 空格/Tab：`train vid` 或单列 `vid`
          - 路径/后缀：取 stem
          - split 同义：dev == val
        """
        want = split.lower()
        if want == "val":
            want_alias = {"val", "dev", "valid", "validation"}
        elif want == "train":
            want_alias = {"train", "tr", "training"}
        elif want == "test":
            want_alias = {"test", "te"}
        else:
            want_alias = {want}

        def norm_vid(x: str) -> str:
            return Path(x.strip().strip('",')).stem

        ids = []
        if not os.path.exists(split_txt):
            # 没 split 文件：直接扫描帧目录
            if os.path.isdir(self.frame_base):
                for p in sorted(os.listdir(self.frame_base)):
                    if os.path.isdir(os.path.join(self.frame_base, p)):
                        ids.append(p)
            return ids

        with open(split_txt, "r", encoding="utf-8") as f:
            first = True
            for raw in f:
                line = raw.strip().lstrip("\ufeff")
                if not line or line.startswith("#") or line.startswith("//"):
                    continue
                # 竖线分隔：name|split
                if "|" in line:
                    if first and ("name" in line and "split" in line):
                        first = False
                        continue
                    first = False
                    name, sp = [x.strip() for x in line.split("|", 1)]
                    if sp.lower() in want_alias:
                        ids.append(norm_vid(name))
                    continue
                # 空格/Tab 分隔
                parts = line.split()
                first = False
                if len(parts) >= 2:
                    sp, vid_raw = parts[0], parts[-1]
                    if sp.lower() in want_alias:
                        ids.append(norm_vid(vid_raw))
                else:
                    ids.append(norm_vid(parts[0]))
        return ids

    def _get_texts(self, vid: str):
        """返回 (raw_text_str, raw_gloss_tokens_or_None)，按 token_level 从 meta 取。"""
        if vid in self.meta:
            m = self.meta[vid]
            if self.token_level == "char":
                return m.get("text_char", ""), None
            elif self.token_level == "word":
                return m.get("text_word", ""), None
            else:  # gloss
                raw = m.get("text_gloss", "")
                toks = raw.split() if raw else None
                return raw, toks
        else:
            # 回退：尝试 sentence/<vid>.txt
            txt_path = os.path.join(self.frame_base, f"{vid}.txt")
            raw = open(txt_path, "r", encoding="utf-8").read().strip() if os.path.exists(txt_path) else ""
            return raw, None

    def encode_text(self, raw: str) -> torch.Tensor:
        if self.token_level == "char":
            if self.char_map:
                ids = [self.char_map.get(ch, self.char_map.get("<unk>", 0)) for ch in list(raw)]
            else:
                ids = list(raw.encode("utf-8"))  # 无词表时用字节占位（仅为避免空）
        elif self.token_level == "word":
            toks = raw.strip().split()
            ids = [self.word_map.get(w, self.word_map.get("<unk>", 0)) for w in toks] if self.word_map else [hash(w) % 100003 for w in toks]
        else:
            toks = raw.strip().split()
            ids = [self.gloss_map.get(g, self.gloss_map.get("<unk>", 0)) for g in toks] if self.gloss_map else [hash(g) % 100003 for g in toks]
        return torch.tensor(ids[:self.max_text_len], dtype=torch.long)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i) -> Dict[str, Any]:
        vid = self.items[i]
        frame_dir = self.id2dir[vid]
        raw_text, raw_gloss = self._get_texts(vid)

        # 文本
        text_ids = self.encode_text(raw_text)
        out = {
            "id": vid,
            "text": text_ids,
            "text_len": int(text_ids.shape[0]),
            "raw_text": raw_text,
        }

        # RGB clip
        if self.use_rgb:
            frames = list_frame_paths(frame_dir)
            n = len(frames)
            if n == 0:
                out["video"] = torch.zeros(self.T, 3, self.img_size, self.img_size)
                out["video_len"] = 0
            else:
                idxs = uniform_sample_indices(n, self.T, random_offset=self.random_offset)
                clip = load_frames_clip(
                    frames, idxs,
                    transform=self.rgb_transform,
                    size=(self.img_size, self.img_size),
                    debug=self.debug
                )
                out["video"] = clip
                out["video_len"] = clip.shape[0]
        else:
            out["video"] = None
            out["video_len"] = 0

        # gloss（若有）
        if raw_gloss:
            toks = raw_gloss if isinstance(raw_gloss, (list, tuple)) else str(raw_gloss).split()
            gloss_ids = [self.gloss_map.get(g, self.gloss_map.get("<unk>", 0)) for g in toks] if self.gloss_map else [hash(g) % 100003 for g in toks]
            out["gloss"] = torch.tensor(gloss_ids, dtype=torch.long)
        else:
            out["gloss"] = None

        out["pose"] = None  # 当前无 pose
        return out


# ======================
# collate
# ======================
def csl_collate(batch, pad_id: int = 0, pad_video: bool = True):
    # text
    text_seqs = [b["text"] for b in batch]
    text_lens = torch.tensor([b["text_len"] for b in batch], dtype=torch.long)
    text_pad = pad_sequence(text_seqs, batch_first=True, padding_value=pad_id)

    out = {
        "ids": [b["id"] for b in batch],
        "text": text_pad,
        "text_len": text_lens,
        "raw_text": [b["raw_text"] for b in batch],
    }

    # video
    has_video = ("video" in batch[0]) and (batch[0]["video"] is not None)
    if has_video:
        if pad_video:
            T_max = max(b["video"].shape[0] for b in batch)
            vids = []
            for b in batch:
                v = b["video"]
                if v.shape[0] < T_max:
                    pad_t = T_max - v.shape[0]
                    v = torch.cat([v, v[-1:].repeat(pad_t, 1, 1, 1)], dim=0)
                vids.append(v)
            out["video"] = torch.stack(vids, dim=0)  # [B,T,C,H,W]
            out["video_len"] = torch.tensor([b["video_len"] for b in batch], dtype=torch.long)
        else:
            out["video"] = [b["video"] for b in batch]
            out["video_len"] = torch.tensor([b["video_len"] for b in batch], dtype=torch.long)
    else:
        out["video"] = None
        out["video_len"] = torch.tensor([0 for _ in batch], dtype=torch.long)

    out["pose"] = [b.get("pose", None) for b in batch]
    out["gloss"] = [b.get("gloss", None) for b in batch]
    return out


# ======================
# 自测
# ======================
if __name__ == "__main__":
    import argparse, time

    def set_seed(seed: int = 42):
        import numpy as _np, random as _rd
        _rd.seed(seed)
        _np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="root contains sentence/ and sentence_label/")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--frame_base", type=str, default="", help="override frames base; default <root>/sentence")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--size", type=int, default=224, help="output spatial size (size x size)")
    parser.add_argument("--token_level", type=str, default="char", choices=["char", "word", "gloss"])
    parser.add_argument("--use_rgb", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--min_frames", type=int, default=1, help="keep ids with at least this many frames")
    parser.add_argument("--skip_empty_text", action="store_true")
    parser.add_argument("--min_text_len", type=int, default=1)

    args = parser.parse_args()
    set_seed(123)

    root = Path(args.root)
    assert root.exists(), f"Root not found: {root}"

    print(f"\n=== Testing split={args.split} | token_level={args.token_level} | T={args.T} | use_rgb={args.use_rgb} ===")

    ds = CSLDailyDataset(
        root=str(root),
        split=args.split,
        token_level=args.token_level,
        T=args.T,
        random_offset=(args.split == "train"),
        use_rgb=args.use_rgb,
        frame_base=(args.frame_base if args.frame_base else None),
        img_size=args.size,
        min_frames=args.min_frames,
        skip_empty_text=args.skip_empty_text,
        min_text_len=args.min_text_len,
        debug=args.debug,
    )

    print(f"[INFO] frame_base = {ds.frame_base}")
    print(f"[INFO] kept items = {len(ds)} (min_frames={args.min_frames})")

    assert len(ds) > 0, f"{args.split} dataset is empty. Check sentence/ dirs and split_1.txt."

    # 取一个样本（debug 模式随机抽一个）
    t0 = time.time()
    idx = random.randrange(len(ds)) if args.debug else 0
    s = ds[idx]
    dt = (time.time() - t0) * 1000
    print(f"Fetched one sample in {dt:.1f} ms ; dataset size = {len(ds)}")
    print(f"[sample] idx={idx} id={s['id']}")
    print(f"  raw_text: {s['raw_text'][:80]}{'...' if len(s['raw_text'])>80 else ''}")
    print(f"  text_len={s['text_len']}  text_ids.shape={tuple(s['text'].shape)}")
    if args.use_rgb:
        print(f"  video.shape={None if s['video'] is None else tuple(s['video'].shape)}  video_len={s['video_len']}")

    # DataLoader & collate
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: csl_collate(b, pad_id=0),
        drop_last=False,
        pin_memory=False,
        persistent_workers=(args.num_workers > 0),
    )
    b = next(iter(loader))
    print(f"Batch text shape: {tuple(b['text'].shape)} ; lens: {b['text_len'].tolist()}")
    if args.use_rgb and b["video"] is not None:
        print(f"Batch video shape: {tuple(b['video'].shape)} ; video_len: {b['video_len'].tolist()}")
    print("\nAll tests finished.")



'''

# 仅文本（最快）
python utils/dataset.py --root /home/pxl416/PeixiLiu/px_proj/pxUni/data/mini_CSL_Daily --batch_size 4 --T 16 --debug

# 加载帧，224 分辨率
python utils/dataset.py --root /home/pxl416/PeixiLiu/px_proj/pxUni/data/mini_CSL_Daily --use_rgb --batch_size 4 --T 16 --size 224 --debug

# 过滤只保留有文本的样本（更稳）
python utils/dataset.py --root /home/pxl416/PeixiLiu/px_proj/pxUni/data/mini_CSL_Daily --use_rgb --skip_empty_text --debug

'''