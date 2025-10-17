# -*- coding: utf-8 -*-
import os, glob, pickle, random, yaml
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


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


def load_frames_clip(frame_paths: List[str], indices: List[int], transform=None, size=(224, 224),
                     debug: bool = False) -> torch.Tensor:
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
            txt_char = "".join(rec.get("label_char", []) or [])
            txt_word = " ".join(rec.get("label_word", []) or [])
            txt_gloss = " ".join(rec.get("label_gloss", []) or [])
            meta_std[rid] = {
                "text_char": txt_char,
                "text_word": txt_word,
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
                    "text_char": rec.get("text_char", ""),
                    "text_word": rec.get("text_word", ""),
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
                    "text_char": rec.get("text_char", ""),
                    "text_word": rec.get("text_word", ""),
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
                "text_char": rec.get("text_char", ""),
                "text_word": rec.get("text_word", ""),
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
                 token_level: str = "char",  # "char" | "word" | "gloss"
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
                print(
                    f"[DEBUG] text filter: kept={len(kept)} dropped={len(dropped)} (min_text_len={self.min_text_len})")
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
            ids = [self.word_map.get(w, self.word_map.get("<unk>", 0)) for w in toks] if self.word_map else [
                hash(w) % 100003 for w in toks]
        else:
            toks = raw.strip().split()
            ids = [self.gloss_map.get(g, self.gloss_map.get("<unk>", 0)) for g in toks] if self.gloss_map else [
                hash(g) % 100003 for g in toks]
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
            gloss_ids = [self.gloss_map.get(g, self.gloss_map.get("<unk>", 0)) for g in toks] if self.gloss_map else [
                hash(g) % 100003 for g in toks]
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


# 4) 工厂
# -------------------------
def _get_from_cfg(cfg, dotted_key: str, default=None):
    """
    从 dict 或属性式配置里取值。
    dotted_key 形如: 'datasets.CSL_Daily.rgb_dir'
    """
    cur = cfg
    for k in dotted_key.split("."):
        if isinstance(cur, dict):
            cur = cur.get(k, default)
        else:
            cur = getattr(cur, k, default)
        if cur is default:
            break
    return cur


def _guess_daily_root_from_cfg(cfg) -> str:
    """
    从 cfg 中推断 CSL_Daily 的 root。
    优先使用 datasets.CSL_Daily.root；
    否则若给了 rgb_dir 且形如 <root>/sentence，则取其父目录作为 root。
    """
    root = _get_from_cfg(cfg, "datasets.CSL_Daily.root", None)
    if root:
        return str(root)
    rgb_dir = _get_from_cfg(cfg, "datasets.CSL_Daily.rgb_dir", None) or \
              _get_from_cfg(cfg, "datasets.CSL_Daily.rgb_dirs", None)
    if rgb_dir:
        p = Path(str(rgb_dir)).resolve()
        if p.name.lower() == "sentence":
            return str(p.parent)
        # 否则直接取其父作为 root（尽力推断）
        return str(p.parent)
    raise ValueError(
        "无法从 cfg 推断 CSL_Daily root。请在 config.yaml 中设置 datasets.CSL_Daily.root 或 rgb_dir=.../sentence")


def load_config_from_yaml(config_path: str):
    """从YAML文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloader(
        dataset_name: str,
        split: str,
        cfg,
        batch_size: int = 4,
        num_workers: int = 4,
        use_rgb: bool = True,
        token_level: str = "char",
        T: int = 16,
        img_size: int = 224,
        min_frames: int = 1,
        skip_empty_text: bool = False,
        min_text_len: int = 1,
        debug: bool = False,
):
    """
    构建 DataLoader：
      - 目前实现 CSL_Daily（帧目录 + sentence_label）
      - 其他数据集可按需扩展
    """
    dataset_name = dataset_name.strip()
    split = split.lower()
    assert split in {"train", "val", "dev", "test"}, f"Unknown split: {split}"

    # 如果是字符串路径，则加载配置
    if isinstance(cfg, str):
        cfg = load_config_from_yaml(cfg)

    if dataset_name == "CSL_Daily":
        # 推断 root：应包含 sentence/ 与 sentence_label/
        root = _guess_daily_root_from_cfg(cfg)

        # 从配置中获取参数
        dataset_cfg = _get_from_cfg(cfg, "datasets.CSL_Daily", {})
        temporal_cfg = _get_from_cfg(cfg, "temporal", {})

        # 计算实际采样帧数
        ratio = temporal_cfg.get("ratio", 0.25)
        jitter = temporal_cfg.get("jitter", True)
        min_frames_cfg = temporal_cfg.get("min_frames", 4)
        max_frames_cfg = temporal_cfg.get("max_frames", 32)

        # 使用配置中的参数覆盖默认参数
        actual_T = T
        if split == "train":
            actual_random_offset = jitter
        else:
            actual_random_offset = False

        ds = CSLDailyDataset(
            root=root,
            split=("val" if split == "dev" else split),
            token_level=token_level,
            T=actual_T,
            random_offset=actual_random_offset,
            max_text_len=dataset_cfg.get("max_text_len", 128),
            use_rgb=use_rgb,
            frame_base=dataset_cfg.get("rgb_dir", None),
            rgb_transform=None,  # 使用类内默认
            img_size=img_size,
            min_frames=max(min_frames, min_frames_cfg),  # 取较大值
            skip_empty_text=skip_empty_text,
            min_text_len=min_text_len,
            debug=debug,
        )

        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=csl_collate,
            drop_last=(split == "train"),
        )
        return loader

    elif dataset_name == "CSL_News":
        raise NotImplementedError("CSL_News 的 DataLoader 请使用你现有的实现或单独适配。")

    else:
        raise NotImplementedError(f"不支持的数据集: {dataset_name}")


# 可选：快速自测
if __name__ == "__main__":
    # 使用正确的配置路径
    config_path = '/home/pxl416/PeixiLiu/px_proj/Uni-SLM/config/config.yaml'

    dl = create_dataloader(
        dataset_name="CSL_Daily",
        split="train",
        cfg=config_path,  # 传递路径字符串，函数内部会加载
        batch_size=2,
        num_workers=0,
        use_rgb=True,
        token_level="char",
        T=16,
        img_size=224,
        min_frames=1,
        debug=True,
    )
    print("len(dl.dataset) =", len(dl.dataset))
    if len(dl.dataset) > 0:
        b = next(iter(dl))
        print("Batch keys:", b.keys())
        print("text:", b["text"].shape, "video:", None if b["video"] is None else b["video"].shape)
    else:
        print("数据集为空，请检查数据路径和配置")