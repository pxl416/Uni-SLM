# utils/dataset.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, get_worker_info
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms


# ======================
# 1) 统一基类（所有数据集继承它）
# ======================
class BaseDataset(Dataset):
    """
    统一数据集基类：
      - 提供可复现的采样 RNG（按 worker 粒度）
      - 提供通用的 collate_fn（对齐 pose/RGB 时间维，输出统一 schema）
      - 预留 train/val transform 的构建钩子（子类可覆盖）
    子类需在 __init__ 中自行解析 cfg["datasets"][<NAME>] 并规范化路径。
    """
    def __init__(self, args, cfg, phase: str):
        super().__init__()
        self.args = args
        self.cfg = cfg
        self.phase = str(phase).lower()

        # —— 常用超参 —— #
        self.max_length  = int(getattr(args, "max_length", 128))
        self.rgb_support = bool(getattr(args, "rgb_support", True))
        self.seed        = int(getattr(args, "seed", getattr(cfg, "seed", 3407)))
        self.enable_aug  = (self.phase == "train") and bool(getattr(args, "use_aug", False))

        # —— 默认 transform（子类可覆写 build_*_transform 来替换）—— #
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        self.data_transform = self.build_train_transform() if self.phase == "train" else self.build_val_transform()
        # 若子类未覆盖，使用基础 transform
        if self.data_transform is None:
            self.data_transform = base_transform

    # ---- 可选：由子类覆盖，构建 train/val transform ---- #
    def build_train_transform(self):
        return None  # 子类可返回包含数据增强的 Compose

    def build_val_transform(self):
        return None  # 子类可返回仅 resize+normalize 的 Compose

    # ---- 可复现 RNG（按 worker 粒度） ---- #
    def _make_rng(self):
        wi = get_worker_info()
        base = int(getattr(self, "seed", 3407))
        wid  = wi.id if wi is not None else 0
        return np.random.default_rng(base + wid)

    # ---- 采样时间索引（子类可覆盖为分段均匀采样等策略） ---- #
    def _sample_indices(self, duration: int) -> np.ndarray:
        """
        返回升序索引，长度最多为 self.max_length。
        子类如需“分段均匀采样+抖动”，可覆写本方法。
        """
        rng = self._make_rng()
        L = int(self.max_length)
        if duration <= 0:
            return np.arange(1, dtype=np.int64)
        if duration <= L:
            return np.arange(duration, dtype=np.int64)
        idx = rng.choice(duration, size=L, replace=False)
        idx.sort()
        return idx.astype(np.int64)

    # ---- 打包 batch（统一输出 schema） ---- #
    def collate_fn(self, batch):
        """
        期望单条样本为：
          (name: str,
           pose_sample: dict{ 'body': [T,9,3], 'left': [T,21,3], 'right':[T,21,3], 'face_all':[T,18,3] }(Tensor),
           text: str,
           time_idx: np.ndarray或list[int]（可选，随子类决定是否提供）,
           support: dict{ 'rgb_img': [T,3,224,224](Tensor)|None, 'rgb_img_indices': list[int]|ndarray|Tensor }
        输出：
          src_input, tgt_input 两个 dict，字段详见下文。
        """
        # 解包
        name_batch, pose_list, text_batch, support_list = [], [], [], []
        for item in batch:
            # 允许 time_idx 缺省（第四位可忽略）
            if len(item) == 5:
                name, pose_sample, text, _, support = item
            else:
                name, pose_sample, text, support = item
            name_batch.append(name)
            pose_list.append(pose_sample)
            text_batch.append(text)
            support_list.append(support)

        # ---------- 1) 对齐 pose ---------- #
        src_input = {}
        pose_parts = ['body', 'left', 'right', 'face_all']  # 固定顺序

        if not pose_list or not pose_list[0]:
            # 空兜底
            B, Tm = len(batch), self.max_length
            src_input['body']   = torch.zeros((B, Tm,  9, 3))
            src_input['left']   = torch.zeros((B, Tm, 21, 3))
            src_input['right']  = torch.zeros((B, Tm, 21, 3))
            src_input['face_all'] = torch.zeros((B, Tm, 18, 3))
            src_input['attention_mask']   = torch.zeros((B, Tm), dtype=torch.long)
            src_input['src_length_batch'] = torch.zeros(B, dtype=torch.long)
        else:
            # 以 body 的长度为基准
            lengths = torch.LongTensor([pose_list[i][pose_parts[0]].shape[0] for i in range(len(pose_list))])
            T_max = int(lengths.max().item())

            for part in pose_parts:
                padded = []
                for i in range(len(pose_list)):
                    x = pose_list[i][part]  # [T, K, 3]
                    T = x.shape[0]
                    if T < T_max:
                        pad = x[-1:].expand(T_max - T, x.shape[1], x.shape[2])
                        x = torch.cat([x, pad], dim=0)
                    padded.append(x)
                src_input[part] = torch.stack(padded, dim=0)  # [B, T_max, K, 3]

            # attention_mask：1=有效，0=pad
            attn_mask = pad_sequence(
                [torch.ones(L, dtype=torch.long) for L in lengths],
                batch_first=True, padding_value=0
            )
            src_input['attention_mask']   = attn_mask
            src_input['src_length_batch'] = lengths

        src_input['name_batch'] = name_batch

        # ---------- 2) 对齐 RGB ---------- #
        if self.rgb_support:
            rgb_list = [s.get('rgb_img', None) for s in support_list]
            for i, x in enumerate(rgb_list):
                if x is None:
                    rgb_list[i] = torch.zeros(1, 3, 224, 224)

            T_rgb_list = [x.shape[0] for x in rgb_list]
            T_rgb_max  = max(T_rgb_list) if T_rgb_list else 1

            rgb_padded = []
            for x in rgb_list:
                T, C, H, W = x.shape
                if T < T_rgb_max:
                    pad = x[-1:].expand(T_rgb_max - T, C, H, W)
                    x = torch.cat([x, pad], dim=0)
                rgb_padded.append(x)

            src_input['rgb_img'] = torch.stack(rgb_padded, dim=0)  # [B, T_rgb_max, 3, 224, 224]

            # 统一成张量（pad 到同长）
            idx_seqs = [torch.as_tensor(s.get('rgb_img_indices', []), dtype=torch.long) for s in support_list]
            if len(idx_seqs) > 0 and idx_seqs[0].numel() > 0:
                src_input['rgb_img_indices'] = pad_sequence(idx_seqs, batch_first=True, padding_value=0)
            else:
                src_input['rgb_img_indices'] = torch.zeros((len(support_list), 1), dtype=torch.long)

            src_input['rgb_img_len'] = torch.as_tensor(T_rgb_list, dtype=torch.long)
            src_input['has_rgb']     = torch.as_tensor([s.get('rgb_img', None) is not None for s in support_list],
                                                       dtype=torch.bool)

        # ---------- 3) 目标 ---------- #
        tgt_input = {
            'gt_sentence': text_batch,  # list[str]
        }

        return src_input, tgt_input


# ======================
# 2) DataLoader 工厂
# ======================
def create_dataloader(args, cfg, phase: str = 'train') -> DataLoader:
    """
    统一创建 DataLoader：
      - 通过延迟导入选择数据集类（避免循环依赖）
      - 设置 worker 种子与生成器，保证可复现
      - 只传入 args/cfg/phase，路径解析留给子类在 __init__ 中完成
    """
    phase = str(phase).lower()

    # 延迟导入，避免在顶层硬依赖具体数据集；新增数据集只需在这里扩一行
    if args.dataset_name == 'CSL_News':
        from CSLNews import CSLNewsDataset
        dataset_cls = CSLNewsDataset
    elif args.dataset_name == 'CSL_Daily':
        from CSLDaily import CSLDailyDataset
        dataset_cls = CSLDailyDataset
    else:
        raise NotImplementedError(f"不支持的数据集: {args.dataset_name}")

    # 由各子类在 __init__ 里解析 cfg["datasets"][NAME] 并规范化路径
    dataset = dataset_cls(args=args, cfg=cfg, phase=phase)

    # DataLoader 随机性与可复现
    def _seed_worker(worker_id: int):
        worker_seed = torch.initial_seed() % (2 ** 32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    num_workers = int(getattr(args, 'num_workers', 4))
    gen_seed = int(getattr(args, "seed", getattr(cfg, "seed", 3407)))
    g = torch.Generator().manual_seed(gen_seed)

    # 仅在多进程时添加 persistent_workers / prefetch_factor，避免 0 worker 报错
    dl_kwargs = dict(
        dataset=dataset,
        batch_size=int(getattr(args, 'batch_size', 4)),
        shuffle=(phase == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        drop_last=(phase == 'train'),
        worker_init_fn=_seed_worker,
        generator=g,
    )
    if num_workers > 0:
        dl_kwargs.update(persistent_workers=True, prefetch_factor=2)

    return DataLoader(**dl_kwargs)
