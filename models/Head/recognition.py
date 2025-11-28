# models/Head/recognition.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple


class SinusoidalPositionalEncoding(nn.Module):
    """
    标准正弦位置编码（Transformer 论文同款）
    输入/输出形状：[B, T, D]
    """
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)                       # [T, D]
        position = torch.arange(0, max_len).unsqueeze(1)         # [T, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)             # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)             # 奇数维
        self.register_buffer("pe", pe)                           # 不参与训练
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.size(1)
        x = x + self.pe[:T, :].unsqueeze(0)  # [1, T, D]
        return self.dropout(x)


class RecognitionHeadCTC(nn.Module):
    """
    连续手语识别（CSLR）CTC 头（可训练版）

    形状契约：
      - forward 输入：seq [B, T, in_dim]，可选 src_key_padding_mask [B, T] (bool，True 表示 padding)
      - forward 输出：logits [T, B, V]  （V = num_classes）
      - compute_loss 期望：
            logits [T, B, V]（未过 softmax），
            targets: 1D LongTensor，拼接后的稀疏标签，
            input_lengths: [B]，为每条样本的有效输入长度（与 logits 的时间步一致），
            target_lengths: [B]，为每条样本的目标长度。
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        nhead: int = 8,
        dropout: float = 0.1,
        blank_id: Optional[int] = None,
        use_positional_encoding: bool = True,
        pos_dropout: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.blank_id = (num_classes - 1) if blank_id is None else int(blank_id)

        # 线性映射到 Transformer 维度
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # 位置编码（可关）
        self.pos_encoding = (
            SinusoidalPositionalEncoding(hidden_dim, pos_dropout) if use_positional_encoding else nn.Identity()
        )

        # Transformer Encoder（batch_first=True 便于与 [B, T, D] 对齐）
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # 分类头
        self.classifier = nn.Linear(hidden_dim, num_classes)

    # --------- 前向 ---------
    def forward(
        self,
        seq: torch.Tensor,                        # [B, T, in_dim]
        src_key_padding_mask: Optional[torch.Tensor] = None  # [B, T] (bool), True=padding
    ) -> torch.Tensor:
        """
        返回 logits，形状 [T, B, V]，以便直接对接 torch.nn.CTCLoss
        """
        # 预处理 & 位置编码
        x = self.input_proj(seq)                  # [B, T, H]
        x = self.pos_encoding(x)                  # [B, T, H]

        # 注意 mask 的 dtype 必须是 bool，True 表示 padding
        if src_key_padding_mask is not None and src_key_padding_mask.dtype != torch.bool:
            src_key_padding_mask = src_key_padding_mask.to(torch.bool)

        # Transformer 编码
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # [B, T, H]

        # 分类到词表
        logits_bt = self.classifier(x)            # [B, T, V]

        # CTC 期望 [T, B, V]
        logits = logits_bt.transpose(0, 1).contiguous()
        return logits

    # --------- 损失 ---------
    def compute_loss(
        self,
        logits: torch.Tensor,             # [T, B, V]
        targets: torch.Tensor,            # [sum(target_lengths)]
        input_lengths: torch.Tensor,      # [B]
        target_lengths: torch.Tensor,     # [B]
    ) -> torch.Tensor:
        """
        计算 CTC 损失。内部做 log_softmax，不要在外面先 softmax。
        """
        # 断言时间维长度匹配
        T, B, _ = logits.shape
        assert int(input_lengths.max()) <= T, \
            f"Input length {int(input_lengths.max())} exceeds logit time {T}"

        # 保证长度张量在同一设备/整型
        dev = logits.device
        input_lengths = input_lengths.to(dev).to(torch.int32)
        target_lengths = target_lengths.to(dev).to(torch.int32)

        log_probs = F.log_softmax(logits, dim=-1)  # [T, B, V]
        loss = F.ctc_loss(
            log_probs,
            targets.to(dev),
            input_lengths,
            target_lengths,
            blank=self.blank_id,
            reduction="mean",
            zero_infinity=True,
        )
        return loss

    # --------- 贪婪解码（用于评测） ---------
    @torch.no_grad()
    def ctc_greedy_decode(
        self,
        logits: torch.Tensor,             # [T, B, V]
        collapse_repeats: bool = True,
    ) -> List[List[int]]:
        """
        返回长度为 B 的列表，每个元素是去重&去 blank 的 token id 序列
        """
        pred = logits.argmax(dim=-1)      # [T, B]
        pred = pred.transpose(0, 1)       # [B, T]
        results: List[List[int]] = []
        for seq in pred:
            out = []
            prev = None
            for p in seq.tolist():
                if collapse_repeats and p == prev:
                    prev = p
                    continue
                if p != self.blank_id:
                    out.append(p)
                prev = p
            results.append(out)
        return results

    # --------- 简单 CER（字符级编辑距离 / 参考可选） ---------
    @staticmethod
    def _edit_distance(a: List[int], b: List[int]) -> int:
        """标准 Levenshtein 距离（O(n*m) 动态规划），避免外部依赖"""
        n, m = len(a), len(b)
        dp = [[0]*(m+1) for _ in range(n+1)]
        for i in range(n+1): dp[i][0] = i
        for j in range(m+1): dp[0][j] = j
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = 0 if a[i-1] == b[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # 删除
                    dp[i][j-1] + 1,      # 插入
                    dp[i-1][j-1] + cost  # 替换
                )
        return dp[n][m]

    @torch.no_grad()
    def compute_metrics(
        self,
        features: torch.Tensor,                 # [B, T, in_dim]
        targets_packed: torch.Tensor,           # [sum(target_lengths)]
        input_lengths: torch.Tensor,            # [B]
        target_lengths: torch.Tensor,           # [B]
        src_key_padding_mask: Optional[torch.Tensor] = None,
        decode_for_cer: bool = False
    ) -> Dict[str, float]:
        """
        计算评测指标：
          - ctc_loss（必有）
          - 可选：CER（需要把 packed target 还原为逐样本目标）
        """
        logits = self.forward(features, src_key_padding_mask=src_key_padding_mask)   # [T, B, V]
        loss = self.compute_loss(logits, targets_packed, input_lengths, target_lengths)

        metrics = {
            "ctc_loss": float(loss.item()),
            "perplexity": float(torch.exp(loss).item()),
        }

        if decode_for_cer:
            # 1) 解码预测
            pred_ids = self.ctc_greedy_decode(logits, collapse_repeats=True)  # List[List[int]]，长度 B

            # 2) 还原目标序列（从 packed 还原）
            tlist: List[List[int]] = []
            offset = 0
            tlens = target_lengths.tolist()
            for L in tlens:
                tlist.append(targets_packed[offset:offset+L].tolist())
                offset += L

            # 3) 计算 CER
            total_edits, total_chars = 0, 0
            for hyp, ref in zip(pred_ids, tlist):
                total_edits += self._edit_distance(hyp, ref)
                total_chars += max(1, len(ref))
            cer = total_edits / total_chars
            metrics["cer"] = float(cer)

        return metrics

    def __repr__(self):
        return (f"RecognitionHeadCTC(in_dim={self.input_proj[0].in_features}, "
                f"num_classes={self.num_classes}, "
                f"hidden_dim={self.input_proj[0].out_features}, "
                f"n_layers={len(self.encoder.layers)}, "
                f"trainable=True, blank_id={self.blank_id})")


class RecognitionFinetuner:
    """
    匹配 CSLDailyDataset 返回格式：
    return vid, pose_sample, gloss_ids, support
    """
    def __init__(self, cfg, device, rgb_encoder, head):
        self.cfg = cfg
        self.device = device
        self.rgb = rgb_encoder.to(device)
        self.head = head.to(device)

        self.optimizer = torch.optim.AdamW(
            list(self.rgb.parameters()) + list(self.head.parameters()),
            lr=float(cfg.lr_head)
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)

    # ----------------------------------------
    # 关键修复1：目标拼接
    # ----------------------------------------
    def _pack_targets(self, gloss_ids_list: List[torch.Tensor]):
        """
        gloss_ids_list: List[tensor([tokens])]
        """
        target_lengths = torch.tensor(
            [len(g) for g in gloss_ids_list], dtype=torch.long
        )
        packed = torch.cat(gloss_ids_list, dim=0) if len(gloss_ids_list) > 0 else torch.tensor([], dtype=torch.long)
        return packed, target_lengths

    # ----------------------------------------
    # 训练
    # ----------------------------------------
    def train_epoch(self, loader):
        self.rgb.train()
        self.head.train()

        total = 0
        for batch in loader:
            # Dataset 返回结构：
            # vids, pose, gloss_ids, support
            vids, pose, gloss_ids, support = batch

            rgb = support["rgb_img"].to(self.device)          # [B,T,3,224,224]
            mask = support["attn_mask"].to(self.device)       # [B,T]

            # 打包 label
            packed_targets, target_lengths = self._pack_targets(gloss_ids)
            input_lengths = mask.sum(dim=1).long()

            with torch.cuda.amp.autocast(enabled=True):
                feat = self.rgb(rgb)                           # [B,T,F]
                logits = self.head(feat, src_key_padding_mask=~mask.bool())
                loss = self.head.compute_loss(
                    logits, packed_targets, input_lengths, target_lengths
                )

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            total += loss.item()

        return {"loss": total / len(loader)}

    # ----------------------------------------
    # 验证
    # ----------------------------------------
    @torch.no_grad()
    def eval_epoch(self, loader):
        self.rgb.eval()
        self.head.eval()

        total = 0
        for batch in loader:
            vids, pose, gloss_ids, support = batch

            rgb = support["rgb_img"].to(self.device)
            mask = support["attn_mask"].to(self.device)

            packed_targets, target_lengths = self._pack_targets(gloss_ids)
            input_lengths = mask.sum(dim=1).long()

            feat = self.rgb(rgb)
            logits = self.head(feat, src_key_padding_mask=~mask.bool())
            loss = self.head.compute_loss(
                logits, packed_targets, input_lengths, target_lengths
            )

            total += loss.item()

        return {"loss": total / len(loader)}

