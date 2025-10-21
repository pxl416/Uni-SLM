# models/Head/translation.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
from transformers import MT5ForConditionalGeneration, T5TokenizerFast  # 推荐 Fast 版


class TranslationHeadMT5(nn.Module):
    """
    将视觉时序特征接入 mT5 编码器，前面加文本 prefix，再用标准解码器生成目标句子。
    - 视觉侧：seq [B,T,in_dim] 先线性映射到 d_model（默认 768）。
    - prefix：以文本 prompt 的 token 嵌入作为“文字前缀”，再拼接视觉嵌入。
    - attention_mask：1=有效，0=padding（务必统一）
    """

    def __init__(
        self,
        mt5_path: str,
        in_dim: int,
        d_model: int = 768,
        label_smoothing: float = 0.1,
        lang_prompt: str = "Chinese",
        max_target_len: int = 50,
    ):
        super().__init__()
        self.mt5 = MT5ForConditionalGeneration.from_pretrained(mt5_path)
        self.tok = T5TokenizerFast.from_pretrained(mt5_path)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(0.1),
        )
        self.ls = float(label_smoothing)
        self.lang_prompt = lang_prompt
        self.max_tgt_len = int(max_target_len)
        # 方便外部引用
        self.pad_id = self.tok.pad_token_id

    def _build_prefix(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        texts = [f"Translate sign language video to {self.lang_prompt}: "] * batch_size
        prefix = self.tok(texts, padding=True, return_tensors="pt")
        prefix = {k: v.to(device) for k, v in prefix.items()}
        # 将 prefix token 转为嵌入
        prefix_emb = self.mt5.encoder.embed_tokens(prefix["input_ids"])  # [B, Lp, d_model]
        return {"emb": prefix_emb, "mask": prefix["attention_mask"]}

    @staticmethod
    def _ensure_attn_mask(mask: Optional[torch.Tensor], length: int, device) -> torch.Tensor:
        """
        将任意形式的 mask 统一为 [B, L]、dtype=torch.long、1=有效、0=padding。
        当 mask=None 时，默认全 1。
        """
        if mask is None:
            return torch.ones((1, length), dtype=torch.long, device=device)  # 会在外层 expand
        if mask.dtype == torch.bool:
            mask = mask.long()
        else:
            # 把非 0 当作有效
            mask = (mask != 0).long()
        return mask

    def forward(
        self,
        vis_seq: torch.Tensor,                 # [B, T, in_dim]
        vis_mask: Optional[torch.Tensor],      # [B, T], 任意 0/1 或 bool，最终会标准化为 1=valid
        tgt_texts: List[str],                  # 目标文本（batch 列表）
    ) -> Dict[str, Any]:

        device = vis_seq.device
        B, T, _ = vis_seq.shape

        # 1) 视觉到 mT5 维度
        vis_emb = self.proj(vis_seq)                         # [B, T, d_model]

        # 2) 文本 prefix
        prefix = self._build_prefix(B, device)               # {"emb": [B,Lp,D], "mask": [B,Lp]}
        prefix_emb, prefix_mask = prefix["emb"], prefix["mask"]

        # 3) 拼接 encoder inputs
        #    attention_mask 统一：1=有效；dtype=long；device 一致
        vis_mask = self._ensure_attn_mask(vis_mask, T, device)
        if vis_mask.size(0) == 1 and vis_mask.size(1) == T:
            vis_mask = vis_mask.expand(B, T)

        inputs_embeds = torch.cat([prefix_emb, vis_emb], dim=1)          # [B, Lp+T, D]
        attn_mask = torch.cat([prefix_mask, vis_mask], dim=1).long()     # [B, Lp+T]

        # 4) 目标标签
        labels = self.tok(
            tgt_texts,
            padding=True,
            truncation=True,
            max_length=self.max_tgt_len,
            return_tensors="pt"
        )["input_ids"].to(device)
        labels[labels == self.pad_id] = -100  # 忽略 pad

        # 5) 前向：传 labels 以便模型做 shift 并返回 logits；我们自己带 smoothing 计算 loss
        out = self.mt5(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            labels=labels,            # 用于自动 shift
            return_dict=True
        )
        logits = out.logits  # [B, L_tgt, V]

        # 6) Label smoothing CE（自己算，覆盖 out.loss）
        ce = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=self.ls)
        loss = ce(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {
            "loss": loss,
            "logits": logits,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attn_mask,
        }

    @torch.no_grad()
    def prepare_inputs(self, vis_seq: torch.Tensor, vis_mask: Optional[torch.Tensor]):
        """
        推理时的准备函数：只构建 encoder 侧的 inputs_embeds 和 attention_mask。
        """
        device = vis_seq.device
        B, T, _ = vis_seq.shape
        vis_emb = self.proj(vis_seq)

        prefix = self._build_prefix(B, device)
        prefix_emb, prefix_mask = prefix["emb"], prefix["mask"]

        vis_mask = self._ensure_attn_mask(vis_mask, T, device)
        if vis_mask.size(0) == 1 and vis_mask.size(1) == T:
            vis_mask = vis_mask.expand(B, T)

        inputs_embeds = torch.cat([prefix_emb, vis_emb], dim=1)      # [B, Lp+T, D]
        attn_mask = torch.cat([prefix_mask, vis_mask], dim=1).long() # [B, Lp+T]

        return {"inputs_embeds": inputs_embeds, "attention_mask": attn_mask}

    @torch.no_grad()
    def generate(
        self,
        prepared: Dict[str, torch.Tensor],
        max_new_tokens: int = 100,
        num_beams: int = 4,
        **gen_kwargs
    ):
        """
        prepared: 由 prepare_inputs 返回的 dict
        """
        return self.mt5.generate(
            inputs_embeds=prepared["inputs_embeds"],
            attention_mask=prepared["attention_mask"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            **gen_kwargs
        )
