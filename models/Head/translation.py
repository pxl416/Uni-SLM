# models/Head/translation.py
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)


class TranslationHeadMT5(nn.Module):
    """
    Unified Translation Head

    - 支持 encoder-decoder (如 mT5/T5/BART...) 和 decoder-only (如 LLaMA...)
    - 通过 AutoConfig 判断模型结构:
        * config.is_encoder_decoder = True  -> Seq2Seq 路线
        * False -> Causal LM 路线
    - 参数 mt5_path 现在其实是通用的 model_name_or_path，
      为了兼容 finetune.py 不改名。
    """

    def __init__(
        self,
        mt5_path: str,              # 通用 model_name_or_path
        in_dim: int,
        d_model: int = 768,         # 会被实际 embedding_dim 覆盖，仅保留作配置兼容
        label_smoothing: float = 0.1,
        lang_prompt: str = "Chinese",
        max_target_len: int = 50,
    ):
        super().__init__()

        self.model_name = mt5_path
        self.config = AutoConfig.from_pretrained(mt5_path)
        self.is_seq2seq = bool(getattr(self.config, "is_encoder_decoder", False))

        # 1) 模型加载：根据结构选择 Seq2Seq 或 Causal LM
        if self.is_seq2seq:
            self.lm = AutoModelForSeq2SeqLM.from_pretrained(mt5_path)
        else:
            self.lm = AutoModelForCausalLM.from_pretrained(mt5_path)

        # 2) Tokenizer（统一用 AutoTokenizer）
        self.tok = AutoTokenizer.from_pretrained(mt5_path, use_fast=True)

        # 对于很多 LLaMA 类模型，没有 pad_token；这里兜底设置
        if self.tok.pad_token_id is None:
            if self.tok.eos_token is not None:
                self.tok.pad_token = self.tok.eos_token
            else:
                # 保险一点，强行加一个 pad token
                self.tok.add_special_tokens({"pad_token": "<pad>"})
                self.lm.resize_token_embeddings(len(self.tok))

        self.pad_id = self.tok.pad_token_id

        # 3) 视觉特征投影到 LM 的 embedding 维度
        emb_dim = self.lm.get_input_embeddings().embedding_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.Dropout(0.1),
        )

        self.ls = float(label_smoothing)
        self.lang_prompt = lang_prompt
        self.max_tgt_len = int(max_target_len)

    # -------------------------
    # internal helpers
    # -------------------------
    def _build_prefix(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        构建文字前缀的 embedding + mask
        - 对 seq2seq：送到 encoder 的 embedding
        - 对 causal LM：作为 input_embeddings 的一部分
        """
        texts = [f"Translate sign language video to {self.lang_prompt}: "] * batch_size
        enc = self.tok(texts, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        if self.is_seq2seq:
            # encoder 侧 embedding（如 T5/BART）
            emb_layer = self.lm.get_encoder().embed_tokens
        else:
            # decoder-only 直接用 input_embeddings
            emb_layer = self.lm.get_input_embeddings()

        prefix_emb = emb_layer(enc["input_ids"])  # [B, Lp, D]
        return {"emb": prefix_emb, "mask": enc["attention_mask"]}  # mask: [B, Lp]

    @staticmethod
    def _ensure_attn_mask(mask: Optional[torch.Tensor], length: int, device) -> torch.Tensor:
        """
        将任意形式的 mask 统一为 [B, L]、dtype=torch.long、1=有效、0=padding。
        当 mask=None 时，默认全 1（后面会根据 batch_size expand）。
        """
        if mask is None:
            return torch.ones((1, length), dtype=torch.long, device=device)
        if mask.dtype == torch.bool:
            mask = mask.long()
        else:
            mask = (mask != 0).long()
        return mask

    def _tokenize_targets(self, tgt_texts: List[str], device: torch.device):
        tok = self.tok(
            tgt_texts,
            padding=True,
            truncation=True,
            max_length=self.max_tgt_len,
            return_tensors="pt",
        )
        input_ids = tok["input_ids"].to(device)          # [B, L_tgt]
        attn_mask = tok["attention_mask"].to(device)     # [B, L_tgt]
        return input_ids, attn_mask

    # -------------------------
    # forward (train)
    # -------------------------
    def forward(
        self,
        vis_seq: torch.Tensor,                 # [B, T, in_dim]
        vis_mask: Optional[torch.Tensor],      # [B, T] 任意 0/1 或 bool
        tgt_texts: List[str],                  # 目标文本
    ) -> Dict[str, Any]:

        device = vis_seq.device
        B, T, _ = vis_seq.shape

        # 1) 视觉特征 -> LM embedding 维度
        vis_emb = self.proj(vis_seq)  # [B, T, D]

        # 2) 文本 prefix
        prefix = self._build_prefix(B, device)
        prefix_emb, prefix_mask = prefix["emb"], prefix["mask"]  # [B,Lp,D], [B,Lp]

        # 3) 视觉 mask 标准化
        vis_mask = self._ensure_attn_mask(vis_mask, T, device)
        if vis_mask.size(0) == 1 and vis_mask.size(1) == T:
            vis_mask = vis_mask.expand(B, T)  # [B,T]

        # 4) 目标文本 token 化
        tgt_ids, tgt_mask = self._tokenize_targets(tgt_texts, device)  # [B,Lt], [B,Lt]

        ce = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=self.ls)

        if self.is_seq2seq:
            # -----------------------------
            # encoder-decoder 路线 (如 mT5)
            # prefix + visual -> encoder
            # labels -> decoder
            # -----------------------------
            inputs_embeds = torch.cat([prefix_emb, vis_emb], dim=1)          # [B, Lp+T, D]
            attn_mask = torch.cat([prefix_mask, vis_mask], dim=1).long()     # [B, Lp+T]

            # labels: pad -> -100
            labels = tgt_ids.clone()
            labels[labels == self.pad_id] = -100

            out = self.lm(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                labels=labels,          # 仅用于自动 shift，loss 我们自己算
                return_dict=True,
            )
            logits = out.logits  # [B, L_tgt, V]

            loss = ce(logits.view(-1, logits.size(-1)), labels.view(-1))

            return {
                "loss": loss,
                "logits": logits,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attn_mask,
            }

        else:
            # -----------------------------
            # decoder-only 路线 (如 LLaMA)
            # [prefix_emb | vis_emb | tgt_emb] -> 输入
            # labels_full 在 prefix/vis 位置为 -100，仅 target 位置监督
            # -----------------------------
            emb_layer = self.lm.get_input_embeddings()
            tgt_emb = emb_layer(tgt_ids)  # [B, L_tgt, D]

            inputs_embeds = torch.cat([prefix_emb, vis_emb, tgt_emb], dim=1)   # [B, Lp+T+Lt, D]

            full_attn = torch.cat([prefix_mask, vis_mask, tgt_mask], dim=1).long()  # [B, L_total]

            # 构造 full labels：前半部分全 -100，只在 target 位置写入 token id
            B_, Lp, _ = prefix_emb.shape
            _, T_, _ = vis_emb.shape
            _, Lt = tgt_ids.shape
            assert B_ == B and T_ == T

            labels_full = torch.full(
                (B, Lp + T + Lt),
                fill_value=-100,
                dtype=torch.long,
                device=device,
            )
            labels_full[:, Lp + T: Lp + T + Lt] = tgt_ids
            labels_full[labels_full == self.pad_id] = -100  # padding 也忽略

            out = self.lm(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attn,
                use_cache=False,
                return_dict=True,
            )
            logits = out.logits  # [B, L_total, V]

            loss = ce(logits.view(-1, logits.size(-1)), labels_full.view(-1))

            return {
                "loss": loss,
                "logits": logits,
                "inputs_embeds": inputs_embeds,
                "attention_mask": full_attn,
            }

    # -------------------------
    # prepare_inputs (inference)
    # -------------------------
    @torch.no_grad()
    def prepare_inputs(
        self,
        vis_seq: torch.Tensor,
        vis_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        推理时只构建上下文（prefix + visual）的 inputs_embeds / attention_mask。
        - 对 Seq2Seq：这些是 encoder 输入
        - 对 Causal LM：这些是 prefix context，generate 会从这里继续生成新 token。
        """
        device = vis_seq.device
        B, T, _ = vis_seq.shape

        vis_emb = self.proj(vis_seq)  # [B,T,D]

        prefix = self._build_prefix(B, device)
        prefix_emb, prefix_mask = prefix["emb"], prefix["mask"]

        vis_mask = self._ensure_attn_mask(vis_mask, T, device)
        if vis_mask.size(0) == 1 and vis_mask.size(1) == T:
            vis_mask = vis_mask.expand(B, T)

        inputs_embeds = torch.cat([prefix_emb, vis_emb], dim=1)      # [B, Lp+T, D]
        attn_mask = torch.cat([prefix_mask, vis_mask], dim=1).long() # [B, Lp+T]

        return {"inputs_embeds": inputs_embeds, "attention_mask": attn_mask}

    # -------------------------
    # generate (inference)
    # -------------------------
    @torch.no_grad()
    def generate(
        self,
        prepared: Dict[str, torch.Tensor],
        max_new_tokens: int = 100,
        num_beams: int = 4,
        **gen_kwargs,
    ):
        """
        prepared: 来自 prepare_inputs 的 dict
        对 Seq2Seq：inputs_embeds/attention_mask -> encoder
        对 Causal LM：作为 context，继续生成 max_new_tokens
        """
        return self.lm.generate(
            inputs_embeds=prepared["inputs_embeds"],
            attention_mask=prepared["attention_mask"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            **gen_kwargs,
        )


# 如果你以后想用一个更中性的名字，可以在别处:
# from models.Head.translation import TranslationHeadMT5 as TranslationHead
