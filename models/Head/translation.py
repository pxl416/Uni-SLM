# models/Head/translation.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MT5ForConditionalGeneration, T5Tokenizer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0))/d_model))
        pe[:, 0::2] = torch.sin(pos*div)
        pe[:, 1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
    def forward(self, x):  # x: [B, L, d_model]
        return x + self.pe[:, :x.size(1), :]

class TranslationHeadMT5(nn.Module):
    def __init__(self, mt5_path, in_dim, d_model=768, pad_id=0, label_smoothing=0.1, lang_prompt="Chinese"):
        super().__init__()
        self.mt5 = MT5ForConditionalGeneration.from_pretrained(mt5_path)
        self.tok = T5Tokenizer.from_pretrained(mt5_path, legacy=False)
        self.proj = nn.Linear(in_dim, d_model)        # 视觉序列→MT5隐藏维
        self.pad_id = pad_id
        self.ls = label_smoothing
        self.lang_prompt = lang_prompt

    def forward(self, vis_seq, vis_mask, tgt_texts):
        # vis_seq: [B,T,Dv]  vis_mask: [B,T] (1=valid or 0=valid都行，注意统一)
        B,T,_ = vis_seq.shape
        vis_emb = self.proj(vis_seq)                  # [B,T,768]

        prefix = self.tok([f"Translate sign language video to {self.lang_prompt}: "]*B,
                          padding=True, return_tensors="pt").to(vis_seq.device)
        prefix_emb = self.mt5.encoder.embed_tokens(prefix["input_ids"])  # [B,Lp,768]

        inputs_embeds = torch.cat([prefix_emb, vis_emb], dim=1)          # [B,Lp+T,768]
        attn_mask = torch.cat([prefix["attention_mask"], vis_mask], dim=1)  # 形状匹配

        lab = self.tok(tgt_texts, padding=True, truncation=True, max_length=50, return_tensors="pt").to(vis_seq.device)["input_ids"]
        lab[lab==self.tok.pad_token_id] = -100

        out = self.mt5(inputs_embeds=inputs_embeds, attention_mask=attn_mask, labels=lab, return_dict=True)
        logits = out.logits.view(-1, out.logits.size(-1))
        loss = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=self.ls)(
            logits, lab.view(-1)
        )
        return {"loss": loss, "inputs_embeds": inputs_embeds, "attention_mask": attn_mask}

    @torch.no_grad()
    def generate(self, pre, max_new_tokens=100, num_beams=4):
        return self.mt5.generate(inputs_embeds=pre["inputs_embeds"],
                                 attention_mask=pre["attention_mask"],
                                 max_new_tokens=max_new_tokens,
                                 num_beams=num_beams)





