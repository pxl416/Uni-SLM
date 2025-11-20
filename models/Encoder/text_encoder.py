# models/Encoder/text_encoder.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # token_embeddings: [B, L, D], attention_mask: [B, L]
    input_mask_expanded = attention_mask.unsqueeze(-1).type_as(token_embeddings)  # [B, L, 1]
    sum_embeddings = (token_embeddings * input_mask_expanded).sum(dim=1)          # [B, D]
    sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)                     # [B, 1]
    return sum_embeddings / sum_mask                                              # [B, D]

class TextEncoder(nn.Module):
    def __init__(self, model_path: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 return_sequence: bool = False,      # False: 返回句向量 [B,D]；True: 返回序列 [B,L,D]
                 max_length: int = 128):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModel.from_pretrained(model_path)
        self.return_sequence = return_sequence
        self.max_length = max_length

    def forward(self, text_list):
        tokenized = self.tokenizer(
            text_list,
            padding=True,                 # 比 "longest" 更稳健
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        device = next(self.model.parameters()).device
        tokenized = {k: v.to(device) for k, v in tokenized.items()}

        outputs = self.model(**tokenized, return_dict=True)   # ✅ 正确调用方式
        token_embeddings = outputs.last_hidden_state          # [B, L, D]
        attention_mask = tokenized["attention_mask"]          # [B, L]

        if self.return_sequence:
            # 兼容旧接口：返回序列隐藏状态 + mask
            return token_embeddings, attention_mask
        else:
            # 默认：返回句向量
            sentence_embeddings = mean_pooling(token_embeddings, attention_mask)  # [B, D]
            # 可选：归一化，方便对比学习
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            return sentence_embeddings, attention_mask




if __name__ == "__main__":
    # enroll TextEncoder
    encoder = TextEncoder(return_sequence=False)  # 默认返回句向量 [B, D]
    texts = ["你好，世界！", "This is a test sentence.", "手语检索项目"]

    # sentence embeddings
    sent_embeds, mask = encoder(texts)
    print("=== Sentence Embeddings Mode ===")
    print("Type:", type(sent_embeds))
    print("Shape:", sent_embeds.shape)  # [B, D]
    print("Mask Shape:", mask.shape)  # [B, L]
    print("Dtype:", sent_embeds.dtype, "\n")

    # 序列模式
    encoder_seq = TextEncoder(return_sequence=True)
    seq_embeds, mask_seq = encoder_seq(texts)
    print("=== Sequence Mode ===")
    print("Type:", type(seq_embeds))
    print("Shape:", seq_embeds.shape)  # [B, L, D]
    print("Mask Shape:", mask_seq.shape)  # [B, L]
    print("Dtype:", seq_embeds.dtype)


