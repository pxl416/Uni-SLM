# models/Head/recognition.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class RecognitionHeadCTC(nn.Module):
    def __init__(self,
                 in_dim: int,
                 num_classes: int,
                 hidden: int = 512,
                 nlayer: int = 2,
                 dropout: float = 0.1,
                 blank_id: int = None):
        """
        Continuous Sign Language Recognition (CSLR) Head with CTC Loss.
        【仅用于评估，不参与训练】

        Args:
            in_dim: 输入特征维度 (来自 Encoder，例如 PoseEncoder/RGBEncoder 融合后)
            num_classes: 词表大小 (含 blank)
            hidden: Transformer 编码维度
            nlayer: Transformer 层数
            dropout: dropout 概率
            blank_id: blank 的索引，默认是 num_classes-1
        """
        super().__init__()
        self.num_classes = num_classes
        self.blank_id = blank_id if blank_id is not None else num_classes - 1

        # 映射到 hidden 维度
        self.proj = nn.Linear(in_dim, hidden)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=8,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayer)

        # 分类器
        self.classifier = nn.Linear(hidden, num_classes)

        # 🚫 冻结所有参数，不参与训练
        self._freeze_parameters()

    def _freeze_parameters(self):
        """冻结所有参数，确保不参与梯度更新"""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()  # 设置为评估模式

    def forward(self, seq: torch.Tensor, src_key_padding_mask=None) -> torch.Tensor:
        """
        Args:
            seq: [B, T, D]  输入序列特征
            src_key_padding_mask: [B, T]  padding mask (True=padding位置)
        Returns:
            logits: [B, T, V]  分类 logits
        """
        # 使用 torch.no_grad() 确保不计算梯度
        with torch.no_grad():
            x = self.proj(seq)  # [B, T, H]
            x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
            logits = self.classifier(x)  # [B, T, V]
        return logits

    def compute_loss(self, logits, targets, input_lengths, target_lengths) -> torch.Tensor:
        """
        Compute CTC loss for evaluation only.

        Args:
            logits: [B, T, V]
            targets: LongTensor, shape [sum(target_lengths)] 稀疏拼接的标签
            input_lengths: LongTensor, shape [B] 每个样本的输入长度
            target_lengths: LongTensor, shape [B] 每个样本的标签长度
        """
        assert logits.size(1) >= int(input_lengths.max()), \
            f"Input length {int(input_lengths.max())} exceeds logit length {logits.size(1)}"

        with torch.no_grad():
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # [T, B, V]
            return F.ctc_loss(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
                blank=self.blank_id,
                zero_infinity=True
            )

    def compute_metrics(self, features, targets, input_lengths, target_lengths):
        """
        计算识别任务的评估指标
        Returns:
            dict: 包含各种评估指标的字典
        """
        with torch.no_grad():
            # 前向传播获取logits
            logits = self.forward(features)

            # 计算CTC损失
            ctc_loss = self.compute_loss(logits, targets, input_lengths, target_lengths)

            # 可以添加更多评估指标
            metrics = {
                "ctc_loss": ctc_loss.item(),
                "perplexity": torch.exp(ctc_loss).item(),
            }

            # 这里可以添加识别准确率、编辑距离等指标
            # metrics.update(self._compute_accuracy(logits, targets, input_lengths))

            return metrics

    def train(self, mode: bool = True):
        """
        重写train方法，确保始终处于评估模式
        防止意外被设置为训练模式
        """
        return super().train(False)  # 强制保持评估模式

    def __repr__(self):
        return f"RecognitionHeadCTC(in_dim={self.proj.in_features}, " \
               f"num_classes={self.num_classes}, " \
               f"hidden={self.proj.out_features}, " \
               f"nlayer={len(self.encoder.layers)}, " \
               f"frozen=True)"