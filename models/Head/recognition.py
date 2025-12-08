# models/Head/recognition.py
import torch
import torch.nn as nn


class RecognitionHead(nn.Module):
    """
    CTC-style recognition head.
    Input:
      - feat: (B, T, D)
    Output:
      - logits: (B, T, num_classes)
    """

    def __init__(self, cfg, hidden_dim: int):
        super().__init__()

        self.num_classes = cfg.num_classes
        self.blank_id = getattr(cfg, "blank_id", 0)
        num_layers = getattr(cfg, "num_layers", 2)
        dropout = getattr(cfg, "dropout", 0.1)

        # BiLSTM over temporal features
        self.rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        # Linear to vocab (2 * hidden_dim//2 == hidden_dim)
        self.classifier = nn.Linear(hidden_dim, self.num_classes)

    def forward(self, feat: torch.Tensor):
        """
        feat: (B, T, D)
        return: logits (B, T, num_classes)
        """
        x, _ = self.rnn(feat)              # (B, T, D)
        logits = self.classifier(x)        # (B, T, num_classes)
        return logits
