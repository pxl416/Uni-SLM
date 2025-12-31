# models/Head/temporal_heatmap.py
import torch
import torch.nn as nn


class TemporalHeatmapHead(nn.Module):
    """
    Temporal Heatmap Head
    ---------------------
    Input:
        rgb_feat: Tensor (B, T, D)
    Output:
        temporal_logits: Tensor (B, T)
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        use_conv: bool = False,
    ):
        super().__init__()

        self.use_conv = use_conv

        if use_conv:
            # temporal conv: (B,T,D) -> (B,D,T)
            self.proj = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_dim, 1, kernel_size=1)
            )
        else:
            # frame-wise linear classifier
            self.proj = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )

    def forward(self, rgb_feat):
        """
        Args:
            rgb_feat: Tensor (B, T, D)
        Returns:
            temporal_logits: Tensor (B, T)
        """
        if self.use_conv:
            # (B,T,D) -> (B,D,T)
            x = rgb_feat.transpose(1, 2)
            logits = self.proj(x)              # (B,1,T)
            logits = logits.squeeze(1)         # (B,T)
        else:
            logits = self.proj(rgb_feat)       # (B,T,1)
            logits = logits.squeeze(-1)        # (B,T)

        return logits
