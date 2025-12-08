# models/Encoder/pose_encoder.py
import torch
import torch.nn as nn


class PoseEncoder(nn.Module):
    """
    Pose Encoder placeholder (simple Transformer)
    Input:  (B, T, 21, 3)
    Output: (B, T, hidden_dim)
    """

    def __init__(self, cfg, hidden_dim):
        super().__init__()

        bb_cfg = cfg.backbone
        proj_cfg = cfg.proj

        input_dim = 21 * 3

        # Backbone = simple transformer encoder
        self.backbone = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
            ),
            num_layers=bb_cfg.output_dim // hidden_dim,
        )

        # Freeze backbone
        if bb_cfg.freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Projection: flatten (21,3) → hidden_dim
        self.proj = nn.Linear(input_dim, hidden_dim)

        if proj_cfg.freeze:
            for p in self.proj.parameters():
                p.requires_grad = False

    def forward(self, keypoints):
        """
        keypoints: (B, T, 21, 3)
        """
        B, T, J, C = keypoints.shape
        x = keypoints.reshape(B, T, -1)  # → (B, T, 63)
        x = self.proj(x)                 # → (B, T, hidden_dim)

        x = x.permute(1, 0, 2)           # Transformer wants (T, B, C)
        x = self.backbone(x)             # → (T, B, hidden_dim)
        x = x.permute(1, 0, 2)

        return x
