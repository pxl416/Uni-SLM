# models/Encoder/text_encoder.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class TextEncoder(nn.Module):
    """
    Unified Text Encoder: backbone + projection
    """

    def __init__(self, cfg, hidden_dim: int):
        super().__init__()

        bb_cfg = cfg.backbone
        proj_cfg = cfg.proj

        self.freeze_backbone = bb_cfg.freeze
        self.freeze_proj = proj_cfg.freeze

        model_name = bb_cfg.name

        # backbone
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)

        backbone_dim = bb_cfg.output_dim
        self.hidden_dim = hidden_dim

        # projection
        if proj_cfg.type == "linear":
            self.proj = nn.Linear(backbone_dim, hidden_dim)
        elif proj_cfg.type == "mlp":
            self.proj = nn.Sequential(
                nn.Linear(backbone_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )
        elif proj_cfg.type == "identity":
            self.proj = nn.Identity()
        else:
            raise ValueError(f"Unknown proj type {proj_cfg.type}")

        # freeze
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        if self.freeze_proj:
            for p in self.proj.parameters():
                p.requires_grad = False

    def forward(self, texts):
        """
        texts: list[str]
        return: (B, hidden_dim)
        """
        device = self.proj.weight.device

        encoded = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=128, return_tensors="pt"
        ).to(device)

        out = self.backbone(**encoded)
        cls_emb = out.last_hidden_state[:, 0]      # (B, backbone_dim)

        return self.proj(cls_emb)
