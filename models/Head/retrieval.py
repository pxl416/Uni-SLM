# models/Head/retrieval.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class RetrievalHead(nn.Module):
    """
    Simple contrastive retrieval head.
    Input:
      - rgb_feat:  (B, T, D)
      - text_feat: (B, D)
    Output:
      - dict with video/text embeddings and similarity logits
    """

    def __init__(self, cfg, hidden_dim: int):
        super().__init__()

        self.proj_dim = cfg.proj_dim
        self.learnable_tau = getattr(cfg, "learnable_tau", False)
        temp = getattr(cfg, "temperature", 0.07)

        # Projection for video & text
        self.video_proj = nn.Linear(hidden_dim, self.proj_dim)
        self.text_proj = nn.Linear(hidden_dim, self.proj_dim)

        if self.learnable_tau:
            self.log_tau = nn.Parameter(torch.log(torch.tensor(temp, dtype=torch.float32)))
        else:
            self.register_buffer("tau", torch.tensor(temp, dtype=torch.float32))

    def get_temperature(self):
        if hasattr(self, "log_tau"):
            return torch.exp(self.log_tau)
        return self.tau

    def forward(self, rgb_feat: torch.Tensor, text_feat: torch.Tensor):
        """
        rgb_feat:  (B, T, D)
        text_feat: (B, D)
        """
        B, T, D = rgb_feat.shape

        # Temporal pool video features
        vid = rgb_feat.mean(dim=1)          # (B, D)

        # Project to retrieval space
        v = self.video_proj(vid)           # (B, proj_dim)
        t = self.text_proj(text_feat)      # (B, proj_dim)

        # L2-normalize
        v = F.normalize(v, dim=-1)
        t = F.normalize(t, dim=-1)

        # Similarity logits
        logits = v @ t.t()                 # (B, B)
        tau = self.get_temperature()
        logits = logits / tau

        return {
            "video_emb": v,
            "text_emb": t,
            "logits": logits,
            "temperature": tau,
        }
