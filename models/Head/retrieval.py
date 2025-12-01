# models/Head/retrieval.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
import numpy as np
import logging

from utils.metrics import t2v_metrics, v2t_metrics

logger = logging.getLogger(__name__)

TensorOrArray = Union[torch.Tensor, np.ndarray]


class RetrievalHead(nn.Module):
    """
    Retrieval Head (CLIP-style)
    - 输入：
        * rgb_feat  : [B, Dv] 或 [B, T, Dv]
        * text_feat : [B, Dt] 或 [B, L, Dt]
    """
    def __init__(
        self,
        rgb_in: int = 512,
        text_in: int = 384,
        proj_dim: int = 256,
        temperature: float = 0.07,
        projection_type: str = "linear",
        dropout: float = 0.1,
        trainable: bool = False,
        learnable_tau: bool = False,
    ):
        super().__init__()

        # Force type conversion
        rgb_in = int(rgb_in)
        text_in = int(text_in)
        proj_dim = int(proj_dim)

        self.proj_dim = proj_dim

        # temperature
        init_tau = float(temperature)
        self.register_buffer("_tau_init", torch.tensor(init_tau))
        self.log_tau = nn.Parameter(torch.log(self._tau_init.clone())) if learnable_tau else None

        # ------------------------
        # Projection
        # ------------------------
        if projection_type == "linear":
            self.rgb_proj = nn.Linear(rgb_in, proj_dim)
            self.text_proj = nn.Linear(text_in, proj_dim)

        elif projection_type == "mlp":
            self.rgb_proj = nn.Sequential(
                nn.Linear(rgb_in, proj_dim * 2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(proj_dim * 2, proj_dim)
            )
            self.text_proj = nn.Sequential(
                nn.Linear(text_in, proj_dim * 2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(proj_dim * 2, proj_dim)
            )
        else:
            raise ValueError(f"Unknown projection type {projection_type}")

        self._init_weights()

        if not trainable:
            for p in self.parameters():
                p.requires_grad = False

    # ------------------
    # Init
    # ------------------
    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_init)

    # ------------------
    # Utility
    # ------------------
    @staticmethod
    def _maybe_pool(x: torch.Tensor, mask: Optional[torch.Tensor]):
        if x.ndim == 2:
            return x  # [B,D]
        if mask is None:
            return x.mean(dim=1)  # [B,D]
        mask = mask.to(dtype=x.dtype).unsqueeze(-1)
        s = (x * mask).sum(dim=1)
        l = mask.sum(dim=1).clamp_min(1e-5)
        return s / l

    def current_tau(self):
        if self.log_tau is None:
            return self._tau_init
        return torch.clamp(self.log_tau.exp(), 1e-3, 10.0)

    # ------------------
    # Forward
    # ------------------
    def forward(self, rgb_feat, text_feat, rgb_mask=None, text_mask=None):
        rgb_vec = self._maybe_pool(rgb_feat, rgb_mask)
        txt_vec = self._maybe_pool(text_feat, text_mask)

        rgb_proj = F.normalize(self.rgb_proj(rgb_vec), dim=-1)
        text_proj = F.normalize(self.text_proj(txt_vec), dim=-1)

        return rgb_proj, text_proj

    # ------------------
    # Loss (InfoNCE)
    # ------------------
    def compute_loss(self, rgb_feat, text_feat, rgb_mask=None, text_mask=None, label_smoothing=0.0):
        rgb_proj, text_proj = self.forward(rgb_feat, text_feat, rgb_mask, text_mask)
        sim = text_proj @ rgb_proj.t()
        B = sim.size(0)
        labels = torch.arange(B, device=sim.device)
        tau = self.current_tau()

        ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        loss_t2v = ce(sim / tau, labels)
        loss_v2t = ce(sim.t() / tau, labels)

        return 0.5 * (loss_t2v + loss_v2t)

    # ------------------
    # Metrics
    # ------------------
    @torch.no_grad()
    def compute_metrics(self, rgb_feat, text_feat, rgb_mask=None, text_mask=None, use_temperature=True):
        rgb_proj, text_proj = self.forward(rgb_feat, text_feat, rgb_mask, text_mask)
        sim = text_proj @ rgb_proj.t()
        if use_temperature:
            sim = sim / self.current_tau()

        sim_np = sim.cpu().numpy()
        t2v, _ = t2v_metrics(sim_np, None)
        v2t, _ = v2t_metrics(sim_np.T, None)

        result = {f"t2v/{k}": v for k, v in t2v.items()}
        result.update({f"v2t/{k}": v for k, v in v2t.items()})
        result["mean_R1"] = 0.5 * (t2v["R1"] + v2t["R1"])
        result["tau"] = float(self.current_tau())

        return result
