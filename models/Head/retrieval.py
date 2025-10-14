# models/Head/retrieval.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np
import logging
from utils.metrics import t2v_metrics, v2t_metrics

logger = logging.getLogger(__name__)


class RetrievalHead(nn.Module):
    def __init__(self,
                 rgb_in: int = 512,
                 text_in: int = 384,     # 匹配 TextEncoder 输出
                 proj_dim: int = 256,
                 temperature: float = 0.07,
                 projection_type: str = 'linear',
                 dropout: float = 0.1):
        """
        Retrieval Head for evaluation only.
        【仅用于评估，不参与训练】
        """
        super().__init__()
        self.proj_dim = proj_dim
        self.temperature = temperature

        # 投影层
        if projection_type == 'linear':
            self.rgb_proj = nn.Linear(rgb_in, proj_dim)
            self.text_proj = nn.Linear(text_in, proj_dim)
        elif projection_type == 'mlp':
            self.rgb_proj = nn.Sequential(
                nn.Linear(rgb_in, proj_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(proj_dim * 2, proj_dim)
            )
            self.text_proj = nn.Sequential(
                nn.Linear(text_in, proj_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(proj_dim * 2, proj_dim)
            )
        else:
            raise ValueError(f"Unknown projection type: {projection_type}")

        self._init_weights()
        self._freeze_parameters()

    def _init_weights(self):
        """Xavier 初始化"""
        for proj in [self.rgb_proj, self.text_proj]:
            if isinstance(proj, nn.Linear):
                nn.init.xavier_uniform_(proj.weight)
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)
            elif isinstance(proj, nn.Sequential):
                for layer in proj:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)

    def _freeze_parameters(self):
        """冻结参数，确保不参与训练"""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, rgb_feat: torch.Tensor, text_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            rgb_feat:  [B, Dv]
            text_feat: [B, Dt]
        Returns:
            rgb_proj, text_proj: [B, proj_dim] 投影后的向量
        """
        rgb_proj = F.normalize(self.rgb_proj(rgb_feat), dim=-1, p=2, eps=1e-12)
        text_proj = F.normalize(self.text_proj(text_feat), dim=-1, p=2, eps=1e-12)
        return rgb_proj, text_proj

    @torch.no_grad()
    def compute_metrics(self,
                        rgb_feat: torch.Tensor,
                        text_feat: torch.Tensor,
                        text_query_mask: Optional[torch.Tensor] = None,
                        video_query_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        计算检索任务的评估指标
        """
        rgb_proj, text_proj = self.forward(rgb_feat, text_feat)
        sim_matrix = compute_similarity(rgb_proj, text_proj, self.temperature)
        metrics, _ = calculate_retrieval_metrics(sim_matrix,
                                                 text_query_mask=text_query_mask,
                                                 video_query_mask=video_query_mask)
        return metrics

    def train(self, mode: bool = True):
        """重写，始终保持 eval 模式"""
        return super().train(False)


# =============== 兼容接口 ===============

@torch.no_grad()
def compute_similarity(rgb_proj: torch.Tensor,
                       text_proj: torch.Tensor,
                       temperature: float = 0.07) -> torch.Tensor:
    """
    计算相似度矩阵
    row = text, col = video
    """
    sim = text_proj @ rgb_proj.T
    return sim / temperature


@torch.no_grad()
def calculate_retrieval_metrics(sim_matrix: torch.Tensor,
                                text_query_mask: Optional[torch.Tensor] = None,
                                video_query_mask: Optional[torch.Tensor] = None):
    """
    计算文本→视频 和 视频→文本 的检索指标
    """
    if isinstance(sim_matrix, torch.Tensor):
        sim_np = sim_matrix.detach().cpu().numpy()
    else:
        sim_np = np.asarray(sim_matrix)

    t2v, t2v_ranks = t2v_metrics(sim_np, text_query_mask)
    v2t, v2t_ranks = v2t_metrics(sim_np.T, video_query_mask)

    metrics = {f"t2v/{k}": float(v) for k, v in t2v.items()}
    metrics.update({f"v2t/{k}": float(v) for k, v in v2t.items()})

    if "R1" in t2v and "R1" in v2t:
        metrics["mean_R1"] = 0.5 * (t2v["R1"] + v2t["R1"])

    return metrics, {"t2v_ranks": t2v_ranks, "v2t_ranks": v2t_ranks}
