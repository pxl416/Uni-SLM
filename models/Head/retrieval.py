# models/Head/retrieval.py
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
    Retrieval Head
    - 输入约定（两种都支持）：
        * 句级：rgb_feat [B, Dv]，text_feat [B, Dt]
        * 时序：rgb_feat [B, T, Dv]，text_feat [B, L, Dt]，可配合 mask 做平均池化
    - 功能：
        * projector -> L2 normalize -> 相似度
        * 评测指标（t2v/v2t）
        * 可选：对称 InfoNCE 训练（仅微调 projector）
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
        learnable_tau: bool = False,   # 新增：让温度可学习（对数形式更稳）
    ):
        super().__init__()
        self.proj_dim = int(proj_dim)
        self.register_buffer("_tau_init", torch.tensor(float(temperature)))
        if learnable_tau:
            # 学习 log_tau，正值约束：tau = exp(log_tau)
            self.log_tau = nn.Parameter(torch.log(self._tau_init.clone()))
        else:
            self.log_tau = None

        # 投影层（与预训练 projector 形状对齐）
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
            raise ValueError(f"Unknown projection type: {projection_type}")

        self._init_weights()

        # 默认只评测：不训练
        if not trainable:
            for p in self.parameters(): p.requires_grad = False

    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        self.apply(_init)

    # ---------- 工具：支持 [B,D] / [B,T,D] 并行 ----------
    @staticmethod
    def _maybe_pool(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        x: [B,D] 或 [B,T,D]；mask: [B,T] (1=valid)
        返回 [B,D]
        """
        if x.ndim == 2:
            return x
        # [B,T,D]
        if mask is None:
            return x.mean(dim=1)
        # 标准化为 float 并避免除零
        m = mask.unsqueeze(-1).to(x.dtype)            # [B,T,1]
        s = (x * m).sum(dim=1)
        d = m.sum(dim=1).clamp_min(1e-5)
        return s / d

    def current_tau(self) -> torch.Tensor:
        if self.log_tau is None:
            return self._tau_init
        # 约束 tau 的范围，避免数值异常
        return torch.clamp(self.log_tau.exp(), min=1e-3, max=10.0)

    # ---------- 前向：返回投影后的单位向量 ----------
    def forward(
        self,
        rgb_feat: torch.Tensor,                  # [B,Dv] 或 [B,T,Dv]
        text_feat: torch.Tensor,                 # [B,Dt] 或 [B,L,Dt]
        rgb_mask: Optional[torch.Tensor] = None, # [B,T] 1=valid
        text_mask: Optional[torch.Tensor] = None # [B,L] 1=valid
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb_vec = self._maybe_pool(rgb_feat, rgb_mask)    # [B,Dv]
        txt_vec = self._maybe_pool(text_feat, text_mask)  # [B,Dt]
        rgb_proj = F.normalize(self.rgb_proj(rgb_vec), dim=-1, eps=1e-12)   # [B,P]
        text_proj = F.normalize(self.text_proj(txt_vec), dim=-1, eps=1e-12) # [B,P]
        return rgb_proj, text_proj

    # ---------- 训练损失（对称 InfoNCE：CLIP 风格） ----------
    def compute_loss(
        self,
        rgb_feat: torch.Tensor,
        text_feat: torch.Tensor,
        rgb_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0
    ) -> torch.Tensor:
        rgb_proj, text_proj = self.forward(rgb_feat, text_feat, rgb_mask, text_mask)  # [B,P]
        sim = text_proj @ rgb_proj.t()                               # [B_text, B_video]
        tau = self.current_tau().to(sim.device)
        B = sim.size(0)
        labels = torch.arange(B, device=sim.device)

        # 行（t->v）+ 列（v->t）平均
        ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        loss_t2v = ce(sim / tau, labels)
        loss_v2t = ce(sim.t() / tau, labels)
        return 0.5 * (loss_t2v + loss_v2t)

    # ---------- 评测指标 ----------
    @torch.no_grad()
    def compute_metrics(
        self,
        rgb_feat: torch.Tensor,
        text_feat: torch.Tensor,
        rgb_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        use_temperature: bool = True
    ) -> Dict[str, float]:
        rgb_proj, text_proj = self.forward(rgb_feat, text_feat, rgb_mask, text_mask)
        sim = text_proj @ rgb_proj.t()
        if use_temperature:
            sim = sim / self.current_tau().to(sim.device)

        sim_np = sim.detach().cpu().numpy()
        t2v, _ = t2v_metrics(sim_np, None)
        v2t, _ = v2t_metrics(sim_np.T, None)

        metrics = {f"t2v/{k}": float(v) for k, v in t2v.items()}
        metrics.update({f"v2t/{k}": float(v) for k, v in v2t.items()})
        if "R1" in t2v and "R1" in v2t:
            metrics["mean_R1"] = 0.5 * (t2v["R1"] + v2t["R1"])
        # 额外记录当前温度
        metrics["tau"] = float(self.current_tau().item())
        return metrics

    # ---------- 从预训练（打包 dict）加载 projector ----------
    def load_proj_from_pretrain(self, proj_state_dict: Dict[str, Dict[str, torch.Tensor]]):
        miss = []
        if "rgb" in proj_state_dict:
            try:
                self.rgb_proj.load_state_dict(proj_state_dict["rgb"], strict=True)
            except Exception as e:
                miss.append(f"rgb_proj load failed: {e}")
        else:
            miss.append("missing key 'rgb' in proj_state_dict")

        if "text" in proj_state_dict:
            try:
                self.text_proj.load_state_dict(proj_state_dict["text"], strict=True)
            except Exception as e:
                miss.appe

# =============== 兼容接口：保留旧的 calculate_retrieval_metrics ===============
@torch.no_grad()
def calculate_retrieval_metrics(
    sim_matrix: torch.Tensor,
    text_query_mask: Optional[torch.Tensor] = None,
    video_query_mask: Optional[torch.Tensor] = None
):
    """
    计算文本→视频 和 视频→文本 的检索指标（兼容旧接口）
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
