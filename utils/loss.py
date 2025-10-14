# utils/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def cosine_similarity_matrix(a, b):
    a_norm = F.normalize(a, dim=1)
    b_norm = F.normalize(b, dim=1)
    return a_norm @ b_norm.T

def contrastive_loss(sim_matrix: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
    return F.cross_entropy(sim_matrix / temperature, labels)

def mutual_prediction_loss(pred_a: torch.Tensor, target_a: torch.Tensor,
                           pred_b: torch.Tensor, target_b: torch.Tensor) -> torch.Tensor:
    loss_a = F.mse_loss(pred_a, target_a)
    loss_b = F.mse_loss(pred_b, target_b)
    return 0.5 * (loss_a + loss_b)

def triplet_loss(anchor, positive, negative, margin=1.0):
    pos_dist = F.cosine_similarity(anchor, positive)
    neg_dist = F.cosine_similarity(anchor, negative)
    return F.relu(neg_dist - pos_dist + margin).mean()

def kl_divergence_loss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

# --- 新增：安全读取嵌套配置 ---
def ns_get(ns, path, default=None):
    cur = ns
    for key in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, None)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur

def build_loss(cfg):
    """根据 config 返回对应的 loss 函数（安全读取配置）"""
    task        = ns_get(cfg, "Pretraining.task", "contrastive")
    loss_type   = ns_get(cfg, "Pretraining.loss", "infoNCE")
    temperature = ns_get(cfg, "Pretraining.temperature", 0.07)

    if task == "contrastive" or str(loss_type).lower() in ["infonce", "contrastive"]:
        def contrastive_fn(features):
            # 期望 features 包含 'sim_matrix'
            sim = features.get('sim_matrix', None)
            if sim is None:
                raise KeyError("features['sim_matrix'] is required for contrastive loss")
            return contrastive_loss(sim, temperature)
        return contrastive_fn

    elif task == "mutual_prediction":
        def mutual_fn(features):
            req = ["pred_a", "target_a", "pred_b", "target_b"]
            missing = [k for k in req if k not in features]
            if missing:
                raise KeyError(f"Missing keys for mutual_prediction_loss: {missing}")
            return mutual_prediction_loss(
                features['pred_a'], features['target_a'],
                features['pred_b'], features['target_b']
            )
        return mutual_fn

    elif str(loss_type).lower() == "triplet":
        margin = ns_get(cfg, "Pretraining.margin", 1.0)
        def triplet_fn(features):
            req = ["anchor", "positive", "negative"]
            missing = [k for k in req if k not in features]
            if missing:
                raise KeyError(f"Missing keys for triplet_loss: {missing}")
            return triplet_loss(
                features['anchor'], features['positive'], features['negative'], margin
            )
        return triplet_fn

    else:
        raise ValueError(f"Unsupported loss type/task: task={task}, loss={loss_type}")
