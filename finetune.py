# finetune.py
import os
import time
import json
import logging
import argparse
from argparse import Namespace
from contextlib import nullcontext
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import wandb
import yaml  # NEW: for YAML loading

# NOTE: 不再强依赖 utils.config.load_train_config，直接本地加载 YAML
# from utils.config import load_train_config
from utils.dataset import create_dataloader
from models.Encoder.rgb_encoder import RGBEncoder
from models.Encoder.text_encoder import TextEncoder
from models.Head.retrieval import RetrievalHead
from models.Head.recognition import RecognitionHeadCTC
from models.Head.translation import TranslationHeadMT5


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------
# config helpers (NEW)
# ------------------------------
def _dict_to_ns(obj):
    """递归把 dict 转 SimpleNamespace，便于点号访问"""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _dict_to_ns(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [_dict_to_ns(v) for v in obj]
    else:
        return obj

def _ns_to_dict(obj):
    """递归把 SimpleNamespace 转 dict（用于 wandb / 保存等）"""
    if isinstance(obj, SimpleNamespace):
        return {k: _ns_to_dict(getattr(obj, k)) for k in vars(obj)}
    elif isinstance(obj, dict):
        return {k: _ns_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_ns_to_dict(v) for v in obj]
    else:
        return obj

def load_yaml_config(path: str) -> SimpleNamespace:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _dict_to_ns(raw if raw is not None else {})


# ------------------------------
# utils
# ------------------------------
def set_seed(s: int = 3407):
    import random, numpy as np, torch
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cfg_get(ns, path, default=None):
    cur = ns
    for k in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(k, None)
        else:
            cur = getattr(cur, k, None)
    return default if cur is None else cur


def freeze_modules(modules):
    for m in modules:
        if m is None:
            continue
        for p in m.parameters():
            p.requires_grad = False


def params_with_lr(modules, lr):
    ps = []
    for m in modules:
        if m is None:
            continue
        ps += [p for p in m.parameters() if p.requires_grad]
    return {"params": ps, "lr": lr}


def _get(obj, key, default=None):
    """对单层对象安全取值（支持 dict / SimpleNamespace）"""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ------------------------------
# build & load
# ------------------------------
def build_backbones(cfg, device):
    Dv = int(cfg_get(cfg, "Encoders.rgb.output_dim", 512))
    Dt = int(cfg_get(cfg, "Encoders.text.output_dim", 384))
    P = int(cfg_get(cfg, "Pretraining.projection_dim", 256))

    rgb = RGBEncoder(pretrained=False, output_dim=Dv).to(device)
    text = TextEncoder().to(device)
    proj = nn.ModuleDict({
        "rgb": nn.Linear(Dv, P),
        "text": nn.Linear(Dt, P)
    }).to(device)

    # Xavier init for projector
    for m in proj.values():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    return rgb, text, proj, Dv, Dt, P


def load_split_weights_if_any(cfg, rgb, text, proj, device):
    if not cfg_get(cfg, "Finetune.weights", None):
        logger.info("[Load] Finetune.weights 未提供，跳过显式加载（可能用随机初始化/继续训练）。")
        return

    def _load(path, module, name):
        if not path:
            logger.warning(f"[Load] 跳过 {name}（路径为空）")
            return
        if not os.path.isfile(path):
            logger.warning(f"[Load] 跳过 {name}（路径不存在）：{path}")
            return
        try:
            sd = torch.load(path, map_location=device)
            module.load_state_dict(sd, strict=True)
            logger.info(f"[Load] {name} <- {path}")
        except Exception as e:
            logger.error(f"[Load] 加载 {name} 失败: {e}")

    _load(cfg_get(cfg, "Finetune.weights.rgb_path"),       rgb,             "rgb_encoder")
    _load(cfg_get(cfg, "Finetune.weights.text_path"),      text,            "text_encoder")
    _load(cfg_get(cfg, "Finetune.weights.proj_rgb_path"),  proj.get("rgb"), "proj_rgb")
    _load(cfg_get(cfg, "Finetune.weights.proj_text_path"), proj.get("text"),"proj_text")


def apply_freeze_from_cfg(cfg, rgb, text, proj):
    freeze_list = cfg_get(cfg, "Finetune.freeze", []) or []
    names = set([s.lower() for s in freeze_list])

    if "rgb" in names:
        freeze_modules([rgb])
        logger.info("[Freeze] RGB encoder frozen")
    if "text" in names:
        freeze_modules([text])
        logger.info("[Freeze] Text encoder frozen")
    if "proj" in names:
        freeze_modules([proj])
        logger.info("[Freeze] Projectors frozen")


# ------------------------------
# dataloaders
# ------------------------------
def phase_alias(name: str) -> str:
    n = (name or "").lower()
    return "dev" if n in ("val", "valid", "validation") else n


def build_loader(cfg, phase, device):
    ds_list = cfg_get(cfg, "active_datasets", []) or []
    if len(ds_list) != 1:
        raise ValueError("finetune 最简实现当前只支持单数据集，请在 active_datasets 中只保留一个。")

    ds_name = ds_list[0]
    ds_cfg = cfg_get(cfg, f"datasets.{ds_name}", {})
    temporal_cfg = _get(ds_cfg, "temporal", {})  # 可能是 dict 或 SimpleNamespace

    # 统一读取 T / max_frames / 兜底
    T = _get(temporal_cfg, "T", None)
    max_frames = _get(temporal_cfg, "max_frames", None)
    max_len = T or max_frames or cfg_get(cfg, "Training.max_length", 64)
    max_len = int(max_len)

    phase_for_split = phase_alias(phase)
    args = Namespace(
        dataset_name=ds_name,
        batch_size=int(cfg_get(cfg, "Training.batch_size", 8)),
        num_workers=int(cfg_get(cfg, "Training.num_workers", 4)),
        max_length=max_len,
        rgb_support=True,
        seed=int(cfg_get(cfg, "seed", 3407)),
    )
    loader = create_dataloader(args, cfg, phase=phase_for_split)
    logger.info(f"[Data] split='{phase}' -> using '{phase_for_split}' | {ds_name}: {len(loader)} batches, max_len={max_len}")
    return loader


# ------------------------------
# evaluation metrics
# ------------------------------
@torch.no_grad()
def evaluate_retrieval(rgb, text, proj_or_head, loader, device, temperature=0.07):
    """评估检索任务性能"""
    rgb.eval()
    text.eval()
    all_rgb, all_text = [], []

    use_head = hasattr(proj_or_head, "forward") and hasattr(proj_or_head, "rgb_proj")

    for (src, tgt) in tqdm(loader, desc="Eval-Retrieval", leave=False):
        # 视频特征
        vseq = rgb(src["rgb_img"].to(device))  # [B,T,Dv]
        v = vseq.mean(dim=1)  # [B,Dv]

        # 文本特征
        tout, amask = text(tgt["gt_sentence"])
        tout = tout.to(device)  # 确保在相同设备

        # 处理序列输出（如果有mask）
        if tout.ndim == 3:
            mask = amask.to(device).unsqueeze(-1).float()
            tout = (tout * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-5)

        # 投影和归一化
        if use_head:
            v_proj = F.normalize(proj_or_head.rgb_proj(v), dim=-1)
            t_proj = F.normalize(proj_or_head.text_proj(tout), dim=-1)
        else:
            v_proj = F.normalize(proj_or_head["rgb"](v), dim=-1)
            t_proj = F.normalize(proj_or_head["text"](tout), dim=-1)

        all_rgb.append(v_proj)
        all_text.append(t_proj)

    rgb_features = torch.cat(all_rgb, dim=0)
    text_features = torch.cat(all_text, dim=0)

    # 计算相似度矩阵
    sim_matrix = text_features @ rgb_features.T
    sim_matrix = sim_matrix / float(temperature)

    # 计算检索指标
    try:
        from utils.metrics import t2v_metrics, v2t_metrics
        sim_np = sim_matrix.detach().cpu().numpy()
        t2v_results, _ = t2v_metrics(sim_np, None)
        v2t_results, _ = v2t_metrics(sim_np.T, None)

        metrics = {f"t2v/{k}": float(v) for k, v in t2v_results.items()}
        metrics.update({f"v2t/{k}": float(v) for k, v in v2t_results.items()})

        if "R1" in t2v_results and "R1" in v2t_results:
            metrics["mean_R1"] = 0.5 * (t2v_results["R1"] + v2t_results["R1"])

    except ImportError:
        logger.warning("utils.metrics not available, using simple accuracy")
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        t2v_acc = (sim_matrix.argmax(dim=1) == labels).float().mean()
        v2t_acc = (sim_matrix.argmax(dim=0) == labels).float().mean()
        metrics = {
            "t2v/R1": float(t2v_acc.item()),
            "v2t/R1": float(v2t_acc.item()),
            "mean_R1": 0.5 * (float(t2v_acc.item()) + float(v2t_acc.item()))
        }

    return metrics


# ------------------------------
# TASK A: Retrieval
# ------------------------------
def finetune_retrieval(cfg, device, rgb, text, proj):
    """微调检索任务"""
    logger.info("Starting retrieval finetuning...")

    # 应用冻结策略
    apply_freeze_from_cfg(cfg, rgb, text, proj)

    # 创建任务头
    task_head = RetrievalHead(
        rgb_in=int(cfg_get(cfg, "Evaluation.retrieval.rgb_in", 512)),
        text_in=int(cfg_get(cfg, "Evaluation.retrieval.text_in", 384)),
        proj_dim=int(cfg_get(cfg, "Evaluation.retrieval.proj_dim", 256)),
        temperature=float(cfg_get(cfg, "Evaluation.retrieval.temperature", 0.07)),
        trainable=True,
    ).to(device)

    # 从主proj复制权重到head
    if proj is not None:
        task_head.rgb_proj.load_state_dict(proj["rgb"].state_dict(), strict=True)
        task_head.text_proj.load_state_dict(proj["text"].state_dict(), strict=True)

    # 数据加载器
    train_split = cfg_get(cfg, "Finetune.train_split", "dev")
    eval_split = cfg_get(cfg, "Finetune.eval_split", "test")
    train_loader = build_loader(cfg, train_split, device)
    val_loader = build_loader(cfg, eval_split, device)

    # 优化器：只训练head
    lr_head = float(cfg_get(cfg, "Training.learning_rate_head", 3e-4))
    optimizer = AdamW([p for p in task_head.parameters() if p.requires_grad], lr=lr_head)

    # 混合精度训练
    amp_enabled = cfg_get(cfg, "Finetune.amp", True) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

    best_R1 = -1.0
    epochs = int(cfg_get(cfg, "Training.epochs", 5))
    eval_every = int(cfg_get(cfg, "Finetune.eval_every", 1))

    for epoch in range(epochs):
        task_head.train()
        if rgb is not None:
            rgb.train()
        if text is not None:
            text.train()

        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[Retrieval] Epoch {epoch + 1}/{epochs}")

        for batch_idx, (src, tgt) in enumerate(pbar):
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                vseq = rgb(src["rgb_img"].to(device)) if rgb is not None else src["rgb_features"].to(device)
                v = vseq.mean(dim=1) if vseq.ndim == 3 else vseq

                if text is not None:
                    tout, amask = text(tgt["gt_sentence"])
                    tout = tout.to(device)
                    if tout.ndim == 3:
                        mask = amask.to(device).unsqueeze(-1).float()
                        tout = (tout * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-5)
                else:
                    tout = tgt["text_features"].to(device)

                v_proj, t_proj = task_head(v, tout)
                sim = t_proj @ v_proj.T

                tau = float(cfg_get(cfg, "Evaluation.retrieval.temperature", 0.07))
                B = sim.size(0)
                labels = torch.arange(B, device=sim.device)

                loss = 0.5 * (F.cross_entropy(sim / tau, labels) + F.cross_entropy(sim.t() / tau, labels))

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(task_head.parameters(), max_norm=float(cfg_get(cfg, "Training.grad_clip", 1.0)))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                clip_grad_norm_(task_head.parameters(), max_norm=float(cfg_get(cfg, "Training.grad_clip", 1.0)))
                optimizer.step()

            epoch_loss += float(loss.item())
            pbar.set_postfix(loss=float(loss.item()))

        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"[Retrieval] Epoch {epoch + 1} train_loss={avg_loss:.4f}")

        if val_loader is not None and (epoch + 1) % eval_every == 0:
            logger.info(f"[Retrieval] Evaluating epoch {epoch + 1}...")
            metrics = evaluate_retrieval(
                rgb, text, task_head, val_loader, device,
                temperature=float(cfg_get(cfg, "Evaluation.retrieval.temperature", 0.07))
            )

            logger.info(f"[Retrieval] Eval metrics: {json.dumps(metrics, ensure_ascii=False, indent=2)}")

            if wandb.run:
                wandb.log({
                    "retrieval/train_loss": avg_loss,
                    **{f"retrieval/{k}": v for k, v in metrics.items()},
                    "epoch": epoch + 1
                })

            current_r1 = metrics.get("mean_R1", metrics.get("t2v/R1", 0.0))
            if current_r1 > best_R1:
                best_R1 = current_r1
                save_dir = cfg_get(cfg, "save_dir", "checkpoints/finetune")
                os.makedirs(save_dir, exist_ok=True)
                torch.save(task_head.state_dict(), os.path.join(save_dir, "best_retrieval_head.pt"))
                logger.info(f"[Retrieval] Best model saved (mean_R1={best_R1:.4f})")

    logger.info(f"[Retrieval] Finetuning completed. Best mean_R1: {best_R1:.4f}")


# ------------------------------
# TASK B: Recognition (CTC)
# ------------------------------
def train_eval_recognition(cfg, device, rgb, proj):
    """训练和评估识别任务（CTC）"""
    logger.info("Starting recognition finetuning...")

    use_proj = bool(cfg_get(cfg, "Finetune.use_proj_for_recog", False))
    in_dim = int(cfg_get(cfg, "Evaluation.recognition.in_dim", 512))

    head = RecognitionHeadCTC(
        in_dim=in_dim,
        num_classes=int(cfg_get(cfg, "Evaluation.recognition.num_classes", 2000)),
        hidden_dim=int(cfg_get(cfg, "Evaluation.recognition.hidden_dim", 512)),
        num_layers=int(cfg_get(cfg, "Evaluation.recognition.num_layers", 2)),
        dropout=float(cfg_get(cfg, "Evaluation.recognition.dropout", 0.1)),
        blank_id=int(cfg_get(cfg, "Evaluation.recognition.blank_id", 0)),
    ).to(device)

    apply_freeze_from_cfg(cfg, rgb, None, proj)

    lr_head = float(cfg_get(cfg, "Training.learning_rate_head", 3e-4))
    lr_back = float(cfg_get(cfg, "Training.learning_rate_backbone", 5e-5))

    optim_groups = [params_with_lr([head], lr_head)]

    if not any("rgb" in s.lower() for s in (cfg_get(cfg, "Finetune.freeze", []) or [])):
        optim_groups.append(params_with_lr([rgb], lr_back))
        logger.info("[Optimizer] Adding RGB backbone parameters for finetuning")

    optimizer = AdamW(optim_groups)
    amp_enabled = cfg_get(cfg, "Finetune.amp", True) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    train_loader = build_loader(cfg, "train", device)
    val_loader = build_loader(cfg, "val", device)

    epochs = int(cfg_get(cfg, "Training.epochs", 10))
    best_val_loss = float('inf')

    for epoch in range(epochs):
        head.train()
        if rgb is not None:
            rgb.train()

        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[Recog] Epoch {epoch + 1}/{epochs}")

        for src, tgt in pbar:
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                feat = rgb(src["rgb_img"].to(device)) if rgb is not None else src["rgb_features"].to(device)
                if use_proj and proj is not None:
                    feat = proj["rgb"](feat)

                logits = head(feat)  # [T,B,V]

                loss = head.compute_loss(
                    logits,
                    tgt["ctc_targets"].to(device),
                    tgt["input_lengths"].to(device) if "input_lengths" in tgt else torch.tensor(
                        [logits.size(0)] * logits.size(1), device=device),
                    tgt["target_lengths"].to(device) if "target_lengths" in tgt else torch.tensor(
                        [len(t) for t in tgt["ctc_targets"]], device=device),
                )

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_([p for group in optim_groups for p in group["params"]],
                                max_norm=float(cfg_get(cfg, "Training.grad_clip", 1.0)))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                clip_grad_norm_([p for group in optim_groups for p in group["params"]],
                                max_norm=float(cfg_get(cfg, "Training.grad_clip", 1.0)))
                optimizer.step()

            total_loss += float(loss.item())
            pbar.set_postfix(loss=float(loss.item()))

        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"[Recog] Epoch {epoch + 1} train_loss={avg_train_loss:.4f}")

        head.eval()
        if rgb is not None:
            rgb.eval()

        val_loss = 0.0
        with torch.no_grad():
            for src, tgt in tqdm(val_loader, desc="[Recog] Validating", leave=False):
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    feat = rgb(src["rgb_img"].to(device)) if rgb is not None else src["rgb_features"].to(device)
                    if use_proj and proj is not None:
                        feat = proj["rgb"](feat)

                    logits = head(feat)
                    loss = head.compute_loss(
                        logits,
                        tgt["ctc_targets"].to(device),
                        tgt["input_lengths"].to(device) if "input_lengths" in tgt else torch.tensor(
                            [logits.size(0)] * logits.size(1), device=device),
                        tgt["target_lengths"].to(device) if "target_lengths" in tgt else torch.tensor(
                            [len(t) for t in tgt["ctc_targets"]], device=device),
                    )
                    val_loss += float(loss.item())

        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"[Recog] Epoch {epoch + 1} val_loss={avg_val_loss:.4f}")

        if wandb.run:
            wandb.log({
                "recog/train_loss": avg_train_loss,
                "recog/val_loss": avg_val_loss,
                "epoch": epoch + 1
            })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_dir = cfg_get(cfg, "save_dir", "checkpoints/finetune")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(head.state_dict(), os.path.join(save_dir, "best_recognition_head.pt"))
            logger.info(f"[Recog] Best model saved (val_loss={best_val_loss:.4f})")

    logger.info(f"[Recog] Finetuning completed. Best val_loss: {best_val_loss:.4f}")


# ------------------------------
# TASK C: Translation (MT5)
# ------------------------------
def train_eval_translation(cfg, device, rgb):
    """训练和评估翻译任务（mT5）"""
    logger.info("Starting translation finetuning...")

    in_dim = int(cfg_get(cfg, "Evaluation.translation.in_dim", 512))
    head = TranslationHeadMT5(
        mt5_path=cfg_get(cfg, "Evaluation.translation.mt5_path", "google/mt5-base"),
        in_dim=in_dim,
        d_model=int(cfg_get(cfg, "Evaluation.translation.hidden_dim", 768)),
        label_smoothing=float(cfg_get(cfg, "Evaluation.translation.label_smoothing", 0.1)),
        lang_prompt=cfg_get(cfg, "Evaluation.translation.lang", "zh"),
        max_target_len=int(cfg_get(cfg, "Evaluation.translation.max_length", 128)),
    ).to(device)

    apply_freeze_from_cfg(cfg, rgb, None, None)

    lr_head = float(cfg_get(cfg, "Training.learning_rate_head", 3e-4))
    lr_back = float(cfg_get(cfg, "Training.learning_rate_backbone", 5e-5))

    optim_groups = [params_with_lr([head], lr_head)]

    if not any("rgb" in s.lower() for s in (cfg_get(cfg, "Finetune.freeze", []) or [])):
        optim_groups.append(params_with_lr([rgb], lr_back))
        logger.info("[Optimizer] Adding RGB backbone parameters for finetuning")

    optimizer = AdamW(optim_groups)
    amp_enabled = cfg_get(cfg, "Finetune.amp", True) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    train_loader = build_loader(cfg, "train", device)
    val_loader = build_loader(cfg, "val", device)

    epochs = int(cfg_get(cfg, "Training.epochs", 10))
    best_val_loss = float('inf')

    for epoch in range(epochs):
        head.train()
        if rgb is not None:
            rgb.train()

        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[Trans] Epoch {epoch + 1}/{epochs}")

        for src, tgt in pbar:
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                feat = rgb(src["rgb_img"].to(device)) if rgb is not None else src["rgb_features"].to(device)
                B, T, _ = feat.shape

                vis_mask = torch.ones((B, T), dtype=torch.long, device=feat.device)

                out = head(
                    vis_seq=feat,
                    vis_mask=vis_mask,
                    tgt_texts=tgt["gt_sentence"]
                )
                loss = out["loss"]

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_([p for group in optim_groups for p in group["params"]],
                                max_norm=float(cfg_get(cfg, "Training.grad_clip", 1.0)))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                clip_grad_norm_([p for group in optim_groups for p in group["params"]],
                                max_norm=float(cfg_get(cfg, "Training.grad_clip", 1.0)))
                optimizer.step()

            total_loss += float(loss.item())
            pbar.set_postfix(loss=float(loss.item()))

        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"[Trans] Epoch {epoch + 1} train_loss={avg_train_loss:.4f}")

        head.eval()
        if rgb is not None:
            rgb.eval()

        val_loss = 0.0
        with torch.no_grad():
            for src, tgt in tqdm(val_loader, desc="[Trans] Validating", leave=False):
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    feat = rgb(src["rgb_img"].to(device)) if rgb is not None else src["rgb_features"].to(device)
                    B, T, _ = feat.shape
                    vis_mask = torch.ones((B, T), dtype=torch.long, device=feat.device)

                    out = head(
                        vis_seq=feat,
                        vis_mask=vis_mask,
                        tgt_texts=tgt["gt_sentence"]
                    )
                    val_loss += float(out["loss"].item())

        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"[Trans] Epoch {epoch + 1} val_loss={avg_val_loss:.4f}")

        if wandb.run:
            wandb.log({
                "trans/train_loss": avg_train_loss,
                "trans/val_loss": avg_val_loss,
                "epoch": epoch + 1
            })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_dir = cfg_get(cfg, "save_dir", "checkpoints/finetune")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(head.state_dict(), os.path.join(save_dir, "best_translation_head.pt"))
            logger.info(f"[Trans] Best model saved (val_loss={best_val_loss:.4f})")

    logger.info(f"[Trans] Finetuning completed. Best val_loss: {best_val_loss:.4f}")


# ------------------------------
# main
# ------------------------------
def main():
    """主函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="config/finetune.yaml",
        help="Path to finetune YAML config (default: config/finetune.yaml)"
    )
    args = parser.parse_args()

    set_seed()
    cfg = load_yaml_config(args.cfg)  # <<< 关键改动：明确加载 config/finetune.yaml
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Loaded config from: {os.path.abspath(args.cfg)}")

    # W&B初始化
    if cfg_get(cfg, "wandb.use", False):
        try:
            wandb.init(
                project=cfg_get(cfg, "wandb.project", "finetune"),
                name=cfg_get(cfg, "wandb.run_name", "run"),
                config=_ns_to_dict(cfg)  # 保证是纯 dict
            )
            logger.info("Initialized Weights & Biases")
        except Exception as e:
            logger.warning(f"W&B init failed: {e}")
            os.environ["WANDB_MODE"] = "disabled"
    else:
        os.environ["WANDB_MODE"] = "disabled"

    # 构建模型
    logger.info("Building models...")
    rgb, text, proj, Dv, Dt, P = build_backbones(cfg, device)
    load_split_weights_if_any(cfg, rgb, text, proj, device)

    # 统计可训练参数
    total_params = sum(p.numel() for p in rgb.parameters() if p.requires_grad)
    logger.info(f'RGB encoder trainable parameters: {total_params:,}')
    total_params = sum(p.numel() for p in text.parameters() if p.requires_grad)
#    logger.info(f'Text encoder trainable parameters: {total_params:,}')
    total_params = sum(p.numel() for p in proj.parameters() if p.requires_grad)
#    logger.info(f'Projector trainable parameters: {total_params:,}')

    # 选择任务
    task = str(cfg_get(cfg, "Finetune.task", "retrieval")).lower()
    logger.info(f"Starting finetuning task: {task}")

    if task == "retrieval":
        finetune_retrieval(cfg, device, rgb, text, proj)
    elif task == "recognition":
        train_eval_recognition(cfg, device, rgb, proj)
    elif task == "translation":
        train_eval_translation(cfg, device, rgb)
    else:
        raise ValueError(f"Unknown Finetune.task: {task}")

    logger.info("Finetune completed successfully!")


if __name__ == "__main__":
    main()
