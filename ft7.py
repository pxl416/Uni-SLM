# finetuner.py
# -*- coding: utf-8 -*-
import os
import logging
import argparse
from types import SimpleNamespace

import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import yaml

# Dataset
from datasets.datasets import create_dataloader

# Encoders
from models.Encoder.rgb_encoder import RGBEncoder
from models.Encoder.pose_encoder import PoseEncoder
from models.Encoder.text_encoder import TextEncoder

# Heads
from models.Head.retrieval import RetrievalHead
from models.Head.translation import TranslationHeadMT5
from models.Head.recognition import RecognitionHeadCTC

from utils.config import cfg_get
from types import SimpleNamespace
from utils.metrics import compute_bleu, compute_rouge, compute_cer, compute_wer
from utils.trainer import init_trainer_common
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# WandB (safe import)
try:
    import wandb

    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False
    print("[Warn] wandb not installed; logging disabled.")


# =========================================================
# Utils
# =========================================================
def set_seed(seed: int = 3407):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_yaml(path):
    """Load YAML exactly like test_data_loading.py"""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def dict_to_ns(d):
    """dict → SimpleNamespace (recursive)"""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
    return d


def params_with_lr(modules, lr):
    ps = []
    for m in modules:
        if m is None:
            continue
        for p in m.parameters():
            if p.requires_grad:
                ps.append(p)
    return {"params": ps, "lr": float(lr)} if ps else None


def save_checkpoint(path, model_dict, optimizer, epoch, best_metric):
    ckpt = {
        "model": model_dict,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
    }
    torch.save(ckpt, path)
    print(f"[Checkpoint] Saved to {path}")


def load_checkpoint(path, model_dict, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    # load model
    for name, module in model_dict.items():
        if module is not None:
            module.load_state_dict(ckpt["model"][name], strict=False)
            print(f"[Checkpoint] Loaded {name}")
    # load optimizer
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("best_metric", None)


class BaseFinetuner:
    """所有 Finetuner 的基类，提供统一的 wandb 支持"""

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.wandb_enabled = WANDB_AVAILABLE

    def log_to_wandb(self, metrics, step=None, prefix=""):
        """统一的 wandb 日志记录方法"""
        if not self.wandb_enabled:
            return

        log_dict = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                key = f"{prefix}/{k}" if prefix else k
                log_dict[key] = v

        if step is not None:
            wandb.log(log_dict, step=step)
        else:
            wandb.log(log_dict)


# =========================================================
# Retrieval Finetuner
# =========================================================

class RetrievalFinetuner(BaseFinetuner):

    def __init__(self, cfg, device):
        super().__init__(cfg, device)

        # ⭐ 统一初始化（AMP、scaler、save_dir、best_metric、wandb）
        init_trainer_common(self, cfg, device)

        # ⭐ 构建 dataloader / model / optimizer
        self.build_dataloaders()
        self.build_models()
        self.build_optimizer()

        logger.info("[Finetune] RetrievalFinetuner ready.")


    def build_dataloaders(self):
        args = SimpleNamespace()
        logger.info("[Retrieval] Building dataloaders...")

        self.train_loader = create_dataloader(args, self.cfg, phase="train")
        self.val_loader   = create_dataloader(args, self.cfg, phase="dev")

        logger.info(f"[Data] train batches = {len(self.train_loader)}")
        logger.info(f"[Data] val   batches = {len(self.val_loader)}")

    # ------------------------------------------------------
    # Build models
    # ------------------------------------------------------
    def build_models(self):
        Dv = cfg_get(self.cfg, "Encoders.rgb.output_dim", 512)
        Dt = cfg_get(self.cfg, "Encoders.text.output_dim", 384)

        self.rgb = RGBEncoder(pretrained=False, output_dim=Dv).to(self.device)

        # 使用 TextEncoder 句向量模式（默认）
        self.text = TextEncoder(
            model_path="sentence-transformers/all-MiniLM-L6-v2",
            return_sequence=False,
            max_length=128
        ).to(self.device)

        self.head = RetrievalHead(
            rgb_in=Dv,
            text_in=Dt,
            proj_dim=cfg_get(self.cfg, "Finetune.retrieval.proj_dim", 256),
            temperature=cfg_get(self.cfg, "Finetune.retrieval.temperature", 0.07),
            learnable_tau=True,
        ).to(self.device)

    # ------------------------------------------------------
    # Build optimizer
    # ------------------------------------------------------
    def build_optimizer(self):
        lr = cfg_get(self.cfg, "Training.learning_rate_head", 3e-4)

        params = list(self.rgb.parameters()) + \
                 list(self.text.parameters()) + \
                 list(self.head.parameters())

        self.optimizer = AdamW(params, lr=lr)
        logger.info(f"[Optimizer] lr={lr}")

    # ------------------------------------------------------
    # Extract batch
    # ------------------------------------------------------
    def extract_batch(self, batch):
        src, tgt = batch

        # RGB
        rgb = src["rgb_img"]
        rgb_len = src["rgb_len"]

        # TEXT —— 自动适配字段名
        possible_keys = ["text", "Text", "sentence", "gt_sentence", "label", "caption"]

        text = None
        for k in possible_keys:
            if k in tgt:
                text = tgt[k]
                break

        if text is None:
            raise KeyError(
                f"[Retrieval] Cannot find text field in tgt_input. "
                f"Available keys = {list(tgt.keys())}"
            )

        # text_len（一般没有）
        text_len = tgt.get("text_len", None)

        return rgb, rgb_len, text, text_len

    # ------------------------------------------------------
    # Encode RGB / Text
    # ------------------------------------------------------
    def encode_rgb(self, rgb, rgb_len):
        rgb = rgb.to(self.device)
        rgb_len = rgb_len.to(self.device)

        feat = self.rgb(rgb)
        B, T, _ = feat.shape

        mask = torch.zeros((B, T), dtype=torch.bool, device=self.device)
        for i in range(B):
            mask[i, : int(rgb_len[i])] = True

        return feat, mask

    def encode_text(self, text, text_len=None):

        # TextEncoder 返回两项
        feat, mask = self.text(text)  # feat: [B,D] 或 [B,L,D]

        # 如果是句向量： [B, D] → 扩展为 [B,1,D]
        if feat.ndim == 2:
            feat = feat.unsqueeze(1)  # [B,1,D]
            mask = None  # Retrieval 不需要 mask 也能跑

        else:
            # feat 是 [B, L, D]
            # mask 是 attention mask [B, L] → bool
            mask = mask.bool()

        return feat, mask

    # ------------------------------------------------------
    # Train one epoch
    # ------------------------------------------------------
    def train_epoch(self):
        self.rgb.train()
        self.text.train()
        self.head.train()

        total_loss = 0
        n_batches = 0

        for batch in tqdm(self.train_loader, desc="Train (retrieval)"):
            rgb, rgb_len, text, text_len = self.extract_batch(batch)

            rgb_feat, rgb_mask = self.encode_rgb(rgb, rgb_len)
            text_feat, text_mask = self.encode_text(text, text_len)


            with torch.amp.autocast('cuda', enabled=self.use_amp):
                loss = self.head.compute_loss(
                    rgb_feat, text_feat,
                    rgb_mask=rgb_mask,
                    text_mask=text_mask
                )

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        metrics = {"loss": avg_loss, "main_metric": -avg_loss}

        self.log_to_wandb(metrics, prefix="train")

        if n_batches == 0:  # 第一批
            print("rgb_feat:", rgb_feat.mean().item(), rgb_feat.std().item())
            print("text_feat:", text_feat.mean().item(), text_feat.std().item())

        return metrics

    # ------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------
    @torch.no_grad()
    def evaluate(self):
        self.rgb.eval()
        self.text.eval()
        self.head.eval()

        rgb_all = []
        text_all = []

        for batch in tqdm(self.val_loader, desc="Eval (retrieval)"):
            rgb, rgb_len, text, text_len = self.extract_batch(batch)

            rgb_feat, rgb_mask = self.encode_rgb(rgb, rgb_len)
            text_feat, text_mask = self.encode_text(text, text_len)

            rgb_p, text_p = self.head(rgb_feat, text_feat, rgb_mask, text_mask)

            rgb_all.append(rgb_p)
            text_all.append(text_p)

        rgb_all = torch.cat(rgb_all, dim=0)
        text_all = torch.cat(text_all, dim=0)

        # 直接计算指标
        metrics = self.compute_metrics_directly(rgb_all, text_all)
        metrics["loss"] = None  # retrieval 没有 val_loss
        metrics["main_metric"] = metrics["mean_R1"]

        # 直接打印而不调用 pretty_print_metrics
        print(
            f"\n[Eval] R@1={metrics['R1']:.4f}, R@5={metrics['R5']:.4f}, R@10={metrics['R10']:.4f} (↑ higher is better)")

        self.log_to_wandb(metrics, prefix="eval")
        return metrics

    def compute_metrics_directly(self, rgb_proj, text_proj):
        """直接计算检索指标"""
        device = rgb_proj.device

        # 计算相似度
        sim = torch.matmul(text_proj, rgb_proj.t())  # [N, N]

        # 计算排名
        ranks = sim.argsort(dim=1, descending=True)
        N = sim.size(0)

        r1 = (ranks[:, 0] == torch.arange(N, device=device)).float().mean().item()
        r5 = (ranks[:, :5] == torch.arange(N, device=device).unsqueeze(1)).any(dim=1).float().mean().item()
        r10 = (ranks[:, :10] == torch.arange(N, device=device).unsqueeze(1)).any(dim=1).float().mean().item()

        return {
            "R1": r1,
            "R5": r5,
            "R10": r10,
            "mean_R1": r1,
        }

    def pretty_print_metrics(self, metrics: dict):
        """专业格式打印检索指标"""
        print("\n" + "=" * 50)
        print("EVALUATION METRICS (Retrieval)")
        print("=" * 50)

        # 检索指标
        if "R1" in metrics:
            print(f"  Recall@1:     {metrics['R1']:.4f} (↑ higher is better)")
        if "R5" in metrics:
            print(f"  Recall@5:     {metrics['R5']:.4f} (↑ higher is better)")
        if "R10" in metrics:
            print(f"  Recall@10:    {metrics['R10']:.4f} (↑ higher is better)")
        if "mean_R1" in metrics:
            print(f"  Mean R1:      {metrics['mean_R1']:.4f} (↑ higher is better)")

        # 通用指标
        for key, value in metrics.items():
            if key not in ["R1", "R5", "R10", "mean_R1", "main_metric"]:
                if isinstance(value, (int, float)):
                    arrow = "↑" if key in ["accuracy", "precision", "recall", "f1"] else "↓"
                    print(f"  {key:12s} {value:.4f} ({arrow} {'higher' if arrow == '↑' else 'lower'} is better)")

        if "main_metric" in metrics:
            print(f"\n  Main Metric:  {metrics['main_metric']:.4f}")
        print("=" * 50 + "\n")

    def compute_metrics_directly(self, rgb_proj, text_proj):
        """直接计算检索指标，不需要再次通过投影层"""
        device = rgb_proj.device

        # 余弦相似度
        # 注意：这里需要确保向量已经归一化（在 head 中已经做了）
        sim = torch.matmul(text_proj, rgb_proj.t())  # [N, N]

        # 计算排名
        ranks = sim.argsort(dim=1, descending=True)
        N = sim.size(0)

        r1 = (ranks[:, 0] == torch.arange(N, device=device)).float().mean().item()
        r5 = (ranks[:, :5] == torch.arange(N, device=device).unsqueeze(1)).any(dim=1).float().mean().item()
        r10 = (ranks[:, :10] == torch.arange(N, device=device).unsqueeze(1)).any(dim=1).float().mean().item()

        return {
            "R1": r1,
            "R5": r5,
            "R10": r10,
            "mean_R1": r1,
        }

# =========================================================
# TranslationFinetuner  ——  RGB → MT5
# =========================================================
class TranslationFinetuner(BaseFinetuner):

    def __init__(self, cfg, device):
        super().__init__(cfg, device)

        init_trainer_common(self, cfg, device)

        self.build_dataloaders()
        self.build_models()
        self.build_optimizer()

        logger.info("[Finetune] TranslationFinetuner ready.")

    # ----------------------------------------
    # Dataloaders
    # ----------------------------------------
    def build_dataloaders(self):
        args = SimpleNamespace()
        self.train_loader = create_dataloader(args, self.cfg, phase="train")
        self.val_loader   = create_dataloader(args, self.cfg, phase="dev")

        logger.info(f"[Data] train batches = {len(self.train_loader)}")
        logger.info(f"[Data] val   batches = {len(self.val_loader)}")

    # ----------------------------------------
    # Models
    # ----------------------------------------
    def build_models(self):
        Dv = cfg_get(self.cfg, "Encoders.rgb.output_dim", 512)

        self.rgb = RGBEncoder(
            pretrained=False,
            output_dim=Dv,
        ).to(self.device)

        mt5_path = cfg_get(self.cfg, "Finetune.translation_model_path", "google/mt5-small")

        self.task_head = TranslationHeadMT5(
            mt5_path=mt5_path,
            in_dim=Dv,
            label_smoothing=cfg_get(self.cfg, "Finetune.label_smoothing", 0.1),
            lang_prompt=cfg_get(self.cfg, "Finetune.lang_prompt", "Chinese"),
            max_target_len=cfg_get(self.cfg, "Finetune.max_target_len", 50),
        ).to(self.device)

        logger.info(f"[Model] RGBEncoder + TranslationHeadMT5 ready")

    # ----------------------------------------
    # Optimizer
    # ----------------------------------------
    def build_optimizer(self):
        lr_head = cfg_get(self.cfg, "Training.learning_rate_head", 1e-4)
        lr_back = cfg_get(self.cfg, "Training.learning_rate_backbone", 1e-5)

        groups = []
        g_head = params_with_lr([self.task_head], lr_head)
        g_rgb  = params_with_lr([self.rgb], lr_back)

        if g_head: groups.append(g_head)
        if g_rgb:  groups.append(g_rgb)

        self.optimizer = AdamW(groups)
        logger.info(f"[Optimizer] head_lr={lr_head}, backbone_lr={lr_back}")

    # ----------------------------------------
    # Extract Batch
    # ----------------------------------------
    def extract_batch(self, batch):
        src, tgt = batch

        rgb = src["rgb_img"]
        rgb_len = src["rgb_len"]

        tgt_texts = tgt["gt_sentence"]

        return rgb, rgb_len, tgt_texts

    # ----------------------------------------
    # Encode RGB
    # ----------------------------------------
    def encode_rgb(self, rgb, rgb_len):
        rgb = rgb.to(self.device)
        rgb_len = rgb_len.to(self.device)

        feat = self.rgb(rgb)
        B, T, _ = feat.shape

        mask = torch.zeros((B, T), dtype=torch.bool, device=self.device)
        for i in range(B):
            mask[i, :int(rgb_len[i])] = True

        return feat, mask

    # ----------------------------------------
    # Train
    # ----------------------------------------
    def train_epoch(self):
        self.rgb.train()
        self.task_head.train()

        tot_loss = 0
        n = 0

        for batch in tqdm(self.train_loader, desc="Train (translation)"):
            rgb, rgb_len, tgt_texts = self.extract_batch(batch)

            vis_seq, vis_mask = self.encode_rgb(rgb, rgb_len)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                out = self.task_head(vis_seq, vis_mask, tgt_texts)
                loss = out["loss"]

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            tot_loss += loss.item()
            n += 1

        avg_loss = tot_loss / max(1, n)
        metrics = {"loss": avg_loss, "main_metric": -avg_loss}

        self.log_to_wandb(metrics, prefix="train")



        return metrics

    # ----------------------------------------
    # Evaluate
    # ----------------------------------------
    @torch.no_grad()
    def evaluate(self):
        self.rgb.eval()
        self.task_head.eval()

        preds, refs = [], []
        tot_loss = 0.0

        for batch in tqdm(self.val_loader, desc="Eval (translation)"):
            rgb, rgb_len, tgt_texts = self.extract_batch(batch)

            vis_seq, vis_mask = self.encode_rgb(rgb, rgb_len)

            out = self.task_head(vis_seq, vis_mask, tgt_texts)
            tot_loss += out["loss"].item()

            # Generate
            prepared = self.task_head.prepare_inputs(vis_seq, vis_mask)
            gen_ids = self.task_head.generate(prepared, max_new_tokens=64)

            pred_texts = self.task_head.tok.batch_decode(gen_ids, skip_special_tokens=True)

            preds.extend(pred_texts)
            refs.extend(tgt_texts)

        avg_loss = tot_loss / max(1, len(self.val_loader))

        bleu  = compute_bleu(preds, refs)
        rouge = compute_rouge(preds, refs)
        cer   = compute_cer(preds, refs)
        wer   = compute_wer(preds, refs)

        metrics = {
            "loss": avg_loss,
            "BLEU": bleu,
            "CER": cer,
            "WER": wer,
            "ROUGE1": rouge["rouge1"],
            "ROUGEL": rouge["rougeL"],
            "main_metric": bleu,  # 可按喜好改成 -WER 或 ROUGE-L
        }

        self.log_to_wandb(metrics, prefix="eval")
        return metrics



# =========================================================
# Recognition Finetuner (CTC)
# =========================================================

class RecognitionFinetuner(BaseFinetuner):

    def __init__(self, cfg, device):
        super().__init__(cfg, device)              # ★ 继承 BaseFinetuner
        init_trainer_common(self, cfg, device)     # ★ 初始化 save_dir / scaler / best_metric

        train_cfg = getattr(cfg, "Training", SimpleNamespace())
        self.grad_clip = getattr(train_cfg, "grad_clip", 1.0)

        # num_classes
        self.num_classes = cfg_get(cfg, "Evaluation.recognition.num_classes", 20000)

        # gloss vocab
        self.gloss2id = {}
        self._next_gloss_id = 0

        # build dataloader / model / optimizer
        self._build_dataloaders()
        self._build_models()
        self._build_optimizer()

    # ---------------------------
    # dataloaders
    # ---------------------------
    def _build_dataloaders(self):
        args = SimpleNamespace()
        self.train_loader = create_dataloader(args, self.cfg, phase="train")
        self.val_loader   = create_dataloader(args, self.cfg, phase="dev")

    # ---------------------------
    # models
    # ---------------------------
    def _build_models(self):
        Dv = cfg_get(self.cfg, "Encoders.rgb.output_dim", 512)

        self.rgb = RGBEncoder(pretrained=False, output_dim=Dv).to(self.device)
        self.head = RecognitionHeadCTC(
            in_dim=Dv,
            num_classes=self.num_classes,
            hidden_dim=cfg_get(self.cfg, "Finetune.recognition.hidden_dim", 512),
            num_layers=cfg_get(self.cfg, "Finetune.recognition.num_layers", 4),
            nhead=cfg_get(self.cfg, "Finetune.recognition.nhead", 8),
        ).to(self.device)

    # ---------------------------
    # optimizer
    # ---------------------------
    def _build_optimizer(self):
        lr_head = cfg_get(self.cfg, "Training.lr_head", 3e-4)
        lr_back = cfg_get(self.cfg, "Training.lr_backbone", 5e-5)

        groups = []
        g_head = params_with_lr([self.head], lr_head)
        g_back = params_with_lr([self.rgb], lr_back)

        if g_head: groups.append(g_head)
        if g_back: groups.append(g_back)

        self.optimizer = AdamW(groups)

    # ---------------------------
    # batch extractor
    # ---------------------------
    def _extract_batch(self, batch):
        src, tgt = batch
        return src["rgb_img"], src["rgb_len"], tgt["gt_gloss"]

    # ---------------------------
    # gloss → ids
    # ---------------------------
    def _gloss_seq_to_ids(self, gloss_seq):

        if isinstance(gloss_seq, torch.Tensor):
            return gloss_seq.long()

        if len(gloss_seq) == 0:
            return torch.zeros(0, dtype=torch.long)

        out = []
        for tok in gloss_seq:
            if isinstance(tok, int):
                out.append(tok)
                continue

            if tok not in self.gloss2id:
                if self._next_gloss_id >= self.num_classes - 1:
                    raise RuntimeError("Gloss vocab exceeded num_classes.")
                self.gloss2id[tok] = self._next_gloss_id
                self._next_gloss_id += 1

            out.append(self.gloss2id[tok])
        return torch.tensor(out, dtype=torch.long)

    # ---------------------------
    # pack CTC target
    # ---------------------------
    def _pack_gloss(self, gloss_list):
        seqs = []
        lens = []
        for g in gloss_list:
            t = self._gloss_seq_to_ids(g)
            seqs.append(t)
            lens.append(len(t))
        packed = torch.cat(seqs) if seqs else torch.zeros(0, dtype=torch.long)
        return packed, torch.tensor(lens, dtype=torch.long)

    # ---------------------------
    # decode CTC ★必须在类里面！！！
    # ---------------------------
    def _decode_ctc(self, logits):
        pred = logits.argmax(-1)  # [T,B]
        T, B = pred.shape
        results = []
        blank = self.num_classes - 1

        for b in range(B):
            seq = []
            prev = -1
            for t in range(T):
                tok = int(pred[t, b])
                if tok != prev and tok != blank:
                    seq.append(str(tok))
                prev = tok
            results.append(" ".join(seq))
        return results

    # ---------------------------
    # encode rgb
    # ---------------------------
    def _encode_rgb(self, rgb, rgb_len):
        rgb = rgb.to(self.device)
        rgb_len = rgb_len.to(self.device)

        feat = self.rgb(rgb)  # [B,T,D]
        B, T, _ = feat.shape

        mask = torch.zeros((B, T), dtype=torch.bool, device=self.device)
        for i in range(B):
            mask[i, :rgb_len[i]] = True
        return feat, mask

    # ---------------------------
    # train epoch
    # ---------------------------
    def train_epoch(self):
        self.rgb.train()
        self.head.train()

        total = 0
        for batch in tqdm(self.train_loader, desc="Train (recognition)"):
            rgb, rgb_len, gloss = self._extract_batch(batch)

            feat, mask = self._encode_rgb(rgb, rgb_len)
            packed, target_len = self._pack_gloss(gloss)
            input_len = mask.sum(1)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.head(feat, src_key_padding_mask=~mask)
                loss = self.head.compute_loss(logits, packed, input_len, target_len)

            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            total += loss.item()

        avg = total / len(self.train_loader)
        return {"loss": avg, "main_metric": -avg}

    def log_to_wandb(self, metrics: dict, prefix=""):
        """
        统一 WandB 日志接口。只有在 trainer.wandb_enabled=True 时才记录。
        """
        if not getattr(self, "wandb_enabled", False):
            return

        log_dict = {}
        for k, v in metrics.items():
            key = f"{prefix}/{k}" if prefix else k
            log_dict[key] = v

        import wandb
        wandb.log(log_dict)

    # ---------------------------
    # evaluate
    # ---------------------------
    @torch.no_grad()
    def evaluate(self):
        self.rgb.eval()
        self.head.eval()

        # ---- 安全检查 ----
        if len(self.val_loader) == 0:
            print("[Eval] WARNING: val_loader is empty! Returning default metrics.")
            metrics = {
                "loss": float("inf"),
                "WER": 1.0,
                "CER": 1.0,
                "main_metric": -1.0,
            }
            self.log_to_wandb(metrics, prefix="eval")
            return metrics

        total = 0
        preds = []
        refs = []

        for batch in tqdm(self.val_loader, desc="Eval (recognition)"):
            rgb, rgb_len, gloss = self._extract_batch(batch)

            feat, mask = self._encode_rgb(rgb, rgb_len)
            packed, target_len = self._pack_gloss(gloss)
            input_len = mask.sum(1)

            logits = self.head(feat, src_key_padding_mask=~mask)
            loss = self.head.compute_loss(logits, packed, input_len, target_len)
            total += loss.item()

            preds.extend(self._decode_ctc(logits))
            refs.extend([" ".join(map(str, g)) for g in gloss])

        # 使用 max(1, len) 防止除以 0
        avg_loss = total / max(1, len(self.val_loader))

        cer = compute_cer(preds, refs) if preds and refs else 1.0
        wer = compute_wer(preds, refs) if preds and refs else 1.0

        metrics = {
            "loss": avg_loss,
            "CER": cer,
            "WER": wer,
            "main_metric": -wer,
        }

        # print(f"[Eval] loss={avg_loss:.4f}, CER={cer:.4f}, WER={wer:.4f}")
        print(
            f"[Eval] "
            f"loss={avg_loss:.4f} ↓ (range: [0, +∞]), "
            f"CER={cer:.4f} ↓ (range: [0.0, 1.0]), "
            f"WER={wer:.4f} ↓ (range: [0.0, 1.0]), "
            f"main_metric={-wer:.4f} ↑ (range: [-1.0, 0.0])"
        )

        # 记录到 wandb
        self.log_to_wandb(metrics, prefix="eval")

        return metrics


# =========================================================
# Factory
# =========================================================
class FinetunerFactory:
    @staticmethod
    def create(cfg, device):
        task = getattr(cfg.Finetune, "task", "retrieval").lower()
        logger.info(f"[Factory] task = {task}")

        if task == "retrieval":
            return RetrievalFinetuner(cfg, device)
        elif task == "translation":
            return TranslationFinetuner(cfg, device)
        elif task == "recognition":
            return RecognitionFinetuner(cfg, device)
        else:
            raise ValueError(f"Unknown finetuner task: {task}")


# =========================================================
# CLI & Main
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Uni-SLM Finetuning")

    parser.add_argument("--config", type=str,
                        default="config/finetune1.yaml")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_backbone", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="0")

    return parser.parse_args()

def main():
    args = parse_args()
    set_seed()

    # ================ Load config ================
    raw_cfg = load_yaml(args.config)
    cfg = dict_to_ns(raw_cfg)

    # ================ Build Device ================
    if args.device == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}")
    cfg.device = args.device
    logger.info(f"[Main] device = {device}")

    # ================ WandB Initialization ================
    wandb_cfg = getattr(cfg, "wandb", None)
    global WANDB_AVAILABLE

    if wandb_cfg and WANDB_AVAILABLE and getattr(wandb_cfg, "use", False):
        api_key = os.getenv("WANDB_API_KEY", None)
        if api_key:
            try:
                wandb.login(key=api_key)

                wandb.init(
                    project=getattr(wandb_cfg, "project", "Uni-SLM"),
                    name=getattr(wandb_cfg, "run_name", f"finetuner-{getattr(cfg.Finetune, 'task', 'unknown')}"),
                    config=raw_cfg,
                )

                print(f"[WandB] Initialized: project={wandb.run.project}, name={wandb.run.name}")
            except Exception as e:
                print(f"[WandB] Initialization failed: {e}")
                WANDB_AVAILABLE = False
        else:
            print("[WandB] No API Key found (export WANDB_API_KEY=...). WandB disabled.")
            WANDB_AVAILABLE = False
    else:
        print("[WandB] WandB disabled by config or unavailable.")
        WANDB_AVAILABLE = False

    # ============================================
    # CLI Override for Training Config
    # ============================================
    if hasattr(cfg, "Training"):
        if args.epochs: cfg.Training.epochs = args.epochs
        if args.batch_size: cfg.Training.batch_size = args.batch_size
        if args.lr_head: cfg.Training.learning_rate_head = args.lr_head
        if args.lr_backbone: cfg.Training.learning_rate_backbone = args.lr_backbone

    # ================ Build Finetuner ================
    finetuner = FinetunerFactory.create(cfg, device)

    # ================ Training Loop ================
    for epoch in range(cfg.Training.epochs):
        logger.info(f"===== Epoch {epoch + 1}/{cfg.Training.epochs} =====")

        train_res = finetuner.train_epoch()
        print(f"Train results: {train_res}")

        eval_res = finetuner.evaluate()
        print(f"Eval results: {eval_res}")

        main_metric = eval_res["main_metric"]

        # ================ WandB Logging ================
        if WANDB_AVAILABLE:
            epoch_log = {"epoch": epoch + 1}
            for k, v in train_res.items():
                epoch_log[f"train/{k}"] = v
            for k, v in eval_res.items():
                epoch_log[f"eval/{k}"] = v
            wandb.log(epoch_log)

        # ================ Save Checkpoints ================
        model_dict = {
            "rgb": getattr(finetuner, "rgb", None),
            "text": getattr(finetuner, "text", None),
            "task_head": getattr(finetuner, "task_head", None),
            "head": getattr(finetuner, "head", None),
        }
        model_state = {k: v.state_dict() for k, v in model_dict.items() if v is not None}

        save_checkpoint(
            os.path.join(finetuner.save_dir, f"epoch_{epoch + 1}.pt"),
            model_state,
            finetuner.optimizer,
            epoch,
            main_metric,
        )

        if main_metric > finetuner.best_metric:
            finetuner.best_metric = main_metric
            save_checkpoint(
                os.path.join(finetuner.save_dir, "best.pt"),
                model_state,
                finetuner.optimizer,
                epoch,
                main_metric,
            )
            print(f"[Main] New best model saved (main_metric={main_metric:.4f})")


if __name__ == "__main__":
    main()