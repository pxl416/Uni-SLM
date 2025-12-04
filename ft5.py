# finetune.py
# -*- coding: utf-8 -*-
import os
import logging
import argparse
from types import SimpleNamespace

import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import torch.cuda.amp as amp

# Dataset
from datasets.datasets import create_dataloader

# Encoders
from models.Encoder.rgb_encoder import RGBEncoder
from models.Encoder.pose_encoder import PoseEncoder  # 现在没用到，但先保留
from models.Encoder.text_encoder import TextEncoder

# Heads
from models.Head.retrieval import RetrievalHead
from models.Head.translation import TranslationHeadMT5
from models.Head.recognition import RecognitionHeadCTC

# Utils
from utils.config import cfg_get, load_yaml, dict_to_ns
from utils.trainer import (
    init_trainer_common,
    encode_rgb_features,
    params_with_lr,
    save_checkpoint,
    load_checkpoint,
    log_metrics_if_enabled,
    set_global_seed,
)

from utils.metrics import compute_bleu, compute_rouge, compute_cer, compute_wer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# WandB (safe import，仅做初始化与开关判断，其它 logging 走 log_metrics_if_enabled)
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False
    print("[Warn] wandb not installed; logging disabled.")


# =========================================================
# Retrieval Finetuner
# =========================================================
class RetrievalFinetuner:
    def __init__(self, cfg, device):
        """
        cfg: SimpleNamespace from config
        device: torch.device
        """
        # 统一基础属性（epochs, grad_clip, batch_size, save_dir, wandb_enabled 等）
        init_trainer_common(self, cfg, device, task="retrieval")

        logger.info("[Finetune][Retrieval] Building dataloaders ...")
        self._build_dataloaders()

        logger.info("[Finetune][Retrieval] Building model ...")
        self._build_models()

        logger.info("[Finetune][Retrieval] Building optimizer ...")
        self._build_optimizer()

    # ------------------------------
    # DataLoader
    # ------------------------------
    def _build_dataloaders(self):
        """
        使用 create_dataloader（与 test_dataloader.py 保持一致）
        这里使用 self.batch_size / self.num_workers 由 init_trainer_common 注入
        """
        args = SimpleNamespace(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            seed=getattr(self.cfg, "seed", 3407),
            cfg=None,
        )

        self.train_loader = create_dataloader(args, self.cfg, phase="train")
        self.val_loader = create_dataloader(args, self.cfg, phase="dev")

        logger.info(f"[Data][Retrieval] train batches = {len(self.train_loader)}")
        logger.info(f"[Data][Retrieval] val   batches = {len(self.val_loader)}")

    # ------------------------------
    # Models
    # ------------------------------
    def _build_models(self):
        Dv = getattr(self.cfg.Encoders.rgb, "output_dim", 512)
        Dt = getattr(self.cfg.Encoders.text, "output_dim", 384)

        self.rgb = RGBEncoder(
            pretrained=False,
            output_dim=Dv,
        ).to(self.device)

        text_model_path = getattr(
            self.cfg.Finetune,
            "text_model_path",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        self.text = TextEncoder(
            model_path=text_model_path,
            return_sequence=False,
            max_length=128,
        ).to(self.device)

        proj_dim = self.cfg.Evaluation.retrieval.proj_dim
        temperature = self.cfg.Evaluation.retrieval.temperature

        self.label_smoothing = getattr(
            self.cfg.Evaluation.retrieval, "label_smoothing", 0.0
        )

        self.task_head = RetrievalHead(
            rgb_in=Dv,
            text_in=Dt,
            proj_dim=proj_dim,
            temperature=temperature,
            projection_type="linear",
            dropout=0.1,
            trainable=True,
            learnable_tau=getattr(
                self.cfg.Evaluation.retrieval, "learnable_tau", False
            ),
        ).to(self.device)

    # ------------------------------
    # Optimizer
    # ------------------------------
    def _build_optimizer(self):
        train_cfg = getattr(self.cfg, "Training", SimpleNamespace())

        # 兼容两种命名：lr_head / learning_rate_head
        lr_head = getattr(
            train_cfg,
            "lr_head",
            getattr(train_cfg, "learning_rate_head", 3e-4),
        )
        lr_backbone = getattr(
            train_cfg,
            "lr_backbone",
            getattr(train_cfg, "learning_rate_backbone", 5e-5),
        )

        groups = []

        g_head = params_with_lr([self.task_head], lr_head)
        if g_head:
            groups.append(g_head)

        g_back = params_with_lr([self.rgb, self.text], lr_backbone)
        if g_back:
            groups.append(g_back)

        if not groups:
            raise RuntimeError("[Retrieval] No parameters to optimize.")

        self.optimizer = AdamW(groups)
        logger.info(
            f"[Optimizer][Retrieval] head lr={lr_head}, backbone lr={lr_backbone}"
        )

    # ------------------------------
    # Batch extractor
    # ------------------------------
    def _extract_batch(self, batch):
        """
        batch is returned by BaseDataset.collate_fn:
            (src_input: dict, tgt_input: dict)
        """
        if not (isinstance(batch, (tuple, list)) and len(batch) == 2):
            raise ValueError(
                f"[Retrieval] Expected batch = (src_input, tgt_input), got {type(batch)}"
            )

        src, tgt = batch
        rgb = src["rgb_img"]
        rgb_len = src["rgb_len"]
        text = tgt["gt_sentence"]

        return rgb, rgb_len, text

    # ------------------------------
    # Train / Eval
    # ------------------------------
    def train_epoch(self):
        self.rgb.train()
        self.text.train()
        self.task_head.train()

        total_loss = 0.0

        for batch_idx, batch in enumerate(
            tqdm(self.train_loader, desc="Train (retrieval)")
        ):
            rgb, rgb_len, text = self._extract_batch(batch)

            vis_seq, vis_mask = encode_rgb_features(
                self.rgb, rgb, rgb_len, self.device
            )
            text_feat, _ = self.text(text)

            loss = self.task_head.compute_loss(
                rgb_feat=vis_seq,
                text_feat=text_feat,
                rgb_mask=vis_mask,
                text_mask=None,
                label_smoothing=self.label_smoothing,
            )

            self.optimizer.zero_grad()
            loss.backward()

            all_params = [
                p for g in self.optimizer.param_groups for p in g["params"]
            ]
            clip_grad_norm_(all_params, max_norm=self.grad_clip)

            self.optimizer.step()
            total_loss += loss.item()

            if batch_idx % 10 == 0:
                step_metrics = {
                    "batch_loss": loss.item(),
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                }
                log_metrics_if_enabled(self, step_metrics, prefix="train/batch")

        avg_loss = total_loss / max(1, len(self.train_loader))
        epoch_metrics = {"loss": avg_loss, "main_metric": -avg_loss}
        log_metrics_if_enabled(self, epoch_metrics, prefix="train/epoch")

        return epoch_metrics

    @torch.no_grad()
    def evaluate(self):
        self.rgb.eval()
        self.text.eval()
        self.task_head.eval()

        total_loss = 0.0
        all_rgb = []
        all_text = []

        for batch in tqdm(self.val_loader, desc="Eval (retrieval)"):
            rgb, rgb_len, text = self._extract_batch(batch)

            vis_seq, vis_mask = encode_rgb_features(
                self.rgb, rgb, rgb_len, self.device
            )
            text_feat, _ = self.text(text)

            loss = self.task_head.compute_loss(
                rgb_feat=vis_seq,
                text_feat=text_feat,
                rgb_mask=vis_mask,
                text_mask=None,
                label_smoothing=self.label_smoothing,
            )
            total_loss += loss.item()

            pooled_rgb = self.task_head._maybe_pool(vis_seq, vis_mask)
            all_rgb.append(pooled_rgb.cpu())
            all_text.append(text_feat.cpu())

        avg_loss = total_loss / max(1, len(self.val_loader))

        if all_rgb:
            rgb_all = torch.cat(all_rgb, dim=0).to(self.device)
            text_all = torch.cat(all_text, dim=0).to(self.device)

            metrics = self.task_head.compute_metrics(
                rgb_feat=rgb_all,
                text_feat=text_all,
                rgb_mask=None,
                text_mask=None,
                use_temperature=True,
            )
            metrics["loss"] = avg_loss
            # 这里假设 mean_R1 是主指标
            metrics["main_metric"] = metrics.get("mean_R1", -avg_loss)
        else:
            metrics = {"loss": avg_loss, "main_metric": -avg_loss}

        log_metrics_if_enabled(self, metrics, prefix="eval")
        return metrics


# =========================================================
# TranslationFinetuner  ——  RGB → MT5
# =========================================================
class TranslationFinetuner:
    def __init__(self, cfg, device):
        self.amp_enabled = getattr(cfg.Finetune, "amp", True)
        init_trainer_common(self, cfg, device, task="translation")

        logger.info("[Finetune][Translation] Building dataloaders ...")
        self._build_dataloaders()

        logger.info("[Finetune][Translation] Building model ...")
        self._build_models()

        logger.info("[Finetune][Translation] Building optimizer ...")
        self._build_optimizer()

        logger.info("[Finetune] TranslationFinetuner ready.")

    # --------------------------------------------------------
    # Build dataloaders
    # --------------------------------------------------------
    def _build_dataloaders(self):
        args = SimpleNamespace(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            seed=getattr(self.cfg, "seed", 3407),
            rgb_support=True,
            pose_support=False,
        )

        self.train_loader = create_dataloader(args, self.cfg, phase="train")
        self.val_loader = create_dataloader(args, self.cfg, phase="dev")

        logger.info(f"[Data][Translation] train batches = {len(self.train_loader)}")
        logger.info(f"[Data][Translation] val   batches = {len(self.val_loader)}")

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------
    def _build_models(self):
        Dv = cfg_get(self.cfg, "Encoders.rgb.output_dim", 512)

        self.rgb = RGBEncoder(
            pretrained=False,
            output_dim=Dv,
        ).to(self.device)

        mt5_path = cfg_get(
            self.cfg,
            "Finetune.translation_model_path",
            "google/mt5-small",
        )

        self.task_head = TranslationHeadMT5(
            mt5_path,
            in_dim=Dv,
            label_smoothing=cfg_get(
                self.cfg, "Finetune.label_smoothing", 0.1
            ),
            lang_prompt=cfg_get(
                self.cfg, "Finetune.lang_prompt", "Chinese"
            ),
            max_target_len=cfg_get(
                self.cfg, "Finetune.max_target_len", 50
            ),
        ).to(self.device)

        logger.info(f"[Model][Translation] RGB encoder = {type(self.rgb).__name__}")
        logger.info(
            f"[Model][Translation] TranslationHead = {type(self.task_head).__name__}"
        )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------
    def _build_optimizer(self):
        train_cfg = getattr(self.cfg, "Training", SimpleNamespace())

        lr_head = getattr(
            train_cfg,
            "lr_head",
            getattr(train_cfg, "learning_rate_head", 1e-4),
        )
        lr_backbone = getattr(
            train_cfg,
            "lr_backbone",
            getattr(train_cfg, "learning_rate_backbone", 1e-5),
        )

        groups = []
        g_head = params_with_lr([self.task_head], lr_head)
        g_back = params_with_lr([self.rgb], lr_backbone)

        if g_head:
            groups.append(g_head)
        if g_back:
            groups.append(g_back)

        if not groups:
            raise RuntimeError("[Translation] No parameters to optimize.")

        self.optimizer = AdamW(groups)
        logger.info(
            f"[Optimizer][Translation] head lr={lr_head}, backbone lr={lr_backbone}"
        )

    # --------------------------------------------------------
    # helpers
    # --------------------------------------------------------
    def _extract_batch(self, batch):
        """
        batch = (src_input, tgt_input)
        """
        if not isinstance(batch, (list, tuple)):
            raise ValueError(
                f"[Translation] batch should be (src, tgt), got {type(batch)}"
            )

        src, tgt = batch

        rgb = src["rgb_img"]  # [B,T,3,H,W]
        rgb_len = src["rgb_len"]  # [B]
        tgt_text = tgt["gt_sentence"]

        return rgb, rgb_len, tgt_text

    # --------------------------------------------------------
    # Train One Epoch
    # --------------------------------------------------------
    def train_epoch(self):
        self.rgb.train()
        self.task_head.train()

        total_loss = 0.0
        n = 0

        pbar = tqdm(self.train_loader, desc="Train (translation)")

        for batch in pbar:
            rgb, rgb_len, tgt_text = self._extract_batch(batch)

            vis_seq, vis_mask = encode_rgb_features(
                self.rgb, rgb, rgb_len, self.device
            )

            with amp.autocast(enabled=self.amp_enabled):
                out = self.task_head(
                    vis_seq=vis_seq,
                    vis_mask=vis_mask,
                    tgt_texts=tgt_text,
                )
                loss = out["loss"]

            self.optimizer.zero_grad()
            loss.backward()

            clip_grad_norm_(self.task_head.parameters(), self.grad_clip)
            clip_grad_norm_(self.rgb.parameters(), self.grad_clip)

            self.optimizer.step()

            total_loss += loss.item()
            n += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg = total_loss / max(1, n)
        metrics = {"loss": avg, "main_metric": -avg}
        log_metrics_if_enabled(self, metrics, prefix="train/epoch")

        return metrics

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------
    @torch.no_grad()
    def evaluate(self):
        self.rgb.eval()
        self.task_head.eval()

        total_loss = 0.0
        preds, refs = [], []

        for batch in tqdm(self.val_loader, desc="Eval (translation)"):
            rgb, rgb_len, tgt_text = self._extract_batch(batch)

            vis_seq, vis_mask = encode_rgb_features(
                self.rgb, rgb, rgb_len, self.device
            )

            with amp.autocast(enabled=self.amp_enabled):
                out = self.task_head(
                    vis_seq=vis_seq,
                    vis_mask=vis_mask,
                    tgt_texts=tgt_text,
                )
                total_loss += out["loss"].item()

            # Decode
            prepared = self.task_head.prepare_inputs(vis_seq, vis_mask)
            gen_ids = self.task_head.generate(prepared, max_new_tokens=64)
            pred_text = self.task_head.tok.batch_decode(
                gen_ids, skip_special_tokens=True
            )

            preds.extend(pred_text)
            refs.extend(tgt_text)

        avg_loss = total_loss / max(1, len(self.val_loader))

        bleu = compute_bleu(preds, refs)
        rouge = compute_rouge(preds, refs)
        cer = compute_cer(preds, refs)
        wer = compute_wer(preds, refs)

        logger.info(
            f"[Eval][Translation] Loss={avg_loss:.4f}, "
            f"BLEU={bleu:.4f}, CER={cer:.4f}, WER={wer:.4f}"
        )

        metrics = {
            "loss": avg_loss,
            "BLEU": bleu,
            "CER": cer,
            "WER": wer,
            "ROUGE1": rouge["rouge1"],
            "ROUGEL": rouge["rougeL"],
            "main_metric": bleu,
        }
        log_metrics_if_enabled(self, metrics, prefix="eval")

        return metrics


# =========================================================
# Recognition Finetuner (CTC)
# =========================================================
class RecognitionFinetuner:
    """
    使用 BaseDataset.collate_fn 的 batch 结构：
        batch = (src_input, tgt_input)
    """

    def __init__(self, cfg: SimpleNamespace, device: torch.device):
        self.amp_enabled = getattr(cfg.Finetune, "amp", True)
        init_trainer_common(self, cfg, device, task="recognition")

        # 识别任务配置
        self.num_classes = cfg_get(
            self.cfg, "Evaluation.recognition.num_classes", 20000
        )

        # gloss vocab
        self.gloss2id = {}
        self._next_gloss_id = 0  # 0..num_classes-2，num_classes-1 留给 blank

        logger.info("[Finetune][Recognition] Building dataloaders ...")
        self._build_dataloaders()

        logger.info("[Finetune][Recognition] Building model ...")
        self._build_models()

        logger.info("[Finetune][Recognition] Building optimizer ...")
        self._build_optimizer()

        # AMP scaler
        self.scaler = amp.GradScaler(enabled=self.amp_enabled)

    # -----------------------------
    # dataloaders
    # -----------------------------
    def _build_dataloaders(self):
        args = SimpleNamespace(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            seed=getattr(self.cfg, "seed", 3407),
            cfg=None,
        )

        self.train_loader = create_dataloader(args, self.cfg, phase="train")
        self.val_loader = create_dataloader(args, self.cfg, phase="dev")

        logger.info(f"[Data][Recognition] train batches = {len(self.train_loader)}")
        logger.info(f"[Data][Recognition] val   batches = {len(self.val_loader)}")

    # -----------------------------
    # models
    # -----------------------------
    def _build_models(self):
        Dv = cfg_get(self.cfg, "Encoders.rgb.output_dim", 512)

        self.rgb = RGBEncoder(
            pretrained=False,
            output_dim=Dv,
        ).to(self.device)

        self.head = RecognitionHeadCTC(
            in_dim=Dv,
            num_classes=self.num_classes,
            hidden_dim=cfg_get(self.cfg, "Finetune.recognition.hidden_dim", 512),
            num_layers=cfg_get(self.cfg, "Finetune.recognition.num_layers", 4),
            nhead=cfg_get(self.cfg, "Finetune.recognition.nhead", 8),
            dropout=0.1,
        ).to(self.device)

        logger.info(f"[Model][Recognition] RGB = {type(self.rgb).__name__}")
        logger.info(
            f"[Model][Recognition] RecognitionHead = {type(self.head).__name__}"
        )

    # -----------------------------
    # optimizer
    # -----------------------------
    def _build_optimizer(self):
        train_cfg = getattr(self.cfg, "Training", SimpleNamespace())

        lr_head = getattr(
            train_cfg,
            "lr_head",
            getattr(train_cfg, "learning_rate_head", 3e-4),
        )
        lr_backbone = getattr(
            train_cfg,
            "lr_backbone",
            getattr(train_cfg, "learning_rate_backbone", 5e-5),
        )

        groups = []
        g_head = params_with_lr([self.head], lr_head)
        g_back = params_with_lr([self.rgb], lr_backbone)

        if g_head:
            groups.append(g_head)
        if g_back:
            groups.append(g_back)

        if not groups:
            raise RuntimeError("[Recognition] No parameters to optimize.")

        self.optimizer = AdamW(groups)
        logger.info(
            f"[Optimizer][Recognition] head lr={lr_head}, backbone lr={lr_backbone}"
        )

    # ======================================================
    # batch 解包
    # ======================================================
    def _extract_batch(self, batch):
        if not (isinstance(batch, (tuple, list)) and len(batch) == 2):
            raise ValueError(
                f"[Recognition] Expect (src_input, tgt_input), got {type(batch)}"
            )

        src, tgt = batch

        rgb = src.get("rgb_img", None)
        rgb_len = src.get("rgb_len", None)
        gloss_raw = tgt.get("gt_gloss", None)

        if rgb is None:
            raise ValueError("[Recognition] Missing 'rgb_img' in src_input")
        if rgb_len is None:
            raise ValueError("[Recognition] Missing 'rgb_len' in src_input")
        if gloss_raw is None:
            raise ValueError("[Recognition] Missing 'gt_gloss' in tgt_input")

        return rgb, rgb_len, gloss_raw

    # ======================================================
    # gloss → id
    # ======================================================
    def _gloss_seq_to_ids(self, gloss_seq):
        # tensor
        if isinstance(gloss_seq, torch.Tensor):
            return gloss_seq.long()

        if len(gloss_seq) == 0:
            return torch.zeros(0, dtype=torch.long)

        first = gloss_seq[0]

        # 已经是数字序列
        if isinstance(first, int):
            return torch.tensor(gloss_seq, dtype=torch.long)

        # 否则认为是字符串序列
        ids = []
        for tok in gloss_seq:
            if isinstance(tok, int):
                ids.append(tok)
                continue

            if tok not in self.gloss2id:
                if self._next_gloss_id >= self.num_classes - 1:
                    raise RuntimeError(
                        f"[Recognition] gloss vocab size exceeded num_classes-1 "
                        f"({self.num_classes - 1}). token='{tok}'"
                    )
                self.gloss2id[tok] = self._next_gloss_id
                self._next_gloss_id += 1

            ids.append(self.gloss2id[tok])

        return torch.tensor(ids, dtype=torch.long)

    def _pack_gloss(self, gloss_list):
        seq_tensors = []
        lengths = []

        for g in gloss_list:
            t = self._gloss_seq_to_ids(g)
            seq_tensors.append(t)
            lengths.append(len(t))

        if seq_tensors:
            packed = torch.cat(seq_tensors, dim=0)
        else:
            packed = torch.zeros(0, dtype=torch.long)

        target_lengths = torch.tensor(lengths, dtype=torch.long)
        return packed, target_lengths

    # ======================================================
    # 训练一个 epoch
    # ======================================================
    def train_epoch(self):
        self.rgb.train()
        self.head.train()

        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(self.train_loader, desc="Train (recognition)"):
            rgb, rgb_len, gloss = self._extract_batch(batch)

            feats, mask = encode_rgb_features(
                self.rgb, rgb, rgb_len, self.device
            )
            input_lengths = mask.sum(dim=1).long()

            packed_targets, target_lengths = self._pack_gloss(gloss)

            with amp.autocast(enabled=self.amp_enabled):
                logits = self.head(feats, src_key_padding_mask=~mask)
                loss = self.head.compute_loss(
                    logits,
                    packed_targets,
                    input_lengths,
                    target_lengths,
                )

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            all_params = []
            for g in self.optimizer.param_groups:
                all_params.extend(g["params"])
            clip_grad_norm_(all_params, max_norm=self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        metrics = {"loss": avg_loss, "main_metric": -avg_loss}
        log_metrics_if_enabled(self, metrics, prefix="train/epoch")

        return metrics

    # ======================================================
    # 验证
    # ======================================================
    @torch.no_grad()
    def evaluate(self):
        self.rgb.eval()
        self.head.eval()

        total_loss = 0.0
        n_batches = 0

        preds_all = []
        refs_all = []

        for batch in tqdm(self.val_loader, desc="Eval (recognition)"):
            rgb, rgb_len, gloss = self._extract_batch(batch)

            feats, mask = encode_rgb_features(
                self.rgb, rgb, rgb_len, self.device
            )
            input_lengths = mask.sum(dim=1).long()
            packed_targets, target_lengths = self._pack_gloss(gloss)

            logits = self.head(feats, src_key_padding_mask=~mask)
            loss = self.head.compute_loss(
                logits,
                packed_targets,
                input_lengths,
                target_lengths,
            )

            total_loss += loss.item()
            n_batches += 1

            # decode CTC logits -> "id id id" 串
            pred_ids = logits.argmax(dim=-1)  # [T,B]
            T, B = pred_ids.shape
            for b in range(B):
                seq = []
                prev = -1
                for t in range(T):
                    tok = int(pred_ids[t, b])
                    if tok != prev and tok != (self.num_classes - 1):
                        seq.append(str(tok))
                    prev = tok
                preds_all.append(" ".join(seq))

            refs = [" ".join([str(x) for x in g]) for g in gloss]
            refs_all.extend(refs)

        avg_loss = total_loss / max(1, n_batches)

        wer = compute_wer(preds_all, refs_all)
        cer = compute_cer(preds_all, refs_all)

        logger.info(
            f"[Eval][Recognition] loss={avg_loss:.4f}, WER={wer:.4f}, CER={cer:.4f}"
        )

        metrics = {
            "loss": avg_loss,
            "WER": wer,
            "CER": cer,
            "main_metric": -wer,
        }
        log_metrics_if_enabled(self, metrics, prefix="eval")

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
            raise ValueError(f"Unknown finetune task: {task}")


# =========================================================
# CLI & Main
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Uni-SLM Finetuning")

    parser.add_argument("--config", type=str, default="config/finetune1.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr_head", type=float, default=None)
    parser.add_argument("--lr_backbone", type=float, default=None)
    parser.add_argument("--device", type=str, default="0")

    return parser.parse_args()


def main():
    args = parse_args()

    # === Load config via utils.config ===
    raw_cfg = load_yaml(args.config)
    cfg = dict_to_ns(raw_cfg)

    # 覆盖 Training 参数（如果 CLI 提供）
    if hasattr(cfg, "Training"):
        if args.epochs is not None:
            cfg.Training.epochs = args.epochs
        if args.batch_size is not None:
            cfg.Training.batch_size = args.batch_size
        if args.lr_head is not None:
            # 同时兼容两种命名
            if hasattr(cfg.Training, "learning_rate_head"):
                cfg.Training.learning_rate_head = args.lr_head
            cfg.Training.lr_head = args.lr_head
        if args.lr_backbone is not None:
            if hasattr(cfg.Training, "learning_rate_backbone"):
                cfg.Training.learning_rate_backbone = args.lr_backbone
            cfg.Training.lr_backbone = args.lr_backbone

    # 设置设备
    if args.device == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}")
    cfg.device = args.device
    logger.info(f"[Main] device={device}")

    # 设定全局随机种子（从 cfg.seed 读）
    seed = getattr(cfg, "seed", 3407)
    set_global_seed(seed)
    logger.info(f"[Main] Global seed = {seed}")

    # WandB 初始化（只负责 init，具体 log 由 log_metrics_if_enabled 处理）
    wandb_available = WANDB_AVAILABLE
    if wandb_available:
        try:
            wandb.init(
                project=getattr(cfg, "wandb_project", "Uni-SLM"),
                name=getattr(
                    cfg,
                    "wandb_run_name",
                    f"finetune-{getattr(cfg.Finetune, 'task', 'unknown')}",
                ),
                config=raw_cfg,
            )
            logger.info(
                f"[WandB] Initialized. Project: {wandb.run.project}, "
                f"Name: {wandb.run.name}"
            )
        except Exception as e:
            logger.warning(f"[WandB] Initialization failed: {e}")
            wandb_available = False

    # 构建 finetuner
    finetuner = FinetunerFactory.create(cfg, device)

    # 训练循环
    for epoch in range(cfg.Training.epochs):
        logger.info(f"===== Epoch {epoch + 1}/{cfg.Training.epochs} =====")

        train_res = finetuner.train_epoch()
        logger.info(f"Train results: {train_res}")

        eval_res = finetuner.evaluate()
        logger.info(f"Eval results: {eval_res}")

        main_metric = eval_res["main_metric"]

        # epoch 汇总写到 wandb（可选）
        if wandb_available:
            epoch_log = {"epoch": epoch + 1}
            for k, v in train_res.items():
                epoch_log[f"summary/train_{k}"] = v
            for k, v in eval_res.items():
                epoch_log[f"summary/eval_{k}"] = v
            wandb.log(epoch_log)

        # ---------- Save last ----------
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

        # ---------- Save best ----------
        if main_metric > finetuner.best_metric:
            finetuner.best_metric = main_metric
            save_checkpoint(
                os.path.join(finetuner.save_dir, "best.pt"),
                model_state,
                finetuner.optimizer,
                epoch,
                main_metric,
            )
            logger.info(
                f"[Main] New best model saved (main_metric={main_metric:.4f})"
            )

    if wandb_available:
        wandb.finish()


if __name__ == "__main__":
    main()
