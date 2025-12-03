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
from utils.metrics import compute_wer, compute_cer
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
    """Load YAML exactly like test_dataloader.py"""
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



# =========================================================
# Retrieval Finetuner
# =========================================================
class RetrievalFinetuner:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

        train_cfg = getattr(cfg, "Training", SimpleNamespace())
        self.epochs = getattr(train_cfg, "epochs", 10)
        self.grad_clip = getattr(train_cfg, "grad_clip", 1.0)

        logger.info("[Finetune] Building dataloaders ...")
        self._build_dataloaders()

        logger.info("[Finetune] Building model ...")
        self._build_models()

        logger.info("[Finetune] Building optimizer ...")
        self._build_optimizer()

        self.save_dir = getattr(cfg, "save_dir", "checkpoints")
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_metric = -1e18

    # ------------------------------
    # DataLoader (same pattern as test_dataloader.py)
    # ------------------------------
    def _build_dataloaders(self):
        args = SimpleNamespace(cfg=None)
        self.train_loader = create_dataloader(args, self.cfg, phase="train")
        self.val_loader = create_dataloader(args, self.cfg, phase="dev")

        logger.info(f"[Data] train batches = {len(self.train_loader)}")
        logger.info(f"[Data] val batches   = {len(self.val_loader)}")

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

        text_model_path = getattr(self.cfg.Finetune, "text_model_path",
                                  "sentence-transformers/all-MiniLM-L6-v2")
        self.text = TextEncoder(
            model_path=text_model_path,
            return_sequence=False,
            max_length=128,
        ).to(self.device)

        proj_dim = self.cfg.Evaluation.retrieval.proj_dim
        temperature = self.cfg.Evaluation.retrieval.temperature

        self.label_smoothing = getattr(self.cfg.Evaluation.retrieval,
                                       "label_smoothing", 0.0)

        self.task_head = RetrievalHead(
            rgb_in=Dv,
            text_in=Dt,
            proj_dim=proj_dim,
            temperature=temperature,
            projection_type="linear",
            dropout=0.1,
            trainable=True,
            learnable_tau=getattr(self.cfg.Evaluation.retrieval,
                                  "learnable_tau", False),
        ).to(self.device)

    # ------------------------------
    # Optimizer
    # ------------------------------
    def _build_optimizer(self):
        train_cfg = self.cfg.Training
        lr_head = getattr(train_cfg, "learning_rate_head", 3e-4)
        lr_back = getattr(train_cfg, "learning_rate_backbone", 5e-5)

        groups = []
        g_head = params_with_lr([self.task_head], lr_head)
        if g_head: groups.append(g_head)

        g_back = params_with_lr([self.rgb, self.text], lr_back)
        if g_back: groups.append(g_back)

        self.optimizer = AdamW(groups)

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
    # Encode RGB & Text
    # ------------------------------
    def _encode_rgb(self, rgb, rgb_len):
        rgb = rgb.to(self.device)
        rgb_len = rgb_len.to(self.device)

        feat = self.rgb(rgb)      # [B,T,D]
        B, T, _ = feat.shape

        mask = torch.zeros((B, T), dtype=torch.bool, device=self.device)
        for i in range(B):
            mask[i, :rgb_len[i]] = True

        return feat, mask

    def _encode_text(self, sentences):
        feat, _ = self.text(sentences)
        return feat

    # ------------------------------
    # Train one epoch
    # ------------------------------
    def train_epoch(self):
        self.rgb.train()
        self.text.train()
        self.task_head.train()

        total_loss = 0
        for batch in tqdm(self.train_loader, desc="Train (retrieval)"):
            rgb, rgb_len, text = self._extract_batch(batch)

            vis_seq, vis_mask = self._encode_rgb(rgb, rgb_len)
            text_feat = self._encode_text(text)

            loss = self.task_head.compute_loss(
                rgb_feat=vis_seq,
                text_feat=text_feat,
                rgb_mask=vis_mask,
                text_mask=None,
                label_smoothing=self.label_smoothing,
            )

            self.optimizer.zero_grad()
            loss.backward()

            all_params = [p for g in self.optimizer.param_groups for p in g["params"]]
            clip_grad_norm_(all_params, max_norm=self.grad_clip)

            self.optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(self.train_loader))
        return {"loss": avg_loss, "main_metric": -avg_loss}

    # ------------------------------
    # Evaluate
    # ------------------------------
    @torch.no_grad()
    def evaluate(self):
        self.rgb.eval()
        self.text.eval()
        self.task_head.eval()

        total_loss = 0
        all_rgb = []
        all_text = []

        for batch in tqdm(self.val_loader, desc="Eval (retrieval)"):
            rgb, rgb_len, text = self._extract_batch(batch)

            vis_seq, vis_mask = self._encode_rgb(rgb, rgb_len)
            text_feat = self._encode_text(text)

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
            metrics["main_metric"] = metrics.get("mean_R1", -avg_loss)
            return metrics

        return {"loss": avg_loss, "main_metric": -avg_loss}



# =========================================================
# TranslationFinetuner  ——  RGB → MT5
# =========================================================
class TranslationFinetuner:

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

        train_cfg = getattr(cfg, "Training", SimpleNamespace())
        self.epochs = getattr(train_cfg, "epochs", 3)
        self.grad_clip = getattr(train_cfg, "grad_clip", 1.0)

        # ---------- dataloader ----------
        self._build_dataloaders()

        # ---------- model ----------
        self._build_models()

        # ---------- optimizer ----------
        self._build_optimizer()

        print("[Finetune] TranslationFinetuner ready.")

    # --------------------------------------------------------
    # Build dataloaders
    # --------------------------------------------------------
    def _build_dataloaders(self):
        args = SimpleNamespace(
            batch_size=getattr(self.cfg.Training, "batch_size", 1),
            num_workers=getattr(self.cfg.Training, "num_workers", 4),
            seed=getattr(self.cfg, "seed", 3407),
            rgb_support=True,
            pose_support=False,
        )

        self.train_loader = create_dataloader(args, self.cfg, phase="train")
        self.val_loader   = create_dataloader(args, self.cfg, phase="dev")

        print(f"[Data] train batches = {len(self.train_loader)}")
        print(f"[Data] val   batches = {len(self.val_loader)}")

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------
    def _build_models(self):
        # RGB feature dim
        Dv = cfg_get(self.cfg, "Encoders.rgb.output_dim", 512)

        # RGB Encoder
        self.rgb = RGBEncoder(
            pretrained=False,
            output_dim=Dv,
        ).to(self.device)

        # MT5 path
        mt5_path = cfg_get(
            self.cfg,
            "Finetune.translation_model_path",
            "google/mt5-small"
        )

        # Translation Head
        self.task_head = TranslationHeadMT5(
            mt5_path,
            in_dim=Dv,
            label_smoothing=cfg_get(self.cfg, "Finetune.label_smoothing", 0.1),
            lang_prompt=cfg_get(self.cfg, "Finetune.lang_prompt", "Chinese"),
            max_target_len=cfg_get(self.cfg, "Finetune.max_target_len", 50),
        ).to(self.device)

        print(f"[Model] RGB encoder = {type(self.rgb).__name__}")
        print(f"[Model] TranslationHead = {type(self.task_head).__name__}")

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------
    def _build_optimizer(self):
        train_cfg = getattr(self.cfg, "Training", SimpleNamespace())
        lr_head = getattr(train_cfg, "learning_rate_head", 1e-4)
        lr_back = getattr(train_cfg, "learning_rate_backbone", 1e-5)

        groups = []
        g_head = params_with_lr([self.task_head], lr_head)
        g_rgb  = params_with_lr([self.rgb], lr_back)

        if g_head: groups.append(g_head)
        if g_rgb:  groups.append(g_rgb)

        self.optimizer = AdamW(groups)

        print(f"[Optimizer] head lr={lr_head}, backbone lr={lr_back}")

    # --------------------------------------------------------
    # helpers
    # --------------------------------------------------------
    def _extract_batch(self, batch):
        """
        batch = (src_input, tgt_input)
        """
        if not isinstance(batch, (list, tuple)):
            raise ValueError(f"[Translation] batch should be (src, tgt), got {type(batch)}")

        src, tgt = batch

        rgb = src["rgb_img"]       # [B,T,C,H,W]
        rgb_len = src["rgb_len"]   # [B]
        tgt_text = tgt["gt_sentence"]

        return rgb, rgb_len, tgt_text

    def _encode_rgb(self, rgb, rgb_len):
        rgb = rgb.to(self.device)
        rgb_len = rgb_len.to(self.device)
        feat = self.rgb(rgb)      # [B,T,D]
        B,T,_ = feat.shape
        mask = torch.zeros((B,T), dtype=torch.bool, device=self.device)
        for i in range(B):
            mask[i, :rgb_len[i]] = True
        return feat, mask

    # --------------------------------------------------------
    # Train One Epoch
    # --------------------------------------------------------
    def train_epoch(self):
        self.rgb.train()
        self.task_head.train()

        tot_loss = 0.0
        n = 0

        pbar = tqdm(self.train_loader, desc="Train (translation)")

        for batch in pbar:
            rgb, rgb_len, tgt_text = self._extract_batch(batch)

            vis_seq, vis_mask = self._encode_rgb(rgb, rgb_len)

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

            tot_loss += loss.item()
            n += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg = tot_loss / max(1, n)
        return {"loss": avg, "main_metric": -avg}

    # --------------------------------------------------------
    # Evaluation (simple BLEU)
    # --------------------------------------------------------

    @torch.no_grad()
    def evaluate(self):
        self.rgb.eval()
        self.task_head.eval()

        tot_loss = 0.0
        preds, refs = [], []

        for batch in tqdm(self.val_loader, desc="Eval (translation)"):
            rgb, rgb_len, tgt_text = self._extract_batch(batch)

            vis_seq, vis_mask = self._encode_rgb(rgb, rgb_len)

            out = self.task_head(
                vis_seq=vis_seq,
                vis_mask=vis_mask,
                tgt_texts=tgt_text,
            )

            tot_loss += out["loss"].item()

            # Decode
            prepared = self.task_head.prepare_inputs(vis_seq, vis_mask)
            gen_ids = self.task_head.generate(prepared, max_new_tokens=64)
            pred_text = self.task_head.tok.batch_decode(gen_ids, skip_special_tokens=True)

            preds.extend(pred_text)
            refs.extend(tgt_text)

        avg_loss = tot_loss / max(1, len(self.val_loader))

        bleu = compute_bleu(preds, refs)
        rouge = compute_rouge(preds, refs)
        cer = compute_cer(preds, refs)
        wer = compute_wer(preds, refs)

        print(f"[Eval] Loss={avg_loss:.4f}, BLEU={bleu:.4f}, CER={cer:.4f}, WER={wer:.4f}")

        metrics = {
            "loss": avg_loss,
            "BLEU": bleu,
            "CER": cer,
            "WER": wer,
            "ROUGE1": rouge["rouge1"],
            "ROUGEL": rouge["rougeL"],
            "main_metric": bleu,  # 可以改成 WER/CER/ROUGE 等你喜欢的
        }
        return metrics
    # @torch.no_grad()
    # def evaluate(self):
    #     self.rgb.eval()
    #     self.task_head.eval()
    #
    #     tot_loss = 0.0
    #     n = 0
    #     preds = []
    #     refs = []
    #
    #     for batch in tqdm(self.val_loader, desc="Eval (translation)"):
    #         rgb, rgb_len, tgt_text = self._extract_batch(batch)
    #         vis_seq, vis_mask = self._encode_rgb(rgb, rgb_len)
    #
    #         out = self.task_head(
    #             vis_seq=vis_seq,
    #             vis_mask=vis_mask,
    #             tgt_texts=tgt_text,
    #         )
    #         loss = out["loss"]
    #
    #         # ==== decode ====
    #         prepared = self.task_head.prepare_inputs(vis_seq, vis_mask)
    #         gen_ids = self.task_head.generate(prepared, max_new_tokens=64)
    #         pred_text = self.task_head.tok.batch_decode(gen_ids, skip_special_tokens=True)
    #
    #         preds.extend(pred_text)
    #         refs.extend(tgt_text)
    #
    #         tot_loss += loss.item()
    #         n += 1
    #
    #     avg = tot_loss / max(1, n)
    #     bleu = self._compute_bleu(preds, refs)
    #
    #     print(f"[Eval] loss={avg:.4f}, BLEU={bleu:.4f}")
    #
    #     return {"loss": avg, "BLEU": bleu, "main_metric": bleu}

    # --------------------------------------------------------
    # Tiny BLEU
    # --------------------------------------------------------
    def _compute_bleu(self, preds, refs):
        try:
            import sacrebleu
            bleu = sacrebleu.corpus_bleu(preds, [refs])
            return float(bleu.score)
        except:
            return 0.0



# =========================================================
# Recognition Finetuner (CTC)
# =========================================================

class RecognitionFinetuner:
    """
    使用 BaseDataset.collate_fn 的 batch 结构：
        batch = (src_input, tgt_input)

    其中：
      src_input = {
          "name": [...],
          "keypoints": Tensor,          # [B, T_pose, ...] （当前不用）
          "kp_len": LongTensor[B],
          "rgb_img": Tensor,            # [B, T_rgb, 3, H, W]
          "rgb_len": LongTensor[B],
          "segments": [...],
      }

      tgt_input = {
          "gt_sentence": List[str],     # 翻译/检索用
          "gt_gloss": List[List[...]],  # 识别用，元素可以是 str / int / 1D Tensor
      }

    本类负责：
      - 用 RGBEncoder 编码 [B,T,3,H,W] -> [B,T,D]
      - 用 RecognitionHeadCTC 做 CTC 识别
      - 自动将 gt_gloss 转成 CTC 需要的 packed target
    """

    def __init__(self, cfg: SimpleNamespace, device: torch.device):
        self.cfg = cfg
        self.device = device

        # ------ 基本训练参数 ------
        train_cfg = getattr(cfg, "Training", SimpleNamespace())
        self.epochs = getattr(train_cfg, "epochs", 10)
        self.grad_clip = getattr(train_cfg, "grad_clip", 1.0)

        finetune_cfg = getattr(cfg, "Finetune", SimpleNamespace())
        self.amp_enabled = getattr(finetune_cfg, "amp", True)

        # ------ 识别任务配置 ------
        # num_classes 必须 >= 实际 gloss vocab size + 1（最后一个留给 CTC blank）
        self.num_classes = cfg_get(self.cfg, "Evaluation.recognition.num_classes", 1500)

        # gloss 词表：字符串 -> id
        self.gloss2id = {}
        # 下一个分配的 id（0..num_classes-2，用于 real tokens；num_classes-1 是 blank）
        self._next_gloss_id = 0

        # 构建各组件
        self._build_dataloaders()
        self._build_models()
        self._build_optimizer()

    # -----------------------------
    # dataloaders
    # -----------------------------
    def _build_dataloaders(self):
        """
        和 test_dataloader.py 一样，通过 create_dataloader(args, cfg, phase)
        这里的 args 只负责一些通用参数，不影响 Training.batch_size 等配置。
        """
        args = SimpleNamespace()  # 可以是空，它只在 BaseDataset 里读取少量可选字段

        logger.info("[Finetune] Building dataloaders (recognition)...")
        self.train_loader = create_dataloader(args, self.cfg, phase="train")
        self.val_loader   = create_dataloader(args, self.cfg, phase="dev")

        logger.info(f"[Data] train batches = {len(self.train_loader)}")
        logger.info(f"[Data] val   batches = {len(self.val_loader)}")

    # -----------------------------
    # models
    # -----------------------------
    def _build_models(self):
        Dv = cfg_get(self.cfg, "Encoders.rgb.output_dim", 512)

        # RGB encoder
        self.rgb = RGBEncoder(
            pretrained=False,
            output_dim=Dv,
        ).to(self.device)

        # CTC head
        self.head = RecognitionHeadCTC(
            in_dim=Dv,
            num_classes=self.num_classes,
            hidden_dim=cfg_get(self.cfg, "Finetune.recognition.hidden_dim", 512),
            num_layers=cfg_get(self.cfg, "Finetune.recognition.num_layers", 4),
            nhead=cfg_get(self.cfg, "Finetune.recognition.nhead", 8),
            dropout=0.1,
        ).to(self.device)

        logger.info(f"[Model] RGB = {type(self.rgb).__name__}")
        logger.info(f"[Model] RecognitionHead = {type(self.head).__name__}")

    # -----------------------------
    # optimizer
    # -----------------------------
    def _build_optimizer(self):
        train_cfg = getattr(self.cfg, "Training", SimpleNamespace())
        lr_head = getattr(train_cfg, "lr_head", 3e-4)
        lr_back = getattr(train_cfg, "lr_backbone", 5e-5)

        groups = []
        g_head = params_with_lr([self.head], lr_head)
        g_back = params_with_lr([self.rgb], lr_back)
        if g_head:
            groups.append(g_head)
        if g_back:
            groups.append(g_back)

        if not groups:
            raise RuntimeError("[Recognition] No parameters to optimize.")

        self.optimizer = AdamW(groups)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

        logger.info(f"[Optimizer] head lr={lr_head}, backbone lr={lr_back}")

    # ======================================================
    # batch 解包
    # ======================================================
    def _extract_batch(self, batch):
        """
        batch: (src_input, tgt_input)
        """
        if not (isinstance(batch, (tuple, list)) and len(batch) == 2):
            raise ValueError(f"[Recognition] Expect (src_input, tgt_input), got {type(batch)}")

        src, tgt = batch

        rgb       = src.get("rgb_img", None)
        rgb_len   = src.get("rgb_len", None)
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
        """
        单条样本的 gloss 序列：
          - List[str]
          - List[int]
          - 1D Tensor
        返回：1D LongTensor
        """
        # 已经是 tensor
        if isinstance(gloss_seq, torch.Tensor):
            return gloss_seq.long()

        # 空样本
        if len(gloss_seq) == 0:
            return torch.zeros(0, dtype=torch.long)

        first = gloss_seq[0]

        # 已经是数字序列
        if isinstance(first, int):
            return torch.tensor(gloss_seq, dtype=torch.long)

        # 否则认为是字符串序列
        ids = []
        for tok in gloss_seq:
            # 如果 dataset 混合了 int/str，保险处理
            if isinstance(tok, int):
                ids.append(tok)
                continue

            # new gloss token
            if tok not in self.gloss2id:
                # 预留最后一个 id 给 CTC blank
                if self._next_gloss_id >= self.num_classes - 1:
                    raise RuntimeError(
                        f"[Recognition] gloss vocab size exceeded num_classes-1 "
                        f"({self.num_classes-1}). token='{tok}'"
                    )
                self.gloss2id[tok] = self._next_gloss_id
                self._next_gloss_id += 1

            ids.append(self.gloss2id[tok])

        return torch.tensor(ids, dtype=torch.long)

    def _decode_ctc(self, logits):
        """
        logits: [T, B, V]
        返回：List[str]，长度 B，每条是空格分隔的 token id 字符串
        """
        # 转为概率最大路径
        # [T,B,V] → [T,B]
        pred = logits.argmax(dim=-1)

        T, B = pred.shape
        results = []

        for b in range(B):
            seq = []
            prev = -1
            for t in range(T):
                tok = int(pred[t, b].item())
                # CTC: 跳过 blank（num_classes-1） 和 重复
                if tok != prev and tok != (self.num_classes - 1):
                    seq.append(str(tok))
                prev = tok
            results.append(" ".join(seq))

        return results

    def _pack_gloss(self, gloss_list):
        """
        gloss_list: List[gloss_seq]
        每个 gloss_seq: List[str] / List[int] / Tensor

        返回:
          packed_targets: 1D LongTensor  (拼接后的 targets)
          target_lengths: LongTensor[B]  (每条样本长度)
        """
        seq_tensors = []
        lengths = []

        for g in gloss_list:
            t = self._gloss_seq_to_ids(g)   # [L]
            seq_tensors.append(t)
            lengths.append(len(t))

        if seq_tensors:
            packed = torch.cat(seq_tensors, dim=0)
        else:
            packed = torch.zeros(0, dtype=torch.long)

        target_lengths = torch.tensor(lengths, dtype=torch.long)
        return packed, target_lengths

    # ======================================================
    # RGB 编码
    # ======================================================
    def _encode_rgb(self, rgb_img, rgb_len):
        """
        rgb_img: [B, T, 3, H, W]
        rgb_len: [B]
        返回：
          feat: [B, T, D]
          mask: [B, T] (bool)，True=有效帧
        """
        rgb_img = rgb_img.to(self.device)
        rgb_len = rgb_len.to(self.device)

        feat = self.rgb(rgb_img)  # [B,T,D]
        if feat.ndim != 3:
            raise ValueError(f"[Recognition] RGBEncoder output must be [B,T,D], got {feat.shape}")

        B, T, _ = feat.shape
        mask = torch.zeros((B, T), dtype=torch.bool, device=self.device)
        for i in range(B):
            valid = int(rgb_len[i].item())
            if valid > T:
                valid = T
            mask[i, :valid] = True

        return feat, mask

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

            # 编码 RGB + 构建 mask
            feats, mask = self._encode_rgb(rgb, rgb_len)
            input_lengths = mask.sum(dim=1).long()

            # 打包 CTC target
            packed_targets, target_lengths = self._pack_gloss(gloss)

            with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                logits = self.head(feats, src_key_padding_mask=~mask)  # [T,B,V]
                loss = self.head.compute_loss(
                    logits,
                    packed_targets,
                    input_lengths,
                    target_lengths,
                )

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # gradient clipping
            all_params = []
            for g in self.optimizer.param_groups:
                all_params.extend(g["params"])
            clip_grad_norm_(all_params, max_norm=self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        return {"loss": avg_loss, "main_metric": -avg_loss}

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

            # ---- encode ----
            feats, mask = self._encode_rgb(rgb, rgb_len)
            input_lengths = mask.sum(dim=1).long()
            packed_targets, target_lengths = self._pack_gloss(gloss)

            # ---- forward ----
            logits = self.head(feats, src_key_padding_mask=~mask)
            loss = self.head.compute_loss(
                logits,
                packed_targets,
                input_lengths,
                target_lengths,
            )

            total_loss += loss.item()
            n_batches += 1

            # ---- decode CTC ----
            pred_seq = self._decode_ctc(logits)  # List[str]
            preds_all.extend(pred_seq)

            # refs：把每个 gloss 序列展开为字符串
            refs = [" ".join([str(x) for x in g]) for g in gloss]
            refs_all.extend(refs)

        # ---- compute metrics ----
        avg_loss = total_loss / max(1, n_batches)

        wer = compute_wer(preds_all, refs_all)
        cer = compute_cer(preds_all, refs_all)

        print(f"[Eval] loss={avg_loss:.4f}, WER={wer:.4f}, CER={cer:.4f}")

        metrics = {
            "loss": avg_loss,
            "WER": wer,
            "CER": cer,
            "main_metric": -wer,  # 越小越好，因此 main_metric = -WER
        }
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
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_backbone", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="0")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed()

    # === Load config EXACTLY like test_dataloader.py ===
    raw_cfg = load_yaml(args.config)
    cfg = dict_to_ns(raw_cfg)

    # -------------------------------
    # Init wandb
    # -------------------------------
    if WANDB_AVAILABLE:
        wandb.init(
            project=getattr(cfg, "wandb_project", "Uni-SLM"),
            name=getattr(cfg, "wandb_run_name", None),
            config=raw_cfg,  # 原始 YAML 直接记录到 wandb config
        )
        print("[WandB] initialized.")

    # Inject CLI overrides
    if hasattr(cfg, "Training"):
        if args.epochs: cfg.Training.epochs = args.epochs
        if args.batch_size: cfg.Training.batch_size = args.batch_size
        if args.lr_head: cfg.Training.learning_rate_head = args.lr_head
        if args.lr_backbone: cfg.Training.learning_rate_backbone = args.lr_backbone

    cfg.device = args.device

    # build device
    if cfg.device == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{cfg.device}")
    logger.info(f"[Main] device={device}")

    finetuner = FinetunerFactory.create(cfg, device)

    for epoch in range(cfg.Training.epochs):
        logger.info(f"===== Epoch {epoch + 1}/{cfg.Training.epochs} =====")

        train_res = finetuner.train_epoch()
        print(train_res)

        eval_res = finetuner.evaluate()
        print(eval_res)

        main_metric = eval_res["main_metric"]

        if WANDB_AVAILABLE:
            log_dict = {}
            for k, v in train_res.items():
                log_dict[f"train/{k}"] = v
            for k, v in eval_res.items():
                log_dict[f"eval/{k}"] = v
            wandb.log(log_dict)

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
            print(f"[Main] New best model saved (main_metric={main_metric:.4f})")


if __name__ == "__main__":
    main()
