# finetune.py
import os
import json
import logging
import argparse
from argparse import Namespace
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import wandb
import yaml

from datasets.datasets import create_dataloader
from models.Encoder.rgb_encoder import RGBEncoder
from models.Encoder.text_encoder import TextEncoder
from models.Head.retrieval import RetrievalHead
from models.Head.recognition import RecognitionHeadCTC  # 预留，将来可实现
from models.Head.translation import TranslationHeadMT5


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------
# Config Helpers
# ------------------------------
def load_yaml_config(path: str) -> dict:
    """加载YAML配置并返回dict"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def cfg_get(cfg: dict, path: str, default=None):
    """从dict配置中安全获取嵌套值，如 cfg_get(cfg, 'Finetune.task', 'translation')"""
    cur = cfg
    for k in path.split("."):
        if not isinstance(cur, dict):
            return default
        if k not in cur:
            return default
        cur = cur[k]
    return cur


def load_finetune_config(args):
    cfg = load_yaml_config(args.config)

    # Override by args
    if args.epochs is not None:
        cfg.setdefault("Training", {})["epochs"] = args.epochs

    if args.batch_size is not None:
        cfg.setdefault("Training", {})["batch_size"] = args.batch_size

    if args.lr_head is not None:
        cfg.setdefault("Training", {})["learning_rate_head"] = args.lr_head

    if args.lr_backbone is not None:
        cfg.setdefault("Training", {})["learning_rate_backbone"] = args.lr_backbone

    # device override
    if args.device is not None:
        cfg["device"] = args.device
    else:
        cfg.setdefault("device", "0")  # default cuda:0

    return cfg



# ------------------------------
# Utils
# ------------------------------
def set_seed(s: int = 3407):
    import random
    import numpy as np
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def freeze_modules(modules):
    """冻结模块参数"""
    for m in modules:
        if m is None:
            continue
        for p in m.parameters():
            p.requires_grad = False


def params_with_lr(modules, lr):
    """获取带学习率的参数组 - 修复版本"""
    ps = []
    for m in modules:
        if m is None:
            continue
        for p in m.parameters():
            if p.requires_grad:
                ps.append(p)
    if not ps:
        return None
    return {"params": ps, "lr": float(lr)}  # 确保lr是float类型


# ------------------------------
# DataLoader Builder
# ------------------------------
def build_loader(cfg_dict: dict, split: str, device: torch.device):
    """
    cfg_dict: YAML 读取出来的 dict（不要转成 SimpleNamespace）
    split: 'train' | 'val' | 'test'
    """
    # 使用 active_datasets 中的第一个数据集
    ds_list = cfg_dict.get("active_datasets", [])
    if not ds_list:
        raise ValueError("No active_datasets specified in config")
    ds_name = ds_list[0]

    ds_cfg = cfg_dict.get("datasets", {}).get(ds_name, {})
    temporal_cfg = ds_cfg.get("temporal", {})

    # 获取时间长度超参
    T = temporal_cfg.get("T")
    max_frames = temporal_cfg.get("max_frames")
    max_len = T if T is not None else max_frames or cfg_dict.get("Training", {}).get("max_length", 64)

    training_cfg = cfg_dict.get("Training", {})

    # 传给 BaseDataset 的 args
    args = Namespace(
        dataset_name=ds_name,
        batch_size=training_cfg.get("batch_size", 1),
        num_workers=training_cfg.get("num_workers", 4),
        max_length=int(max_len),
        rgb_support=True,
        seed=cfg_dict.get("seed", 3407),
        use_aug=False,  # 增强策略在 Dataset 内部自己控制
    )

    # split -> phase 映射（YAML 里 Finetune.train_split / eval_split 可以自定义）
    split_mapping = {
        "train": cfg_get(cfg_dict, "Finetune.train_split", "train"),
        "val":   cfg_get(cfg_dict, "Finetune.eval_split", "dev"),
        "test":  "test",
    }
    phase_for_split = split_mapping.get(split, split)

    loader = create_dataloader(args, cfg_dict, phase=phase_for_split)
    logger.info(
        f"[Data] split='{split}' -> phase='{phase_for_split}' | dataset='{ds_name}', "
        f"{len(loader)} batches, max_len={max_len}"
    )
    return loader


# ------------------------------
# Finetuner Base Class
# ------------------------------
class BaseFinetuner:
    """微调任务基类"""

    def __init__(self, cfg: dict, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.epochs = cfg_get(cfg, "Training.epochs", 30)
        self.grad_clip = cfg_get(cfg, "Training.grad_clip", 1.0)
        self.amp_enabled = bool(cfg_get(cfg, "Finetune.amp", True)) and torch.cuda.is_available()

        # 构建模型 / 优化器 / DataLoader
        self._build_models()
        self._build_optimizer()
        self._build_dataloaders()

        self.current_epoch = 0
        self.best_metric = -float("inf")

    # ---- 子类必须实现 ----
    def _build_models(self):
        raise NotImplementedError

    def _build_optimizer(self):
        raise NotImplementedError

    def train_epoch(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    # ---- 通用组件 ----
    def _build_dataloaders(self):
        self.train_loader = build_loader(self.cfg, "train", self.device)
        self.val_loader = build_loader(self.cfg, "val", self.device)

    def _apply_freeze(self):
        """根据 Finetune.freeze 冻结部分模块"""
        freeze_list = cfg_get(self.cfg, "Finetune.freeze", []) or []
        names = {s.lower() for s in freeze_list}

        if "rgb" in names and hasattr(self, "rgb"):
            freeze_modules([self.rgb])
            logger.info("[Freeze] RGB encoder frozen")
        if "text" in names and hasattr(self, "text"):
            freeze_modules([self.text])
            logger.info("[Freeze] Text encoder frozen")
        if "proj" in names and hasattr(self, "proj"):
            freeze_modules([self.proj])
            logger.info("[Freeze] projection layers frozen")

    def _load_pretrained_weights(self):
        """
        从 cfg['Finetune']['weights'] 或 cfg['active_weights'] 中加载预训练权重（如果存在）
        建议在 YAML 中使用：
        Finetune:
          weights:
            rgb_path:  path/to/rgb_encoder.pt
            text_path: path/to/text_encoder.pt
            proj_rgb_path:  path/to/proj_rgb.pt
            proj_text_path: path/to/proj_text.pt
        """
        weights_cfg = cfg_get(self.cfg, "Finetune.weights", {}) or {}

        # 兼容：如果 top-level 有 active_weights 也尝试读取
        active_weights = self.cfg.get("active_weights", {})

        def _clean_state_dict(sd):
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            if any(k.startswith("module.") for k in sd.keys()):
                sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
            return sd

        def _load(path, module, name):
            if not module or not path:
                return
            if not os.path.isfile(path):
                logger.warning(f"[Load] {name} path not found: {path}")
                return
            try:
                sd = torch.load(path, map_location=self.device)
                sd = _clean_state_dict(sd)
                module.load_state_dict(sd, strict=False)
                logger.info(f"[Load] {name} <- {path}")
            except Exception as e:
                logger.error(f"[Load] Failed to load {name} from {path}: {e}")

        # rgb
        rgb_path = weights_cfg.get("rgb_path") or active_weights.get("rgb_encoder")
        text_path = weights_cfg.get("text_path") or active_weights.get("text_encoder")
        proj_rgb_path = weights_cfg.get("proj_rgb_path") or active_weights.get("proj_rgb")
        proj_text_path = weights_cfg.get("proj_text_path") or active_weights.get("proj_text")

        if hasattr(self, "rgb"):
            _load(rgb_path, self.rgb, "rgb_encoder")
        if hasattr(self, "text"):
            _load(text_path, self.text, "text_encoder")
        if hasattr(self, "proj"):
            if isinstance(self.proj, nn.ModuleDict):
                rgb_proj_module = self.proj["rgb"] if "rgb" in self.proj else None
                text_proj_module = self.proj["text"] if "text" in self.proj else None

                _load(proj_rgb_path, rgb_proj_module, "proj_rgb")
                _load(proj_text_path, text_proj_module, "proj_text")

    def save_checkpoint(self, metric: float, is_best: bool = False):
        """保存检查点"""
        save_dir = self.cfg.get("save_dir", "checkpoints/finetune")
        os.makedirs(save_dir, exist_ok=True)

        ckpt = {
            "epoch": self.current_epoch,
            "best_metric": self.best_metric,
        }
        if hasattr(self, "task_head"):
            ckpt["task_head"] = self.task_head.state_dict()
        if hasattr(self, "rgb"):
            ckpt["rgb"] = self.rgb.state_dict()
        if hasattr(self, "text"):
            ckpt["text"] = self.text.state_dict()
        if hasattr(self, "proj"):
            ckpt["proj"] = self.proj.state_dict()
        if hasattr(self, "optimizer"):
            ckpt["optimizer"] = self.optimizer.state_dict()

        latest_path = os.path.join(save_dir, "latest.pt")
        torch.save(ckpt, latest_path)

        if is_best:
            best_path = os.path.join(save_dir, "best.pt")
            torch.save(ckpt, best_path)
            logger.info(f"[Checkpoint] Best model saved to {best_path}, metric={metric:.4f}")

    def train(self):
        """通用训练循环"""
        logger.info(f"Starting training for {self.epochs} epochs")

        for epoch in range(self.epochs):
            self.current_epoch = epoch
            logger.info(f"Epoch {epoch + 1}/{self.epochs}")

            train_metrics = self.train_epoch()

            # 按频率评估
            if (epoch + 1) % cfg_get(self.cfg, "Finetune.eval_every", 1) == 0:
                eval_metrics = self.evaluate()

                # wandb 记录
                if wandb.run:
                    log_dict = {"epoch": epoch + 1}
                    log_dict.update({f"train/{k}": v for k, v in train_metrics.items()})
                    log_dict.update({f"eval/{k}": v for k, v in eval_metrics.items()})
                    wandb.log(log_dict)

                cur = eval_metrics.get("main_metric", 0.0)
                if cur > self.best_metric:
                    self.best_metric = cur
                    self.save_checkpoint(cur, is_best=True)

        logger.info(f"Training completed. Best main_metric={self.best_metric:.4f}")


# ------------------------------
# Retrieval Finetuner
# ------------------------------
class RetrievalFinetuner(BaseFinetuner):
    """检索任务微调器"""

    def _build_models(self):
        Dv = cfg_get(self.cfg, "Encoders.rgb.output_dim", 512)
        Dt = cfg_get(self.cfg, "Encoders.text.output_dim", 384)
        P = cfg_get(self.cfg, "Pretraining.projection_dim", 256)

        self.rgb = RGBEncoder(pretrained=False, output_dim=Dv).to(self.device)
        self.text = TextEncoder().to(self.device)
        self.proj = nn.ModuleDict({
            "rgb": nn.Linear(Dv, P),
            "text": nn.Linear(Dt, P),
        }).to(self.device)

        # 初始化投影层
        for m in self.proj.values():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.task_head = RetrievalHead(
            rgb_in=Dv,
            text_in=Dt,
            proj_dim=P,
            temperature=cfg_get(self.cfg, "Evaluation.retrieval.temperature", 0.07),
            trainable=True,
        ).to(self.device)

        # 把主 proj 的权重拷给 head（方便 warm start）
        self.task_head.rgb_proj.load_state_dict(self.proj["rgb"].state_dict())
        self.task_head.text_proj.load_state_dict(self.proj["text"].state_dict())

        # self._load_pretrained_weights()
        if not cfg_get(self.cfg, "Finetune.load_pretrained", True):
            logger.info("[Load] Skipping pretrained weights (load_pretrained=False)")
            # return

        self._apply_freeze()

    def _build_optimizer(self):
        lr_head = cfg_get(self.cfg, "Training.learning_rate_head", 3e-4)
        lr_back = cfg_get(self.cfg, "Training.learning_rate_backbone", 5e-5)

        groups = []
        g_head = params_with_lr([self.task_head], lr_head)
        if g_head is not None:
            groups.append(g_head)

        freeze_list = cfg_get(self.cfg, "Finetune.freeze", []) or []
        names = {s.lower() for s in freeze_list}

        if "rgb" not in names and "text" not in names and "proj" not in names:
            g_back = params_with_lr([self.rgb, self.text, self.proj], lr_back)
            if g_back is not None:
                groups.append(g_back)
                logger.info("[Optimizer] Adding RGB+Text+Proj for finetuning")

        if not groups:
            raise RuntimeError("No parameters to optimize for retrieval finetuning.")

        self.optimizer = AdamW(groups)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

    def train_epoch(self):
        self.task_head.train()
        if self.rgb is not None:
            self.rgb.train()
        if self.text is not None:
            self.text.train()

        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc="Training (retrieval)")

        for batch_idx, (src, tgt) in enumerate(pbar):
            with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                rgb = src["rgb_img"].to(self.device)        # [B,T,3,H,W]
                vseq = self.rgb(rgb)                        # [B,T,Dv]
                v = vseq.mean(dim=1)                        # [B,Dv]

                tout, amask = self.text(tgt["gt_sentence"]) # tout: [B,L,Dt] or [B,Dt]
                tout = tout.to(self.device)

                if tout.ndim == 3:
                    mask = amask.to(self.device).unsqueeze(-1).float()
                    tout = (tout * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-5)  # [B,Dt]

                v_proj, t_proj = self.task_head(v, tout)    # [B,P], [B,P]
                sim = t_proj @ v_proj.T                    # [B,B]

                tau = cfg_get(self.cfg, "Evaluation.retrieval.temperature", 0.07)
                B = sim.size(0)
                labels = torch.arange(B, device=sim.device)

                loss = 0.5 * (
                    F.cross_entropy(sim / tau, labels) +
                    F.cross_entropy(sim.t() / tau, labels)
                )

            self.optimizer.zero_grad()
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                # clip 所有参数
                all_params = []
                for g in self.optimizer.param_groups:
                    all_params.extend(g["params"])
                clip_grad_norm_(all_params, max_norm=self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                all_params = []
                for g in self.optimizer.param_groups:
                    all_params.extend(g["params"])
                clip_grad_norm_(all_params, max_norm=self.grad_clip)
                self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / max(1, len(self.train_loader))
        return {"loss": avg_loss}

    @torch.no_grad()
    def evaluate(self):
        self.task_head.eval()
        if self.rgb is not None:
            self.rgb.eval()
        if self.text is not None:
            self.text.eval()

        all_rgb, all_text = [], []

        for src, tgt in tqdm(self.val_loader, desc="Evaluating (retrieval)"):
            with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                rgb = src["rgb_img"].to(self.device)
                vseq = self.rgb(rgb)
                v = vseq.mean(dim=1)

                tout, amask = self.text(tgt["gt_sentence"])
                tout = tout.to(self.device)
                if tout.ndim == 3:
                    mask = amask.to(self.device).unsqueeze(-1).float()
                    tout = (tout * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-5)

                v_proj, t_proj = self.task_head(v, tout)
                all_rgb.append(v_proj)
                all_text.append(t_proj)

        rgb_features = torch.cat(all_rgb, dim=0)
        text_features = torch.cat(all_text, dim=0)

        sim_matrix = text_features @ rgb_features.T
        sim_matrix = sim_matrix / cfg_get(self.cfg, "Evaluation.retrieval.temperature", 0.07)

        # 指标
        try:
            from utils.metrics import t2v_metrics, v2t_metrics
            sim_np = sim_matrix.cpu().numpy()
            t2v_results, _ = t2v_metrics(sim_np, None)
            v2t_results, _ = v2t_metrics(sim_np.T, None)

            metrics = {f"t2v/{k}": v for k, v in t2v_results.items()}
            metrics.update({f"v2t/{k}": v for k, v in v2t_results.items()})

            if "R1" in t2v_results and "R1" in v2t_results:
                metrics["main_metric"] = 0.5 * (t2v_results["R1"] + v2t_results["R1"])
        except Exception:
            labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
            t2v_acc = (sim_matrix.argmax(dim=1) == labels).float().mean()
            v2t_acc = (sim_matrix.argmax(dim=0) == labels).float().mean()
            metrics = {
                "t2v/R1": t2v_acc.item(),
                "v2t/R1": v2t_acc.item(),
                "main_metric": 0.5 * (t2v_acc.item() + v2t_acc.item()),
            }

        logger.info(f"Retrieval metrics:\n{json.dumps(metrics, indent=2, ensure_ascii=False)}")
        return metrics


# ------------------------------
# Translation Finetuner
# ------------------------------
class TranslationFinetuner(BaseFinetuner):
    """翻译任务微调器（RGB -> 文本）"""

    def _build_models(self):
        Dv = cfg_get(self.cfg, "Encoders.rgb.output_dim", 512)

        self.rgb = RGBEncoder(pretrained=False, output_dim=Dv).to(self.device)

        self.task_head = TranslationHeadMT5(
            mt5_path=cfg_get(self.cfg, "Evaluation.translation.model_path", "google/mt5-base"),
            in_dim=cfg_get(self.cfg, "Evaluation.translation.in_dim", Dv),
            d_model=cfg_get(self.cfg, "Evaluation.translation.d_model", 768),
            label_smoothing=cfg_get(self.cfg, "Evaluation.translation.label_smoothing", 0.1),
            lang_prompt=cfg_get(self.cfg, "Evaluation.translation.lang", "zh"),
            max_target_len=cfg_get(self.cfg, "Evaluation.translation.max_target_len", 128),
        ).to(self.device)

        self._load_pretrained_weights()
        self._apply_freeze()

    def _build_optimizer(self):
        lr_head = cfg_get(self.cfg, "Training.learning_rate_head", 3e-4)
        lr_back = cfg_get(self.cfg, "Training.learning_rate_backbone", 5e-5)

        groups = []
        g_head = params_with_lr([self.task_head], lr_head)
        if g_head is not None:
            groups.append(g_head)

        freeze_list = cfg_get(self.cfg, "Finetune.freeze", []) or []
        names = {s.lower() for s in freeze_list}
        if "rgb" not in names:
            g_back = params_with_lr([self.rgb], lr_back)
            if g_back is not None:
                groups.append(g_back)
                logger.info("[Optimizer] Adding RGB backbone for finetuning")

        if not groups:
            raise RuntimeError("No parameters to optimize for translation finetuning.")

        self.optimizer = AdamW(groups)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

    def train_epoch(self):
        self.rgb.train()
        self.task_head.train()

        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc="Training (recognition)")

        for src, tgt in pbar:
            # ----------- 1) 取出输入 -----------
            rgb = src["rgb_img"].to(self.device)  # [B,T,C,H,W]
            rgb_len = src["rgb_len"].to(self.device)  # [B]
            gloss_ids = tgt["gloss_ids"].to(self.device)  # [B,Lg]

            # ----------- 2) 提取视觉特征 -----------
            feat = self.rgb(rgb)  # [B,T,Dv]
            B, T, _ = feat.shape

            # 视频 mask：CTC 必须用真实长度
            vis_mask = torch.zeros((B, T), dtype=torch.bool, device=self.device)
            for i in range(B):
                vis_mask[i, :rgb_len[i]] = True

            # gloss_len：CTC 必须要 label 的真实长度
            gloss_len = torch.tensor(
                [len(g[g != 0]) for g in gloss_ids],
                dtype=torch.long,
                device=self.device
            )

            # ----------- 3) 前向 (CTC) -----------
            out = self.task_head(
                vis_seq=feat,
                vis_mask=vis_mask,
                gloss_ids=gloss_ids,
                gloss_len=gloss_len,
            )

            loss = out["loss"]

            # ----------- 4) 反向传播 -----------
            self.optimizer.zero_grad()

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)

                # gradient clipping
                all_params = []
                for g in self.optimizer.param_groups:
                    all_params.extend(g["params"])
                clip_grad_norm_(all_params, max_norm=self.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                all_params = []
                for g in self.optimizer.param_groups:
                    all_params.extend(g["params"])
                clip_grad_norm_(all_params, max_norm=self.grad_clip)

                self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / max(1, len(self.train_loader))
        return {"loss": avg_loss}

    @torch.no_grad()
    def evaluate(self):
        self.task_head.eval()
        if self.rgb is not None:
            self.rgb.eval()

        total_loss = 0.0

        for src, tgt in tqdm(self.val_loader, desc="Evaluating (translation)"):
            with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                rgb = src["rgb_img"].to(self.device)
                feat = self.rgb(rgb)
                B, T, _ = feat.shape
                vis_mask = torch.ones((B, T), dtype=torch.long, device=feat.device)

                out = self.task_head(
                    vis_seq=feat,
                    vis_mask=vis_mask,
                    tgt_texts=tgt["gt_sentence"],
                )
                total_loss += out["loss"].item()

        avg_loss = total_loss / max(1, len(self.val_loader))
        metrics = {"loss": avg_loss, "main_metric": -avg_loss}  # 损失越小越好

        logger.info(f"Translation eval loss: {avg_loss:.4f}")
        return metrics

# ------------------------------
# Recognition Finetuner
# ------------------------------
# class RecognitionFinetuner(BaseFinetuner):
#
#     def _build_models(self):
#         # RGB encoder
#         Dv = cfg_get(self.cfg, "Encoders.rgb.output_dim", 512)
#         self.rgb = RGBEncoder(pretrained=False, output_dim=Dv).to(self.device)
#
#         # vocab
#         vocab_path = cfg_get(self.cfg, "Evaluation.recognition.vocab_path")
#         with open(vocab_path, "r", encoding="utf-8") as f:
#             vocab = [line.strip() for line in f if line.strip()]
#         num_classes = len(vocab) + 1  # +blank
#
#         self.task_head = RecognitionHeadCTC(
#             in_dim=Dv,
#             num_classes=num_classes,
#             hidden_dim=cfg_get(self.cfg, "Evaluation.recognition.hidden_dim", 512),
#             num_layers=cfg_get(self.cfg, "Evaluation.recognition.num_layers", 2),
#             dropout=cfg_get(self.cfg, "Evaluation.recognition.dropout", 0.1),
#             blank_id=cfg_get(self.cfg, "Evaluation.recognition.blank_id", 0),
#         ).to(self.device)
#
#         self._load_pretrained_weights()
#         self._apply_freeze()
#
#     def _build_optimizer(self):
#         lr_head = cfg_get(self.cfg, "Training.learning_rate_head", 3e-4)
#         lr_back = cfg_get(self.cfg, "Training.learning_rate_backbone", 5e-5)
#
#         groups = []
#
#         g_head = params_with_lr([self.task_head], lr_head)
#         if g_head:
#             groups.append(g_head)
#
#         freeze_list = cfg_get(self.cfg, "Finetune.freeze", [])
#         if "rgb" not in freeze_list:
#             g_back = params_with_lr([self.rgb], lr_back)
#             if g_back:
#                 groups.append(g_back)
#
#         self.optimizer = AdamW(groups)
#         self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
#
#     def train_epoch(self):
#         self.task_head.train()
#         self.rgb.train()
#
#         total_loss = 0.0
#         pbar = tqdm(self.train_loader, desc="Training (recognition)")
#
#         for vids, poses, gloss_ids, support in pbar:
#             rgb = support["rgb_img"].to(self.device)         # [B,T,C,H,W]
#             mask = support["attn_mask"].to(self.device)      # [B,T]
#             gloss_list = [g.to(self.device) for g in gloss_ids]
#
#             packed_targets = torch.cat(gloss_list, dim=0)
#             target_lengths = torch.tensor([len(g) for g in gloss_list],
#                                           dtype=torch.long,
#                                           device=self.device)
#             input_lengths = mask.sum(dim=1).long()
#
#             with torch.amp.autocast("cuda", enabled=self.amp_enabled):
#                 feat = self.rgb(rgb)                         # [B,T,D]
#                 logits = self.task_head(feat, src_key_padding_mask=~mask.bool())
#                 loss = self.task_head.compute_loss(
#                     logits, packed_targets, input_lengths, target_lengths
#                 )
#
#             self.optimizer.zero_grad()
#             if self.scaler.is_enabled():
#                 self.scaler.scale(loss).backward()
#                 self.scaler.unscale_(self.optimizer)
#                 clip_grad_norm_(self.optimizer.param_groups[0]["params"], max_norm=self.grad_clip)
#                 self.scaler.step(self.optimizer)
#                 self.scaler.update()
#             else:
#                 loss.backward()
#                 clip_grad_norm_(self.optimizer.param_groups[0]["params"], max_norm=self.grad_clip)
#                 self.optimizer.step()
#
#             total_loss += loss.item()
#             pbar.set_postfix(loss=loss.item())
#
#         return {"loss": total_loss / len(self.train_loader)}
#
#     @torch.no_grad()
#     def evaluate(self):
#         self.task_head.eval()
#         self.rgb.eval()
#
#         total_loss = 0.0
#
#         for vids, poses, gloss_ids, support in tqdm(self.val_loader, desc="Evaluating (recognition)"):
#             rgb = support["rgb_img"].to(self.device)
#             mask = support["attn_mask"].to(self.device)
#             gloss_list = [g.to(self.device) for g in gloss_ids]
#
#             packed_targets = torch.cat(gloss_list, dim=0)
#             target_lengths = torch.tensor([len(g) for g in gloss_list],
#                                           dtype=torch.long,
#                                           device=self.device)
#             input_lengths = mask.sum(dim=1).long()
#
#             feat = self.rgb(rgb)
#             logits = self.task_head(feat, src_key_padding_mask=~mask.bool())
#             loss = self.task_head.compute_loss(
#                 logits, packed_targets, input_lengths, target_lengths
#             )
#             total_loss += loss.item()
#
#         avg = total_loss / len(self.val_loader)
#         return {"loss": avg, "main_metric": -avg}


# ------------------------------
# Recognition Finetuner
# ------------------------------
class RecognitionFinetuner(BaseFinetuner):

    def _build_models(self):
        # RGB encoder
        Dv = cfg_get(self.cfg, "Encoders.rgb.output_dim", 512)
        self.rgb = RGBEncoder(pretrained=False, output_dim=Dv).to(self.device)

        # vocab
        vocab_path = cfg_get(self.cfg, "Evaluation.recognition.vocab_path")
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = [line.strip() for line in f if line.strip()]
        num_classes = len(vocab) + 1  # +blank

        self.task_head = RecognitionHeadCTC(
            in_dim=Dv,
            num_classes=num_classes,
            hidden_dim=cfg_get(self.cfg, "Evaluation.recognition.hidden_dim", 512),
            num_layers=cfg_get(self.cfg, "Evaluation.recognition.num_layers", 2),
            dropout=cfg_get(self.cfg, "Evaluation.recognition.dropout", 0.1),
            blank_id=cfg_get(self.cfg, "Evaluation.recognition.blank_id", 0),
        ).to(self.device)

        self._load_pretrained_weights()
        self._apply_freeze()

    def _build_optimizer(self):
        lr_head = cfg_get(self.cfg, "Training.learning_rate_head", 3e-4)
        lr_back = cfg_get(self.cfg, "Training.learning_rate_backbone", 5e-5)

        groups = []

        g_head = params_with_lr([self.task_head], lr_head)
        if g_head:
            groups.append(g_head)

        freeze_list = cfg_get(self.cfg, "Finetune.freeze", [])
        if "rgb" not in freeze_list:
            g_back = params_with_lr([self.rgb], lr_back)
            if g_back:
                groups.append(g_back)

        self.optimizer = AdamW(groups)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

    def train_epoch(self):
        self.task_head.train()
        self.rgb.train()

        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc="Training (recognition)")

        for batch in pbar:
            # 修复：适配数据加载器的实际返回格式
            # 根据之前的代码，数据加载器返回的是 (src, tgt) 元组
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                src, tgt = batch
                # 从src中获取RGB数据
                rgb = src["rgb_img"].to(self.device) if isinstance(src, dict) else src[0].to(self.device)
                # 从src中获取mask
                if isinstance(src, dict):
                    mask = src.get("attn_mask", src.get("rgb_len"))
                    if mask is not None:
                        mask = mask.to(self.device)
                    else:
                        # 如果没有mask，创建一个全1的mask
                        B, T, _, _, _ = rgb.shape
                        mask = torch.ones(B, T, dtype=torch.bool, device=self.device)
                else:
                    # 如果src是tensor，创建默认mask
                    B, T, _, _, _ = rgb.shape
                    mask = torch.ones(B, T, dtype=torch.bool, device=self.device)

                # 从tgt中获取gloss_ids
                if isinstance(tgt, dict):
                    gloss_ids = tgt.get("gloss_ids", [])
                else:
                    gloss_ids = tgt
            else:
                # 如果格式不符合预期，跳过或报错
                logger.warning(f"Unexpected batch format: {type(batch)}")
                continue

            # 确保gloss_ids是tensor列表
            if not isinstance(gloss_ids, list):
                gloss_ids = [gloss_ids]

            gloss_list = [g.to(self.device) if isinstance(g, torch.Tensor) else torch.tensor(g, device=self.device)
                          for g in gloss_ids]

            # 准备CTC需要的输入
            packed_targets = torch.cat(gloss_list, dim=0)
            target_lengths = torch.tensor([len(g) for g in gloss_list],
                                          dtype=torch.long,
                                          device=self.device)
            input_lengths = mask.sum(dim=1).long() if mask is not None else torch.tensor([rgb.shape[1]] * rgb.shape[0],
                                                                                         device=self.device)

            with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                feat = self.rgb(rgb)  # [B,T,D]
                logits = self.task_head(feat, src_key_padding_mask=~(mask.bool() if mask is not None else None))
                loss = self.task_head.compute_loss(
                    logits, packed_targets, input_lengths, target_lengths
                )

            self.optimizer.zero_grad()
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                # 修复：正确获取所有参数进行梯度裁剪
                all_params = []
                for g in self.optimizer.param_groups:
                    all_params.extend(g["params"])
                clip_grad_norm_(all_params, max_norm=self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                all_params = []
                for g in self.optimizer.param_groups:
                    all_params.extend(g["params"])
                clip_grad_norm_(all_params, max_norm=self.grad_clip)
                self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        return {"loss": total_loss / len(self.train_loader)}

    @torch.no_grad()
    def evaluate(self):
        self.task_head.eval()
        self.rgb.eval()

        total_loss = 0.0

        for batch in tqdm(self.val_loader, desc="Evaluating (recognition)"):
            # 同样的修复应用于验证集
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                src, tgt = batch
                rgb = src["rgb_img"].to(self.device) if isinstance(src, dict) else src[0].to(self.device)

                if isinstance(src, dict):
                    mask = src.get("attn_mask", src.get("rgb_len"))
                    if mask is not None:
                        mask = mask.to(self.device)
                    else:
                        B, T, _, _, _ = rgb.shape
                        mask = torch.ones(B, T, dtype=torch.bool, device=self.device)
                else:
                    B, T, _, _, _ = rgb.shape
                    mask = torch.ones(B, T, dtype=torch.bool, device=self.device)

                if isinstance(tgt, dict):
                    gloss_ids = tgt.get("gloss_ids", [])
                else:
                    gloss_ids = tgt
            else:
                continue

            if not isinstance(gloss_ids, list):
                gloss_ids = [gloss_ids]

            gloss_list = [g.to(self.device) if isinstance(g, torch.Tensor) else torch.tensor(g, device=self.device)
                          for g in gloss_ids]

            packed_targets = torch.cat(gloss_list, dim=0)
            target_lengths = torch.tensor([len(g) for g in gloss_list],
                                          dtype=torch.long,
                                          device=self.device)
            input_lengths = mask.sum(dim=1).long() if mask is not None else torch.tensor([rgb.shape[1]] * rgb.shape[0],
                                                                                         device=self.device)

            feat = self.rgb(rgb)
            logits = self.task_head(feat, src_key_padding_mask=~(mask.bool() if mask is not None else None))
            loss = self.task_head.compute_loss(
                logits, packed_targets, input_lengths, target_lengths
            )
            total_loss += loss.item()

        avg = total_loss / len(self.val_loader)
        return {"loss": avg, "main_metric": -avg}


# ------------------------------
# Finetuner Factory
# ------------------------------
class FinetunerFactory:
    @staticmethod
    def create_finetuner(cfg: dict, device: torch.device) -> BaseFinetuner:
        task = cfg_get(cfg, "Finetune.task", "retrieval").lower()
        logger.info(f"Starting finetuning task: {task}")

        if task == "retrieval":
            return RetrievalFinetuner(cfg, device)
        elif task == "translation":
            return TranslationFinetuner(cfg, device)
        elif task == "recognition":
            return RecognitionFinetuner(cfg, device)

        else:
            raise ValueError(f"Unknown finetune task: {task}")

def parse_args():
    parser = argparse.ArgumentParser(description="Sign language finetuning")

    # --- Config file ---
    parser.add_argument("--config", type=str, default="config/finetune_newtask_mini_1.yaml", help="Path to YAML config file")
    # --- Training overrides ---
    parser.add_argument("--epochs", type=int, default=5, help="Override Training.epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Override Training.batch_size")
    parser.add_argument("--lr_head", type=float, default=3e-4, help="Override Training.learning_rate_head")
    parser.add_argument("--lr_backbone", type=float, default=5e-5, help="Override Training.learning_rate_backbone")
    # --- Device settings ---
    parser.add_argument("--device", type=str, default=0, help="CUDA device id(s), e.g. '0' or '0,1' or 'cpu'")
    parser.add_argument("--distributed", action="store_true", help="Enable DDP")
    parser.add_argument("--local_rank", type=int, default=-1, help="DDP local rank")

    args = parser.parse_args()

    print("\n=== Parsed args ===")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("===================\n")

    return args

# ------------------------------
# Main
# ------------------------------
def main():
    args = parse_args()
    set_seed()

    cfg = load_finetune_config(args)
    print('/*-/*-/*-')
    print(cfg["Evaluation"]["recognition"])
    print('/*-/*-/*-')

    # device
    if cfg["device"] == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{cfg['device']}")

    logger.info(f"Using device: {device}")

    # wandb
    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg.get("use", False):
        wandb.init(
            project=wandb_cfg.get("project", "uni-slm-finetune"),
            name=wandb_cfg.get("run_name", "finetune-run"),
            config=cfg,
        )
    else:
        os.environ["WANDB_MODE"] = "disabled"

    finetuner = FinetunerFactory.create_finetuner(cfg, device)
    finetuner.train()



if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    main()


# os.environ["WANDB_MODE"]= "0c563568ac7ebb0941d93b54803b7101d16280b6"