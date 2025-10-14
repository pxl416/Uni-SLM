# train.py
import os
import time
import logging
from argparse import Namespace
from contextlib import nullcontext

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_

from utils.config import load_config, load_train_config
from utils.dataset2 import create_dataloader
from utils.loss import build_loss, cosine_similarity_matrix
from models.Encoder.rgb_encoder import RGBEncoder
from models.Encoder.text_encoder import TextEncoder
from models.Head.retrieval import RetrievalHead
from models.Head.recognition import RecognitionHeadCTC
from models.Head.translation import TranslationHeadMT5

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(s: int = 42):
    """设置随机种子"""
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cfg_get(ns, path, default=None):
    """
    安全获取嵌套配置；ns 是 SimpleNamespace 或 dict；path 用 'A.B.C'。
    例子：cfg_get(self.config, "Evaluation.retrieval.enabled", False)
    """
    cur = ns
    for key in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, default if key == path.split(".")[-1] else None)
        else:
            if not hasattr(cur, key):
                return default
            cur = getattr(cur, key)
    return cur



class Trainer:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.scaler = GradScaler(enabled=torch.cuda.is_available())
        self.best_metric = 0.0

        self.setup_directories()
        self.setup_models()
        self.setup_optimizers()
        self.setup_loss()
        self.load_data()

    def setup_directories(self):
        """初始化保存目录"""
        cfg_dir = getattr(self.config, "save_dir", None)
        if cfg_dir and isinstance(cfg_dir, str) and cfg_dir != "PATH":
            base_dir = cfg_dir
        else:
            run_id = getattr(getattr(wandb, "run", None), "id", None)
            if not run_id:
                run_id = time.strftime("%Y%m%d-%H%M%S")
            base_dir = os.path.join("saved_models", run_id)

        self.save_dir = base_dir
        self.plot_dir = os.path.join(self.save_dir, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)
        logger.info(f"Save directory: {self.save_dir}")

    def setup_models(self):
        self.models = {}
        self.eval_heads = {}

        # ====== 1) 可训练主编码器 ======
        Dv = 512
        self.models["rgb"] = RGBEncoder(pretrained=True, output_dim=Dv).to(self.device)
        logger.info("Initialized RGBEncoder")

        # 仅当对比学习需要文本分支
        if cfg_get(self.config, "Pretraining.task", "contrastive") == "contrastive":
            self.models["text"] = TextEncoder().to(self.device)
            logger.info("Initialized TextEncoder for contrastive learning")

        # ====== 2) 预训练用 projector（可训练） ======
        text_dim = getattr(self.models.get("text", None), "embedding_dim", 384)
        proj_dim = cfg_get(self.config, "Pretraining.projection_dim", 256)

        self.proj = nn.ModuleDict({
            "rgb": nn.Linear(Dv, proj_dim),
            "text": nn.Linear(text_dim, proj_dim),
        }).to(self.device)

        # 初始化（可选）
        for m in self.proj.values():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        logger.info(f"Initialized pretraining projector: rgb {Dv}->{proj_dim}, text {text_dim}->{proj_dim}")

        # ====== 3) 冻结评估头（只在验证用） ======
        # Retrieval
        if cfg_get(self.config, "Evaluation.retrieval.enabled", False):
            from models.Head.retrieval import RetrievalHead
            self.eval_heads["retrieval"] = RetrievalHead(
                rgb_in=Dv,
                text_in=text_dim,
                proj_dim=cfg_get(self.config, "Evaluation.retrieval.proj_dim", 256),
                temperature=cfg_get(self.config, "Evaluation.retrieval.temperature", 0.07),
            ).to(self.device)
            logger.info("Initialized RetrievalHead (frozen)")  # ✅ 只有这一行

        # Recognition
        if cfg_get(self.config, "Evaluation.recognition.enabled", False):
            from models.Head.recognition import RecognitionHeadCTC
            self.eval_heads["recognition"] = RecognitionHeadCTC(
                in_dim=Dv,
                num_classes=cfg_get(self.config, "Evaluation.recognition.num_classes", 2000),
                hidden=cfg_get(self.config, "Evaluation.recognition.hidden_dim", 512),
                nlayer=cfg_get(self.config, "Evaluation.recognition.num_layers", 2),
                dropout=cfg_get(self.config, "Evaluation.recognition.dropout", 0.1),
                blank_id=cfg_get(self.config, "Evaluation.recognition.blank_id", None),
            ).to(self.device)
            logger.info("Initialized RecognitionHeadCTC (frozen)")  # ✅ 只有这一行

        # Translation
        if cfg_get(self.config, "Evaluation.translation.enabled", False):
            from models.Head.translation import TranslationHeadMT5
            self.eval_heads["translation"] = TranslationHeadMT5(
                mt5_path=cfg_get(self.config, "Evaluation.translation.mt5_path", "google/mt5-base"),
                in_dim=Dv,
                d_model=cfg_get(self.config, "Evaluation.translation.hidden_dim", 768),
                label_smoothing=cfg_get(self.config, "Evaluation.translation.label_smoothing", 0.1),
                lang_prompt=cfg_get(self.config, "Evaluation.translation.lang", "Chinese"),
            ).to(self.device)
            logger.info("Initialized TranslationHeadMT5 (frozen)")

        logger.info(
            f"Trainable models: {list(self.models.keys())}; "
            f"Eval heads: {list(self.eval_heads.keys())}; "
            f"Pretraining.task={cfg_get(self.config, 'Pretraining.task', None)}"
        )

    def setup_optimizers(self):
        params = []
        for m in self.models.values():
            params += list(m.parameters())
        # projector 参数
        params += list(self.proj.parameters())

        if not params:
            raise ValueError("No trainable parameters found!")

        lr = cfg_get(self.config, "Training.learning_rate", 1e-4)
        opt_type = cfg_get(self.config, "optimizer.type", "adam")
        if opt_type == "adam":
            self.optimizer = Adam(params, lr=lr)
            logger.info(f"Initialized Adam optimizer with lr={lr}")
        else:
            raise ValueError(f"Unsupported optimizer: {opt_type}")

        if cfg_get(self.config, "optimizer.scheduler") == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=cfg_get(self.config, "Training.epochs", 20),
                eta_min=lr * 0.01,
            )
            logger.info("Initialized CosineAnnealingLR scheduler")
        else:
            self.scheduler = None

    def setup_loss(self):
        self.loss_fn = build_loss(self.config)
        logger.info(f"Initialized loss function: {cfg_get(self.config, 'Pretraining.task', 'contrastive')}")


    def load_data(self):
        """加载数据"""
        args = Namespace(
            dataset_name="CSL_News",
            batch_size=cfg_get(self.config, "Training.batch_size", 4),
            num_workers=2,
            max_length=64,
            rgb_support=True,
        )

        # 尝试加载数据配置，如果失败则使用主配置
        try:
            cfg = load_config("config/data_config.yaml")
        except:
            cfg = self.config
            logger.warning("Failed to load data_config.yaml, using main config")

        self.train_loader = create_dataloader(args, cfg, phase="train")
        self.val_loader = create_dataloader(args, cfg, phase="val")
        logger.info(f"Loaded data: {len(self.train_loader)} train batches, {len(self.val_loader)} val batches")

    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        # 设置模式：主模型训练，评估头评估
        for model in self.models.values():
            model.train()
        for eval_head in self.eval_heads.values():
            eval_head.eval()

        total_loss = 0.0
        # progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{getattr(self.config.Training, 'epochs', 20)}")
        epochs = int(cfg_get(self.config, "Training.epochs", 20))
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        # 收集所有可训练参数
        # trainable_params = []
        # for model in self.models.values():
        #     trainable_params.extend(list(model.parameters()))

        # for batch_idx, (src_input, tgt_input) in enumerate(progress_bar):
            # 自动混合精度上下文
            # ctx = autocast() if torch.cuda.is_available() else nullcontext()

            # with ctx:
            #     loss = self.compute_pretrain_loss(src_input, tgt_input)
            #
            # # 反向传播和优化
            # self.optimizer.zero_grad(set_to_none=True)
            # self.scaler.scale(loss).backward()
            # # clip_grad_norm_(trainable_params, max_norm=getattr(self.config.Training, 'gradient_clip', 1.0))
            # trainable_params.extend(list(self.proj.parameters()))
            # clip_grad_norm_(trainable_params, max_norm=cfg_get(self.config, 'Training.gradient_clip', 1.0))
            #
            # self.scaler.step(self.optimizer)
            # self.scaler.update()

        # 收集所有可训练参数
        trainable_params = []
        for model in self.models.values():
            trainable_params.extend(list(model.parameters()))
        trainable_params.extend(list(self.proj.parameters()))  # ← 加在这里，别放循环里

        for batch_idx, (src_input, tgt_input) in enumerate(progress_bar):
            # 自动混合精度上下文
            ctx = autocast() if torch.cuda.is_available() else nullcontext()
            with ctx:
                loss = self.compute_pretrain_loss(src_input, tgt_input)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            clip_grad_norm_(trainable_params, max_norm=cfg_get(self.config, 'Training.gradient_clip', 1.0))
            self.scaler.step(self.optimizer)
            self.scaler.update()


            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

            # 记录到wandb
            if wandb.run and batch_idx % 10 == 0:
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "epoch": epoch,
                    "batch": batch_idx,
                    "learning_rate": self.optimizer.param_groups[0]["lr"]
                })

        return total_loss / max(1, len(self.train_loader))

    def compute_pretrain_loss(self, src_input, tgt_input):
        """计算预训练主损失"""
        task = cfg_get(self.config, 'Pretraining.task', 'contrastive')

        if task == "contrastive":
            # 视觉特征 [B,T,D] → [B,D]
            rgb_feat = self.models["rgb"](src_input["rgb_img"].to(self.device))
            rgb_pooled = rgb_feat.mean(dim=1)  # [B, 512]

            if "text" not in self.models:
                raise ValueError("TextEncoder required for contrastive learning")

            text_out, attn_mask = self.models["text"](tgt_input["gt_sentence"])

            # 文本特征：句向量 or 序列平均
            if text_out.ndim == 2:
                text_pooled = text_out  # [B, Dt]
            else:
                mask = attn_mask.unsqueeze(-1)  # [B,L,1]
                valid_mask = (mask.sum(dim=1) > 0).squeeze(-1)  # [B]
                if not valid_mask.any():
                    # 防止出现空 batch
                    return torch.tensor(0.0, device=self.device, requires_grad=True)
                rgb_pooled = rgb_pooled[valid_mask]
                text_feat = text_out[valid_mask]
                mask = mask[valid_mask]
                text_pooled = (text_feat * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-5)  # [B', Dt]

            # ---- 关键：两侧投影到同一维度，再归一化，再做相似度 ----
            v = F.normalize(self.proj["rgb"](rgb_pooled), dim=1)  # [B', P]
            t = F.normalize(self.proj["text"](text_pooled), dim=1)  # [B', P]
            sim_matrix = v @ t.T  # [B', B']

            temperature = cfg_get(self.config, "Pretraining.temperature", 0.07)
            return self.loss_fn({'sim_matrix': sim_matrix, 'temperature': temperature})

        else:
            raise ValueError(f"Unsupported pretraining task: {task}")

    @torch.no_grad()
    def validate(self, epoch: int = None):
        """验证：在所有下游任务上评估"""
        results = {}

        # 开始验证的详细日志
        logger.info(f"🚀 Starting validation for epoch {epoch}...")

        # 设置模型模式
        for name, model in self.models.items():
            model.eval()
            logger.debug(f"Set {name} model to eval mode")

        for name, eval_head in self.eval_heads.items():
            eval_head.eval()
            logger.debug(f"Set {name} head to eval mode")

        # 分别评估每个任务并记录详细信息
        task_results = {}

        # Retrieval 评估
        if "retrieval" in self.eval_heads:
            logger.info("📊 Evaluating Retrieval task...")
            start_time = time.time()
            try:
                retrieval_metrics = self.evaluate_retrieval()
                task_results["retrieval"] = retrieval_metrics
                results.update(retrieval_metrics)
                elapsed = time.time() - start_time

                # 详细记录检索指标
                r1 = retrieval_metrics.get("retrieval/R1", 0)
                r5 = retrieval_metrics.get("retrieval/R5", 0)
                r10 = retrieval_metrics.get("retrieval/R10", 0)
                # mean_r1 = retrieval_metrics.get("retrieval/mean_R1", 0)
                mean_r1 = retrieval_metrics.get("retrieval/mean_R1", None)
                if mean_r1 is None:
                    r1_i2t = retrieval_metrics.get("retrieval/i2t_R1", 0.0)
                    r1_t2i = retrieval_metrics.get("retrieval/t2i_R1", 0.0)
                    mean_r1 = 0.5 * (r1_i2t + r1_t2i)

                logger.info(f"   Retrieval Results - R1: {r1:.2%}, R5: {r5:.2%}, R10: {r10:.2%}, MeanR1: {mean_r1:.2%}")
                logger.info(f"   Retrieval evaluation completed in {elapsed:.2f}s")

            except Exception as e:
                logger.error(f"❌ Retrieval evaluation failed: {e}")
                import traceback
                logger.error(traceback.format_exc())

        # Recognition 评估
        if "recognition" in self.eval_heads:
            logger.info("📊 Evaluating Recognition task...")
            start_time = time.time()
            try:
                recognition_metrics = self.evaluate_recognition()
                task_results["recognition"] = recognition_metrics
                results.update(recognition_metrics)
                elapsed = time.time() - start_time

                ctc_loss = recognition_metrics.get("recognition/ctc_loss", float('inf'))
                logger.info(f"   Recognition CTC Loss: {ctc_loss:.4f}")
                logger.info(f"   Recognition evaluation completed in {elapsed:.2f}s")

            except Exception as e:
                logger.error(f"❌ Recognition evaluation failed: {e}")

        # Translation 评估
        if "translation" in self.eval_heads:
            logger.info("📊 Evaluating Translation task...")
            start_time = time.time()
            try:
                translation_metrics = self.evaluate_translation()
                task_results["translation"] = translation_metrics
                results.update(translation_metrics)
                elapsed = time.time() - start_time

                # 记录翻译指标
                for metric_name, value in translation_metrics.items():
                    if "loss" in metric_name:
                        logger.info(f"   {metric_name}: {value:.4f}")
                    else:
                        logger.info(f"   {metric_name}: {value:.2f}")
                logger.info(f"   Translation evaluation completed in {elapsed:.2f}s")

            except Exception as e:
                logger.error(f"❌ Translation evaluation failed: {e}")

        # 汇总日志
        logger.info(f"🎯 === Epoch {epoch} Validation Summary ===")
        for task_name, metrics in task_results.items():
            logger.info(f"   {task_name.upper()}:")
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    if 0 <= value <= 1:  # 百分比格式
                        logger.info(f"     {metric_name}: {value:.2%}")
                    else:  # 普通数值
                        logger.info(f"     {metric_name}: {value:.4f}")
                else:
                    logger.info(f"     {metric_name}: {value}")
        r1 = retrieval_metrics.get("retrieval/R1", None)
        r5 = retrieval_metrics.get("retrieval/R5", None)
        r10 = retrieval_metrics.get("retrieval/R10", None)

        # 若没有单向汇总键，则用双向平均
        def avg(key):
            a = retrieval_metrics.get(f"retrieval/i2t_{key}", None)
            b = retrieval_metrics.get(f"retrieval/t2i_{key}", None)
            if a is None and b is None: return None
            if a is None: return b
            if b is None: return a
            return 0.5 * (a + b)

        if r1 is None:  r1 = avg("R1")
        if r5 is None:  r5 = avg("R5")
        if r10 is None:  r10 = avg("R10")

        mean_r1 = retrieval_metrics.get("retrieval/mean_R1", None)
        if mean_r1 is None:
            i2t = retrieval_metrics.get("retrieval/i2t_R1", 0.0)
            t2i = retrieval_metrics.get("retrieval/t2i_R1", 0.0)
            mean_r1 = 0.5 * (i2t + t2i)

        logger.info(
            f"   Retrieval Results - R1: {(r1 or 0):.2%}, R5: {(r5 or 0):.2%}, R10: {(r10 or 0):.2%}, MeanR1: {mean_r1:.2%}")

        # 记录到wandb
        if wandb.run:
            wandb.log(results, step=epoch)

        return results

    @torch.no_grad()
    def evaluate_retrieval(self):
        """评估检索任务"""
        all_rgb, all_text = [], []

        for src_input, tgt_input in self.val_loader:
            # 提取特征
            rgb_feat = self.models["rgb"](src_input["rgb_img"].to(self.device))
            rgb_pooled = rgb_feat.mean(dim=1)  # [B, D]

            if "text" in self.models:
                text_out, attn_mask = self.models["text"](tgt_input["gt_sentence"])

                # 处理文本特征
                if text_out.ndim == 2:
                    text_pooled = text_out
                else:
                    mask = attn_mask.unsqueeze(-1)
                    text_pooled = (text_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-5)

                all_rgb.append(rgb_pooled)
                all_text.append(text_pooled)

        if all_rgb and all_text:
            rgb_features = torch.cat(all_rgb, dim=0)
            text_features = torch.cat(all_text, dim=0)

            return self.eval_heads["retrieval"].compute_metrics(rgb_features, text_features)

        return {}

    @torch.no_grad()
    def evaluate_recognition(self):
        """评估识别任务（CTC）"""
        if "recognition" not in self.eval_heads:
            return {}

        total_loss, count = 0.0, 0

        for src_input, tgt_input in self.val_loader:
            # 必须有标签
            if "gt_gloss" not in tgt_input:
                continue

            # 视觉时序特征与 mask
            rgb_seq = self.models["rgb"](src_input["rgb_img"].to(self.device))  # [B,T,Dv]
            src_mask = src_input.get("attention_mask", None)
            if src_mask is None:
                # 没提供就默认全有效
                src_mask = torch.ones(rgb_seq.size(0), rgb_seq.size(1), device=self.device, dtype=torch.long)

            # CTC 需要 True=pad 的 mask
            src_key_padding_mask = (src_mask == 0)
            input_lengths = (~src_key_padding_mask).sum(dim=1).to(torch.long)  # [B]

            # 目标标签：兼容 List[List[int]] 或 (targets, target_lengths)
            gt_gloss = tgt_input["gt_gloss"]
            if isinstance(gt_gloss, torch.Tensor) and gt_gloss.ndim == 1:
                # 已是拼接好的一维 targets，同步获取 lengths
                if "gt_gloss_lengths" not in tgt_input:
                    # 没有 lengths，无法计算 CTC
                    continue
                targets = gt_gloss.to(self.device)
                target_lengths = tgt_input["gt_gloss_lengths"].to(self.device).to(torch.long)
            else:
                # 认为是 List[List[int]]
                if not isinstance(gt_gloss, (list, tuple)) or not isinstance(gt_gloss[0], (list, tuple)):
                    # 格式异常，跳过
                    continue
                target_lengths = torch.tensor([len(x) for x in gt_gloss], dtype=torch.long, device=self.device)
                flat = [tid for seq in gt_gloss for tid in seq]
                targets = torch.tensor(flat, dtype=torch.long, device=self.device)

            # 前向 + loss
            logits = self.eval_heads["recognition"](rgb_seq, src_key_padding_mask=src_key_padding_mask)  # [B,T,V]
            loss = self.eval_heads["recognition"].compute_loss(
                logits, targets, input_lengths, target_lengths
            )
            total_loss += float(loss.item());
            count += 1

        return {"recognition/ctc_loss": (total_loss / max(1, count))} if count > 0 else {}

    @torch.no_grad()
    def evaluate_translation(self):
        """评估翻译任务"""
        if "translation" not in self.eval_heads:
            return {}

        sums = {}  # 指标名 -> 累加和
        counts = {}  # 指标名 -> 累计样本/批次数（这里按批计数）

        for src_input, tgt_input in self.val_loader:
            if "gt_sentence" not in tgt_input:
                continue

            # 提取特征
            rgb_seq = self.models["rgb"](src_input["rgb_img"].to(self.device))

            # 计算指标（TranslationHead.compute_metrics 返回如 {"translation_loss":..., "perplexity":...}）
            batch_metrics = self.eval_heads["translation"].compute_metrics(
                vis_seq=rgb_seq,
                tgt_texts=tgt_input["gt_sentence"]
            )

            # 累积
            for k, v in batch_metrics.items():
                sums[k] = sums.get(k, 0.0) + float(v)
                counts[k] = counts.get(k, 0) + 1

        # 取均值并加前缀
        out = {}
        for k in sums:
            out[f"translation/{k}"] = sums[k] / max(1, counts[k])
        return out

    def save_checkpoint(self, state: dict, filename: str):
        """保存检查点"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(state, filename)

        try:
            if wandb.run:
                wandb.save(filename)
        except Exception as e:
            logger.warning(f"Failed to upload to wandb: {e}")


    def train(self):
        """主训练循环"""
        logger.info("Starting training...")

        epochs = int(cfg_get(self.config, "Training.epochs", 20))
        eval_freq = int(cfg_get(self.config, "Training.eval_freq", 1))

        for epoch in range(epochs):
            # 1) 训练一个 epoch
            train_loss = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch + 1}/{epochs}: train_loss = {train_loss:.4f}")

            if wandb.run:
                wandb.log({"train/epoch_loss": train_loss, "epoch": epoch})

            # 2) 验证（按频率）
            if (epoch % eval_freq) == 0:
                eval_metrics = self.validate(epoch)

                # 用检索的 mean_R1 作为“最佳模型”的度量（若不存在则跳过）
                mean_r1 = eval_metrics.get("retrieval/mean_R1", None)
                if mean_r1 is not None and mean_r1 > self.best_metric:
                    self.best_metric = mean_r1

                    # 组装 checkpoint
                    state = {
                        "epoch": epoch,
                        "best_metric": self.best_metric,
                        # 注意：vars(self.config) 只适用于 Namespace；
                        # 如果是 SimpleNamespace 也可以，但若是嵌套对象复杂，建议自己序列化。
                        "config": vars(self.config) if hasattr(self.config, "__dict__") else {},
                    }
                    for name, model in self.models.items():
                        state[f"{name}_state_dict"] = model.state_dict()
                    state["optimizer_state_dict"] = self.optimizer.state_dict()
                    if self.scheduler is not None:
                        state["scheduler_state_dict"] = self.scheduler.state_dict()

                    self.save_checkpoint(state, os.path.join(self.save_dir, "best_model.pt"))
                    logger.info(f"🔥 New best model: mean_R1 = {self.best_metric:.4f}")

            # 3) 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()

        logger.info("Training completed!")


def main():
    """主函数"""
    set_seed(42)

    # 加载配置
    cfg = load_train_config()
    logger.info("Loaded configuration")

    # 初始化W&B
    if getattr(cfg.wandb, 'use', False):
        try:
            wandb.init(
                project=getattr(cfg.wandb, 'project', 'sign-language-pretraining'),
                name=getattr(cfg.wandb, 'run_name', 'experiment'),
                config=vars(cfg)
            )
            logger.info("Initialized Weights & Biases")
        except Exception as e:
            logger.warning(f"W&B initialization failed: {e}")
            os.environ["WANDB_MODE"] = "disabled"
    else:
        os.environ["WANDB_MODE"] = "disabled"
        logger.info("W&B disabled")

    # 开始训练
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()