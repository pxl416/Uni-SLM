# pretrain.py
# ========= 通用工具 =========
import os
import time
import logging
from argparse import Namespace
import argparse

# ========= 第三方库 =========
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb

# ========= 项目内部模块 =========
from utils.amp_compat import make_autocast, make_scaler
from utils.config import load_train_config, cfg_get
from datasets.datasets import create_dataloader
from utils.loss import build_loss
from utils.optimizer import build_optimizer
from models.Encoder.rgb_encoder import RGBEncoder
from models.Encoder.text_encoder import TextEncoder
# from utils.distributed import setup_distributed
# from models.build_model import build_model





# 推荐避免 tokenizer 多线程抢占
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------
# 小工具
# ------------------------------
def set_seed(s: int = 42):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cfg_get(ns, path, default=None):
    """
    安全获取嵌套配置；支持 SimpleNamespace 和 dict
    """
    cur = ns
    keys = path.split(".")
    for i, key in enumerate(keys):
        is_last = (i == len(keys) - 1)
        if cur is None:
            return default

        # 处理 SimpleNamespace
        if hasattr(cur, key):
            cur = getattr(cur, key)
        # 处理 dict
        elif isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default if is_last else None

    return cur


def parse_args():
    parser = argparse.ArgumentParser(description="Sign language pretraining")
    # ====== 训练相关 ======
    parser.add_argument("--config", type=str, default='config/pretrain_newtask_mini_1.yaml', help="Path to YAML config file. If None, use default in load_train_config().")
    parser.add_argument("--epochs", type=int, default=20, help="Override number of training epochs.")
    parser.add_argument("--batch_size", ype=int, default=4, help="Override training batch size.")
    parser.add_argument("--optimizer", type=str, default="adamW", help="Override optimizer name in config, e.g. adam / adamw / sgd.")
    # ====== 设备 & 分布式相关 ======
    parser.add_argument("--device", type=str, default="0", help="CUDA device(s), e.g. '0' or '0,1,2' or 'cpu'. " "If empty, use all visible GPUs." )
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training (DDP) over the selected device(s).")
    parser.add_argument("--local_rank", type=int, default=-1, help="DDP local rank, usually set automatically by torch.distributed.launch/torchrun.")

    args = parser.parse_args()
    print("=== Parsed args ===")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("===================")
    return args


# ------------------------------
# 训练器（无评估、无下游监测）
# ------------------------------
class Trainer:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.amp_dtype = cfg_get(self.config, "AMP.dtype", None)  # 'bf16' / 'fp16' / None
        self.amp_enabled = bool(cfg_get(self.config, "AMP.enabled", True))

        self.scaler = make_scaler(enabled=self.amp_enabled, dtype=self.amp_dtype)

        self.setup_directories()
        self.setup_models()
        self.setup_optimizers()
        self.setup_loss()
        self.load_data()  # 多数据集（可选），内部会做轮询混合

    # ---------- 路径 ----------
    def setup_directories(self):
        # 优先用 config.save_dir，否则 fallback 到 saved_models/<run_id 或时间戳>
        cfg_dir = getattr(self.config, "save_dir", None)
        if cfg_dir and isinstance(cfg_dir, str) and cfg_dir != "PATH":
            base_dir = cfg_dir
        else:
            run_id = getattr(getattr(wandb, "run", None), "id", None)
            if not run_id:
                run_id = time.strftime("%Y%m%d-%H%M%S")
            base_dir = os.path.join("saved_models", run_id)

        self.save_dir = base_dir
        self.ckpt_dir = os.path.join(self.save_dir, "ckpt")
        self.wts_dir = os.path.join(self.save_dir, "weights")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.wts_dir, exist_ok=True)
        logger.info(f"Save directory: {self.save_dir}")

    def save_split_weights(self, epoch: int, tag: str):
        """
        按模态/分支分别保存纯权重（state_dict），便于下游精确加载。
        目录：weights/<tag>/{rgb_encoder.pt, text_encoder.pt, proj_rgb.pt, proj_text.pt}
        """
        out_dir = os.path.join(self.wts_dir, tag)
        os.makedirs(out_dir, exist_ok=True)

        # 1) Encoders
        if "rgb" in self.models:
            torch.save(self.models["rgb"].state_dict(), os.path.join(out_dir, "rgb_encoder.pt"))
        if "text" in self.models:
            torch.save(self.models["text"].state_dict(), os.path.join(out_dir, "text_encoder.pt"))

        # 2) Projectors（与预训练形状一致）
        if isinstance(self.proj, nn.ModuleDict):
            if "rgb" in self.proj:
                torch.save(self.proj["rgb"].state_dict(), os.path.join(out_dir, "proj_rgb.pt"))
            if "text" in self.proj:
                torch.save(self.proj["text"].state_dict(), os.path.join(out_dir, "proj_text.pt"))

        # 3) 记录一条日志 &（可选）W&B artifact
        logger.info(f"[SplitSaved] {out_dir}")
        if wandb.run:
            try:
                # 直接保存目录下的文件（W&B 会逐个文件记录）
                for fname in os.listdir(out_dir):
                    wandb.save(os.path.join(out_dir, fname))
            except Exception as e:
                logger.warning(f"W&B split save failed: {e}")

    # ---------- 模型 ----------
    def setup_models(self):
        self.models = {}
        Dv = 512
        # 视觉编码器（可训练）
        self.models["rgb"] = RGBEncoder(pretrained=True, output_dim=Dv).to(self.device)

        # 文本编码器（对比学习需要）
        if cfg_get(self.config, "Pretraining.task", "contrastive") == "contrastive":
            self.models["text"] = TextEncoder().to(self.device)

        # projector（可训练）：统一投影到同一维度
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

        logger.info(f"Initialized encoders & projector: rgb {Dv}->{proj_dim}, text {text_dim}->{proj_dim}")

    # ---------- 优化器 / 调度 ----------
    def setup_optimizers(self):
        params = []
        for m in self.models.values():
            params += list(m.parameters())
        params += list(self.proj.parameters())

        if not params:
            raise ValueError("No trainable parameters found!")

        lr = cfg_get(self.config, "Training.learning_rate", 1e-4)
        # self.optimizer = Adam(params, lr=lr)
        self.optimizer = build_optimizer(cfg, params)
        logger.info(f"Initialized Adam optimizer with lr={lr}")

        if cfg_get(self.config, "optimizer.scheduler") == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=cfg_get(self.config, "Training.epochs", 20),
                eta_min=lr * 0.01,
            )
            logger.info("Initialized CosineAnnealingLR scheduler")
        else:
            self.scheduler = None

    # ---------- 损失 ----------
    def setup_loss(self):
        self.loss_fn = build_loss(self.config)  # 损失函数的设定
        logger.info(f"Initialized loss for task: {cfg_get(self.config, 'Pretraining.task', 'contrastive')}")

    # ---------- 数据 ----------
    def load_data(self):
        """加载多个数据集 -> self.train_loaders（dict）。无验证/评估。"""
        self.train_loaders = {}

        ds_list = cfg_get(self.config, "active_datasets", []) or []

        if not ds_list:
            raise ValueError(
                "active_datasets 为空，请在 pretrain.yaml 顶层设置 active_datasets: [CSL_Daily, CSL_News] 等")

        for ds_name in ds_list:
            # 修复：正确处理 SimpleNamespace 配置
            ds_config = cfg_get(self.config, f"datasets.{ds_name}", {})
            temporal_config = getattr(ds_config, "temporal", {}) if hasattr(ds_config, "temporal") else {}

            # 安全获取 max_length
            max_len = None
            if hasattr(temporal_config, 'T'):
                max_len = temporal_config.T
            elif hasattr(temporal_config, 'max_frames'):
                max_len = temporal_config.max_frames
            elif isinstance(temporal_config, dict):
                max_len = temporal_config.get('T') or temporal_config.get('max_frames')

            if max_len is None:
                max_len = cfg_get(self.config, "Training.max_length", 64)

            max_len = int(max_len)

            args = Namespace(
                dataset_name=ds_name,
                batch_size=cfg_get(self.config, "Training.batch_size", 4),
                num_workers=cfg_get(self.config, "Training.num_workers", 2),
                max_length=max_len,
                rgb_support=True,
                seed=cfg_get(self.config, "seed", 3407),
            )

            loader = create_dataloader(args, self.config, phase="train")
            self.train_loaders[ds_name] = loader
            logger.info(f"[Data] {ds_name}: train batches={len(loader)}")

        # 训练时轮询各数据集
        self._train_order = list(self.train_loaders.keys())

    def _roundrobin_min(self):
        """将多个 loader 以"数据集轮询"的方式混合，Step 数对齐到 **最短** 数据集。"""
        names = list(self.train_loaders.keys())
        iters = {n: iter(self.train_loaders[n]) for n in names}
        min_len = min(len(ld) for ld in self.train_loaders.values())
        for _ in range(min_len):
            for n in names:
                yield n, next(iters[n])

    # ---------- 训练 ----------
    def compute_pretrain_loss(self, src_input, tgt_input):
        """仅支持对比学习（contrastive）"""
        task = cfg_get(self.config, 'Pretraining.task', 'contrastive')
        if task != "contrastive":
            raise ValueError(f"Only 'contrastive' is supported in this clean pretrain script, got: {task}")

        # 视觉 [B,T,D] -> [B,D]
        rgb_feat = self.models["rgb"](src_input["rgb_img"].to(self.device))  # [B, T, 512]
        rgb_pooled = rgb_feat.mean(dim=1)  # [B, 512]

        # 文本
        text_in = tgt_input["gt_sentence"].to(self.device)
        text_out, attn_mask = self.models["text"](text_in)

        # 文本池化：句向量 or 按 mask 平均
        if text_out.ndim == 2:
            text_pooled = text_out  # [B, Dt]
        else:
            mask = attn_mask.unsqueeze(-1)  # [B,L,1]
            valid_mask = (mask.sum(dim=1) > 0).squeeze(-1)
            if not valid_mask.any():
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            rgb_pooled = rgb_pooled[valid_mask]
            text_feat = text_out[valid_mask]
            mask = mask[valid_mask]
            text_pooled = (text_feat * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-5)  # [B', Dt]

        # projector + 归一化 + 相似度矩阵
        v = F.normalize(self.proj["rgb"](rgb_pooled), dim=1)
        t = F.normalize(self.proj["text"](text_pooled), dim=1)
        sim_matrix = v @ t.T
        temperature = cfg_get(self.config, "Pretraining.temperature", 0.07)
        return self.loss_fn({'sim_matrix': sim_matrix, 'temperature': temperature})

    def save_both_formats(self, epoch: int, tag: str):
        """
        同时保存 ckpt（可恢复）与 pt（纯权重）。
        tag 可以是 'epoch_003' / 'latest' / 'final'
        """
        # 1) 组织 state（ckpt）
        state = {
            "epoch": epoch,
            "config": vars(self.config) if hasattr(self.config, "__dict__") else {},
            "optimizer": self.optimizer.state_dict(),
            "scheduler": (self.scheduler.state_dict() if self.scheduler is not None else None),
        }
        for name, model in self.models.items():
            state[f"{name}_state_dict"] = model.state_dict()
        state["proj_state_dict"] = self.proj.state_dict()

        # 2) 组织纯权重（pt）
        weights = {name: m.state_dict() for name, m in self.models.items()}
        weights["proj"] = self.proj.state_dict()

        # 3) 写盘
        ckpt_path = os.path.join(self.ckpt_dir, f"{tag}.ckpt")
        pt_path = os.path.join(self.wts_dir, f"{tag}.pt")
        torch.save(state, ckpt_path)
        torch.save(weights, pt_path)

        # 4) W&B 记录（可选）
        if wandb.run:
            try:
                wandb.save(ckpt_path)
                wandb.save(pt_path)
            except Exception as e:
                logger.warning(f"W&B artifact save failed: {e}")

        logger.info(f"Saved: {ckpt_path} | {pt_path}")

    def train_epoch(self, epoch: int):
        """一个 epoch，轮询多个数据集混合训练"""
        for m in self.models.values(): m.train()
        self.proj.train()
        total_loss = 0.0

        # 收集可训练参数一次
        trainable_params = []
        for m in self.models.values():
            trainable_params.extend(list(m.parameters()))
        trainable_params.extend(list(self.proj.parameters()))

        rr = self._roundrobin_min()
        names = list(self.train_loaders.keys())
        min_len = min(len(ld) for ld in self.train_loaders.values())
        pbar = tqdm(rr, total=min_len * len(names),
                    desc=f"Epoch {epoch + 1}/{cfg_get(self.config, 'Training.epochs', 20)}")

        for step, (ds_name, (src_input, tgt_input)) in enumerate(pbar):

            # if torch.cuda.is_available() and amp_enabled:
            #     with make_autocast(dtype=amp_dtype):
            if torch.cuda.is_available() and self.amp_enabled:
                with make_autocast(dtype=self.amp_dtype):
                    loss = self.compute_pretrain_loss(src_input, tgt_input)
            else:
                loss = self.compute_pretrain_loss(src_input, tgt_input)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            # 先反缩放再裁剪
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(trainable_params, max_norm=cfg_get(self.config, 'Training.grad_clip', 1.0))
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(loss.item())
            pbar.set_postfix(loss=loss.item(), ds=ds_name)

            if wandb.run and step % 10 == 0:
                wandb.log({
                    "train/batch_loss": float(loss.item()),
                    "epoch": epoch,
                    "step": step,
                    "dataset": ds_name,
                    "learning_rate": self.optimizer.param_groups[0]["lr"]
                })

        avg = total_loss / max(1, sum(len(ld) for ld in self.train_loaders.values()))
        return avg

    def train(self):
        logger.info("Starting training...")
        epochs = int(cfg_get(self.config, "Training.epochs", 20))
        split_flag = cfg_get(self.config, "Saving.split_modalities", True)

        for epoch in range(epochs):
            # 1) 训练
            train_loss = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch + 1}/{epochs}: train_loss = {train_loss:.4f}")
            if wandb.run:
                wandb.log({"train/epoch_loss": float(train_loss), "epoch": epoch})

            # 2) 保存
            tag = f"epoch_{epoch + 1:03d}"
            self.save_both_formats(epoch=epoch, tag=tag)
            self.save_both_formats(epoch=epoch, tag="latest")

            if split_flag:
                self.save_split_weights(epoch=epoch, tag=tag)
                self.save_split_weights(epoch=epoch, tag="latest")

            # 3) 调度
            if self.scheduler is not None:
                self.scheduler.step()

        # 4) 训练结束再保存 final
        self.save_both_formats(epoch=epochs, tag="final")
        if split_flag:
            self.save_split_weights(epoch=epochs, tag="final")
        logger.info("Training completed!")


# ------------------------------
# 入口
# ------------------------------
def main():
    set_seed(42)

    cfg = load_train_config()
    logger.info("Loaded configuration")

    # 安全拿到 wandb 子配置
    wandb_cfg = getattr(cfg, "wandb", None)

    if wandb_cfg is not None and getattr(wandb_cfg, "use", False):
        try:
            wandb.init(
                project=getattr(wandb_cfg, "project", "sign-language-pretraining"),
                name=getattr(wandb_cfg, "run_name", "experiment"),
                config=vars(cfg),
            )
            logger.info("Initialized Weights & Biases")
        except Exception as e:
            logger.warning(f"W&B initialization failed: {e}")
            os.environ["WANDB_MODE"] = "disabled"
    else:
        os.environ["WANDB_MODE"] = "disabled"
        logger.info("W&B disabled")

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    main()
