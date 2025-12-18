# finetuner/recognition_finetuner.py

import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from types import SimpleNamespace
from tqdm import tqdm

from finetuner.base_finetuner import BaseFinetuner
from models.Head.recognition import RecognitionHead
from utils.optimizer import build_optimizer


# (1) CER / WER — 评价函数
def edit_distance(a, b):
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1): dp[i][0] = i
    for j in range(len(b)+1): dp[0][j] = j

    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[-1][-1]


def compute_cer(pred_list, gt_list):
    total_ed, total_ref = 0, 0
    for p, g in zip(pred_list, gt_list):
        total_ed += edit_distance(p, g)
        total_ref += len(g)
    return total_ed / max(1, total_ref)


class RecognitionFinetuner(BaseFinetuner):

    def __init__(self, cfg, model, dataset, device):
        """
        model 允许为 None：未来可以让 Finetuner 自己 build_model
        dataset 对 recognition 是必须的（因为 gloss2id）
        """
        # 0) allow model=None (future-proof)
        if model is None:
            from utils.config import load_yaml_as_ns
            from models.build_model import build_model
            model = build_model(load_yaml_as_ns(cfg.model)).to(device)

        if dataset is None:
            raise ValueError("RecognitionFinetuner requires dataset (for gloss2id).")

        super().__init__(cfg, model, device)

        self.use_amp = False

        # Gloss vocabulary
        if not hasattr(dataset, "gloss2id"):
            raise ValueError("Dataset must contain gloss2id!")

        self.gloss2id = dataset.gloss2id
        vocab_size = len(self.gloss2id) + 1  # UNK
        blank_id = 0
        self.unk_id = vocab_size - 1

        print(f"[Info] Recognition vocab_size={vocab_size}, blank={blank_id}, unk={self.unk_id}")

        # 反向词表（只建一次）
        self.inv_vocab = {v: k for k, v in self.gloss2id.items()}

        # 1) 替换 recognition_head
        self.model.recognition_head = RecognitionHead(
            hidden_dim=self.model.hidden_dim,
            num_classes=vocab_size,
            blank_id=blank_id
        ).to(device)

        self.blank_id = blank_id
        self.vocab_size = vocab_size

        # 2) 重建 optimizer / scheduler
        train_cfg = getattr(cfg, "Training", None)
        if train_cfg is None:
            raise ValueError("cfg.Training not found")

        self.optimizer, self.scheduler = build_optimizer(self.model, train_cfg)

        # Loss
        self.criterion = nn.CTCLoss(blank=self.blank_id, zero_infinity=True)

        self.grad_clip = getattr(train_cfg, "grad_clip", 1.0)

        # Track best
        self.best_eval_loss = float("inf")



    # (2) Save best checkpoint
    def save_if_best(self, eval_loss, epoch):
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            filename = f"best_epoch_{epoch}.pt"
            self.save_checkpoint(filename)
            print(f"[Checkpoint] Saved best model → {filename}")


    #                 TRAIN LOOP
    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(loader, desc="Train", ncols=100)

        for src, tgt in pbar:
            # --------------------------
            # Move data to GPU
            # --------------------------
            src = self._move_batch_to_device(src)
            gloss_ids, gloss_len = self._prepare_gloss(tgt["gt_gloss"])

            self.optimizer.zero_grad()

            # Forward with AMP
            with autocast(enabled=self.use_amp):
                out = self.model(batch=src, task="recognition")
                logits = out["logits"]  # (B, T_real, C)

                B, T_real, C = logits.shape

                # CTC format: (T, B, C)
                logits_tbc = logits.permute(1, 0, 2)

                # True input lengths
                input_len = torch.full(
                    (B,), T_real,
                    dtype=torch.long,
                    device=self.device
                )

                loss = self.criterion(
                    logits_tbc,
                    gloss_ids.to(self.device),
                    input_len,
                    gloss_len.to(self.device)
                )

            # Backward with scaler
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Bookkeeping
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            self.global_step += 1

            # Release memory
            del logits, logits_tbc, loss
            torch.cuda.empty_cache()

        return total_loss / len(loader)

    #                 EVAL LOOP
    def eval_epoch(self, loader):
        self.model.eval()
        total_loss = 0.0

        pred_all, gt_all = [], []

        pbar = tqdm(loader, desc="Eval", ncols=100)

        with torch.no_grad():
            for src, tgt in pbar:
                src = self._move_batch_to_device(src)

                gloss_ids, gloss_len = self._prepare_gloss(tgt["gt_gloss"])

                out = self.model(batch=src, task="recognition")
                logits = out["logits"]

                B, T_real, C = logits.shape
                logits_tbc = logits.permute(1, 0, 2)

                input_len = torch.full((B,), T_real, dtype=torch.long, device=self.device)

                loss = self.criterion(
                    logits_tbc,
                    gloss_ids.to(self.device),
                    input_len,
                    gloss_len.to(self.device)
                )
                total_loss += loss.item()

                # Greedy decode
                pred_ids = torch.argmax(logits, dim=-1).cpu()

                for i in range(B):
                    seq = pred_ids[i]
                    decoded = []
                    last = -1

                    for tok in seq:
                        tok = tok.item()
                        if tok != self.blank_id and tok != last:
                            decoded.append(tok)
                        last = tok

                    # ID → gloss
                    pred_gloss = [
                        self.inv_vocab.get(tok, "<unk>") for tok in decoded
                    ]

                    pred_all.append(pred_gloss)
                    gt_all.append(tgt["gt_gloss"][i])

        cer = compute_cer(pred_all, gt_all)
        print(f"[Eval] CER={cer:.4f}")

        return total_loss / len(loader)


    # CTC target preparation
    def _prepare_gloss(self, gloss_list):
        ids_list = []
        len_list = []

        for seq in gloss_list:
            if len(seq) == 0:
                ids = [self.blank_id]
            else:
                ids = [self.gloss2id.get(x, self.unk_id) for x in seq]

            ids_list.append(torch.tensor(ids, dtype=torch.long))
            len_list.append(len(ids))

        return torch.cat(ids_list, dim=0), torch.tensor(len_list, dtype=torch.long)





