# pretrainer/temporal_pretrainer.py
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from pretrainer.base_pretrainer import BasePretrainer
from utils.loss import temporal_bce_loss, temporal_mse_loss


class TemporalPretrainer(BasePretrainer):
    def __init__(self, cfg, model, device):
        super().__init__(cfg, model, device)

        pre_cfg = getattr(cfg, "Pretrain", None)
        self.loss_type = getattr(pre_cfg, "loss", "bce")   # bce | mse
        self.loss_weight = getattr(pre_cfg, "loss_weight", 1.0)
        self.log_interval = getattr(pre_cfg, "log_interval", 50)

        print(f"[Info] TemporalPretrainer initialized | loss={self.loss_type}, amp={self.use_amp}")

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(loader, desc="Train-Temporal", ncols=100)

        for step, (src, tgt) in enumerate(pbar):
            # src is dict, tgt is dict (or unused)
            src = self._move_batch_to_device(src)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                out = self.model(batch=src, task="temporal")
                logits = out["temporal_logits"]  # (B, T_pred)

                rgb_mask = src.get("rgb_mask", None)
                if rgb_mask is None:
                    raise RuntimeError("src['rgb_mask'] is None, cannot build temporal target yet.")

                if self.loss_type == "bce":
                    loss = temporal_bce_loss(logits, rgb_mask)
                elif self.loss_type == "mse":
                    loss = temporal_mse_loss(logits, rgb_mask)
                else:
                    raise ValueError(f"Unknown loss type: {self.loss_type}")

                loss = loss * self.loss_weight

            # backward
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(loss.item())
            self.global_step += 1

            if step % self.log_interval == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total_loss/(step+1):.4f}")

        return total_loss / max(len(loader), 1)

    @torch.no_grad()
    def eval_epoch(self, loader):
        self.model.eval()
        total_loss = 0.0

        pbar = tqdm(loader, desc="Eval-Temporal", ncols=100)

        for step, (src, tgt) in enumerate(pbar):
            src = self._move_batch_to_device(src)

            out = self.model(batch=src, task="temporal")
            logits = out["temporal_logits"]

            rgb_mask = src.get("rgb_mask", None)
            if rgb_mask is None:
                raise RuntimeError("src['rgb_mask'] is None, cannot build temporal target yet.")

            if self.loss_type == "bce":
                loss = temporal_bce_loss(logits, rgb_mask)
            else:
                loss = temporal_mse_loss(logits, rgb_mask)

            total_loss += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(len(loader), 1)
        print(f"[Eval] Temporal loss = {avg_loss:.4f}")
        return avg_loss
