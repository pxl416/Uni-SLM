# pretrainer/temporal_pretrainer.py
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm

from pretrainer.base_pretrainer import BasePretrainer


class TemporalPretrainer(BasePretrainer):
    """
    Temporal Heatmap Pretrainer

    Objective:
        Learn frame-wise temporal importance / activity heatmap
        via self-supervised or weakly-supervised signals.
    """

    def __init__(self, cfg, model, device):
        super().__init__(cfg, model, device)

        # loss type (easy to extend later)
        pre_cfg = getattr(cfg, "Pretrain", None)
        self.loss_type = getattr(pre_cfg, "loss", "bce")  # bce | mse

        print(f"[Info] TemporalPretrainer initialized | loss={self.loss_type}")

    # --------------------------------------------------
    # Target construction (baseline version)
    # --------------------------------------------------
    def build_temporal_target(self, batch, T: int):
        """
        Build temporal supervision target.

        Current baseline:
          - If rgb_mask exists: use it as binary mask
          - Else: assume all frames valid (ones)

        Returns:
            target: Tensor (B, T) in [0, 1]
        """
        if "rgb_mask" in batch and batch["rgb_mask"] is not None:
            # rgb_mask: (B, T) or (B, T, 1)
            mask = batch["rgb_mask"]
            if mask.dim() == 3:
                mask = mask.squeeze(-1)
            target = mask.float()
        else:
            B = batch["rgb_img"].size(0)
            target = torch.ones((B, T), device=self.device)

        return target

    # --------------------------------------------------
    # Train
    # --------------------------------------------------
    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(loader, desc="Train-Temporal", ncols=100)

        for batch in pbar:
            batch = self._move_batch_to_device(batch)

            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                out = self.model(batch=batch, task="temporal")

                logits = out["temporal_logits"]      # (B, T)
                B, T = logits.shape

                target = self.build_temporal_target(batch, T)

                if self.loss_type == "bce":
                    loss = F.binary_cross_entropy_with_logits(
                        logits, target
                    )
                elif self.loss_type == "mse":
                    loss = F.mse_loss(torch.sigmoid(logits), target)
                else:
                    raise ValueError(f"Unknown loss type: {self.loss_type}")

            # backward
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            self.global_step += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}")

            del loss
            torch.cuda.empty_cache()

        return total_loss / len(loader)

    # --------------------------------------------------
    # Eval (lightweight sanity check)
    # --------------------------------------------------
    def eval_epoch(self, loader):
        self.model.eval()
        total_loss = 0.0

        pbar = tqdm(loader, desc="Eval-Temporal", ncols=100)

        with torch.no_grad():
            for batch in pbar:
                batch = self._move_batch_to_device(batch)

                out = self.model(batch=batch, task="temporal")
                logits = out["temporal_logits"]
                B, T = logits.shape

                target = self.build_temporal_target(batch, T)

                if self.loss_type == "bce":
                    loss = F.binary_cross_entropy_with_logits(
                        logits, target
                    )
                else:
                    loss = F.mse_loss(torch.sigmoid(logits), target)

                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"[Eval] Temporal loss = {avg_loss:.4f}")

        return avg_loss
