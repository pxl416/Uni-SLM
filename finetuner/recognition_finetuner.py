# finetuner/recognition_finetuner.py
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast
from types import SimpleNamespace

from finetuner.base_finetuner import BaseFinetuner


class RecognitionFinetuner(BaseFinetuner):
    def __init__(self, cfg, model, device):
        super().__init__(cfg, model, device)

        self.model = model
        self.blank_id = model.recognition_head.blank_id
        self.criterion = nn.CTCLoss(blank=self.blank_id, zero_infinity=True)

        train_cfg = getattr(cfg, "Training", SimpleNamespace())
        self.grad_clip = getattr(train_cfg, "grad_clip", 1.0)

    # =============================================================
    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0

        for src, tgt in loader:

            gloss = tgt["gt_gloss"]
            gloss_ids, gloss_len = self._prepare_gloss(gloss)

            rgb_len = src["rgb_len"].to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=True):
                out = self.model(src, task="recognition")   # logits: (B, T_out, C)
                logits = out["logits"].permute(1, 0, 2)     # (T_out, B, C)

                loss = self.criterion(
                    logits,
                    gloss_ids.to(self.device),
                    rgb_len.to(self.device),
                    gloss_len.to(self.device)
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()
            self.global_step += 1

        return total_loss / len(loader)

    # =============================================================
    def eval_epoch(self, loader):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for src, tgt in loader:

                gloss = tgt["gt_gloss"]
                gloss_ids, gloss_len = self._prepare_gloss(gloss)
                rgb_len = src["rgb_len"].to(self.device)

                out = self.model(src, task="recognition")
                logits = out["logits"].permute(1, 0, 2)

                loss = self.criterion(
                    logits,
                    gloss_ids.to(self.device),
                    rgb_len.to(self.device),
                    gloss_len.to(self.device)
                )

                total_loss += loss.item()

        return total_loss / len(loader)

    # =============================================================
    # Convert gloss list → CTC target format
    # =============================================================
    def _prepare_gloss(self, gloss_list):
        """
        gloss_list: [["天气", "好"], ["你", "好"], ...]
        return:
            concat_ids: (sum_len,)
            gloss_len:  (B,)
        """
        ids_list = []
        len_list = []

        for g in gloss_list:
            ids = [self.model.gloss2id[x] for x in g]
            ids_list.append(torch.tensor(ids, dtype=torch.long))
            len_list.append(len(ids))

        # concat without padding → required by CTC
        concat_ids = torch.cat(ids_list, dim=0)
        gloss_len = torch.tensor(len_list, dtype=torch.long)

        return concat_ids, gloss_len
