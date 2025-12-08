# # finetuner/recognition_finetuner.py
# import torch
# import torch.nn as nn
# from torch.cuda.amp import autocast
# from types import SimpleNamespace
#
# from finetuner.base_finetuner import BaseFinetuner
# from models.Head.recognition import RecognitionHead
#
#
# class RecognitionFinetuner(BaseFinetuner):
#
#     def __init__(self, cfg, model, dataset, device):
#         super().__init__(cfg, model, device)
#
#         # ---- AMP 必须关掉（非常重要）----
#         self.use_amp = False
#
#         # -----------------------------
#         # Gloss vocabulary
#         # -----------------------------
#         if not hasattr(dataset, "gloss2id"):
#             raise ValueError("Dataset must have gloss2id for recognition task!")
#
#         self.gloss2id = dataset.gloss2id
#         vocab_size = len(self.gloss2id) + 1   # UNK token
#         blank_id = 0
#         self.unk_id = vocab_size - 1
#
#         print(f"[Info] RecognitionFinetuner: vocab_size={vocab_size}, blank_id={blank_id}, unk_id={self.unk_id}")
#
#         # -----------------------------
#         # Replace Recognition Head
#         # -----------------------------
#         # ① 必须使用 RecognitionHead(...) 而不是 cfg
#         model.recognition_head = RecognitionHead(
#             hidden_dim=model.hidden_dim,
#             num_classes=vocab_size,
#             blank_id=blank_id
#         ).to(device)
#
#         self.blank_id = blank_id
#         self.vocab_size = vocab_size
#
#         # -----------------------------
#         # Loss function
#         # -----------------------------
#         self.criterion = nn.CTCLoss(blank=self.blank_id, zero_infinity=True)
#
#         train_cfg = getattr(cfg, "Training", SimpleNamespace())
#         self.grad_clip = getattr(train_cfg, "grad_clip", 1.0)
#
#     # =============================================================
#     def train_epoch(self, loader):
#         self.model.train()
#         total_loss = 0.0
#
#         # for src, tgt in loader:
#         for src, tgt in loader:
#             src = self._move_batch_to_device(src)
#             gloss = tgt["gt_gloss"]
#             gloss_ids, gloss_len = self._prepare_gloss(gloss)
#
#             self.optimizer.zero_grad()
#
#             with autocast(enabled=self.use_amp):
#                 out = self.model(batch=src, task="recognition")
#                 logits = out["logits"]       # (B, T, C)
#
#                 B, T, C = logits.shape
#                 logits_tbc = logits.permute(1, 0, 2)  # (T, B, C)
#
#                 input_len = torch.full(
#                     size=(B,),
#                     fill_value=T,
#                     dtype=torch.long,
#                     device=self.device
#                 )
#
#                 loss = self.criterion(
#                     logits_tbc,
#                     gloss_ids.to(self.device),
#                     input_len,
#                     gloss_len.to(self.device)
#                 )
#
#             # ---- ② 当 use_amp=False 时，不要用 scaler ----
#             if self.use_amp:
#                 self.scaler.scale(loss).backward()
#                 self.scaler.unscale_(self.optimizer)
#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
#                 self.scaler.step(self.optimizer)
#                 self.scaler.update()
#             else:
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
#                 self.optimizer.step()
#
#             if self.scheduler:
#                 self.scheduler.step()
#
#             total_loss += loss.item()
#             self.global_step += 1
#
#         return total_loss / len(loader)
#
#     # =============================================================
#     def eval_epoch(self, loader):
#         self.model.eval()
#         total_loss = 0.0
#
#         with torch.no_grad():
#             for src, tgt in loader:
#                 # ⭐⭐ 这里同样要搬到 GPU ⭐⭐
#                 src = self._move_batch_to_device(src)
#
#                 gloss = tgt["gt_gloss"]
#                 gloss_ids, gloss_len = self._prepare_gloss(gloss)
#
#                 out = self.model(batch=src, task="recognition")
#                 logits = out["logits"]
#
#                 B, T, C = logits.shape
#                 logits_tbc = logits.permute(1, 0, 2)
#
#                 input_len = torch.full(
#                     size=(B,),
#                     fill_value=T,
#                     dtype=torch.long,
#                     device=self.device
#                 )
#
#                 loss = self.criterion(
#                     logits_tbc,
#                     gloss_ids.to(self.device),
#                     input_len,
#                     gloss_len.to(self.device)
#                 )
#                 total_loss += loss.item()
#
#         return total_loss / len(loader)
#
#
#     # =============================================================
#     def _prepare_gloss(self, gloss_list):
#         ids_list = []
#         len_list = []
#
#         for seq in gloss_list:
#             if len(seq) == 0:
#                 ids = [self.blank_id]
#             else:
#                 ids = [self.gloss2id.get(tok, self.unk_id) for tok in seq]
#
#             ids_list.append(torch.tensor(ids, dtype=torch.long))
#             len_list.append(len(ids))
#
#         concat_ids = torch.cat(ids_list, dim=0)
#         gloss_len = torch.tensor(len_list, dtype=torch.long)
#
#         return concat_ids, gloss_len
# finetuner/recognition_finetuner.py
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from types import SimpleNamespace
from tqdm import tqdm

from finetuner.base_finetuner import BaseFinetuner
from models.Head.recognition import RecognitionHead


# =============================================================
#                (1) CER / WER — 解耦独立函数
# =============================================================
def edit_distance(a, b):
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1):
        dp[i][0] = i
    for j in range(len(b)+1):
        dp[0][j] = j

    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j], dp[i][j-1], dp[i-1][j-1]
                )
    return dp[-1][-1]


def compute_cer(pred_list, gt_list):
    """pred_list, gt_list: list[list[str]]"""
    total_ed, total_ref = 0, 0

    for pred, gt in zip(pred_list, gt_list):
        ed = edit_distance(pred, gt)
        total_ed += ed
        total_ref += len(gt)

    return total_ed / max(total_ref, 1)


# =============================================================
#                (2) Recognition Finetuner
# =============================================================
class RecognitionFinetuner(BaseFinetuner):

    def __init__(self, cfg, model, dataset, device):
        super().__init__(cfg, model, device)

        # AMP OFF for I3D
        self.use_amp = False

        # -----------------------------
        # Gloss vocabulary
        # -----------------------------
        if not hasattr(dataset, "gloss2id"):
            raise ValueError("Dataset must contain gloss2id!")

        self.gloss2id = dataset.gloss2id
        vocab_size = len(self.gloss2id) + 1  # UNK
        blank_id = 0
        self.unk_id = vocab_size - 1

        print(f"[Info] RecognitionHead vocab_size={vocab_size}, blank={blank_id}, unk={self.unk_id}")

        # -----------------------------
        # Replace recognition head
        # -----------------------------
        model.recognition_head = RecognitionHead(
            hidden_dim=model.hidden_dim,
            num_classes=vocab_size,
            blank_id=blank_id
        ).to(device)

        self.blank_id = blank_id
        self.vocab_size = vocab_size

        # -----------------------------
        # Loss
        # -----------------------------
        self.criterion = nn.CTCLoss(blank=self.blank_id, zero_infinity=True)

        train_cfg = getattr(cfg, "Training", SimpleNamespace())
        self.grad_clip = getattr(train_cfg, "grad_clip", 1.0)

        # -----------------------------
        # Best checkpoint tracking
        # -----------------------------
        self.best_eval_loss = 1e9


    # =============================================================
    #        (3) 解耦后的 best checkpoint 保存
    # =============================================================
    def save_if_best(self, eval_loss, epoch):
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            filename = f"best_epoch_{epoch}.pt"
            self.save_checkpoint(filename)
            print(f"[Checkpoint] New best ({eval_loss:.4f}) → saved {filename}")


    #                 TRAIN LOOP
    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(loader, desc="Train", ncols=100)

        for src, tgt in pbar:

            # Move tensor inputs to device
            src = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                   for k, v in src.items()}

            gloss = tgt["gt_gloss"]
            gloss_ids, gloss_len = self._prepare_gloss(gloss)

            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                out = self.model(batch=src, task="recognition")
                logits = out["logits"]  # (B, T_real, C)

                B, T_real, C = logits.shape

                # CTC expects (T, B, C)
                logits_tbc = logits.permute(1, 0, 2)

                # ---- FIX: CTC requires input_len ≤ T_real ----
                input_len = torch.full(
                    (B,),
                    T_real,
                    dtype=torch.long,
                    device=self.device
                )

                loss = self.criterion(
                    logits_tbc,
                    gloss_ids.to(self.device),
                    input_len,
                    gloss_len.to(self.device)
                )

            # Backward
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(loader)

    #                   EVAL LOOP
    def eval_epoch(self, loader):
        self.model.eval()
        total_loss = 0.0

        pred_all = []
        gt_all = []

        pbar = tqdm(loader, desc="Eval", ncols=100)

        with torch.no_grad():
            for src, tgt in pbar:

                src = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                       for k, v in src.items()}

                gloss = tgt["gt_gloss"]
                gloss_ids, gloss_len = self._prepare_gloss(gloss)

                out = self.model(batch=src, task="recognition")
                logits = out["logits"]

                B, T_real, C = logits.shape
                logits_tbc = logits.permute(1, 0, 2)

                # ---- FIX: CTC requires input_len ≤ T_real ----
                input_len = torch.full(
                    (B,),
                    T_real,
                    dtype=torch.long,
                    device=self.device
                )

                loss = self.criterion(
                    logits_tbc,
                    gloss_ids.to(self.device),
                    input_len,
                    gloss_len.to(self.device)
                )
                total_loss += loss.item()

                # ----------------------------
                # Greedy decode
                # ----------------------------
                pred_tokens = torch.argmax(logits, dim=-1).cpu()  # (B,T_real)

                for i in range(B):
                    decoded_ids = []
                    last = -1
                    for t in range(T_real):
                        tok = pred_tokens[i, t].item()
                        if tok != self.blank_id and tok != last:
                            decoded_ids.append(tok)
                        last = tok

                    # ID → gloss
                    pred_gloss = []
                    inv_vocab = {v: k for k, v in self.gloss2id.items()}

                    for tok in decoded_ids:
                        if tok == self.unk_id:
                            pred_gloss.append("<unk>")
                        else:
                            pred_gloss.append(inv_vocab.get(tok, "<unk>"))

                    pred_all.append(pred_gloss)
                    gt_all.append(gloss[i])

        # ----------------------------
        # CER（解耦函数）
        # ----------------------------
        cer = compute_cer(pred_all, gt_all)
        print(f"[Eval] CER={cer:.4f}")

        return total_loss / len(loader)


    #          gloss → concat ids (for CTC)
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


