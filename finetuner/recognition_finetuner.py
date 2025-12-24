# finetuner/recognition_finetuner.py
import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm
from finetuner.base_finetuner import BaseFinetuner
from utils.optimizer import build_optimizer


#  Metrics (baseline stage: kept locally, will be moved later)
def edit_distance(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1],
                )
    return dp[-1][-1]

def compute_cer(pred_list, gt_list):
    """
    pred_list / gt_list: List[List[str]]
    """
    total_ed, total_ref = 0, 0
    for p, g in zip(pred_list, gt_list):
        total_ed += edit_distance(p, g)
        total_ref += len(g)
    return total_ed / max(1, total_ref)


def compute_wer(pred_list, gt_list):
    """
    WER (Sequence Error Rate):
    一个序列只要不完全一致，就记为 1 个错误
    """
    assert len(pred_list) == len(gt_list)

    num_err = 0
    for p, g in zip(pred_list, gt_list):
        if p != g:
            num_err += 1

    return num_err / max(1, len(gt_list))


def compute_ser(pred_list, gt_list):
    """
    Sentence Error Rate:
    完整序列是否完全匹配
    """
    return compute_wer(pred_list, gt_list)


def compute_token_accuracy(pred_list, gt_list):
    """
    Token-level accuracy:
    对齐到最短长度，逐 token 比较
    """
    correct, total = 0, 0

    for p, g in zip(pred_list, gt_list):
        L = min(len(p), len(g))
        for i in range(L):
            if p[i] == g[i]:
                correct += 1
        total += max(len(g), 1)

    return correct / max(1, total)




#  Recognition Finetuner
class RecognitionFinetuner(BaseFinetuner):
    """
    CTC-based recognition finetuner.

    Assumptions:
    - model is already built via build_model
    - model has recognition_head
    - model(batch, task="recognition") returns {"logits": (B, T, C)}
    """
    def __init__(self, cfg, model, dataset, device):
        if model is None:
            raise ValueError("RecognitionFinetuner requires a pre-built model.")

        if dataset is None or not hasattr(dataset, "gloss2id"):
            raise ValueError("RecognitionFinetuner requires dataset with gloss2id.")

        super().__init__(cfg, model, device)

        self.use_amp = getattr(cfg.Training, "amp", False)  # AMP
        self.gloss2id = dataset.gloss2id  # Vocabulary
        self.inv_vocab = {v: k for k, v in self.gloss2id.items()}
        self.blank_id = 0
        self.vocab_size = len(self.gloss2id) + 1  # + UNK
        self.unk_id = self.vocab_size - 1
        print(f"[Info] Recognition vocab_size={self.vocab_size}, "
              f"blank_id={self.blank_id}, unk_id={self.unk_id}")
        self.optimizer, self.scheduler = build_optimizer(self.model, cfg.Training) # Optimizer / Scheduler
        self.criterion = nn.CTCLoss(blank=self.blank_id, zero_infinity=True)  # Loss
        self.grad_clip = getattr(cfg.Training, "grad_clip", 1.0)
        self.best_eval_loss = float("inf")  # Best tracking
        self._resize_recognition_head(self.vocab_size)

    #  Training
    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc="Train", ncols=100)
        for src, tgt in pbar:
            src = self._move_batch_to_device(src)
            gloss_ids, gloss_len = self._prepare_gloss(tgt["gt_gloss"])
            self.optimizer.zero_grad()
            with autocast(enabled=self.use_amp):
                out = self.model(batch=src, task="recognition")
                logits = out["logits"]  # (B, T, C)

                # ✅ 关键：CTC 需要 log-probabilities
                log_probs = torch.log_softmax(logits, dim=-1)

                B, T, C = log_probs.shape
                log_probs_tbc = log_probs.permute(1, 0, 2)

                input_len = torch.full((B,), T, dtype=torch.long, device=self.device)

                loss = self.criterion(
                    log_probs_tbc,
                    gloss_ids.to(self.device),
                    input_len,
                    gloss_len.to(self.device),
                )
                if torch.isnan(loss) or loss.item() < 0:  # px. 如果我的loss设计有问题这里会标出来
                    print(
                        "DEBUG CTC:",
                        "loss =", loss.item(),
                        "logits min/max =", logits.min().item(), logits.max().item(),
                        "gloss_len max =", gloss_len.max().item(),
                        "input_len =", input_len[0].item()
                    )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
            self.global_step += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(loader)

    #  Evaluation
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
                logits = out["logits"]  # (B, T, C)

                # ====== CTC 用 log-probabilities ======
                log_probs = torch.log_softmax(logits, dim=-1)

                B, T, C = log_probs.shape
                log_probs_tbc = log_probs.permute(1, 0, 2)

                input_len = torch.full(
                    (B,), T, dtype=torch.long, device=self.device
                )

                loss = self.criterion(
                    log_probs_tbc,
                    gloss_ids.to(self.device),
                    input_len,
                    gloss_len.to(self.device),
                )
                total_loss += loss.item()

                # ====== Greedy Decode（仍然用 logits） ======
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

                    pred_gloss = [
                        self.inv_vocab.get(tok, "<unk>")
                        for tok in decoded
                    ]

                    pred_all.append(pred_gloss)
                    gt_all.append(tgt["gt_gloss"][i])

        cer = compute_cer(pred_all, gt_all)
        wer = compute_wer(pred_all, gt_all)
        ser = compute_ser(pred_all, gt_all)
        acc = compute_token_accuracy(pred_all, gt_all)

        print(
            f"[Eval] CER = {cer:.4f} | "
            f"WER = {wer:.4f} | "
            f"SER = {ser:.4f} | "
            f"TokenAcc = {acc:.4f}"
        )

        # ===== WandB logging (if enabled) =====
        if hasattr(self, "cfg") and getattr(self.cfg, "wandb", None):
            try:
                import wandb
                if self.cfg.wandb.use:
                    # random qualitative sample
                    if len(gt_all) > 0:
                        import random
                        idx = random.randrange(len(gt_all))
                        sample_gt = " ".join(gt_all[idx])
                        sample_pred = " ".join(pred_all[idx])
                    else:
                        sample_gt = ""
                        sample_pred = ""

                    wandb.log({
                        "eval/loss": total_loss / len(loader),
                        "eval/cer": cer,
                        "eval/wer": wer,
                        "eval/ser": ser,
                        "eval/token_acc": acc,
                        "eval/sample_gt": sample_gt,
                        "eval/sample_pred": sample_pred,
                    })
            except Exception as e:
                print(f"[Warn] WandB logging failed: {e}")

        return total_loss / len(loader)

    #  Checkpoint helper
    def save_if_best(self, eval_loss, epoch):
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            tag = f"recog_best_epoch_{epoch}"

            # 1) 完整 checkpoint（用于 resume）
            self.save_checkpoint(tag + ".pt")
            print(f"[Checkpoint] Saved full model → {tag}.pt")

            # 2) 子模块权重（用于复用）
            self.save_pretrained_submodules(tag=tag)

    def save_pretrained_submodules(self, tag: str):
        """
        Save encoder submodules for reuse (baseline-friendly).
        """
        os.makedirs("checkpoints/pretrained", exist_ok=True)
        base = f"checkpoints/pretrained/{tag}"

        # RGB encoder
        torch.save(
            self.model.rgb_encoder.backbone.state_dict(),
            base + "_rgb_backbone.pt"
        )

        if hasattr(self.model.rgb_encoder, "proj") and self.model.rgb_encoder.proj is not None:
            torch.save(
                self.model.rgb_encoder.proj.state_dict(),
                base + "_rgb_proj.pt"
            )

        # Pose encoder (if exists)
        if hasattr(self.model, "pose_encoder") and self.model.pose_encoder is not None:
            torch.save(
                self.model.pose_encoder.backbone.state_dict(),
                base + "_pose_backbone.pt"
            )
            if hasattr(self.model.pose_encoder, "proj") and self.model.pose_encoder.proj is not None:
                torch.save(
                    self.model.pose_encoder.proj.state_dict(),
                    base + "_pose_proj.pt"
                )

        # Text encoder (if exists)
        if hasattr(self.model, "text_encoder") and self.model.text_encoder is not None:
            torch.save(
                self.model.text_encoder.backbone.state_dict(),
                base + "_text_backbone.pt"
            )
            if hasattr(self.model.text_encoder, "proj") and self.model.text_encoder.proj is not None:
                torch.save(
                    self.model.text_encoder.proj.state_dict(),
                    base + "_text_proj.pt"
                )

        print(f"[Pretrained] Saved encoder submodules for tag={tag}")

    #  CTC target preparation
    def _prepare_gloss(self, gloss_list):
        ids_list = []
        len_list = []

        for seq in gloss_list:
            if len(seq) == 0:
                ids = [self.blank_id]
            else:
                ids = [
                    self.gloss2id.get(x, self.unk_id)
                    for x in seq
                ]

            ids_list.append(torch.tensor(ids, dtype=torch.long))
            len_list.append(len(ids))

        return (
            torch.cat(ids_list, dim=0),
            torch.tensor(len_list, dtype=torch.long),
        )

    def _resize_recognition_head(self, num_classes: int):
        head = self.model.recognition_head
        if head is None:
            raise RuntimeError("model.recognition_head is None")
        if getattr(head, "num_classes", None) == num_classes:
            return

        in_dim = head.fc.in_features
        new_fc = nn.Linear(in_dim, num_classes).to(self.device)

        # 可选：尽量拷贝旧权重
        old_fc = head.fc
        copy_dim = min(old_fc.out_features, num_classes)
        with torch.no_grad():
            new_fc.weight[:copy_dim].copy_(old_fc.weight[:copy_dim])
            new_fc.bias[:copy_dim].copy_(old_fc.bias[:copy_dim])

        head.fc = new_fc
        head.num_classes = num_classes




