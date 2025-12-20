# finetuner/retrieval_finetuner.py
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.cuda.amp import autocast

from finetuner.base_finetuner import BaseFinetuner
from utils.optimizer import build_optimizer


class RetrievalFinetuner(BaseFinetuner):
    """
    Finetuner for Video-Text Retrieval (batch-wise InfoNCE).
    """
    def __init__(self, cfg, model, dataset, device):
        if model is None:
            from utils.config import load_yaml_as_ns
            from models.build_model import build_model
            model = build_model(load_yaml_as_ns(cfg.model)).to(device)

        super().__init__(cfg, model, device)

        self.use_amp = bool(getattr(cfg, "amp", False))

        train_cfg = getattr(cfg, "Training", None)
        if train_cfg is None:
            raise ValueError("cfg.Training not found")

        self.optimizer, self.scheduler = build_optimizer(self.model, train_cfg)
        self.grad_clip = getattr(train_cfg, "grad_clip", 1.0)

        self.best_r1 = -1.0
        print("[Info] RetrievalFinetuner initialized.")

    # Train
    def train_epoch(self, loader):
        self.model.train()
        total_loss = total_t2v = total_v2t = 0.0
        n = 0

        pbar = tqdm(loader, desc="Train-Retrieval", ncols=100)
        for src, tgt in pbar:
            src = self._move_batch_to_device(src)

            # ⭐ 核心：拼 retrieval task batch
            task_batch = dict(src)
            task_batch["gt_sentence"] = tgt["gt_sentence"]

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                out = self.model(batch=task_batch, task="retrieval")
                sim = out.get("sim_logits", out.get("logits", None))
                if sim is None:
                    raise KeyError(
                        f"Retrieval forward must return sim_logits/logits, got {list(out.keys())}"
                    )

                B = sim.size(0)
                targets = torch.arange(B, device=self.device)

                loss_t2v = F.cross_entropy(sim, targets)
                if getattr(self.cfg, "symmetric", True):
                    loss_v2t = F.cross_entropy(sim.t(), targets)
                    loss = 0.5 * (loss_t2v + loss_v2t)
                else:
                    loss_v2t = torch.zeros_like(loss_t2v)
                    loss = loss_t2v

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(loss.item())
            total_t2v += float(loss_t2v.item())
            total_v2t += float(loss_v2t.item())
            n += 1
            self.global_step += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        metrics = {
            "loss": total_loss / max(1, n),
            "loss_t2v": total_t2v / max(1, n),
            "loss_v2t": total_v2t / max(1, n),
            "lr": self.optimizer.param_groups[0]["lr"],
            "grad_norm": float(grad_norm),
            "main_metric": -(total_loss / max(1, n)),
        }

        if "temperature" in out:
            metrics["tau"] = float(out["temperature"].detach().cpu())

        return metrics

    # Eval
    @torch.no_grad()
    def eval_epoch(self, loader):
        self.model.eval()

        sum_t2v = {"R1": 0.0, "R5": 0.0, "R10": 0.0}
        sum_v2t = {"R1": 0.0, "R5": 0.0, "R10": 0.0}
        n = 0

        for src, tgt in tqdm(loader, desc="Eval-Retrieval", ncols=100):
            src = self._move_batch_to_device(src)

            # ⭐ 同样拼 batch
            task_batch = dict(src)
            task_batch["gt_sentence"] = tgt["gt_sentence"]

            out = self.model(batch=task_batch, task="retrieval")
            sim = out.get("sim_logits", out.get("logits", None))
            if sim is None:
                raise KeyError(
                    f"Retrieval forward must return sim_logits/logits, got {list(out.keys())}"
                )

            m_t2v = self._recall_at_k_square(sim, ks=(1, 5, 10))
            m_v2t = self._recall_at_k_square(sim.t(), ks=(1, 5, 10))

            for k in sum_t2v:
                sum_t2v[k] += m_t2v[k]
                sum_v2t[k] += m_v2t[k]
            n += 1

        if n == 0:
            return {
                "t2v/R1": 0.0, "t2v/R5": 0.0, "t2v/R10": 0.0,
                "v2t/R1": 0.0, "v2t/R5": 0.0, "v2t/R10": 0.0,
                "main_metric": 0.0,
            }

        metrics = {
            "t2v/R1": sum_t2v["R1"] / n,
            "t2v/R5": sum_t2v["R5"] / n,
            "t2v/R10": sum_t2v["R10"] / n,
            "v2t/R1": sum_v2t["R1"] / n,
            "v2t/R5": sum_v2t["R5"] / n,
            "v2t/R10": sum_v2t["R10"] / n,
        }
        metrics["main_metric"] = metrics["t2v/R1"]

        print(
            f"[Eval] t2v R@1={metrics['t2v/R1']:.4f}, "
            f"v2t R@1={metrics['v2t/R1']:.4f}"
        )
        return metrics

    @staticmethod
    def _recall_at_k_square(sim_logits: torch.Tensor, ks=(1, 5, 10)):
        B = sim_logits.size(0)
        device = sim_logits.device
        ranks = torch.argsort(sim_logits, dim=1, descending=True)
        targets = torch.arange(B, device=device).unsqueeze(1)

        out = {}
        for k in ks:
            k = min(k, ranks.size(1))
            hit = (ranks[:, :k] == targets).any(dim=1).float().mean().item()
            out[f"R{k}"] = hit

        return {
            "R1": out.get("R1", 0.0),
            "R5": out.get("R5", 0.0),
            "R10": out.get("R10", 0.0),
        }

    def save_if_best(self, eval_metrics: dict, epoch: int):
        ckpt_dir = self._get_ckpt_dir()
        os.makedirs(ckpt_dir, exist_ok=True)

        main = float(eval_metrics.get("main_metric", 0.0))
        if main > self.best_r1:
            self.best_r1 = main
            filename = os.path.join(ckpt_dir, f"best_epoch_{epoch}.pt")

            torch.save(
                {
                    "model": self.model.state_dict(),
                    "epoch": epoch,
                    "metrics": eval_metrics,
                },
                filename,
            )

            print(f"[Checkpoint] Saved best retrieval model → {filename}")

    def _get_ckpt_dir(self):
        """
        Directory for retrieval checkpoints.
        Priority:
          1) cfg.Finetune.save_dir
          2) default: checkpoints/retrieval
        """
        base = None
        if hasattr(self.cfg, "Finetune") and hasattr(self.cfg.Finetune, "save_dir"):
            base = self.cfg.Finetune.save_dir
        if base is None:
            base = "checkpoints"
        return os.path.join(base, "retrieval")


