# finetune.py
import argparse
import inspect
import random
from types import SimpleNamespace

import numpy as np
import torch

from utils.config import load_yaml_as_ns
from datasets.datasets import create_dataloader
from models.build_model import build_model

from finetuner.recognition_finetuner import RecognitionFinetuner
from finetuner.translation_finetuner import TranslationFinetuner

# Optional tasks (if you have them)
try:
    from finetuner.retrieval_finetuner import RetrievalFinetuner
except Exception:
    RetrievalFinetuner = None

try:
    from finetuner.segment_finetuner import SegmentFinetuner
except Exception:
    SegmentFinetuner = None

try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False


# -------------------------
# Args
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/ft.yaml")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# -------------------------
# Helpers
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_training_cfg(cfg):
    """
    Do NOT override cfg.Training if it exists.
    Only fill missing fields.
    """
    if not hasattr(cfg, "Training") or cfg.Training is None:
        cfg.Training = SimpleNamespace()

    # Fill defaults if missing
    defaults = {
        "epochs": 30,
        "batch_size": 4,
        "num_workers": 4,
        "grad_clip": 1.0,
    }
    for k, v in defaults.items():
        if not hasattr(cfg.Training, k):
            setattr(cfg.Training, k, v)

    return cfg


def call_maybe_with_loader(func, loader):
    """
    Compatible call:
      - func(loader)
      - func()
    """
    sig = inspect.signature(func)
    if len(sig.parameters) == 0:
        return func()
    return func(loader)


def normalize_metrics(out, default_key: str):
    """
    Normalize outputs into a dict for printing / wandb.
    - If out is float/int -> {default_key: out}
    - If out is dict -> out
    - Else -> {default_key: None}
    """
    if isinstance(out, (float, int)):
        return {default_key: float(out)}
    if isinstance(out, dict):
        return out
    return {default_key: None}


def save_if_best_compat(finetuner, eval_out, epoch: int):
    """
    Compatible with:
      - save_if_best(eval_loss: float, epoch: int)   (legacy)
      - save_if_best(eval_metrics: dict, epoch: int) (recommended)
    If finetuner has no save_if_best, do nothing.
    """
    if not hasattr(finetuner, "save_if_best"):
        return

    fn = finetuner.save_if_best
    sig = inspect.signature(fn)
    n_params = len(sig.parameters)

    # Most common:
    #   save_if_best(eval_loss, epoch)
    #   save_if_best(eval_metrics, epoch)
    if n_params == 2:
        fn(eval_out, epoch)
        return

    # If someone wrote save_if_best(self, epoch) style:
    if n_params == 1:
        fn(epoch)
        return

    # Fallback: try best effort
    try:
        fn(eval_out, epoch)
    except Exception as e:
        print(f"[Warn] save_if_best call failed: {e}")


# Main
def main():
    args = parse_args()

    cfg = load_yaml_as_ns(args.config)
    cfg = ensure_training_cfg(cfg)

    # device
    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )
    cfg.device = device
    print(f"[Config] Using device: {device}")

    # seed
    if args.seed is not None:
        set_seed(args.seed)
        print(f"[Config] seed={args.seed}")

    # override CLI epochs/batch_size (if provided)
    if args.epochs is not None:
        cfg.Training.epochs = args.epochs
    if args.batch_size is not None:
        cfg.Training.batch_size = args.batch_size

    print(f"[Config] epochs={cfg.Training.epochs}, batch_size={cfg.Training.batch_size}")

    # Dataloaders
    # create_dataloader() in your project expects an args-like object
    dl_args = SimpleNamespace(
        batch_size=getattr(cfg.Training, "batch_size", 4),
        num_workers=getattr(cfg.Training, "num_workers", 4),
    )

    # Split names: prefer cfg.Finetune.train_split / eval_split; fallback to "train"/"dev"
    train_split = getattr(getattr(cfg, "Finetune", SimpleNamespace()), "train_split", "train")
    eval_split = getattr(getattr(cfg, "Finetune", SimpleNamespace()), "eval_split", "dev")

    train_loader = create_dataloader(dl_args, cfg, phase=train_split)
    eval_loader = create_dataloader(dl_args, cfg, phase=eval_split)

    # Model
    # cfg.model can be either a path or an already-loaded namespace (your code uses path)
    model_cfg = load_yaml_as_ns(cfg.model) if isinstance(cfg.model, str) else cfg.model
    model = build_model(model_cfg).to(device)

    # Finetuner
    task = getattr(cfg.Finetune, "task", None)
    if task is None:
        raise ValueError("cfg.Finetune.task not found")
    task = task.lower()

    if task == "recognition":
        finetuner = RecognitionFinetuner(
            cfg=cfg,
            model=model,
            dataset=train_loader.dataset,
            device=device
        )

    elif task == "translation":
        finetuner = TranslationFinetuner(
            cfg=cfg,
            model=model,
            dataset=train_loader.dataset,
            device=device
        )

    elif task == "retrieval":
        if RetrievalFinetuner is None:
            raise ImportError("RetrievalFinetuner not found. Create finetuner/retrieval_finetuner.py first.")
        finetuner = RetrievalFinetuner(
            cfg=cfg,
            model=model,
            dataset=train_loader.dataset,
            device=device
        )

    # elif task == "segment":
    #     if SegmentFinetuner is None:
    #         raise ImportError("SegmentFinetuner not found. Create finetuner/segment_finetuner.py first.")
    #     finetuner = SegmentFinetuner(
    #         cfg=cfg,
    #         model=model,
    #         dataset=train_loader.dataset,
    #         device=device
    #     )

    else:
        raise ValueError(f"Unknown finetune task: {task}")

    # WandB
    if getattr(cfg, "wandb", None) and cfg.wandb.use and WANDB_AVAILABLE:
        wandb.init(project=cfg.wandb.project, name=cfg.wandb.run_name)
        # wandb.config.update(cfg) may fail for namespaces; safest:
        try:
            wandb.config.update(vars(cfg))
        except Exception:
            pass

    # Train loop
    epochs = int(cfg.Training.epochs)

    for epoch in range(1, epochs + 1):
        print(f"\n===== Epoch {epoch}/{epochs} | task={task} =====")

        train_out = call_maybe_with_loader(finetuner.train_epoch, train_loader)
        eval_out = call_maybe_with_loader(finetuner.eval_epoch, eval_loader)

        train_metrics = normalize_metrics(train_out, default_key="train/loss")
        eval_metrics = normalize_metrics(eval_out, default_key="eval/loss")

        # Print in a robust way (retrieval may not return loss)
        print(f"[Epoch {epoch}] Train metrics: {train_metrics}")
        print(f"[Epoch {epoch}] Eval  metrics: {eval_metrics}")

        # WandB log
        if WANDB_AVAILABLE and getattr(cfg, "wandb", None) and cfg.wandb.use:
            flat = {"epoch": epoch}
            # Prefix keys if user returned raw metric keys
            for k, v in train_metrics.items():
                flat[k if k.startswith("train/") else f"train/{k}"] = v
            for k, v in eval_metrics.items():
                flat[k if k.startswith("eval/") else f"eval/{k}"] = v
            wandb.log(flat)

        # Save best (compatible with both legacy and dict-based)
        # Pass eval_out directly (could be float or dict); finetuner decides what "best" means.
        save_if_best_compat(finetuner, eval_out, epoch)


if __name__ == "__main__":
    main()
