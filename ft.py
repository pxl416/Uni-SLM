# finetune.py
import argparse
import torch
from types import SimpleNamespace

from utils.config import load_yaml_as_ns
from datasets.datasets import create_dataloader
from models.build_model import build_model

from finetuner.recognition_finetuner import RecognitionFinetuner
from finetuner.translation_finetuner import TranslationFinetuner
from finetuner.retrieval_finetuner import RetrievalFinetuner

try:
    import wandb
    WANDB_AVAILABLE = True
except:
    WANDB_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/ft.yaml")
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="0")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_yaml_as_ns(args.config)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    cfg.device = device
    print(f"[Config] Using device: {device}")

    # ===== DEFAULT TRAINING CONFIG =====
    default_training = SimpleNamespace(
        epochs=3,
        batch_size=4,
        lr=1e-4,
        grad_clip=1.0,
        num_workers=4,
    )

    cfg.Training = getattr(cfg, "Training", default_training)

    if args.epochs is not None:
        cfg.Training.epochs = args.epochs
    if args.batch_size is not None:
        cfg.Training.batch_size = args.batch_size

    print(f"[Config] epochs={cfg.Training.epochs}, batch_size={cfg.Training.batch_size}")

    # ===== DATALOADER =====
    train_loader = create_dataloader(args, cfg, phase=cfg.Finetune.train_split)
    eval_loader = create_dataloader(args, cfg, phase=cfg.Finetune.eval_split)

    # ===== MODEL =====
    model = build_model(load_yaml_as_ns(cfg.model)).to(device)

    # ===== FINETUNER =====
    task = cfg.Finetune.task.lower()

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
        finetuner = RetrievalFinetuner(
            cfg=cfg,
            model=model,
            dataset=train_loader.dataset,
            device=device
        )

    else:
        raise ValueError(f"Unknown finetune task: {task}")

    # ===== WANDB =====
    if getattr(cfg, "wandb", None) and cfg.wandb.use and WANDB_AVAILABLE:
        wandb.init(project=cfg.wandb.project, name=cfg.wandb.run_name)
        wandb.config.update(cfg)

    # ===== TRAIN LOOP =====
    for epoch in range(cfg.Training.epochs):
        print(f"\n===== Epoch {epoch + 1}/{cfg.Training.epochs} | task={task} =====")

        train_out = finetuner.train_epoch(train_loader)
        eval_out = finetuner.eval_epoch(eval_loader)

        if task in ["recognition", "translation"]:
            # === legacy behavior (unchanged) ===
            train_loss = train_out
            eval_loss = eval_out

            print(
                f"[Epoch {epoch + 1}] "
                f"train_loss={train_loss:.4f} | "
                f"eval_loss={eval_loss:.4f}"
            )

            if WANDB_AVAILABLE and getattr(cfg, "wandb", None) and cfg.wandb.use:
                wandb.log({
                    "train/loss": train_loss,
                    "eval/loss": eval_loss
                })

            finetuner.save_if_best(eval_loss, epoch + 1)

        elif task == "retrieval":
            # === retrieval uses metrics dict ===
            print(f"[Epoch {epoch + 1}] Train metrics: {train_out}")
            print(f"[Epoch {epoch + 1}] Eval  metrics: {eval_out}")

            if WANDB_AVAILABLE and getattr(cfg, "wandb", None) and cfg.wandb.use:
                wandb.log({
                    **{f"train/{k}": v for k, v in train_out.items()},
                    **{f"eval/{k}": v for k, v in eval_out.items()},
                })

            finetuner.save_if_best(eval_out, epoch + 1)


if __name__ == "__main__":
    main()
