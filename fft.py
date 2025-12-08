# fft.py (finetune.py)
import argparse
import torch
import os
from types import SimpleNamespace

from utils.config import load_yaml_as_ns
from datasets.datasets import create_dataloader
from models.build_model import build_model

from finetuner.recognition_finetuner import RecognitionFinetuner
#from finetuner.retrieval_finetuner import RetrievalFinetuner
#from finetuner.translation_finetuner import TranslationFinetuner

try:
    import wandb
    WANDB_AVAILABLE = True
except:
    WANDB_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/ft.yaml")
    parser.add_argument("--epochs", type=int, default=3)
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

    task = cfg.Finetune.task.lower()
    if task == "recognition":
        # finetuner = RecognitionFinetuner(cfg, model, device)
        train_dataset = train_loader.dataset  # 取 dataset，用于 gloss2id

        finetuner = RecognitionFinetuner(
            cfg=cfg,
            model=model,
            dataset=train_dataset,
            device=cfg.device
        )

    # elif task == "retrieval":
    #     finetuner = RetrievalFinetuner(cfg, model, device)
    # elif task == "translation":
    #     finetuner = TranslationFinetuner(cfg, model, device)
    else:
        raise ValueError(f"Unknown finetune task: {task}")

    # ===== WANDB =====
    if getattr(cfg, "wandb", None) and cfg.wandb.use and WANDB_AVAILABLE:
        wandb.init(project=cfg.wandb.project, name=cfg.wandb.run_name)
        wandb.config.update(cfg)

    # ===== TRAIN LOOP =====
    for epoch in range(cfg.Training.epochs):
        print(f"\n===== Epoch {epoch+1}/{cfg.Training.epochs} =====")

        train_loss = finetuner.train_epoch(train_loader)
        eval_loss = finetuner.eval_epoch(eval_loader)

        print(f"[Epoch {epoch+1}] train_loss={train_loss:.4f} | eval_loss={eval_loss:.4f}")

        if WANDB_AVAILABLE and getattr(cfg, "wandb", None) and cfg.wandb.use:
            wandb.log({"train_loss": train_loss, "eval_loss": eval_loss})

        finetuner.save_checkpoint(os.path.join(cfg.save_dir, f"epoch_{epoch+1}.pt"))


if __name__ == "__main__":
    main()
