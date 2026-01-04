# pretrain.py
import argparse
import torch
from types import SimpleNamespace

from utils.config import load_yaml_as_ns
from datasets.datasets import create_dataloader
from models.build_model import build_model

from pretrainer.temporal_pretrainer import TemporalPretrainer

try:
    import wandb
    WANDB_AVAILABLE = True
except:
    WANDB_AVAILABLE = False


# Args
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/pretrain.yaml")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="0")
    return parser.parse_args()


# Main
def main():
    args = parse_args()

    # Load config
    cfg = load_yaml_as_ns(args.config)

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )
    cfg.device = device
    print(f"[Config] Using device: {device}")

    # Default Training Config (safe fallback)
    default_training = SimpleNamespace(
        epochs=10,
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

    print(
        f"[Config] epochs={cfg.Training.epochs}, "
        f"batch_size={cfg.Training.batch_size}"
    )

    # Dataloader
    train_loader = create_dataloader(
        args, cfg, phase=cfg.Pretrain.train_split
    )
    eval_loader = create_dataloader(
        args, cfg, phase=cfg.Pretrain.eval_split
    )

    # Model
    # Resolve model yaml path → namespace
    cfg.model = load_yaml_as_ns(cfg.model)
    model = build_model(cfg).to(device)

    # Pretrainer
    pretrainer = TemporalPretrainer(
        cfg=cfg,
        model=model,
        device=device,
    )

    # WandB
    if getattr(cfg, "wandb", None) and cfg.wandb.use and WANDB_AVAILABLE:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
        )
        wandb.config.update(cfg)

    # Train Loop
    best_loss = float("inf")

    for epoch in range(cfg.Training.epochs):
        print(
            f"\n===== Epoch {epoch + 1}/{cfg.Training.epochs} | "
            f"task=temporal_pretrain ====="
        )

        train_loss = pretrainer.train_epoch(train_loader)
        eval_loss = pretrainer.eval_epoch(eval_loader)

        print(
            f"[Epoch {epoch + 1}] "
            f"train_loss={train_loss:.4f} | "
            f"eval_loss={eval_loss:.4f}"
        )

        # WandB log
        if WANDB_AVAILABLE and getattr(cfg, "wandb", None) and cfg.wandb.use:
            wandb.log({
                "train/loss": train_loss,
                "eval/loss": eval_loss,
                "epoch": epoch + 1,
            })

        # Save best
        if eval_loss < best_loss:
            best_loss = eval_loss
            pretrainer.save_checkpoint(
                tag="best",
                epoch=epoch + 1,
                metric=best_loss,
            )

    print("[Done] Temporal pretraining finished.")


if __name__ == "__main__":
    main()
