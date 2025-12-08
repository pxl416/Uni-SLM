# finetune.py
import os
import yaml
import torch
from types import SimpleNamespace

from utils.config import load_yaml_as_ns, dict_to_ns
from datasets.datasets import create_dataloader
from models.build_model import build_model

from finetuner.recognition_finetuner import RecognitionFinetuner


import argparse

def parse_args():
    parser = argparse.ArgumentParser("Uni-SLM Finetuning")

    parser.add_argument("--config", type=str, default="config/ft.yaml",
                        help="Path to finetune config yaml")

    # 覆盖 Training 里的 epochs、batch_size
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")

    # 指定 GPU，例如 0 → cuda:0
    parser.add_argument("--device", type=str, default="0",
                        help="Device id, e.g. 0 or cuda:0 or cpu")

    return parser.parse_args()

def parse_device(device_str):
    """
    device_str: "0", "cuda:0", "cpu"
    """
    if device_str.lower() == "cpu":
        return torch.device("cpu")

    if device_str.isdigit():
        return torch.device(f"cuda:{device_str}")

    if device_str.startswith("cuda"):
        return torch.device(device_str)

    raise ValueError(f"Invalid device string: {device_str}")

def safe_init_training(cfg, args):
    if not hasattr(cfg, "Training"):
        cfg.Training = SimpleNamespace()

    if not hasattr(cfg.Training, "epochs"):
        cfg.Training.epochs = args.epochs

    if not hasattr(cfg.Training, "batch_size"):
        cfg.Training.batch_size = args.batch_size

    if not hasattr(cfg.Training, "num_workers"):
        cfg.Training.num_workers = 4   # default

    return cfg


def main():
    args = parse_args()
    cfg = load_yaml_as_ns(args.config)

    cfg = safe_init_training(cfg, args)

    cfg.device = parse_device(args.device)
    print(f"[Config] Using device: {cfg.device}")
    print(f"[Config] epochs={cfg.Training.epochs}, batch_size={cfg.Training.batch_size}")

    train_loader = create_dataloader(args, cfg, phase=cfg.Finetune.train_split)
    val_loader   = create_dataloader(args, cfg, phase=cfg.Finetune.eval_split)

    model_cfg = load_yaml_as_ns(cfg.model)
    model = build_model(model_cfg)

    task = cfg.Finetune.task
    if task == "recognition":
        from finetuner.recognition_finetuner import RecognitionFinetuner
        finetuner = RecognitionFinetuner(cfg, model, cfg.device)
    # elif task == "retrieval":
    #     from finetuner.retrieval_finetuner import RetrievalFinetuner
    #     finetuner = RetrievalFinetuner(cfg, model, cfg.device)
    # elif task == "translation":
    #     from finetuner.translation_finetuner import TranslationFinetuner
    #     finetuner = TranslationFinetuner(cfg, model, cfg.device)
    else:
        raise NotImplementedError(task)

    for epoch in range(cfg.Training.epochs):
        train_loss = finetuner.train_epoch(train_loader)
        val_loss   = finetuner.eval_epoch(val_loader)

        print(f"Epoch {epoch+1}: train={train_loss:.4f} | val={val_loss:.4f}")

        finetuner.save_checkpoint("last.pt")
        if finetuner.best_metric is None or val_loss < finetuner.best_metric:
            finetuner.best_metric = val_loss
            finetuner.save_checkpoint("best.pt")



if __name__ == "__main__":
    main()
