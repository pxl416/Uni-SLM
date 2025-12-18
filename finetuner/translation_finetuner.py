# finetuner/translation_finetuner.py

import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
import sacrebleu
from rouge_score import rouge_scorer


from finetuner.base_finetuner import BaseFinetuner
from models.Head.translation import TranslationHead
from utils.optimizer import build_optimizer


class TranslationFinetuner(BaseFinetuner):
    """
    Finetuner for Sign Language Translation
    RGB video -> encoder -> MT5 decoder
    """

    def __init__(self, cfg, model, dataset, device):
        """
        model: can be None (future-proof)
        dataset: not required for translation, but kept for unified interface
        """

        # allow model=None
        if model is None:
            from utils.config import load_yaml_as_ns
            from models.build_model import build_model
            model = build_model(load_yaml_as_ns(cfg.model)).to(device)

        super().__init__(cfg, model, device)

        self.use_amp = False

        # ===== Translation Head =====
        self.model.translation_head = TranslationHead(
            cfg=cfg,
            hidden_dim=self.model.hidden_dim
        ).to(device)

        self.tokenizer = self.model.translation_head.tokenizer

        # ===== Optimizer =====
        train_cfg = getattr(cfg, "Training", None)
        if train_cfg is None:
            raise ValueError("cfg.Training not found")

        self.optimizer, self.scheduler = build_optimizer(self.model, train_cfg)
        self.grad_clip = getattr(train_cfg, "grad_clip", 1.0)

        self.best_eval_loss = float("inf")

        print("[Info] TranslationFinetuner initialized.")

    # ------------------------------------------------
    # helper: get translation text safely
    # ------------------------------------------------
    def _get_text_from_tgt(self, tgt):
        for key in ["gt_sentence", "gt_text", "text", "sentence", "translation"]:
            if key in tgt:
                return tgt[key]
        raise ValueError(
            f"No translation sentence found in tgt keys: {list(tgt.keys())}"
        )

    # ------------------------------------------------
    # save best checkpoint
    # ------------------------------------------------
    def save_if_best(self, eval_loss, epoch):
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            filename = f"best_epoch_{epoch}.pt"
            self.save_checkpoint(filename)
            print(f"[Checkpoint] Saved best model → {filename}")

    # ------------------------------------------------
    # Train Loop
    # ------------------------------------------------
    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(loader, desc="Train-Translation", ncols=100)

        for src, tgt in pbar:
            # src: tensor dict → device
            src = self._move_batch_to_device(src)
            # tgt: contains strings, DO NOT move whole tgt to device

            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                # encoder forward
                out = self.model(batch=src, task="translation")
                rgb_feat = out["rgb_feat"]  # (B, T, D)

                # -------- prepare text_input_ids --------
                if "text_input_ids" not in tgt:
                    texts = self._get_text_from_tgt(tgt)  # List[str]

                    tokenized = self.tokenizer(
                        texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=getattr(self.cfg, "max_target_len", 128),
                    )

                    tgt["text_input_ids"] = tokenized.input_ids.to(self.device)
                    tgt["attention_mask"] = tokenized.attention_mask.to(self.device)

                # -------- translation forward (teacher forcing) --------
                trans_out = self.model.translation_head(
                    rgb_feat=rgb_feat,
                    batch=tgt,
                    mode="train"
                )

                loss = trans_out["loss"]

            # backward
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            self.global_step += 1

            del loss, rgb_feat
            torch.cuda.empty_cache()

        return total_loss / len(loader)

    # ------------------------------------------------
    # Eval Loop
    # ------------------------------------------------
    def eval_epoch(self, loader):
        self.model.eval()
        total_loss = 0.0

        all_preds = []
        all_gts = []

        pbar = tqdm(loader, desc="Eval-Translation", ncols=100)

        with torch.no_grad():
            for src, tgt in pbar:
                src = self._move_batch_to_device(src)
                # tgt stays on CPU

                out = self.model(batch=src, task="translation")
                rgb_feat = out["rgb_feat"]

                # -------- loss (teacher forcing) --------
                if "text_input_ids" not in tgt:
                    texts = self._get_text_from_tgt(tgt)

                    tokenized = self.tokenizer(
                        texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=getattr(self.cfg, "max_target_len", 128),
                    )

                    tgt["text_input_ids"] = tokenized.input_ids.to(self.device)
                    tgt["attention_mask"] = tokenized.attention_mask.to(self.device)

                trans_out = self.model.translation_head(
                    rgb_feat=rgb_feat,
                    batch=tgt,
                    mode="train"
                )
                loss = trans_out["loss"]
                total_loss += loss.item()

                # -------- generation --------
                gen_out = self.model.translation_head(
                    rgb_feat=rgb_feat,
                    batch=tgt,
                    mode="eval"
                )

                preds = gen_out["pred_text"]
                gts = self._get_text_from_tgt(tgt)

                all_preds.extend(preds)
                all_gts.extend(gts)

        avg_loss = total_loss / len(loader)

        if len(all_preds) > 0 and len(all_gts) > 0:
            # -------- BLEU --------
            bleu = sacrebleu.corpus_bleu(all_preds, [all_gts])

            # -------- ROUGE --------
            scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"],
                use_stemmer=False
            )

            r1, r2, rL = 0.0, 0.0, 0.0
            n = len(all_preds)

            for pred, gt in zip(all_preds, all_gts):
                scores = scorer.score(gt, pred)
                r1 += scores["rouge1"].fmeasure
                r2 += scores["rouge2"].fmeasure
                rL += scores["rougeL"].fmeasure

            r1 /= n
            r2 /= n
            rL /= n

            print(
                f"[Eval] Translation samples: {n} | "
                f"BLEU={bleu.score:.2f} | "
                f"ROUGE-1={r1:.4f} ROUGE-2={r2:.4f} ROUGE-L={rL:.4f}"
            )
        else:
            print("[Eval] No valid translation pairs, skip metrics.")

        print("[Eval Sample]")
        print("GT  :", all_gts[0])
        print("PRED:", all_preds[0])

        return avg_loss


