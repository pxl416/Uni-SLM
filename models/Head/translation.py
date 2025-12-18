# models/Head/translation.py

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput


class TranslationHead(nn.Module):
    """
    Translation head using MT5.
    Supports:
      - mode="train": teacher-forcing, return loss
      - mode="eval": generation, return pred_text
    """

    def __init__(self, cfg, hidden_dim: int):
        super().__init__()

        self.cfg = cfg
        self.hidden_dim = hidden_dim
        model_path = getattr(cfg, "model_path", "google/mt5-base")

        try:
            from transformers import MT5ForConditionalGeneration, T5Tokenizer

            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
            self.mt5 = MT5ForConditionalGeneration.from_pretrained(model_path)
            self.d_model = self.mt5.config.d_model

            self.video_proj = nn.Linear(hidden_dim, self.d_model)
            self.use_mt5 = True

            print(f"[Info] TranslationHead: MT5 loaded from {model_path}")

        except Exception as e:
            self.mt5 = None
            self.tokenizer = None
            self.video_proj = nn.Linear(hidden_dim, hidden_dim)
            self.d_model = hidden_dim
            self.use_mt5 = False
            print(f"[Warning] MT5 not available: using dummy head ({e})")

        self.max_target_len = getattr(cfg, "max_target_len", 128)
        self.num_beams = getattr(cfg, "num_beams", 4)
        self.prompt = getattr(cfg, "prompt", "Translate sign language video to Chinese:")

    def forward(self, rgb_feat: torch.Tensor, batch: dict, mode: str = "train"):
        """
        rgb_feat: (B, T, D)
        batch:
          - train: must contain text_input_ids
        """

        # ---------- Dummy fallback ----------
        if not self.use_mt5:
            pooled = rgb_feat.mean(dim=1)
            return {
                "video_repr": self.video_proj(pooled),
                "mt5_used": False
            }

        device = rgb_feat.device
        B, T, D = rgb_feat.shape

        # 1) Project video features
        video_enc = self.video_proj(rgb_feat)  # (B, T, d_model)
        encoder_outputs = BaseModelOutput(last_hidden_state=video_enc)

        # ================= TRAIN =================
        if mode == "train":
            if "text_input_ids" not in batch:
                raise ValueError("batch must contain 'text_input_ids' for translation training")

            labels = batch["text_input_ids"].to(device)

            outputs = self.mt5(
                encoder_outputs=encoder_outputs,
                labels=labels,
            )

            return {
                "loss": outputs.loss,
                "mt5_used": True
            }

        # ================= EVAL / GENERATION =================
        elif mode == "eval":
            prompt_tokens = self.tokenizer(
                [self.prompt] * B,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            outputs = self.mt5.generate(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=prompt_tokens.input_ids,
                max_length=self.max_target_len,
                num_beams=self.num_beams,
            )

            decoded = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            return {
                "pred_text": decoded,
                "mt5_used": True
            }

        else:
            raise ValueError(f"Unknown mode: {mode}")
