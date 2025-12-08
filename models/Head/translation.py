# models/Head/translation.py
import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput


class TranslationHead(nn.Module):
    """
    Translation head using MT5. Minimal but correct bridging:
    video_feat -> mapped to d_model -> used as encoder hidden states
    """
    def __init__(self, cfg, hidden_dim: int):
        super().__init__()

        self.cfg = cfg
        self.hidden_dim = hidden_dim
        model_path = getattr(cfg, "model_path", "google/mt5-base")

        # Try loading MT5
        try:
            from transformers import MT5ForConditionalGeneration, T5Tokenizer

            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
            self.mt5 = MT5ForConditionalGeneration.from_pretrained(model_path)
            self.d_model = self.mt5.config.d_model

            # Map video feat -> MT5 encoder dim
            self.video_proj = nn.Linear(hidden_dim, self.d_model)

            self.use_mt5 = True
            print(f"[Info] TranslationHead: MT5 loaded from {model_path}")

        except Exception as e:
            # dummy fallback
            self.mt5 = None
            self.tokenizer = None
            self.video_proj = nn.Linear(hidden_dim, hidden_dim)
            self.d_model = hidden_dim
            self.use_mt5 = False
            print(f"[Warning] MT5 not available: using dummy head ({e})")

        self.max_target_len = getattr(cfg, "max_target_len", 128)
        self.num_beams = getattr(cfg, "num_beams", 4)
        self.prompt = getattr(cfg, "prompt", "Translate sign language video to Chinese:")

    def forward(self, rgb_feat: torch.Tensor, batch: dict):
        """
        rgb_feat: (B, T, D)
        """

        # -------- Dummy fallback ----------
        if not self.use_mt5:
            pooled = rgb_feat.mean(dim=1)
            return {"video_repr": self.video_proj(pooled), "mt5_used": False}

        B, T, D = rgb_feat.shape
        device = rgb_feat.device

        # -------- 1) Project video features to MT5 encoder dimension ----------
        video_enc = self.video_proj(rgb_feat)  # (B, T, d_model)

        # -------- 2) Use prompt as decoder input ----------
        prompt_tokens = self.tokenizer(
            [self.prompt] * B,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        # -------- 3) Build BaseModelOutput for MT5 encoder ----------
        encoder_outputs = BaseModelOutput(last_hidden_state=video_enc)

        # -------- 4) Run generation ----------
        outputs = self.mt5.generate(
            encoder_outputs=encoder_outputs,
            max_length=self.max_target_len,
            num_beams=self.num_beams,
            decoder_input_ids=prompt_tokens.input_ids,
        )

        decoded = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return {"pred_text": decoded, "mt5_used": True}
