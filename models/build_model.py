# models/build_model.py

import torch
import torch.nn as nn

from models.Encoder.rgb_encoder import RGBEncoder
from models.Encoder.pose_encoder import PoseEncoder
from models.Encoder.text_encoder import TextEncoder

from models.Head.recognition import RecognitionHead
from models.Head.retrieval import RetrievalHead
from models.Head.translation import TranslationHead

# Optional pretext head (only used if enabled in cfg)
from models.Head.temporal_heatmap import TemporalHeatmapHead


def load_pretrained_submodule(submodule, path: str, strict: bool = False):
    if path is None:
        return
    state = torch.load(path, map_location="cpu")
    missing, unexpected = submodule.load_state_dict(state, strict=strict)
    print(
        f"[Pretrained] Loaded {path} | "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )


def requires_mask(backbone_name: str):
    name = backbone_name.lower()
    if any(x in name for x in ["i3d", "resnet3d", "r3d", "3d", "slowfast"]):
        return False
    if any(x in name for x in ["vit", "transformer", "timesformer", "swin"]):
        return True
    return False


def build_rgb_encoder(cfg, hidden_dim):
    enc = RGBEncoder(cfg, hidden_dim)
    enc.use_mask = requires_mask(cfg.backbone.name)
    return enc


def build_pose_encoder(cfg, hidden_dim):
    return PoseEncoder(cfg, hidden_dim)


def build_text_encoder(cfg, hidden_dim):
    return TextEncoder(cfg, hidden_dim)


# -----------------------------
# Compat helpers (SAFE fallback)
# -----------------------------
def _resolve_hidden_dim(cfg):
    """
    Backward compatible:
      - old finetune: cfg.model.hidden_dim
      - new/pretrain: cfg.model.model.hidden_dim
    """
    if hasattr(cfg, "model") and hasattr(cfg.model, "hidden_dim"):
        return cfg.model.hidden_dim
    if hasattr(cfg, "model") and hasattr(cfg.model, "model") and hasattr(cfg.model.model, "hidden_dim"):
        return cfg.model.model.hidden_dim
    raise AttributeError("Cannot find hidden_dim in cfg.model.hidden_dim or cfg.model.model.hidden_dim")


def _resolve_encoder_cfg(cfg, name: str):
    """
    Backward compatible:
      - old finetune: cfg.rgb_encoder / cfg.pose_encoder / cfg.text_encoder
      - future: cfg.model.rgb_encoder ...
    """
    if hasattr(cfg, name):
        return getattr(cfg, name)
    if hasattr(cfg, "model") and hasattr(cfg.model, name):
        return getattr(cfg.model, name)
    if hasattr(cfg, "model") and hasattr(cfg.model, "model") and hasattr(cfg.model.model, name):
        return getattr(cfg.model.model, name)
    raise AttributeError(f"Missing encoder config: {name}")


class MultiModalModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # ✅ This keeps finetune behavior identical when cfg.model.hidden_dim exists
        self.hidden_dim = _resolve_hidden_dim(cfg)

        # ✅ Prefer old finetune layout (top-level encoders)
        rgb_cfg = _resolve_encoder_cfg(cfg, "rgb_encoder")
        pose_cfg = _resolve_encoder_cfg(cfg, "pose_encoder")
        text_cfg = _resolve_encoder_cfg(cfg, "text_encoder")

        self.rgb_encoder = build_rgb_encoder(rgb_cfg, self.hidden_dim)
        self.pose_encoder = build_pose_encoder(pose_cfg, self.hidden_dim)
        self.text_encoder = build_text_encoder(text_cfg, self.hidden_dim)

        # (Optional) unified view for pretraining convenience; does not affect finetune
        self.encoders = {
            "rgb": self.rgb_encoder,
            "pose": self.pose_encoder,
            "text": self.text_encoder,
        }
        self.encoder = self.rgb_encoder

        # ---------------- Heads (unchanged) ----------------
        # Retrieval
        if hasattr(cfg, "retrieval_head"):
            self.retrieval_head = RetrievalHead(cfg.retrieval_head, self.hidden_dim)
        else:
            self.retrieval_head = None

        # Recognition
        if hasattr(cfg, "recognition_head"):
            num_classes = getattr(cfg.recognition_head, "num_classes", 5000)
            self.recognition_head = RecognitionHead(
                hidden_dim=self.hidden_dim,
                num_classes=num_classes,
                blank_id=getattr(cfg.recognition_head, "blank_id", 0),
            )
        else:
            self.recognition_head = None

        # Translation
        if hasattr(cfg, "translation_head"):
            self.translation_head = TranslationHead(
                cfg.translation_head,
                self.hidden_dim
            )
        else:
            self.translation_head = None

        # ---------------- Temporal head (new, optional) ----------------
        tem_cfg = getattr(cfg, "temporal_head", None)
        if tem_cfg is not None and getattr(tem_cfg, "enabled", True):
            self.temporal_head = TemporalHeatmapHead(
                hidden_dim=self.hidden_dim,
                dropout=getattr(tem_cfg, "dropout", 0.1),
                use_conv=getattr(tem_cfg, "use_conv", False),
            )
        else:
            self.temporal_head = None

    def forward(self, batch, task: str):
        if task is None:
            raise RuntimeError("Must specify task")

        if getattr(self.rgb_encoder, "use_mask", False):
            video = self.rgb_encoder(
                batch["rgb_img"],
                batch.get("rgb_mask", None)
            )
        else:
            video = self.rgb_encoder(batch["rgb_img"])

        # ✅ Original finetune behavior (unchanged)

        if task == "retrieval":
            if self.retrieval_head is None:
                raise RuntimeError("retrieval_head is not initialized (finetune-only head)")
            text = self.text_encoder(batch["gt_sentence"])
            return self.retrieval_head(video, text)


        elif task == "recognition":
            return self.recognition_head(video)

        elif task == "translation":
            return {"rgb_feat": video}

        # ✅ New task branch only used by pretrain/test
        elif task == "temporal":
            if self.temporal_head is None:
                raise RuntimeError("temporal_head is not enabled in config")
            temporal_logits = self.temporal_head(video)  # (B, T)
            return {
                "rgb_feat": video,
                "temporal_logits": temporal_logits
            }

        else:
            raise ValueError(f"Unknown task: {task}")

def build_model(cfg):
    model = MultiModalModel(cfg)

    # -------- Encoders (UNCHANGED) --------
    rgb_cfg = _resolve_encoder_cfg(cfg, "rgb_encoder")
    load_pretrained_submodule(
        model.rgb_encoder.backbone,
        rgb_cfg.backbone.pretrained_path,
        strict=False
    )
    if getattr(rgb_cfg, "proj", None) and rgb_cfg.proj.enabled:
        load_pretrained_submodule(
            model.rgb_encoder.proj,
            rgb_cfg.proj.pretrained_path,
            strict=False
        )

    pose_cfg = _resolve_encoder_cfg(cfg, "pose_encoder")
    load_pretrained_submodule(
        model.pose_encoder.backbone,
        pose_cfg.backbone.pretrained_path,
        strict=False
    )
    if pose_cfg.proj.enabled:
        load_pretrained_submodule(
            model.pose_encoder.proj,
            pose_cfg.proj.pretrained_path,
            strict=False
        )

    text_cfg = _resolve_encoder_cfg(cfg, "text_encoder")
    load_pretrained_submodule(
        model.text_encoder.backbone,
        text_cfg.backbone.pretrained_path,
        strict=False
    )
    if text_cfg.proj.enabled:
        load_pretrained_submodule(
            model.text_encoder.proj,
            text_cfg.proj.pretrained_path,
            strict=False
        )

    # -------- Fusion (UNCHANGED) --------
    if getattr(cfg, "fusion", None) and cfg.fusion.type != "none":
        load_pretrained_submodule(
            model.fusion,
            cfg.fusion.pretrained_path,
            strict=False
        )

    # -------- Temporal head (PRETRAIN ONLY) --------
    if (
        model.temporal_head is not None
        and hasattr(cfg, "temporal_head")
        and getattr(cfg.temporal_head, "resume_path", None)
    ):
        load_pretrained_submodule(
            model.temporal_head,
            cfg.temporal_head.resume_path,
            strict=False
        )

    # -------- Retrieval head (FINETUNE ONLY) --------
    if (
        model.retrieval_head is not None
        and hasattr(cfg, "retrieval_head")
        and getattr(cfg.retrieval_head, "pretrained_path", None)
    ):
        load_pretrained_submodule(
            model.retrieval_head,
            cfg.retrieval_head.pretrained_path,
            strict=False
        )

    # -------- Recognition head (FINETUNE ONLY) --------
    if (
        model.recognition_head is not None
        and hasattr(cfg, "recognition_head")
        and getattr(cfg.recognition_head, "pretrained_path", None)
    ):
        load_pretrained_submodule(
            model.recognition_head,
            cfg.recognition_head.pretrained_path,
            strict=False
        )

    # -------- Translation head (FINETUNE ONLY) --------
    if (
        model.translation_head is not None
        and hasattr(cfg, "translation_head")
        and getattr(cfg.translation_head, "resume_path", None)
    ):
        load_pretrained_submodule(
            model.translation_head,
            cfg.translation_head.resume_path,
            strict=False
        )

    return model



# def build_model(cfg):
#     model = MultiModalModel(cfg)
#
#     # ✅ Keep EXACTLY your original loading behavior (top-level cfg.* preferred)
#     rgb_cfg = _resolve_encoder_cfg(cfg, "rgb_encoder")
#     load_pretrained_submodule(
#         model.rgb_encoder.backbone,
#         rgb_cfg.backbone.pretrained_path,
#         strict=False
#     )
#     if getattr(rgb_cfg, "proj", None) and rgb_cfg.proj.enabled:
#         load_pretrained_submodule(
#             model.rgb_encoder.proj,
#             rgb_cfg.proj.pretrained_path,
#             strict=False
#         )
#
#     pose_cfg = _resolve_encoder_cfg(cfg, "pose_encoder")
#     load_pretrained_submodule(
#         model.pose_encoder.backbone,
#         pose_cfg.backbone.pretrained_path,
#         strict=False
#     )
#     if pose_cfg.proj.enabled:
#         load_pretrained_submodule(
#             model.pose_encoder.proj,
#             pose_cfg.proj.pretrained_path,
#             strict=False
#         )
#
#     text_cfg = _resolve_encoder_cfg(cfg, "text_encoder")
#     load_pretrained_submodule(
#         model.text_encoder.backbone,
#         text_cfg.backbone.pretrained_path,
#         strict=False
#     )
#     if text_cfg.proj.enabled:
#         load_pretrained_submodule(
#             model.text_encoder.proj,
#             text_cfg.proj.pretrained_path,
#             strict=False
#         )
#
#     # fusion (unchanged)
#     if getattr(cfg, "fusion", None) and cfg.fusion.type != "none":
#         load_pretrained_submodule(
#             model.fusion,
#             cfg.fusion.pretrained_path,
#             strict=False
#         )
#
#     # temporal head resume (optional)
#     if getattr(cfg, "temporal_head", None) and getattr(cfg.temporal_head, "resume_path", None):
#         load_pretrained_submodule(
#             model.temporal_head,
#             cfg.temporal_head.resume_path,
#             strict=False
#         )
#
#     # heads (unchanged)
#     load_pretrained_submodule(
#         model.retrieval_head,
#         cfg.retrieval_head.pretrained_path,
#         strict=False
#     )
#
#     if getattr(cfg, "recognition_head", None):
#         load_pretrained_submodule(
#             model.recognition_head,
#             cfg.recognition_head.pretrained_path,
#             strict=False
#         )
#
#     load_pretrained_submodule(
#         model.translation_head,
#         cfg.translation_head.resume_path,
#         strict=False
#     )
#
#     return model
