# models/build_model.py
import torch
import torch.nn as nn

# Encoders
from models.Encoder.rgb_encoder import RGBEncoder
from models.Encoder.pose_encoder import PoseEncoder
from models.Encoder.text_encoder import TextEncoder

# Heads
from models.Head.retrieval import RetrievalHead
from models.Head.translation import TranslationHead
from models.Head.recognition import RecognitionHead


def load_pretrained_submodule(submodule, path: str, strict: bool = False):
    """
    Load a 'pure' state_dict into a submodule.
    The .pt file should contain keys matching submodule.state_dict().
    """
    if path is None:
        return
    state = torch.load(path, map_location="cpu")
    missing, unexpected = submodule.load_state_dict(state, strict=strict)

    print(
        f"[Pretrained] Loaded {path} | "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )


# 判断是否需要 mask（用于 transformer 视频模型）
def requires_mask(backbone_name: str):
    name = backbone_name.lower()
    # CNN / 3D卷积 → 不需要 mask
    if any(x in name for x in ["i3d", "resnet3d", "r3d", "3d", "slowfast"]):
        return False
    # Transformer 视频模型 → 需要 mask
    if any(x in name for x in ["vit", "transformer", "timesformer", "swin"]):
        return True
    return False


# Encoders builder
def build_rgb_encoder(cfg, hidden_dim):
    enc = RGBEncoder(cfg, hidden_dim)
    enc.use_mask = requires_mask(cfg.backbone.name)
    return enc


def build_pose_encoder(cfg, hidden_dim):
    return PoseEncoder(cfg, hidden_dim)


def build_text_encoder(cfg, hidden_dim):
    return TextEncoder(cfg, hidden_dim)


# Multi-modal model container
class MultiModalModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.hidden_dim = cfg.model.hidden_dim

        # Encoders
        self.rgb_encoder  = build_rgb_encoder(cfg.rgb_encoder,  self.hidden_dim)
        self.pose_encoder = build_pose_encoder(cfg.pose_encoder, self.hidden_dim)
        self.text_encoder = build_text_encoder(cfg.text_encoder, self.hidden_dim)

        # Heads
        self.retrieval_head = RetrievalHead(cfg.retrieval_head, self.hidden_dim)

        # ❗ RecognitionHead 延迟初始化（由 RecognitionFinetuner 负责构建）
        self.recognition_head = None

        self.translation_head = TranslationHead(cfg.translation_head, self.hidden_dim)

    # ----------------------------------------------------------
    def forward(self, batch, task: str = None):
        if task is None:
            raise RuntimeError("❌ Must specify task='recognition'/'retrieval'/'translation'")

        # RGB Encoder (with or without mask)
        if getattr(self.rgb_encoder, "use_mask", False):
            video = self.rgb_encoder(batch["rgb_img"], batch.get("rgb_mask", None))
        else:
            video = self.rgb_encoder(batch["rgb_img"])

        # Task branches
        if task == "retrieval":
            text = self.text_encoder(batch["gt_sentence"])
            return self.retrieval_head(video, text)

        elif task == "recognition":
            if self.recognition_head is None:
                raise RuntimeError(
                    "RecognitionHead not initialized. "
                    "Use RecognitionFinetuner to attach the head."
                )
            return self.recognition_head(video)

        # elif task == "translation":
        #     return self.translation_head(video, batch)
        elif task == "translation":
            return {
                "rgb_feat": video
            }


        else:
            raise ValueError(f"Unknown task: {task}")


# ===============================================================

def build_model(cfg):
    model = MultiModalModel(cfg)

    # ---------------- RGB Encoder ----------------
    rgb_cfg = cfg.rgb_encoder

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

    # ---------------- Pose Encoder ----------------
    pose_cfg = cfg.pose_encoder

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

    # ---------------- Text Encoder ----------------
    text_cfg = cfg.text_encoder

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

    # ---------------- Fusion ----------------
    if getattr(cfg, "fusion", None) and cfg.fusion.type != "none":
        load_pretrained_submodule(
            model.fusion,
            cfg.fusion.pretrained_path,
            strict=False
        )

    # ---------------- Heads (optional) ----------------
    # Retrieval head
    load_pretrained_submodule(
        model.retrieval_head,
        cfg.retrieval_head.pretrained_path,
        strict=False
    )

    # Translation head（如果你希望支持 resume）
    load_pretrained_submodule(
        model.translation_head,
        cfg.translation_head.resume_path,
        strict=False
    )

    return model

