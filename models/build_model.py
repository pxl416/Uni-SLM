# models/build_model.py
import torch
import torch.nn as nn
from types import SimpleNamespace

# Encoders
from models.Encoder.rgb_encoder import RGBEncoder
from models.Encoder.pose_encoder import PoseEncoder
from models.Encoder.text_encoder import TextEncoder

# Heads
from models.Head.recognition import RecognitionHead
from models.Head.retrieval import RetrievalHead
from models.Head.translation import TranslationHead



# ===============================================================
#   Build Functions
# ===============================================================

def build_rgb_encoder(cfg, hidden_dim):
    return RGBEncoder(cfg, hidden_dim)


def build_pose_encoder(cfg, hidden_dim):
    return PoseEncoder(cfg, hidden_dim)


def build_text_encoder(cfg, hidden_dim):
    return TextEncoder(cfg, hidden_dim)



# ===============================================================
#   Multi-modal model container
# ===============================================================

class MultiModalModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.hidden_dim = cfg.model.hidden_dim

        # ------------------------------
        # Encoders
        # ------------------------------
        self.rgb_encoder  = build_rgb_encoder(cfg.rgb_encoder,  self.hidden_dim)
        self.pose_encoder = build_pose_encoder(cfg.pose_encoder, self.hidden_dim)
        self.text_encoder = build_text_encoder(cfg.text_encoder, self.hidden_dim)

        # ------------------------------
        # Heads
        # ------------------------------
        self.retrieval_head  = RetrievalHead(cfg.retrieval_head, self.hidden_dim)
        self.recognition_head = RecognitionHead(cfg.recognition_head, self.hidden_dim)
        self.translation_head = TranslationHead(cfg.translation_head, self.hidden_dim)

    # ----------------------------------------------------------
    def forward(self, batch, task="retrieval"):
        """
        task = 'retrieval' | 'recognition' | 'translation'
        """
        rgb = batch["rgb_img"]
        pose = batch["keypoints"]
        text = batch["gt_sentence"]

        rgb_feat  = self.rgb_encoder(rgb)
        pose_feat = self.pose_encoder(pose)
        text_feat = self.text_encoder(text)

        if task == "retrieval":
            return self.retrieval_head(rgb_feat, text_feat)

        if task == "recognition":
            return self.recognition_head(rgb_feat)

        if task == "translation":
            return self.translation_head(rgb_feat, batch)

        raise ValueError(f"Unknown task {task}")


# ===============================================================
def build_model(cfg):
    return MultiModalModel(cfg)
