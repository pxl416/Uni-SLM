# models/Encoder/rgb_encoder.py
import torch
import torch.nn as nn
import torchvision.models.video as video_models
from torchvision.models.video import R3D_18_Weights
from einops import rearrange


class RGBEncoder(nn.Module):
    """
    RGB Encoder = backbone (3D CNN) + projection
    Forward output: (B, T_out, hidden_dim)
    """

    def __init__(self, cfg, hidden_dim):
        super().__init__()

        bb_cfg = cfg.backbone
        proj_cfg = cfg.proj

        self.hidden_dim = hidden_dim
        self.backbone_dim = bb_cfg.output_dim

        # ---------------------------------------------------------
        # Backbone (r3d_18)
        # ---------------------------------------------------------
        if bb_cfg.pretrained_path:
            weights = None
        else:
            weights = R3D_18_Weights.KINETICS400_V1

        backbone = video_models.r3d_18(weights=weights)

        self.backbone = nn.Sequential(
            backbone.stem,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        # Spatial pooling only (preserve temporal length)
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        # Load custom pretrained backbone
        if bb_cfg.pretrained_path:
            state = torch.load(bb_cfg.pretrained_path, map_location="cpu")
            self.backbone.load_state_dict(state, strict=False)

        # Freeze backbone if needed
        if bb_cfg.freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # ---------------------------------------------------------
        # Projection Layer
        # ---------------------------------------------------------
        if proj_cfg.enabled:
            if proj_cfg.type == "linear":
                self.proj = nn.Linear(self.backbone_dim, hidden_dim)

            elif proj_cfg.type == "mlp":
                self.proj = nn.Sequential(
                    nn.Linear(self.backbone_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim),
                )

            elif proj_cfg.type == "identity":
                self.proj = nn.Identity()
            else:
                raise ValueError(f"Unknown proj type: {proj_cfg.type}")
        else:
            self.proj = nn.Identity()

        # Freeze proj if needed
        if proj_cfg.freeze:
            for p in self.proj.parameters():
                p.requires_grad = False

    # ---------------------------------------------------------
    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = rearrange(x, "b t c h w -> b c t h w")

        # Backbone
        x = self.backbone(x)  # (B, 512, T', H', W')

        x = self.spatial_pool(x).squeeze(-1).squeeze(-1)  # → (B, 512, T')

        x = x.permute(0, 2, 1)  # → (B, T', 512)

        # Projection
        x = self.proj(x)  # → (B, T', hidden_dim)

        return x
