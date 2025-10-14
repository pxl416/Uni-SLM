import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.efficientnet import efficientnet_b0, EfficientNet_B0_Weights
from einops import rearrange


class RGBEncoder(nn.Module):
    def __init__(self, pretrained=False, output_dim=512):
        super().__init__()

        # 加载 EfficientNet B0 backbone
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)

        # 去掉分类头，仅保留特征提取部分
        self.backbone = nn.Sequential(*list(backbone.features.children()))

        # 自定义一个全局池化 + 线性映射到统一维度
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))  # 输出 B, C, 1, 1
        self.proj = nn.Linear(1280, output_dim)  # EfficientNet-B0 最终通道数为 1280

    def forward(self, x):
        """
        输入：
            x: RGB 视频张量，形状为 (B, T, C, H, W)，其中 C=3
        输出：
            features: B, T, D 的张量，表示每帧的特征向量
        """

        B, T, C, H, W = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w')  # 合并 batch 和时间维
        x = self.backbone(x)  # (B*T, C', H', W')
        x = self.pooling(x).squeeze(-1).squeeze(-1)  # (B*T, C')
        x = self.proj(x)  # (B*T, D)
        x = rearrange(x, '(b t) d -> b t d', b=B, t=T)  # 恢复 batch 时间维

        return x
