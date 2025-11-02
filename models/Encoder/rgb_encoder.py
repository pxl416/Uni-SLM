import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.efficientnet import efficientnet_b0, EfficientNet_B0_Weights
from einops import rearrange


# class RGBEncoder(nn.Module):
#     def __init__(self, pretrained=False, output_dim=512):
#         super().__init__()
#
#         # 加载 EfficientNet B0 backbone
#         backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
#
#         # 去掉分类头，仅保留特征提取部分
#         self.backbone = nn.Sequential(*list(backbone.features.children()))
#
#         # 自定义一个全局池化 + 线性映射到统一维度
#         self.pooling = nn.AdaptiveAvgPool2d((1, 1))  # 输出 B, C, 1, 1
#         self.proj = nn.Linear(1280, output_dim)  # EfficientNet-B0 最终通道数为 1280
#
#     def forward(self, x):
#         """
#         输入：
#             x: RGB 视频张量，形状为 (B, T, C, H, W)，其中 C=3
#         输出：
#             features: B, T, D 的张量，表示每帧的特征向量
#         """
#
#         B, T, C, H, W = x.shape
#         x = rearrange(x, 'b t c h w -> (b t) c h w')  # 合并 batch 和时间维
#         x = self.backbone(x)  # (B*T, C', H', W')
#         x = self.pooling(x).squeeze(-1).squeeze(-1)  # (B*T, C')
#         x = self.proj(x)  # (B*T, D)
#         x = rearrange(x, '(b t) d -> b t d', b=B, t=T)  # 恢复 batch 时间维
#
#         return x


import torch
import torch.nn as nn
import torchvision.models as tvm
from torchvision.models.video import R3D_18_Weights
from einops import rearrange

class RGBEncoder(nn.Module):
    """
    I3D-style 3D CNN encoder based on torchvision r3d_18.
    - 输入:  (B, T, C, H, W),  C=3
    - 输出:  (B, T_out, D)
      * 默认 T_out = 模型的时间下采样后长度（通常是 T/8）
      * 若 keep_temporal=True，则线性插值回原始 T
    """
    def __init__(self, pretrained: bool = False, output_dim: int = 512, keep_temporal: bool = True):
        super().__init__()
        self.keep_temporal = keep_temporal
        # 加载 3D ResNet-18（R3D-18），可选用 Kinetics400 预训练
        weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        backbone = tvm.video.r3d_18(weights=weights)
        # 去掉 avgpool + fc，保留卷积干部分
        # r3d_18 结构: stem -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool -> fc
        self.stem   = backbone.stem
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.out_channels = 512  # r3d_18 的最后通道数

        # 仅做空间池化，保留时间维
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # 输出 (B, C, T', 1, 1)

        # 投影到统一维度
        self.proj = nn.Linear(self.out_channels, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            feats: (B, T_out, D)  若 keep_temporal=True，则 T_out == 输入的 T
        """
        B, T, C, H, W = x.shape
        x = rearrange(x, 'b t c h w -> b c t h w')  # (B, C, T, H, W)

        # 3D CNN 前向（r3d_18 会在时间维做下采样，通常 stride 总计 8）
        x = self.stem(x)      # -> (B, 64,  T/2, H/2,  W/2)  视实现细节而定
        x = self.layer1(x)    # -> (B, 64,  T/2, ...)
        x = self.layer2(x)    # -> (B, 128, T/4, ...)
        x = self.layer3(x)    # -> (B, 256, T/8, ...)
        x = self.layer4(x)    # -> (B, 512, T/8, ...)

        # 仅做空间池化，保留时间维 (B, 512, T', 1, 1)
        x = self.spatial_pool(x).squeeze(-1).squeeze(-1)  # (B, 512, T')

        # 置换到 (B, T', C)
        x = x.permute(0, 2, 1).contiguous()  # (B, T', 512)

        # 线性投影到 D
        x = self.proj(x)  # (B, T', D)

        # 可选：把时间维插值回原始 T，便于与旧管线对齐
        if self.keep_temporal and x.size(1) != T:
            # 使用线性插值到原始 T
            x = x.permute(0, 2, 1)                    # (B, D, T')
            x = nn.functional.interpolate(x, size=T, mode="linear", align_corners=False)
            x = x.permute(0, 2, 1).contiguous()       # (B, T, D)

        return x
