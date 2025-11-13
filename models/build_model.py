# models/build_model.py

import torch
import torch.nn as nn

from models.Encoder.rgb_encoder import RGBEncoder
from models.Encoder.pose_encoder import PoseEncoder
from models.Encoder.text_encoder import TextEncoder


def freeze_module(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = False


def load_pretrained_weights(module: nn.Module, weight_path: str):
    if not weight_path:
        return
    try:
        state_dict = torch.load(weight_path, map_location="cpu")
        # 如果是保存了 {"model": state_dict} 这种，可在这里做 unwrap
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        module.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded pretrained weights from {weight_path}")
    except Exception as e:
        print(f"⚠️ Failed to load weights from {weight_path}: {e}")


def build_model(cfg) -> nn.Module:
    """
    根据 cfg.Model 构建并列 encoders：
      - RGBEncoder
      - PoseEncoder
      - TextEncoder
    支持 per-encoder 的 freeze / pretrained
    """
    encoders = {}

    # 先保护一下，防止没有 Model 字段直接炸
    model_cfg = getattr(cfg, "Model", None)

    # ---- RGB Encoder ----
    if model_cfg is not None and hasattr(model_cfg, "RGBEncoder"):
        rgb_cfg = model_cfg.RGBEncoder
        if getattr(rgb_cfg, "enable", False):
            out_dim = getattr(rgb_cfg, "output_dim", 512)
            # 这里保持和你原来的 RGBEncoder 接口一致：
            # RGBEncoder(pretrained=True, output_dim=Dv)
            rgb = RGBEncoder(pretrained=True, output_dim=out_dim)

            if getattr(rgb_cfg, "freeze", False):
                freeze_module(rgb)

            load_pretrained_weights(rgb, getattr(rgb_cfg, "pretrained", ""))

            encoders["rgb"] = rgb

    # ---- Pose Encoder ----
    if model_cfg is not None and hasattr(model_cfg, "PoseEncoder"):
        pose_cfg = model_cfg.PoseEncoder
        if getattr(pose_cfg, "enable", False):
            # TODO: 按你的 PoseEncoder 实际 __init__ 签名改
            pose = PoseEncoder()  # 如果需要 cfg，就改成 PoseEncoder(pose_cfg)

            if getattr(pose_cfg, "freeze", False):
                freeze_module(pose)

            load_pretrained_weights(pose, getattr(pose_cfg, "pretrained", ""))

            encoders["pose"] = pose

    # ---- Text Encoder ----
    if model_cfg is not None and hasattr(model_cfg, "TextEncoder"):
        text_cfg = model_cfg.TextEncoder
        if getattr(text_cfg, "enable", False):
            # TODO: 按 TextEncoder 实际签名改。你之前是 TextEncoder()。
            text = TextEncoder()

            if getattr(text_cfg, "freeze", False):
                freeze_module(text)

            load_pretrained_weights(text, getattr(text_cfg, "pretrained", ""))

            encoders["text"] = text

    model = nn.ModuleDict(encoders)

    # ---- Fusion 占位（暂时注释掉，避免 NameError） ----
    # fusion_cfg = getattr(getattr(cfg, "Fusion", None), "rgb_text", None)
    # if fusion_cfg is not None and getattr(fusion_cfg, "enabled", False):
    #     model["fusion"] = SomeFusionModule(fusion_cfg)

    return model


# ========== 测试入口（显示维度 / freeze状态 / 输入输出） ==========
if __name__ == "__main__":
    import argparse
    from utils.config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="../config/pretrain_newtask_mini_1.yaml",
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    # 加载 config
    cfg = load_config(args.config)

    # 构建模型
    model = build_model(cfg)
    model.eval()

    print("=== 模型子模块列表 ===")
    for name, module in model.items():
        print(f"[{name.upper()}] Module: {module.__class__.__name__}")
        n_params = sum(p.numel() for p in module.parameters())
        n_frozen = sum(p.numel() for p in module.parameters() if not p.requires_grad)
        ratio = 0.0 if n_params == 0 else n_frozen / n_params
        print(f"  Total Params: {n_params:,} | Frozen: {n_frozen:,} ({ratio:.1%})")

    print("\n=== 测试 Dummy 输入与输出维度 ===")
    with torch.no_grad():
        if "rgb" in model:
            dummy_rgb = torch.randn(2, 26, 3, 224, 224)  # B,T,C,H,W
            out = model["rgb"](dummy_rgb)
            print(f"[RGB] Input: {dummy_rgb.shape} → Output: {out.shape}")

        if "pose" in model:
            dummy_pose = torch.randn(2, 256, 256, 21)  # 这个 shape 需要和你实际 PoseEncoder 对齐
            out = model["pose"](dummy_pose)
            print(f"[Pose] Input: {dummy_pose.shape} → Output: {out.shape}")

        if "text" in model:
            dummy_text = ["we go to school .", "they study hard ."]
            out = model["text"](dummy_text)
            if isinstance(out, (list, tuple)):
                text_feat = out[0]
            else:
                text_feat = out
            print(f"[Text] Input: list[str] ({len(dummy_text)}) → Output: {text_feat.shape}")

    print("\n✅ 模型结构加载与检查完毕。")
