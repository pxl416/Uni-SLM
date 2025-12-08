# tools/test_model_building.py
import os
import sys
import torch
import yaml
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from utils.config import dict_to_ns
from models.build_model import build_model


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    print("========================================")
    print(" Test: Model Building System")
    print("========================================\n")

    # 1) 加载 finetuner.yaml
    ft_path = os.path.join(ROOT, "config/ft.yaml")
    raw_ft = load_yaml(ft_path)
    ft_cfg = dict_to_ns(raw_ft)
    print("[Info] finetuner.yaml keys:", list(vars(ft_cfg).keys()))

    # 2) 加载 model.yaml
    model_path = os.path.join(ROOT, ft_cfg.model)
    raw_model = load_yaml(model_path)
    model_cfg = dict_to_ns(raw_model)
    print("[Info] model.yaml keys:", list(vars(model_cfg).keys()))

    # 3) 构建模型
    model = build_model(model_cfg)
    model.eval()

    # 简单打印参数量
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Info] Total parameters: {n_params/1e6:.2f} M")

    # 4) 构造一个 dummy batch
    B, T, C, H, W = 2, 4, 3, 224, 224
    device = torch.device("cpu")

    batch = {
        # src_input 部分
        "rgb_img": torch.randn(B, T, C, H, W, device=device),
        "keypoints": torch.randn(B, T, 21, 3, device=device),
        "kp_len": torch.tensor([T, T], dtype=torch.long, device=device),
        "rgb_len": torch.tensor([T, T], dtype=torch.long, device=device),

        # tgt_input 部分
        "gt_sentence": ["你好，世界。", "团队内只有分工合作才能提高工作效率。"],
        "gt_gloss": [["你们", "好"], ["团队", "分工", "合作"]],
    }

    # 5) Retrieval 测试
    with torch.no_grad():
        print("\n[Check] Forward: retrieval")
        out_ret = model(batch, task="retrieval")
        print("  video_emb:", out_ret["video_emb"].shape)
        print("  text_emb:", out_ret["text_emb"].shape)
        print("  logits:", out_ret["logits"].shape)

    # 6) Recognition 测试
    with torch.no_grad():
        print("\n[Check] Forward: recognition")
        out_rec = model(batch, task="recognition")
        print("  logits:", out_rec.shape)  # (B, T, num_classes)

    # 7) Translation 测试
    with torch.no_grad():
        print("\n[Check] Forward: translation")
        out_tr = model(batch, task="translation")
        if out_tr.get("mt5_used", False):
            print("  pred_text example:", out_tr["pred_text"][:2])
        else:
            print("  dummy video_repr:", out_tr["video_repr"].shape)

    print("\n✅ Model building test finished.\n")


if __name__ == "__main__":
    main()
