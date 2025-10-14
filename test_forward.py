import torch
from utils.dataset2 import create_dataloader
import types, yaml

# --------- 1) 加载 cfg 并做适配 ---------
def load_cfg_with_adapter(yaml_path, dataset_name):
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg_raw = yaml.safe_load(f)
    cfg = types.SimpleNamespace(**cfg_raw)
    ds_map = cfg_raw.get("datasets", {})
    data_path = types.SimpleNamespace(**ds_map[dataset_name])
    cfg.data_path = data_path
    return cfg

def build_args(dataset_name="CSL_News"):
    A = types.SimpleNamespace()
    A.dataset_name = dataset_name
    A.batch_size = 2
    A.num_workers = 0
    A.max_length = 64
    A.rgb_support = True
    A.use_aug = False
    return A

# --------- 2) Dummy 模型 ---------
class DummyPoseEncoder(torch.nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.proj = torch.nn.Linear(9*3, out_dim)  # 假设 body 用 9 关节
    def forward(self, x):
        B, T, K, C = x.shape  # [B,T,9,3]
        return self.proj(x.view(B, T, -1))  # [B,T,out_dim]

class DummyRGBEncoder(torch.nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.proj = torch.nn.Linear(112*112*3, out_dim)
    def forward(self, x):
        B, T, C, H, W = x.shape
        return self.proj(x.view(B, T, -1))  # [B,T,out_dim]

# --------- 3) 主流程 ---------
def main():
    dataset_name = "CSL_News"
    yaml_path = "config/config.yaml"  # 改成你的路径
    cfg = load_cfg_with_adapter(yaml_path, dataset_name)
    args = build_args(dataset_name)

    dl = create_dataloader(args, cfg, phase="train")
    src_input, tgt_input = next(iter(dl))

    # 准备模型
    pose_encoder = DummyPoseEncoder()
    rgb_encoder  = DummyRGBEncoder()

    # 输入
    x_pose = src_input["body"]      # [B,T,9,3]
    x_rgb  = src_input["rgb_img"]   # [B,T,3,112,112]
    mask   = src_input["attention_mask"]

    # 前向
    out_pose = pose_encoder(x_pose)
    out_rgb  = rgb_encoder(x_rgb)

    print("Pose input:", x_pose.shape, "-> out:", out_pose.shape)
    print("RGB  input:", x_rgb.shape, "-> out:", out_rgb.shape)
    print("Mask:", mask.shape)
    print("Sentences:", tgt_input["gt_sentence"])

if __name__ == "__main__":
    main()
