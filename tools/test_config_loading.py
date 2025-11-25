import yaml
from pathlib import Path

from datasets.CSLDaily import CSLDailyDataset
from torch.utils.data import DataLoader
from types import SimpleNamespace


def load_cfg(path):
    path = Path(path)
    assert path.exists(), f"配置文件不存在：{path}"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_load_yaml():
    cfg = load_cfg("../config/finetune_newtask_mini_1.yaml")

    assert "datasets" in cfg
    assert "CSL_Daily" in cfg["datasets"]
    assert "paths" in cfg["datasets"]["CSL_Daily"]
    assert "root" in cfg["datasets"]["CSL_Daily"]["paths"]

    print("YAML 配置加载正常。")


def test_csl_daily_getitem():
    cfg = load_cfg("../config/finetune_newtask_mini_1.yaml")
    print("cfg keys:", cfg.keys())

    ds_cfg = cfg["datasets"]["CSL_Daily"]
    print("paths:", ds_cfg["paths"])

    args = SimpleNamespace(
        max_length=32,
        rgb_support=True,
        seed=3407,
        use_aug=False
    )

    ds = CSLDailyDataset(args=args, cfg=cfg, phase="train")

    name, pose, text, support = ds[0]
    print("name:", name)
    print("text:", text)
    print("rgb shape:", support["rgb_img"].shape)

    dl = DataLoader(ds, batch_size=2, collate_fn=ds.collate_fn)
    batch_src, batch_tgt = next(iter(dl))

    print("batch_src keys:", batch_src.keys())
    print("rgb batch shape:", batch_src["rgb_img"].shape)
    print("attn mask shape:", batch_src["rgb_attn_mask"].shape)


if __name__ == "__main__":
    test_load_yaml()
    test_csl_daily_getitem()
