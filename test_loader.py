
def create_dataloader(
        dataset_name: str,
        split: str,
        cfg,
        batch_size: int = 4,
        num_workers: int = 4,
        use_rgb: bool = True,
        token_level: str = "char",
        T: int = 16,
        img_size: int = 224,
        min_frames: int = 1,
        skip_empty_text: bool = False,
        min_text_len: int = 1,
        debug: bool = False,
):
    """
    构建 DataLoader：
      - 目前实现 CSL_Daily（帧目录 + sentence_label）
      - 其他数据集可按需扩展
    """
    dataset_name = dataset_name.strip()
    split = split.lower()
    assert split in {"train", "val", "dev", "test"}, f"Unknown split: {split}"

    # 如果是字符串路径，则加载配置
    if isinstance(cfg, str):
        config_path = cfg
        cfg = load_config_from_yaml(config_path)
    else:
        config_path = None

    if dataset_name == "CSL_Daily":
        # 推断 root：使用明确的根目录
        root = _guess_daily_root_from_cfg(cfg, config_path)

        # 从配置中获取参数
        dataset_cfg = _get_from_cfg(cfg, "datasets.CSL_Daily", {})
        temporal_cfg = _get_from_cfg(cfg, "temporal", {})

        # 计算实际采样帧数
        ratio = temporal_cfg.get("ratio", 0.25)
        jitter = temporal_cfg.get("jitter", True)
        min_frames_cfg = temporal_cfg.get("min_frames", 4)
        max_frames_cfg = temporal_cfg.get("max_frames", 32)

        # 使用配置中的split文件
        split_file = dataset_cfg.get("split_file")

        # 处理frame_base路径
        frame_base = dataset_cfg.get("rgb_dir")

        # 使用配置中的参数覆盖默认参数
        actual_T = T
        if split == "train":
            actual_random_offset = jitter
        else:
            actual_random_offset = False

        if debug:
            print(f"[DEBUG] Creating CSL_Daily dataset with:")
            print(f"  root: {root}")
            print(f"  split: {split}")
            print(f"  split_file: {split_file}")
            print(f"  frame_base: {frame_base}")
            print(f"  T: {actual_T}")
            print(f"  min_frames: {max(min_frames, min_frames_cfg)}")

        ds = CSLDailyDataset(
            root=root,
            split=("val" if split == "dev" else split),
            token_level=token_level,
            T=actual_T,
            random_offset=actual_random_offset,
            max_text_len=dataset_cfg.get("max_text_len", 128),
            use_rgb=use_rgb,
            frame_base=frame_base,
            rgb_transform=None,  # 使用类内默认
            img_size=img_size,
            min_frames=max(min_frames, min_frames_cfg),  # 取较大值
            skip_empty_text=skip_empty_text,
            min_text_len=min_text_len,
            split_file=split_file,  # 传递split文件路径
            debug=debug,
        )

        if len(ds) == 0:
            print(f"警告: {split} 数据集为空!")
            print("请检查以下路径是否存在:")
            print(f"  root: {root}")
            print(f"  frame_base: {ds.frame_base}")
            print(f"  split_file: {split_file}")
            if os.path.exists(root):
                print("root目录内容:", os.listdir(root))
            else:
                print(f"root目录不存在: {root}")
            if os.path.exists(ds.frame_base):
                print("frame_base目录内容:", os.listdir(ds.frame_base)[:10])
            else:
                print(f"frame_base目录不存在: {ds.frame_base}")
            if split_file and os.path.exists(split_file):
                print(f"split_file存在: {split_file}")
            else:
                print(f"split_file不存在: {split_file}")

        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=csl_collate,
            drop_last=(split == "train"),
        )
        return loader

    elif dataset_name == "CSL_News":
        raise NotImplementedError("CSL_News 的 DataLoader 请使用你现有的实现或单独适配。")

    else:
        raise NotImplementedError(f"不支持的数据集: {dataset_name}")


# 可选：快速自测
if __name__ == "__main__":

    import os, glob

    root = "/home/pxl416/PeixiLiu/px_proj/Uni-SLM/data/mini_CSL_Daily"
    split_file = os.path.join(root, "sentence_label", "split_2.txt")
    frame_base = os.path.join(root, "sentence")

    print("root exists:", os.path.exists(root), root)
    print("split exists:", os.path.exists(split_file), split_file)
    print("frame_base exists:", os.path.isdir(frame_base), frame_base)

    if os.path.isdir(frame_base):
        subs = [d for d in sorted(os.listdir(frame_base)) if os.path.isdir(os.path.join(frame_base, d))]
        print("num video dirs under sentence/:", len(subs))
        print("sample dirs:", subs[:5])
        if subs:
            first = os.path.join(frame_base, subs[0])
            imgs = []
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                imgs += glob.glob(os.path.join(first, ext))
            print("first video dir:", first, "num imgs:", len(imgs))
