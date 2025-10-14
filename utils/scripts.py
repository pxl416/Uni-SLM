import os
import random
import shutil
import json


def select_and_copy_rgb(folder_rgb, folder_rgb_out, ratio=0.1):
    os.makedirs(folder_rgb_out, exist_ok=True)

    rgb_files = [f for f in os.listdir(folder_rgb) if f.endswith('.mp4')]
    selected_files = random.sample(rgb_files, int(len(rgb_files) * ratio))

    total_size = 0
    for file_name in selected_files:
        src = os.path.join(folder_rgb, file_name)
        dst = os.path.join(folder_rgb_out, file_name)
        shutil.copy2(src, dst)
        total_size += os.path.getsize(dst)

    print(f"Copied {len(selected_files)} RGB files to {folder_rgb_out}")
    print(f"Total size of copied RGB files: {total_size / (1024 * 1024):.2f} MB")
    return [os.path.splitext(f)[0] for f in selected_files]  # 返回不带扩展名的文件名列表


def copy_matching_pose(folder_pose, folder_pose_out, selected_names):
    os.makedirs(folder_pose_out, exist_ok=True)

    total_size = 0
    for name in selected_names:
        pose_file = f"{name}.pkl"
        src = os.path.join(folder_pose, pose_file)
        dst = os.path.join(folder_pose_out, pose_file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            total_size += os.path.getsize(dst)
        else:
            print(f"Warning: Pose file {pose_file} not found.")

    print(f"Copied {len(selected_names)} pose files to {folder_pose_out}")
    print(f"Total size of copied pose files: {total_size / (1024 * 1024):.2f} MB")



def split_labels(input_path, train_path, val_path, test_path, seed=42):
    # 加载原始数据
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 打乱顺序（可重复）
    random.seed(seed)
    random.shuffle(data)

    # 计算划分数量
    total = len(data)
    train_end = int(total * 0.7)
    val_end = train_end + int(total * 0.2)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # 保存为json
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=4)
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)

    # 打印信息
    print(f"Total samples: {total}")
    print(f"Train: {len(train_data)} → {train_path}")
    print(f"Val:   {len(val_data)} → {val_path}")
    print(f"Test:  {len(test_data)} → {test_path}")



if __name__ == "__main__":
    # 1. 第一步，合并
    # folder = r"E:\csl-daily\sentence"  # 分卷所在文件夹
    # output_file = os.path.join(folder, "csl-daily-frames-512x512.tar.gz")  # 分卷合并后的地方
    # with open(output_file, "wb") as wfd:
    #     for i in range(10):  # 0~9 共10个分卷
    #         part_file = os.path.join(folder, f"csl-daily-frames-512x512.tar.gz_{i:02d}")
    #         print("Merging", part_file)
    #         with open(part_file, "rb") as fd:
    #             wfd.write(fd.read())
    #
    # print("合并完成:", output_file)
    # 2. 第二步，解压
    import tarfile

    tar_path = r"E:\csl-daily\sentence\csl-daily-frames-512x512.tar.gz"
    extract_dir = r"E:\csl-daily\sentence\frames"  # 解压后的目录

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    print("解压完成:", extract_dir)


