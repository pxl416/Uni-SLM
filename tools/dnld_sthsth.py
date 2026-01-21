# from pathlib import Path
# import tarfile
# import shutil
#
# # 这两个就是你现在已有的文件
# parts = [
#     Path("/home/pxl416/PeixiLiu/px_proj/px_data/sthsth/20bn-something-something-v2-00"),
#     Path("/home/pxl416/PeixiLiu/px_proj/px_data/sthsth/20bn-something-something-v2-01"),
# ]
#
# # 输出目录
# out_dir = Path("ssv2_videos")
# out_dir.mkdir(exist_ok=True)
#
# # 临时拼接文件
# combined = Path("ssv2_full.tgz")
#
# print("Concatenating parts...")
# with open(combined, "wb") as w:
#     for p in parts:
#         print("  adding", p.name)
#         with open(p, "rb") as r:
#             shutil.copyfileobj(r, w)
#
# print("Concatenation done.")
#
# print("Extracting tar.gz ...")
# with tarfile.open(combined, "r:gz") as tar:
#     tar.extractall(out_dir)
#
# print("All done.")

# import imageio
#
# vid = imageio.get_reader("/home/pxl416/PeixiLiu/px_proj/px_data/sthsth/20bn-something-something-v2/1.webm", format="ffmpeg")
# frame = vid.get_data(0)
# print(frame.shape, frame.dtype)

# import random
# import shutil
# from pathlib import Path
#
# # ====== 配置区 ======
# SRC_DIR = Path("/home/pxl416/PeixiLiu/px_proj/px_data/sthsth/20bn-something-something-v2")
# DST_DIR = Path("/home/pxl416/PeixiLiu/px_proj/Uni-SLM/data/mini_sthsth_v2")
#
# NUM_SAMPLES = 200        # 👈 你可以改成 50 / 100 / 500 / 1000
# SEED = 42                # 固定随机种子，保证可复现
# EXT = ".webm"
# # ====================
#
# random.seed(SEED)
# DST_DIR.mkdir(parents=True, exist_ok=True)
#
# # 收集所有视频
# videos = sorted(SRC_DIR.glob(f"*{EXT}"))
# assert len(videos) > 0, "No videos found!"
#
# print(f"Found {len(videos)} videos total.")
#
# # 随机采样
# selected = random.sample(videos, k=min(NUM_SAMPLES, len(videos)))
#
# print(f"Sampling {len(selected)} videos...")
#
# for v in selected:
#     dst = DST_DIR / v.name
#     shutil.copy2(v, dst)
#
# print("Done.")
# print(f"Subset saved to: {DST_DIR}")


import random
import shutil
from pathlib import Path

# ===== 配置 =====
SRC_ROOT = Path("/home/pxl416/PeixiLiu/px_proj/px_data/UCF-101")
DST_ROOT = Path("/home/pxl416/PeixiLiu/px_proj/Uni-SLM/data/mini_ucf101")

VIDEOS_PER_CLASS = 5     # 👈 每个类别抽几个（2–5 都很合适）
SEED = 42
EXT = ".avi"
# =================

random.seed(SEED)
DST_ROOT.mkdir(parents=True, exist_ok=True)

class_dirs = sorted([d for d in SRC_ROOT.iterdir() if d.is_dir()])
print(f"Found {len(class_dirs)} classes.")

total_copied = 0

for cls_dir in class_dirs:
    videos = sorted(cls_dir.glob(f"*{EXT}"))
    if not videos:
        continue

    k = min(VIDEOS_PER_CLASS, len(videos))
    selected = random.sample(videos, k=k)

    dst_cls_dir = DST_ROOT / cls_dir.name
    dst_cls_dir.mkdir(parents=True, exist_ok=True)

    for v in selected:
        shutil.copy2(v, dst_cls_dir / v.name)
        total_copied += 1

print(f"Done. Copied {total_copied} videos.")
print(f"Subset saved to: {DST_ROOT}")






