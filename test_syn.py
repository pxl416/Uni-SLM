from synthetic_world.loaders.csl_daily import load_csl_daily_as_assets
from synthetic_world.loaders.ucf101 import load_ucf101_as_assets
from synthetic_world.stream import SignWorldStream

signs = load_csl_daily_as_assets(
    root="/home/pxl416/PeixiLiu/px_proj/px_data/csl-daily-frames-512x512",
    rgb_dir="sentence",
    anno_pkl="sentence_label/csl2020ct_v2.pkl",
    split_file="sentence_label/split_1_train.txt",
    max_samples=50,
)

bgs = load_ucf101_as_assets(
    root="/home/pxl416/PeixiLiu/px_proj/px_data/UCF-101",
    max_samples=20,
)

ds = SignWorldStream(signs, bgs)
from synthetic_world.loaders.csl_daily import load_csl_daily_as_assets
from synthetic_world.loaders.ucf101 import load_ucf101_as_assets
from synthetic_world.stream import SignWorldStream

signs = load_csl_daily_as_assets(
    root="/home/pxl416/PeixiLiu/px_proj/px_data/csl-daily-frames-512x512",
    rgb_dir="sentence",
    anno_pkl="sentence_label/csl2020ct_v2.pkl",
    split_file="sentence_label/split_1_train.txt",
    max_samples=50,
)

bgs = load_ucf101_as_assets(
    root="/home/pxl416/PeixiLiu/px_proj/px_data/UCF-101",
    max_samples=20,
)

ds = SignWorldStream(signs, bgs)

sample = ds[0]

print(sample["rgb"].shape)
print(sample["temporal_gt"].shape)
print(sample["temporal_gt"].sum())
print(sample["segments"])

sample = ds[0]

print(sample["rgb"].shape)
print(sample["temporal_gt"].shape)
print(sample["temporal_gt"].sum())
print(sample["segments"])
