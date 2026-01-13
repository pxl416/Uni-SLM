from synthetic_world.audit import audit_sample
from synthetic_world.assets import AssetPool
from synthetic_world.loaders.csl_daily import load_csl_daily_as_assets
from synthetic_world.loaders.ucf101 import load_ucf101_as_assets

# 加载数据
pool = AssetPool()
signs = load_csl_daily_as_assets(..., max_samples=50)
bgs = load_ucf101_as_assets(..., max_samples=20)

for s in signs: pool.add_sign(s)
for b in bgs: pool.add_background(b)

# 审计第一个样本
outputs = audit_sample(
    asset_pool=pool,
    sample_idx=0,
    output_dir="./audit_results",
    sampler_config={
        'min_signs': 1,
        'max_signs': 3,
        'target_duration': 5.0,
    },
    renderer_config={
        'output_size': (320, 240),
        'fps': 10,
        'seed': 42,
    },
    save_all=True,
)