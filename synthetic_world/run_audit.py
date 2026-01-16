# synthetic_world/run_audit.py
import sys

sys.path.append('.')

from synthetic_world.world_sampler import WorldSampler
from synthetic_world.renderer import WorldRenderer
from synthetic_world.label_emitter import LabelEmitter
from synthetic_world.audit import VideoAuditor
from synthetic_world.assets import AssetPool

# 根据你的实际路径调整
CSL_ROOT = "/home/pxl416/PeixiLiu/px_proj/px_data/csl-daily-frames-512x512"
UCF_ROOT = "/home/pxl416/PeixiLiu/px_proj/px_data/UCF-101"


def main():
    print("=== Real Data Audit ===")
    print(f"CSL path: {CSL_ROOT}")
    print(f"UCF path: {UCF_ROOT}")
    print("=" * 50 + "\n")

    try:
        # 1. 加载真实资产
        print("1. Loading assets...")

        from synthetic_world.loaders.csl_daily import load_csl_daily_as_assets
        from synthetic_world.loaders.ucf101 import load_ucf101_as_assets

        # 加载少量测试数据
        signs = load_csl_daily_as_assets(
            root=CSL_ROOT,
            rgb_dir="sentence",
            anno_pkl="sentence_label/csl2020ct_v2.pkl",
            split_file="sentence_label/split_1_train.txt",
            max_samples=30,  # 少量用于测试
            verbose=True,
        )

        bgs = load_ucf101_as_assets(
            root=UCF_ROOT,
            max_samples=10,  # 少量用于测试
            verbose=True,
        )

        print(f"   ✓ Loaded {len(signs)} sign assets")
        print(f"   ✓ Loaded {len(bgs)} background assets")

        # 2. 创建资产池
        print("\n2. Creating asset pool...")
        pool = AssetPool()

        for asset in signs:
            pool.add_sign(asset, category=getattr(asset, 'semantic_category', 'general'))

        for asset in bgs:
            pool.add_background(asset)

        print(f"   ✓ Asset pool created")
        print(f"   {pool.summary()}")

        # 3. 构建世界采样器
        print("\n3. Creating world sampler...")
        sampler = WorldSampler(
            pool=pool,
            min_signs=1,
            max_signs=3,
            target_duration=6.0,
        )

        # 4. 构建渲染器
        print("\n4. Creating renderer...")
        renderer = WorldRenderer(
            output_size=(320, 240),  # 中等分辨率
            fps=15,  # 较低帧率加速测试
            seed=42,
            spatial_config={
                "position_mode": "grid",  # 使用语义布局
                "scale_range": (0.8, 1.2),
                "blend_mode": "alpha",
            }
        )

        # 5. 生成一个世界
        print("\n5. Sampling world...")
        timeline = sampler.sample_world()

        print(f"   Background: {timeline.background.asset_id}")
        print(f"   Num segments: {len(timeline.segments)}")
        for i, seg in enumerate(timeline.segments):
            sign = seg["sign"]
            print(f"   Segment {i}: '{sign.text}' at {seg['start_sec']:.1f}s-{seg['end_sec']:.1f}s")

        # 6. 渲染
        print("\n6. Rendering...")
        render_result = renderer.render(timeline, clear_cache=True)

        print(f"   Rendered {len(render_result.rgb)} frames")
        print(f"   Frames with sign: {render_result.temporal_gt.sum()}/{len(render_result.temporal_gt)}")

        # 7. 生成标签
        print("\n7. Generating labels...")
        emitter = LabelEmitter(include_masks=True)
        labels = emitter.emit(render_result)

        print(f"   Labels generated: {len(labels.text_alignments)} segments")
        print(f"   Vocabulary: {labels.vocabulary}")

        # 8. 审计 & 可视化
        print("\n8. Auditing and visualizing...")

        # 把 labels 挂到 render_result 上
        render_result.labels = labels

        auditor = VideoAuditor(output_dir="./audit_real")
        outputs = auditor.audit_render_result(
            render_result,
            base_name="real_sample",
            fps=renderer.fps,
            save_all=True
        )

        print("\n" + "=" * 50)
        print("AUDIT COMPLETE!")
        print("=" * 50)
        print("\nGenerated files:")
        for key, path in outputs.items():
            if path:  # 只显示非空路径
                print(f"  {key}: {path}")

        print("\nOpen the following files to check:")
        print(f"  1. {outputs.get('overlay_video', '')} - 完整覆盖视频")
        print(f"  2. {outputs.get('bbox', '')} - 仅显示bbox")
        print(f"  3. {outputs.get('temporal_analysis', '')} - 时间分析图")
        print(f"  4. {outputs.get('spatial_analysis', '')} - 空间分析图")
        print(f"  5. {outputs.get('metadata_report', '')} - 元数据报告")

        # 额外：保存原始数据用于手动检查
        print("\n9. Saving raw data for manual inspection...")
        import numpy as np
        import cv2

        # 保存第一帧用于检查
        if len(render_result.rgb) > 0:
            first_frame = render_result.rgb[0]
            cv2.imwrite("./audit_real/first_frame.jpg",
                        cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
            print(f"   ✓ Saved first frame: audit_real/first_frame.jpg")

            # 如果有mask，保存第一个mask
            if render_result.spatial_masks and len(render_result.spatial_masks[0]) > 0:
                first_mask = render_result.spatial_masks[0][0]
                cv2.imwrite("./audit_real/first_mask.png", first_mask)
                print(f"   ✓ Saved first mask: audit_real/first_mask.png")

        print("\nDone! Check the 'audit_real' folder for all outputs.")

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: File not found - {e}")
        print("\nPlease check your data paths:")
        print(f"  CSL_ROOT: {CSL_ROOT}")
        print(f"  UCF_ROOT: {UCF_ROOT}")
        print("\nMake sure these directories exist and contain the expected files.")

    except ImportError as e:
        print(f"\n❌ ERROR: Import failed - {e}")
        print("\nMake sure you're running from the correct directory and all modules are installed.")
        print("Try: python -m synthetic_world.run_audit")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


def main1():
    print("=== Real Data Audit ===")
    print(f"CSL path: {CSL_ROOT}")
    print(f"UCF path: {UCF_ROOT}")
    print("=" * 50 + "\n")

    try:
        # 1. 加载真实资产
        print("1. Loading assets...")

        from synthetic_world.loaders.csl_daily import load_csl_daily_as_assets
        from synthetic_world.loaders.ucf101 import load_ucf101_as_assets

        # 加载少量测试数据
        signs = load_csl_daily_as_assets(
            root=CSL_ROOT,
            rgb_dir="sentence",
            anno_pkl="sentence_label/csl2020ct_v2.pkl",
            split_file="sentence_label/split_1_train.txt",
            max_samples=30,  # 少量用于测试
            verbose=True,
        )

        bgs = load_ucf101_as_assets(
            root=UCF_ROOT,
            max_samples=10,  # 少量用于测试
            verbose=True,
        )

        print(f"   ✓ Loaded {len(signs)} sign assets")
        print(f"   ✓ Loaded {len(bgs)} background assets")

        # 2. 创建资产池
        print("\n2. Creating asset pool...")
        pool = AssetPool()

        for asset in signs:
            pool.add_sign(asset, category=getattr(asset, 'semantic_category', 'general'))

        for asset in bgs:
            pool.add_background(asset)

        print(f"   ✓ Asset pool created")
        print(f"   {pool.summary()}")

        # 3. 构建世界采样器
        print("\n3. Creating world sampler...")
        sampler = WorldSampler(
            pool=pool,
            min_signs=1,
            max_signs=3,
            target_duration=6.0,
        )

        # 4. 构建渲染器
        print("\n4. Creating renderer...")
        renderer = WorldRenderer(
            output_size=(512, 512),  # 大背景尺寸
            fps=15,
            seed=42,
            spatial_config={
                "position_mode": "grid",
                "scale_range": (0.4, 0.6),
                "blend_mode": "alpha",
                # "keep_inside_frame": True,
                # "bbox_padding": 0.05
            }

        )

        # 5. 生成一个世界
        print("\n5. Sampling world...")
        timeline = sampler.sample_world()

        print(f"   Background: {timeline.background.asset_id}")
        print(f"   Num segments: {len(timeline.segments)}")
        for i, seg in enumerate(timeline.segments):
            sign = seg["sign"]
            print(f"   Segment {i}: '{sign.text}' at {seg['start_sec']:.1f}s-{seg['end_sec']:.1f}s")

        # 6. 渲染
        print("\n6. Rendering...")
        render_result = renderer.render(timeline, clear_cache=True)

        print(f"   Rendered {len(render_result.rgb)} frames")
        print(f"   Frames with sign: {render_result.temporal_gt.sum()}/{len(render_result.temporal_gt)}")

        # 7. 生成标签
        print("\n7. Generating labels...")
        emitter = LabelEmitter(include_masks=True)
        labels = emitter.emit(render_result)

        print(f"   Labels generated: {len(labels.text_alignments)} segments")
        print(f"   Vocabulary: {labels.vocabulary}")

        # 8. 审计 & 可视化
        print("\n8. Auditing and visualizing...")

        # 把 labels 挂到 render_result 上
        render_result.labels = labels

        auditor = VideoAuditor(output_dir="./audit_real")
        outputs = auditor.audit_render_result(
            render_result,
            base_name="real_sample",
            fps=renderer.fps,
            save_all=True
        )

        print("\n" + "=" * 50)
        print("AUDIT COMPLETE!")
        print("=" * 50)
        print("\nGenerated files:")
        for key, path in outputs.items():
            if path:  # 只显示非空路径
                print(f"  {key}: {path}")

        print("\nOpen the following files to check:")
        print(f"  1. {outputs.get('overlay_video', '')} - 完整覆盖视频")
        print(f"  2. {outputs.get('bbox', '')} - 仅显示bbox")
        print(f"  3. {outputs.get('temporal_analysis', '')} - 时间分析图")
        print(f"  4. {outputs.get('spatial_analysis', '')} - 空间分析图")
        print(f"  5. {outputs.get('metadata_report', '')} - 元数据报告")

        # 额外：保存原始数据用于手动检查
        print("\n9. Saving raw data for manual inspection...")
        import numpy as np
        import cv2

        # 保存第一帧用于检查
        if len(render_result.rgb) > 0:
            first_frame = render_result.rgb[0]
            cv2.imwrite("./audit_real/first_frame.jpg",
                        cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
            print(f"   ✓ Saved first frame: audit_real/first_frame.jpg")

            # 如果有mask，保存第一个mask
            if render_result.spatial_masks and len(render_result.spatial_masks[0]) > 0:
                first_mask = render_result.spatial_masks[0][0]
                cv2.imwrite("./audit_real/first_mask.png", first_mask)
                print(f"   ✓ Saved first mask: audit_real/first_mask.png")

        print("\nDone! Check the 'audit_real' folder for all outputs.")

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: File not found - {e}")
        print("\nPlease check your data paths:")
        print(f"  CSL_ROOT: {CSL_ROOT}")
        print(f"  UCF_ROOT: {UCF_ROOT}")
        print("\nMake sure these directories exist and contain the expected files.")

    except ImportError as e:
        print(f"\n❌ ERROR: Import failed - {e}")
        print("\nMake sure you're running from the correct directory and all modules are installed.")
        print("Try: python -m synthetic_world.run_audit")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # main()
    main1()
