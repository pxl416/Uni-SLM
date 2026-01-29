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
        render_result.labels = labels        # 把 labels 挂到 render_result 上
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

def main2():
    """
    v1-aligned audit entry:
    Real assets → WorldSampler → TimelinePlanner → Renderer → LabelEmitter → VideoAuditor
    """
    print("=== Real Data Audit (v1 aligned) ===")
    print(f"CSL path: {CSL_ROOT}")
    print(f"UCF path: {UCF_ROOT}")
    print("=" * 60)

    try:
        # 1) Load assets
        print("\n[1] Loading assets...")

        from synthetic_world.loaders.csl_daily import load_csl_daily_as_assets_v1
        from synthetic_world.loaders.ucf101 import load_ucf101_as_assets_v1
        from synthetic_world.timeline import TimelinePlanner

        signs = load_csl_daily_as_assets_v1(
            root=CSL_ROOT,
            rgb_dir="sentence",
            anno_pkl="sentence_label/csl2020ct_v2.pkl",
            split_file="sentence_label/split_1_train.txt",
            max_samples=30,
            verbose=True,
        )
        bgs = load_ucf101_as_assets_v1(
            root=UCF_ROOT,
            max_samples=10,
            verbose=True,
        )

        print(f"  ✓ Signs: {len(signs)}")
        print(f"  ✓ Backgrounds: {len(bgs)}")

        # 2) Build AssetPool
        print("\n[2] Building asset pool...")
        pool = AssetPool()
        for s in signs:
            # 你之前 pool.add_sign(asset, category=...) 也可以，这里用最简
            pool.add_sign(s, category=getattr(s, "semantic_category", "unknown"))
        for b in bgs:
            pool.add_background(b)

        print("  ✓ Pool summary:", pool.summary())

        # 3) WorldSampler (abstract plan)
        print("\n[3] Creating WorldSampler...")
        sampler = WorldSampler(
            pool=pool,
            min_signs=1,
            max_signs=3,
        )

        # 4) TimelinePlanner
        print("\n[4] Creating TimelinePlanner...")
        planner = TimelinePlanner(
            target_duration=6.0,    # seconds
            allow_overlap=False,
        )

        # 5) Renderer
        print("\n[5] Creating renderer...")
        renderer = WorldRenderer(
            output_size=(320, 240),
            fps=15,
            seed=42,
            # 你要限制 sign 的 jitter 在 0.95~1.05，可以在 spatial_config 里传给 SpatialComposer
            # spatial_config={"spatial_cfg": {...}} 视你 _parse_spatial_cfg 的接口而定
        )

        # 6) Sample plan → concrete timeline
        print("\n[6] Sampling world...")
        plan = sampler.sample_world()

        timeline = planner.plan(
            background=plan.background,
            signs=plan.signs,
        )

        # 打印 segments（兼容 dataclass / dict）
        segs = getattr(timeline, "segments", None)
        if segs is None:
            segs = getattr(timeline, "sign_segments", [])
        segs = list(segs or [])

        print(f"  Background: {getattr(timeline.background, 'asset_id', 'unknown')}")
        print(f"  Num sign segments: {len(segs)}")

        for i, seg in enumerate(segs):
            if isinstance(seg, dict):
                sign = seg.get("sign", None)
                s0 = float(seg.get("start_sec", seg.get("start_time", seg.get("start", 0.0))))
                s1 = float(seg.get("end_sec", seg.get("end_time", seg.get("end", 0.0))))
            else:
                sign = getattr(seg, "sign", None)
                s0 = float(getattr(seg, "start_sec", getattr(seg, "start_time", getattr(seg, "start", 0.0))))
                s1 = float(getattr(seg, "end_sec", getattr(seg, "end_time", getattr(seg, "end", 0.0))))

            sid = getattr(sign, "asset_id", "unknown_sign") if sign is not None else "None"
            print(f"  Segment {i}: {sid} | {s0:.2f}s → {s1:.2f}s")

        # 7) Render
        print("\n[7] Rendering...")
        render_result = renderer.render(timeline, clear_cache=True)

        T = int(render_result.rgb.shape[0])
        print(f"  Frames: {T}")
        print(f"  Frames with sign: {int(render_result.temporal_gt.sum())} / {len(render_result.temporal_gt)}")

        # 8) Emit labels (v1)
        print("\n[8] Emitting labels...")
        emitter = LabelEmitter(include_masks=True)

        labels = emitter.emit_from_render_result(
            render_result=render_result,
            fps=renderer.fps,
        )

        print(f"  ✓ Segments: {int(labels.segment_spans.shape[0])}")
        print(f"  ✓ Vocabulary size: {len(labels.vocabulary)}")
        print("  DEBUG spans:", labels.segment_spans)

        # ✅ 关键：必须挂回去，否则 auditor 读不到 segment_spans
        render_result.labels = labels

        # 9) Audit
        print("\n[9] Auditing...")
        auditor = VideoAuditor(output_dir="./audit_real_v1")
        outputs = auditor.audit_render_result(
            render_result,
            base_name="real_v1_sample",
            fps=renderer.fps,
        )

        print("\n=== AUDIT COMPLETE ===")
        for k, v in outputs.items():
            if v:
                print(f"  {k}: {v}")

        # 10) Save sanity-check artifacts
        print("\n[10] Saving sanity-check artifacts...")
        import cv2

        if T > 0:
            cv2.imwrite(
                "./audit_real_v1/first_frame.jpg",
                cv2.cvtColor(render_result.rgb[0], cv2.COLOR_RGB2BGR),
            )
            print("  ✓ first_frame.jpg")

            # 保存第一帧的第一个 mask（如果有）
            if getattr(render_result, "spatial_masks", None):
                if len(render_result.spatial_masks) > 0 and len(render_result.spatial_masks[0]) > 0:
                    cv2.imwrite("./audit_real_v1/first_mask.png", render_result.spatial_masks[0][0])
                    print("  ✓ first_mask.png")

        print("\nAll done. Check ./audit_real_v1/")

    except Exception as e:
        print("\n❌ ERROR:", e)
        import traceback
        traceback.print_exc()




if __name__ == "__main__":
    # main()
    # main1()
    main2()
