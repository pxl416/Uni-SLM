# synthetic_world/run_audit.py
import sys
import os
import numpy as np
import cv2
from pathlib import Path


sys.path.append('.')

from synthetic_world.world_sampler import WorldSampler
from synthetic_world.renderer import WorldRenderer
from synthetic_world.label_emitter import LabelEmitter
from synthetic_world.audit import VideoAuditor
from synthetic_world.assets import AssetPool
from synthetic_world.loaders.csl_daily import load_csl_daily_as_assets_v1
from synthetic_world.loaders.ucf101 import load_ucf101_as_assets_v1
from synthetic_world.timeline import TimelinePlanner

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



def main3():
    """
    v1 Spatial Occlusion Audit
    Compare: occlusion OFF vs ON
    """

    print("=== Spatial Occlusion Audit ===")
    print(f"Seed: 42 | FPS: 25")
    print("=" * 60)

    import os
    import copy
    import cv2
    import numpy as np
    import traceback

    from synthetic_world.loaders.csl_daily import load_csl_daily_as_assets_v1
    from synthetic_world.loaders.ucf101 import load_ucf101_as_assets_v1
    from synthetic_world.timeline import TimelinePlanner

    OUT_ROOT = "./audit_occlusion_v1"
    os.makedirs(OUT_ROOT, exist_ok=True)

    stage = "init"


    try:
        # 1. Load assets
        stage = "load_assets"
        print("\n[1] Loading assets...")

        signs = load_csl_daily_as_assets_v1(
            root=CSL_ROOT,
            rgb_dir="sentence",
            anno_pkl="sentence_label/csl2020ct_v2.pkl",
            split_file="sentence_label/split_1_train.txt",
            max_samples=10,
            verbose=False,
        )

        bgs = load_ucf101_as_assets_v1(
            root=UCF_ROOT,
            max_samples=5,
            verbose=False,
        )

        print(f"  ✓ Signs: {len(signs)}")
        print(f"  ✓ Bgs: {len(bgs)}")

        # 2. Build pool
        stage = "build_pool"
        print("\n[2] Building pool...")

        pool = AssetPool()

        for s in signs:
            pool.add_sign(s, category=getattr(s, "semantic_category", "unknown"))

        for b in bgs:
            pool.add_background(b)

        print("  ✓ Pool ready")

        # 3. Sampler / Planner
        stage = "sampler_planner"
        print("\n[3] Sampler / Planner...")

        sampler = WorldSampler(
            pool=pool,
            min_signs=1,
            max_signs=2,
        )

        planner = TimelinePlanner(
            target_duration=5.0,
            allow_overlap=False,
        )

        plan = sampler.sample_world()

        timeline = planner.plan(
            background=plan.background,
            signs=plan.signs,
        )

        print(f"  ✓ Segments: {len(timeline.segments)}")

        # 4. Prepare two configs (OFF / ON)
        stage = "prepare_cfg"
        print("\n[4] Preparing configs...")

        base_cfg = {
            "sign_ops": {
                "occlusion": {
                    "enabled": True,
                    "mode": "patch",
                    "patch_size": 6,
                    "ratio": 0.25,
                    "fill": "zero",
                }
            }
        }

        cfg_off = copy.deepcopy(base_cfg)

        cfg_on = copy.deepcopy(base_cfg)
        cfg_on["sign_ops"]["occlusion"]["enabled"] = True

        cfgs = {
            "off": cfg_off,
            "on": cfg_on,
        }

        # 5. Render (OFF / ON)
        stage = "render"
        print("\n[5] Rendering...")

        results = {}

        for name, spatial_cfg in cfgs.items():

            print(f"  → Rendering: {name}")

            renderer = WorldRenderer(
                output_size=(640, 480),
                fps=25,
                seed=42,
                spatial_config=spatial_cfg
            )

            render_result = renderer.render(
                timeline,
                clear_cache=True,
            )

            results[name] = render_result

            print(f"    Frames: {render_result.rgb.shape[0]}")

        # 6. Save comparison frames
        stage = "save_frames"
        print("\n[6] Saving comparison frames...")

        def save_frame(name, t, img):
            path = os.path.join(
                OUT_ROOT,
                f"{name}_t{t}.jpg"
            )
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        T = min(
            results["off"].rgb.shape[0],
            results["on"].rgb.shape[0],
            5,
        )

        for t in range(T):

            save_frame("off", t, results["off"].rgb[t])
            save_frame("on", t, results["on"].rgb[t])

        print(f"  ✓ Saved {T*2} frames")

        # ------------------------------------------------
        # 7. Find first frame with sign + check mask
        # ------------------------------------------------
        stage = "mask_check"
        print("\n[7] Checking masks...")

        for name in ["off", "on"]:
            print(f"\n  --- {name.upper()} ---")

            rr = results[name]
            masks = rr.spatial_masks
            gt = rr.temporal_gt

            print("  total frames:", len(masks))
            print("  frames with sign:", int(gt.sum()))

            found = False

            for t in range(len(masks)):
                if gt[t] < 0.5:
                    continue
                if len(masks[t]) == 0:
                    print(f"  ⚠️ frame {t}: active but no mask")
                    continue
                m = masks[t][0]
                area = np.sum(m > 0)
                cv2.imwrite(
                    os.path.join(OUT_ROOT, f"{name}_mask_t{t}.png"),
                    m
                )
                print(f"  ✓ first active frame: {t}, mask area = {area}")
                found = True
                break

            if not found:
                print("  ❌ No valid mask found in any active frame")

        # 8. Difference visualization
        stage = "diff"
        print("\n[8] Computing diff...")

        for t in range(T):

            a = results["off"].rgb[t].astype(np.int16)
            b = results["on"].rgb[t].astype(np.int16)

            diff = np.abs(a - b).astype(np.uint8)

            path = os.path.join(
                OUT_ROOT,
                f"diff_t{t}.png"
            )

            cv2.imwrite(path, diff)

        print("  ✓ Diff saved")

        # Done
        print("\n=== OCCLUSION AUDIT DONE ===")
        print(f"Check: {OUT_ROOT}")

    except Exception as e:

        print(f"\n❌ ERROR at stage: {stage}")
        print(e)

        traceback.print_exc()

    # ------------------------------------------------
    # 9. Export one full occluded sample (paper-ready)
    # ------------------------------------------------
    print("\n[9] Exporting occluded demo sample...")

    demo_dir = "./audit_occlusion_v1/demo"
    os.makedirs(demo_dir, exist_ok=True)

    rr_on = results["on"]

    auditor = VideoAuditor(output_dir=demo_dir)

    outputs = auditor.audit_render_result(
        render_result=rr_on,
        base_name="occluded_demo",
        fps=25,
    )

    print("  ✓ Demo exported:")
    for k, v in outputs.items():
        print(f"    {k}: {v}")



# def main4():
#
#     import traceback
#
#     print("=== Spatial Transform Audit ===")
#     print("Seed: 42 | FPS: 25")
#     print("=" * 60)
#
#     stage = "init"
#
#     try:
#         # 1. Load assets
#         stage = "load_assets"
#         print("\n[1] Loading assets...")
#         signs = load_csl_daily_as_assets_v1(
#             root=CSL_ROOT,
#             rgb_dir="sentence",
#             anno_pkl="sentence_label/csl2020ct_v2.pkl",
#             split_file="sentence_label/split_1_train.txt",
#             max_samples=10,
#             verbose=False,
#         )
#         bgs = load_ucf101_as_assets_v1(
#             root=UCF_ROOT,
#             max_samples=5,
#             verbose=False,
#         )
#         print(f"  ✓ Signs: {len(signs)}")
#         print(f"  ✓ Bgs: {len(bgs)}")
#
#
#         # 2. Build pool
#         stage = "build_pool"
#         print("\n[2] Building pool...")
#         pool = AssetPool()
#         for s in signs:
#             pool.add_sign(s, category=getattr(s, "semantic_category", "unknown"))
#         for b in bgs:
#             pool.add_background(b)
#         print("  ✓ Pool ready")
#
#
#         # 3. Sampler / Timeline
#         stage = "sampler"
#         print("\n[3] Sampler / Planner...")
#         sampler = WorldSampler(
#             pool=pool,
#             min_signs=1,
#             max_signs=2,
#             # seed=42,
#         )
#         timeline = sampler.sample_world()
#         n_seg = len(getattr(timeline, "segments", []))
#         print(f"  ✓ Segments: {n_seg}")
#         assert n_seg > 0, "No sign sampled! Abort transform audit."
#
#         # 4. Prepare configs
#         stage = "prepare_cfg"
#         print("\n[4] Preparing configs...")
#         cfg_off = {
#             "transform": {},
#         }
#         cfg_on = {
#             "transform": {
#                 "scale_range": (0.95, 1.05),
#                 "allow_horizontal_flip": False,
#                 "translate_frac": (0.0, 0.01),
#                 "brightness": (-0.03, 0.03),
#                 "contrast": (0.97, 1.03),
#                 "gamma": (0.97, 1.03),
#             }
#         }
#
#         cfgs = {
#             "off": cfg_off,
#             "on": cfg_on,
#         }
#
#         # 5. Render
#         stage = "render"
#         print("\n[5] Rendering...")
#         results = {}
#
#         for name, spatial_cfg in cfgs.items():
#             print(f"  → Rendering: {name}")
#             renderer = WorldRenderer(
#                 output_size=(320, 240),
#                 fps=25,
#                 seed=42,
#                 spatial_config=spatial_cfg
#             )
#             rr = renderer.render(timeline)
#             results[name] = rr
#             print(f"    Frames: {rr.rgb.shape[0]}")
#
#
#         # 6. Save frames
#         stage = "save"
#         print("\n[6] Saving frames...")
#
#         out_dir = Path("./audit_transform_v1")
#         out_dir.mkdir(exist_ok=True)
#
#         # (你原来的 frame 保存逻辑)
#
#         print("  ✓ Saved frames")
#
#         # 7. Save videos (NEW)
#         stage = "video"
#         print("\n[7] Saving videos...")
#
#         from synthetic_world.audit import VideoAuditor
#
#         auditor = VideoAuditor(output_dir="./audit_transform_v1")
#
#         rr_off = results["off"]
#         rr_on = results["on"]
#
#         auditor.audit_render_result(
#             rr_off,
#             base_name="transform_off",
#             fps=25,
#         )
#
#         auditor.audit_render_result(
#             rr_on,
#             base_name="transform_on",
#             fps=25,
#         )
#
#         print("  ✓ Videos saved")


def main4():

    import traceback
    from pathlib import Path

    print("=== Spatial Transform Audit ===")
    print("Seed: 42 | FPS: 25")
    print("=" * 60)

    stage = "init"

    try:

        # =====================================================
        # 1. Load assets
        # =====================================================
        stage = "load_assets"
        print("\n[1] Loading assets...")

        signs = load_csl_daily_as_assets_v1(
            root=CSL_ROOT,
            rgb_dir="sentence",
            anno_pkl="sentence_label/csl2020ct_v2.pkl",
            split_file="sentence_label/split_1_train.txt",
            max_samples=10,
            verbose=False,
        )

        bgs = load_ucf101_as_assets_v1(
            root=UCF_ROOT,
            max_samples=5,
            verbose=False,
        )

        print(f"  ✓ Signs: {len(signs)}")
        print(f"  ✓ Bgs: {len(bgs)}")


        # =====================================================
        # 2. Build pool
        # =====================================================
        stage = "build_pool"
        print("\n[2] Building pool...")

        pool = AssetPool()

        for s in signs:
            pool.add_sign(s, category=getattr(s, "semantic_category", "unknown"))

        for b in bgs:
            pool.add_background(b)

        print("  ✓ Pool ready")


        # =====================================================
        # 3. Sampler + TimelinePlanner  ✅ 关键修复点
        # =====================================================
        stage = "sampler"
        print("\n[3] Sampler / Planner...")

        sampler = WorldSampler(
            pool=pool,
            min_signs=1,
            max_signs=1,
        )

        planner = TimelinePlanner(
            target_duration=None,
            allow_overlap=False,
        )

        # Retry until valid timeline
        max_retry = 10

        for i in range(max_retry):

            plan = sampler.sample_world()

            timeline = planner.plan(
                background=plan.background,
                signs=plan.signs,
            )

            n_seg = len(timeline.segments)

            if n_seg > 0:
                break

            print(f"  ⚠️ Resample {i+1}/{max_retry}: empty timeline")

        else:
            raise RuntimeError("Failed to sample valid timeline")

        print(f"  ✓ Segments: {n_seg}")


        # =====================================================
        # 4. Prepare configs
        # =====================================================
        stage = "prepare_cfg"
        print("\n[4] Preparing configs...")

        cfg_off = {
            "transform": {},
        }

        cfg_on = {
            "transform": {
                "scale_range": (0.05, 0.25),
                "allow_horizontal_flip": True,

                # optional (future expand)
                # "translate_frac": (0.0, 0.01),
                # "brightness": (-0.03, 0.03),
                # "contrast": (0.97, 1.03),
                # "gamma": (0.97, 1.03),
            }
        }

        cfgs = {
            "off": cfg_off,
            "on": cfg_on,
        }


        # =====================================================
        # 5. Render
        # =====================================================
        stage = "render"
        print("\n[5] Rendering...")

        results = {}

        for name, spatial_cfg in cfgs.items():

            print(f"  → Rendering: {name}")

            renderer = WorldRenderer(
                output_size=(320, 240),
                fps=25,
                seed=42,
                spatial_config=spatial_cfg,
            )

            rr = renderer.render(timeline)

            results[name] = rr

            print(f"    Frames: {rr.rgb.shape[0]}")


        # =====================================================
        # 6. Save frames (optional)
        # =====================================================
        stage = "save"
        print("\n[6] Saving frames...")

        out_dir = Path("./audit_transform_v1")
        out_dir.mkdir(exist_ok=True)

        print("  ✓ Output dir ready")


        # =====================================================
        # 7. Save videos  ✅ 和 main3 对齐
        # =====================================================
        stage = "video"
        print("\n[7] Saving videos...")

        from synthetic_world.audit import VideoAuditor

        auditor = VideoAuditor(
            output_dir="./audit_transform_v1"
        )

        auditor.audit_render_result(
            results["off"],
            base_name="transform_off",
            fps=25,
        )

        auditor.audit_render_result(
            results["on"],
            base_name="transform_on",
            fps=25,
        )

        print("  ✓ Videos saved")


        print("\n=== TRANSFORM AUDIT DONE ===")


    except Exception:

        print("\n❌ ERROR at stage:", stage)
        traceback.print_exc()


def main_pixel_mask_audit():

    import cv2
    import numpy as np
    from pathlib import Path

    print("=== Pixel Mask Audit ===")
    print("=" * 60)

    # --------------------------------------------------
    # Config
    # --------------------------------------------------
    OUT_DIR = Path("./audit_yolo_mask")
    OUT_DIR.mkdir(exist_ok=True)

    MAX_SAMPLES = 200
    FPS = 10

    VIDEO_PATH = OUT_DIR / "audit_yolo.mp4"

    # --------------------------------------------------
    # Build mask provider
    # --------------------------------------------------
    from synthetic_world.sign_mask import MaskConfig, SignMaskProvider

    cfg = MaskConfig(
        method="yolo",
        yolo_model="../pretrained-model/yolov8n-seg.pt",
        device="cpu",          # ⚠️ 强制 CPU（避免 CUDA 报错）
        conf_thres=0.25,
    )

    provider = SignMaskProvider(cfg)

    print(f"Mask method: {cfg.method}")

    # --------------------------------------------------
    # Load sign assets
    # --------------------------------------------------
    from synthetic_world.loaders.csl_daily import load_csl_daily_as_assets_v1

    signs = load_csl_daily_as_assets_v1(
        root=CSL_ROOT,
        rgb_dir="sentence",
        anno_pkl="sentence_label/csl2020ct_v2.pkl",
        split_file="sentence_label/split_1_train.txt",
        max_samples=3,
        verbose=False,
    )

    print(f"Loaded {len(signs)} sign videos")

    # --------------------------------------------------
    # Video Writer
    # --------------------------------------------------
    writer = None
    saved = 0

    # --------------------------------------------------
    # Main Loop
    # --------------------------------------------------
    for s in signs:

        frames = s.load_frames()

        print(f"Processing: frames={len(frames)}")

        for t, frame in enumerate(frames):

            if saved >= MAX_SAMPLES:
                break

            frame = frame.astype(np.uint8)

            # ----------------------------
            # Get mask
            # ----------------------------
            mask01, info = provider.get_mask(frame)

            # ----------------------------
            # Resize mask if mismatch
            # ----------------------------
            if mask01.shape != frame.shape[:2]:

                mask01 = cv2.resize(
                    mask01,
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

            # ----------------------------
            # Visualization
            # ----------------------------

            mask_vis = (mask01 * 255).astype(np.uint8)
            mask_rgb = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)

            overlay = frame.copy()

            idx = mask01 == 1

            overlay[idx] = (
                0.6 * overlay[idx] +
                0.4 * np.array([0, 255, 0])
            )

            overlay = overlay.astype(np.uint8)

            panel = np.concatenate(
                [frame, mask_rgb, overlay],
                axis=1
            )

            # ----------------------------
            # Init video writer (once)
            # ----------------------------
            if writer is None:

                h, w = panel.shape[:2]

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                writer = cv2.VideoWriter(
                    str(VIDEO_PATH),
                    fourcc,
                    FPS,
                    (w, h),
                )

                print(f"Video initialized: {VIDEO_PATH}")
                print(f"Resolution: {w}x{h}, FPS={FPS}")

            # ----------------------------
            # Save image
            # ----------------------------
            out_path = OUT_DIR / f"pixel_{saved:04d}.jpg"

            cv2.imwrite(str(out_path), panel)

            # ----------------------------
            # Write video
            # ----------------------------
            writer.write(panel)

            print(f"Saved: {out_path.name} | {info}")

            saved += 1

        if saved >= MAX_SAMPLES:
            break

    # --------------------------------------------------
    # Release video
    # --------------------------------------------------
    if writer is not None:
        writer.release()
        print(f"\nVideo saved: {VIDEO_PATH}")

    print("\n=== PIXEL MASK AUDIT DONE ===")
    print(f"Results in: {OUT_DIR}")

def main5():
    """
    v1 Spatial Combo Audit
    YOLO mask + Occlusion + Transform
    """

    print("=== Spatial Combo Audit (YOLO + Occl + Trans) ===")
    print("Seed: 42 | FPS: 25")
    print("=" * 60)

    import os
    import copy
    import cv2
    import numpy as np
    import traceback
    from pathlib import Path

    from synthetic_world.loaders.csl_daily import load_csl_daily_as_assets_v1
    from synthetic_world.loaders.ucf101 import load_ucf101_as_assets_v1
    from synthetic_world.timeline import TimelinePlanner
    from synthetic_world.audit import VideoAuditor
    from synthetic_world.sign_mask import MaskConfig, SignMaskProvider


    OUT_ROOT = Path("./audit_spatial_combo_v1")
    OUT_ROOT.mkdir(exist_ok=True)

    FPS = 25

    stage = "init"


    try:
        # =====================================================
        # 1. Build YOLO Mask Provider
        # =====================================================
        stage = "mask"

        print("\n[1] Build YOLO mask provider...")

        mask_cfg = MaskConfig(
            method="yolo",
            yolo_model="../pretrained-model/yolov8n-seg.pt",
            device="cpu",
            conf_thres=0.25,
        )

        mask_provider = SignMaskProvider(mask_cfg)

        print("  ✓ YOLO mask ready")


        # =====================================================
        # 2. Load Assets
        # =====================================================
        stage = "load_assets"

        print("\n[2] Loading assets...")

        signs = load_csl_daily_as_assets_v1(
            root=CSL_ROOT,
            rgb_dir="sentence",
            anno_pkl="sentence_label/csl2020ct_v2.pkl",
            split_file="sentence_label/split_1_train.txt",
            max_samples=6,
            verbose=False,
        )

        bgs = load_ucf101_as_assets_v1(
            root=UCF_ROOT,
            max_samples=3,
            verbose=False,
        )

        print(f"  ✓ Signs: {len(signs)}")
        print(f"  ✓ Bgs: {len(bgs)}")


        # =====================================================
        # 3. Build Pool
        # =====================================================
        stage = "build_pool"

        print("\n[3] Building pool...")

        pool = AssetPool()

        for s in signs:
            pool.add_sign(s)

        for b in bgs:
            pool.add_background(b)

        print("  ✓ Pool ready")


        # =====================================================
        # 4. Sample Timeline
        # =====================================================
        stage = "timeline"

        print("\n[4] Sampling timeline...")

        sampler = WorldSampler(
            pool=pool,
            min_signs=1,
            max_signs=1,
        )

        planner = TimelinePlanner(
            target_duration=None,
            allow_overlap=False,
        )

        max_retry = 10

        for i in range(max_retry):

            plan = sampler.sample_world()

            timeline = planner.plan(
                background=plan.background,
                signs=plan.signs,
            )

            if len(timeline.segments) > 0:
                break

            print(f"  ⚠️ retry {i+1}")

        else:
            raise RuntimeError("Empty timeline")

        print(f"  ✓ Segments: {len(timeline.segments)}")


        # =====================================================
        # 5. Prepare Spatial Configs
        # =====================================================
        stage = "cfg"

        print("\n[5] Preparing configs...")

        # ----- baseline -----
        cfg_base = {
            "sign_mask": {
                "method": "yolo",
            },
            "sign_ops": {},
            "transform": {},
        }


        # ----- full combo -----
        cfg_combo = {
            "sign_mask": {
                "method": "yolo",
            },

            "sign_ops": {
                "occlusion": {
                    "enabled": True,
                    "mode": "patch",
                    "patch_size": 8,
                    "ratio": 0.25,
                    "fill": "zero",
                }
            },

            "transform": {
                "scale_range": (0.08, 0.25),
                "allow_horizontal_flip": True,
            },
        }


        cfgs = {
            "base": cfg_base,
            "combo": cfg_combo,
        }


        # =====================================================
        # 6. Render
        # =====================================================
        stage = "render"

        print("\n[6] Rendering...")

        results = {}

        for name, spatial_cfg in cfgs.items():

            print(f"  → {name}")

            renderer = WorldRenderer(
                output_size=(640, 480),
                fps=FPS,
                seed=42,
                spatial_config=spatial_cfg,
                sign_mask_provider=mask_provider,   # ⭐ 核心
            )

            rr = renderer.render(
                timeline,
                clear_cache=True,
            )

            results[name] = rr

            print(f"    Frames: {rr.rgb.shape[0]}")


        # =====================================================
        # 7. Export Videos
        # =====================================================
        stage = "video"

        print("\n[7] Exporting videos...")

        auditor = VideoAuditor(output_dir=str(OUT_ROOT))


        auditor.audit_render_result(
            results["base"],
            base_name="spatial_base",
            fps=FPS,
        )

        auditor.audit_render_result(
            results["combo"],
            base_name="spatial_combo",
            fps=FPS,
        )


        # =====================================================
        # 8. Diff (Optional)
        # =====================================================
        stage = "diff"

        print("\n[8] Computing diff...")

        diff_dir = OUT_ROOT / "diff"
        diff_dir.mkdir(exist_ok=True)

        T = min(
            results["base"].rgb.shape[0],
            results["combo"].rgb.shape[0],
            50,
        )

        for t in range(T):

            a = results["base"].rgb[t].astype(np.int16)
            b = results["combo"].rgb[t].astype(np.int16)

            diff = np.abs(a - b).astype(np.uint8)

            cv2.imwrite(
                str(diff_dir / f"diff_{t:04d}.png"),
                diff,
            )

        print("  ✓ Diff saved")


        # =====================================================
        # Done
        # =====================================================
        print("\n=== SPATIAL COMBO AUDIT DONE ===")
        print(f"Check: {OUT_ROOT}")


    except Exception:

        print(f"\n❌ ERROR at stage: {stage}")
        traceback.print_exc()





if __name__ == "__main__":
    # main()
    # main1()
    # main2()
    # main3()
    # main4()
    # main_pixel_mask_audit()
    main5()
