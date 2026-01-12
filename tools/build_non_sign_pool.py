# tools/build_non_sign_pool.py
import os
import json
import time
from typing import Dict, Any, List, Optional

import cv2


def _is_video_file(name: str) -> bool:
    name = name.lower()
    return name.endswith(".avi") or name.endswith(".mp4") or name.endswith(".mkv")


def _probe_video(path: str, fallback_fps: int = 25) -> Dict[str, Any]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {
            "ok": False,
            "fps": fallback_fps,
            "num_frames": -1,
            "width": -1,
            "height": -1,
            "error": "cv2.VideoCapture open failed",
        }

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(round(fps)) if fps and fps > 0 else fallback_fps

    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    num_frames = int(num_frames) if num_frames and num_frames > 0 else -1

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = int(width) if width and width > 0 else -1
    height = int(height) if height and height > 0 else -1

    cap.release()

    return {
        "ok": True,
        "fps": fps,
        "num_frames": num_frames,
        "width": width,
        "height": height,
        "error": None,
    }


def build_ucf101_index(
    ucf_root: str,
    out_json: str,
    max_per_class: Optional[int] = None,
    min_frames: int = 1,
    allowed_classes: Optional[List[str]] = None,
    fallback_fps: int = 25,
    verbose_every: int = 500,
) -> Dict[str, Any]:
    """
    Build a non-sign video pool index for UCF-101.

    Output schema:
    {
      "dataset": "UCF101",
      "root": "...",
      "created_at": "...",
      "total_videos": N,
      "classes": { "ClassName": count, ... },
      "videos": [
        {
          "video_id": "...",
          "class": "ClassName",
          "path": "...(absolute)...",
          "relpath": "...(relative to root)...",
          "fps": 25,
          "num_frames": 110,
          "width": 320,
          "height": 240
        },
        ...
      ]
    }
    """
    ucf_root = os.path.abspath(ucf_root)
    assert os.path.isdir(ucf_root), f"UCF root not found: {ucf_root}"

    allowed_set = set(allowed_classes) if allowed_classes else None

    classes = sorted([d for d in os.listdir(ucf_root) if os.path.isdir(os.path.join(ucf_root, d))])
    if allowed_set is not None:
        classes = [c for c in classes if c in allowed_set]

    videos: List[Dict[str, Any]] = []
    class_counts: Dict[str, int] = {}

    t0 = time.time()
    processed = 0
    skipped_open_fail = 0
    skipped_short = 0

    for cls in classes:
        cls_dir = os.path.join(ucf_root, cls)
        files = sorted([f for f in os.listdir(cls_dir) if _is_video_file(f)])

        if max_per_class is not None:
            files = files[: int(max_per_class)]

        class_counts[cls] = 0

        for f in files:
            abs_path = os.path.join(cls_dir, f)
            rel_path = os.path.relpath(abs_path, ucf_root)

            probe = _probe_video(abs_path, fallback_fps=fallback_fps)
            processed += 1

            if not probe["ok"]:
                skipped_open_fail += 1
                continue

            # frame_count may be -1; only enforce min_frames when we know it
            if probe["num_frames"] != -1 and probe["num_frames"] < min_frames:
                skipped_short += 1
                continue

            video_id = os.path.splitext(os.path.basename(f))[0]

            item = {
                "video_id": video_id,
                "class": cls,
                "path": abs_path,
                "relpath": rel_path,
                "fps": int(probe["fps"]),
                "num_frames": int(probe["num_frames"]),
                "width": int(probe["width"]),
                "height": int(probe["height"]),
            }
            videos.append(item)
            class_counts[cls] += 1

            if verbose_every and (processed % verbose_every == 0):
                dt = time.time() - t0
                print(
                    f"[Indexing] processed={processed} kept={len(videos)} "
                    f"skipped_open={skipped_open_fail} skipped_short={skipped_short} "
                    f"elapsed={dt/60:.1f}min"
                )

    meta = {
        "dataset": "UCF101",
        "root": ucf_root,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "total_videos": len(videos),
        "processed_files": processed,
        "skipped_open_fail": skipped_open_fail,
        "skipped_short": skipped_short,
        "classes": class_counts,
        "videos": videos,
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[Done] Saved index: {out_json}")
    print(f"[Done] Total kept videos: {len(videos)} | Classes: {len(class_counts)}")
    return meta


def main():
    # You can directly edit these defaults.
    ucf_root = "/home/pxl416/PeixiLiu/px_proj/px_data/UCF-101"
    out_json = "/home/pxl416/PeixiLiu/px_proj/px_data/ucf101_non_sign_index.json"

    # For quick smoke test, you can set max_per_class=5 first.
    build_ucf101_index(
        ucf_root=ucf_root,
        out_json=out_json,
        max_per_class=None,   # e.g. 5 for quick test
        min_frames=1,
        allowed_classes=None, # e.g. ["BoxingPunchingBag", "Basketball"]
        fallback_fps=25,
        verbose_every=500,
    )


if __name__ == "__main__":
    main()
