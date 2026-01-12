# tools/build_csl_sign_index.py
import os
import json
import cv2
from tqdm import tqdm

# ------------------------
# Config (按需修改)
# ------------------------
DATASET_NAME = "CSL-Daily"

CSL_ROOT = "/home/pxl416/PeixiLiu/px_proj/px_data/csl-daily-rgb"
SEGMENT_FILE = "/home/pxl416/PeixiLiu/px_proj/px_data/csl-daily-annotations/video_map.txt"
SAVE_PATH = "/home/pxl416/PeixiLiu/px_proj/px_data/csl_sign_index.json"

MIN_FRAMES = 16   # 太短的不要
EXPECTED_FPS = 25


# ------------------------
# Utils
# ------------------------
def read_segments(path):
    """
    video_id start end
    """
    segs = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            vid = parts[0]
            try:
                s = int(parts[1])
                e = int(parts[2])
            except ValueError:
                continue
            segs.setdefault(vid, []).append([s, e])
    return segs


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(round(fps)) if fps > 0 else EXPECTED_FPS
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, frames


# ------------------------
# Main Index Builder
# ------------------------
def build_index():
    segments = read_segments(SEGMENT_FILE)

    index = []
    skipped_open = 0
    skipped_short = 0

    for vid, seg_list in tqdm(segments.items(), desc="Indexing CSL"):
        video_path = os.path.join(CSL_ROOT, vid + ".mp4")
        if not os.path.exists(video_path):
            skipped_open += 1
            continue

        info = get_video_info(video_path)
        if info is None:
            skipped_open += 1
            continue

        fps, num_frames = info
        if num_frames < MIN_FRAMES:
            skipped_short += 1
            continue

        entry = {
            "dataset": DATASET_NAME,
            "video_id": vid,
            "video_path": video_path,
            "fps": fps,
            "num_frames": num_frames,
            "segments": seg_list,
            "has_rgb": True,
            "has_pose": True,   # CSL-Daily 有 pose
        }
        index.append(entry)

    print(f"[Done] kept={len(index)} skipped_open={skipped_open} skipped_short={skipped_short}")
    return index


if __name__ == "__main__":
    idx = build_index()
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=2, ensure_ascii=False)

    print(f"[Saved] CSL sign index → {SAVE_PATH}")
