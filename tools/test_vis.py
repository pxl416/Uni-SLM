# tools/test_vis.py
import os
import cv2
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt


def list_avi_files(root):
    avi_files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(".avi"):
                avi_files.append(os.path.join(dirpath, f))
    return avi_files


def inspect_video(video_path, show_frames=True, max_show=3):
    print(f"\n[Video] {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Failed to open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  FPS        : {fps:.2f}")
    print(f"  Frames     : {frame_count}")
    print(f"  Resolution : {width} x {height}")

    if frame_count <= 0:
        cap.release()
        return

    # indices to sample
    indices = [
        0,
        frame_count // 2,
        frame_count - 1,
    ]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((idx, frame))

    cap.release()

    if show_frames and len(frames) > 0:
        n = min(len(frames), max_show)
        plt.figure(figsize=(5 * n, 4))
        for i, (idx, frame) in enumerate(frames[:n]):
            plt.subplot(1, n, i + 1)
            plt.imshow(frame)
            plt.title(f"Frame {idx}")
            plt.axis("off")
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default='/home/pxl416/PeixiLiu/px_proj/px_data/UCF-101/BalanceBeam',
        help="Path to UCF101 root directory",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=3,
        help="Number of random videos to inspect",
    )
    parser.add_argument(
        "--no_vis",
        action="store_true",
        help="Only print stats, do not visualize frames",
    )

    args = parser.parse_args()

    avi_files = list_avi_files(args.path)
    print(f"[Info] Found {len(avi_files)} .avi files")

    if len(avi_files) == 0:
        print("❌ No .avi files found.")
        return

    sample_files = random.sample(
        avi_files, min(args.num, len(avi_files))
    )

    for vp in sample_files:
        inspect_video(
            vp,
            show_frames=not args.no_vis,
        )


if __name__ == "__main__":
    main()
