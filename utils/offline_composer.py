# utils/offline_composer.py
import os
import cv2
import json
import numpy as np
from typing import List, Tuple, Dict
from utils.temporal_composer import TemporalComposer
import torch
from datasets.datasets import create_dataloader
from tools.test_data_loading import load_yaml, dict_to_ns

def load_one_csl_sample(cfg_path):
    raw_cfg = load_yaml(cfg_path)
    cfg = dict_to_ns(raw_cfg)

    class DummyArgs:
        pass

    args = DummyArgs()
    args.cfg = cfg_path

    dataloader = create_dataloader(args, cfg, phase="train")

    for src, tgt in dataloader:
        # 取第一个 batch 的第一个样本
        rgb = src["rgb_img"][0]        # (T,C,H,W)
        keypoints = src["keypoints"][0]  # (T,21,3)
        name = src["name"][0]
        break

    # 转成 numpy video
    rgb = rgb.permute(0,2,3,1).cpu().numpy()  # (T,H,W,C)
    rgb = (rgb * 255).astype(np.uint8)

    return VideoSample(rgb, fps=25), keypoints.numpy(), name


class VideoSample:
    def __init__(self, frames: np.ndarray, fps: int):
        assert frames.ndim == 4  # (T,H,W,C)
        self.frames = frames.astype(np.uint8)
        self.fps = int(fps)
        self.T, self.H, self.W, self.C = frames.shape

    def info(self) -> Dict:
        return {
            "T": self.T,
            "H": self.H,
            "W": self.W,
            "C": self.C,
            "fps": self.fps,
        }


def load_video_avi(path: str, max_frames: int = None) -> VideoSample:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(round(fps)) if fps > 0 else 25

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break

    cap.release()
    frames = np.stack(frames, axis=0)
    return VideoSample(frames, fps)


def resize_video(video: VideoSample, size=(112, 112)) -> VideoSample:
    resized = np.stack(
        [cv2.resize(f, size, interpolation=cv2.INTER_LINEAR)
         for f in video.frames],
        axis=0
    )
    return VideoSample(resized, video.fps)


def temporal_concat(
    videos: List[VideoSample],
    labels: List[int]
) -> Tuple[VideoSample, np.ndarray]:
    assert len(videos) == len(labels)

    frames = []
    temporal_label = []

    fps = videos[0].fps
    for v in videos:
        assert v.fps == fps, "fps mismatch in temporal concat"

    for v, lab in zip(videos, labels):
        frames.append(v.frames)
        temporal_label.append(
            np.full(v.T, lab, dtype=np.int64)
        )

    frames = np.concatenate(frames, axis=0)
    temporal_label = np.concatenate(temporal_label, axis=0)

    return VideoSample(frames, fps), temporal_label


def spatial_identity(video: VideoSample) -> Tuple[VideoSample, np.ndarray]:
    T, H, W, _ = video.frames.shape
    spatial_prior = np.zeros((T, H, W), dtype=np.float32)
    return video, spatial_prior


def save_sample(
    video: VideoSample,
    temporal_label: np.ndarray,
    spatial_prior: np.ndarray,
    save_dir: str,
    name: str
):
    os.makedirs(save_dir, exist_ok=True)
    sample_dir = os.path.join(save_dir, name)
    os.makedirs(sample_dir, exist_ok=True)

    video_path = os.path.join(sample_dir, "video.mp4")
    label_path = os.path.join(sample_dir, "temporal_label.npy")
    spatial_path = os.path.join(sample_dir, "spatial_prior.npy")
    meta_path = os.path.join(sample_dir, "meta.json")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        video_path,
        fourcc,
        video.fps,
        (video.W, video.H)
    )
    for f in video.frames:
        writer.write(f)
    writer.release()

    np.save(label_path, temporal_label)
    np.save(spatial_path, spatial_prior)

    with open(meta_path, "w") as f:
        json.dump(video.info(), f, indent=2)

    print(f"[Saved] {sample_dir}")


def visualize_temporal_label(label: np.ndarray, max_len=200):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 2))
    plt.plot(label[:max_len])
    plt.title("Temporal label")
    plt.xlabel("frame")
    plt.ylabel("sign")
    plt.tight_layout()
    plt.show()




def main():
    cfg_path = "../config/ft.yaml"

    # 取一个 CSL 样本
    sign_video, kpts, name = load_one_csl_sample(cfg_path)

    # 取两个背景 clip（用 UCF 先顶着）
    ucf_path = "/home/pxl416/PeixiLiu/px_proj/px_data/UCF-101/BaseballPitch"
    vids = sorted([os.path.join(ucf_path,f) for f in os.listdir(ucf_path) if f.endswith(".avi")])

    bg1 = resize_video(load_video_avi(vids[0]), (sign_video.W, sign_video.H))
    bg2 = resize_video(load_video_avi(vids[1]), (sign_video.W, sign_video.H))

    # Temporal synthetic
    temporal = TemporalComposer(mode="synthetic")
    # merged_video, t_label = temporal(sign_video, meta={"bg1": bg1, "bg2": bg2})
    meta = {
        "bg1": bg1,
        "bg2": bg2,
    }
    frames, fps, t_label = temporal(sign_video, meta)
    merged_video = VideoSample(frames, fps)

    # Spatial: 先 identity
    _, spatial = spatial_identity(merged_video)

    visualize_temporal_label(t_label)

    save_sample(
        merged_video,
        t_label,
        spatial,
        save_dir="./offline_debug",
        name=f"csl_{name}"
    )


if __name__ == "__main__":
    main()
