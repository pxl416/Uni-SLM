import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import torchvision.transforms as T
from torchvision.transforms import functional as F
import PIL.Image as Image
from augment1 import MultiModalAugmentor

def make_fake_sample(T_len=12, K=21, H=256, W=456):
    # 伪造 RGB 帧：list of PIL
    frames = [Image.fromarray((np.random.rand(H, W, 3) * 255).astype('uint8')) for _ in range(T_len)]
    # 伪造 Pose： [T,K,3]，(x,y) 为像素坐标，c=置信度
    pose = np.concatenate([
        np.random.rand(T_len, K, 2) * np.array([W, H]),
        np.ones((T_len, K, 1)) * 0.9
    ], axis=-1)
    text = "你好，世界。Hello world."
    return frames, pose, text, (W, H)

def test_smoke():
    frames, pose, text, (W, H) = make_fake_sample()
    cfg = {
        "rgb": { "hflip_p": 0.0, "blur_p": 0.0, "erase_p": 0.0,
                 "color_p": 0.0, "color_jitter": {"b":0.0,"c":0.0,"s":0.0,"h":0.0}},
        "temporal": {"max_frames": 8, "strategy": "uniform"},
        "pose": {"conf_thr": 0.3, "noise_level": 0.0},
        "text": {"prob": 0.0},
    }
    aug = MultiModalAugmentor(cfg, image_size=(224, 224))
    rgb_aug, pose_aug, text_aug, params = aug(frames, pose, text,
                                              sample_index=0, global_seed=123,
                                              orig_size=(W, H))
    assert isinstance(rgb_aug, torch.Tensor)
    assert rgb_aug.shape == (8, 3, 224, 224), rgb_aug.shape
    assert pose_aug.shape == (8, pose.shape[1], 3)
    assert isinstance(text_aug, str)
    print("Smoke OK:", rgb_aug.shape, pose_aug.shape, text_aug[:8], params.frame_indices)

def test_determinism():
    frames, pose, text, (W, H) = make_fake_sample()
    cfg = {
        "rgb": { "hflip_p": 0.5, "blur_p": 0.5, "erase_p": 0.0, "color_p": 0.5 },
        "temporal": {"max_frames": 8, "strategy": "random"},
        "pose": {"conf_thr": 0.3, "noise_level": 0.01},
        "text": {"prob": 0.5, "modes": ["noise"]},
    }
    aug = MultiModalAugmentor(cfg, image_size=(224, 224))
    out1 = aug(frames, pose, text, sample_index=7, global_seed=123, orig_size=(W, H))
    out2 = aug(frames, pose, text, sample_index=7, global_seed=123, orig_size=(W, H))
    out3 = aug(frames, pose, text, sample_index=8, global_seed=123, orig_size=(W, H))

    rgb1, pose1, txt1, _ = out1
    rgb2, pose2, txt2, _ = out2
    rgb3, pose3, txt3, _ = out3

    # 同样本索引 + 同全局种子 => 完全一致
    assert torch.allclose(rgb1, rgb2), "Determinism failed for rgb"
    assert np.allclose(pose1, pose2), "Determinism failed for pose"
    assert txt1 == txt2, "Determinism failed for text"
    # 不同样本索引 => 大概率不同（不强制断言值不同，仅打印）
    print("Determinism OK; different sample_index likely differs:",
          not torch.allclose(rgb1, rgb3) or not np.allclose(pose1, pose3) or (txt1 != txt3))

def test_flip_effect():
    # 强制水平翻转，检查 x 是否镜像
    T_len, K, H, W = 5, 4, 100, 200
    frames = [Image.fromarray((np.random.rand(H, W, 3) * 255).astype('uint8')) for _ in range(T_len)]
    pose = np.zeros((T_len, K, 3))
    # 人为设置一个点的 x=10
    pose[..., 0, 0] = 10
    pose[..., 0, 1] = 20
    pose[..., 0, 2] = 1.0  # conf
    text = "flip test"

    cfg = {
        "rgb": { "hflip_p": 1.0, "blur_p": 0.0, "erase_p": 0.0, "color_p": 0.0 },
        "temporal": {"max_frames": T_len, "strategy": "uniform"},
        "pose": {"conf_thr": 0.3, "noise_level": 0.0},
        "text": {"prob": 0.0},
    }
    aug = MultiModalAugmentor(cfg, image_size=(W, H))  # 输出尺寸与原图一致，便于直观检查
    rgb_aug, pose_aug, text_aug, params = aug(frames, pose, text,
                                              sample_index=0, global_seed=1, orig_size=(W, H))
    # 镜像后 x 应为 W-1-10
    expected_x = (W - 1) - 10
    got = pose_aug[..., 0, 0]
    # 因为还做了crop->resize，如果 image_size==(W,H) 且 crop 是全图，值应匹配
    assert np.allclose(got, expected_x), (expected_x, got.mean())
    print("Flip OK:", expected_x, got[0])

if __name__ == "__main__":
    test_smoke()
    test_determinism()
    test_flip_effect()
    print("All tests passed ✅")