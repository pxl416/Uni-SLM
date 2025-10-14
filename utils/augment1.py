# augment1.py

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import torchvision.transforms as T
from torchvision.transforms import functional as F


from types import SimpleNamespace

def _ns_to_dict(x):
    if isinstance(x, SimpleNamespace):
        return {k: _ns_to_dict(v) for k, v in x.__dict__.items()}
    if isinstance(x, dict):
        return {k: _ns_to_dict(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_ns_to_dict(v) for v in x]
    return x


# ========== 工具函数 ==========

def make_sample_rng(global_seed: int, sample_index: int) -> np.random.RandomState:
    """
    为每个样本生成可复现的 RNG。
    约定：global_seed 为整型，sample_index >= 0。
    """
    # 统一转为 Python int，避免 numpy int 与 python int 混用时的极端行为
    base = int(global_seed) & 0x7FFFFFFF
    idx  = int(sample_index) & 0x7FFFFFFF
    # 使用一个大的素数做混合，减少简单相加的碰撞概率
    mixed = (base * 1_000_003 + idx) % (2**31 - 1)
    if mixed <= 0:
        mixed = 42  # 保底非零
    return np.random.RandomState(mixed)


def bbox_from_keypoints(
    kpts_xyc: np.ndarray,
    conf_thr: float = 0.2,
    margin: float = 0.05,
    min_box: float = 1.0
) -> Optional[Tuple[float, float, float, float]]:
    """
    计算关键点包围框，返回 (xmin, ymin, xmax, ymax)，与输入坐标系一致（像素或归一化由上游统一）。
    - kpts_xyc: [T, K, 3] 或 [K, 3]，最后一维为 (x, y, conf)。
    - conf_thr: 置信度阈值，<= 阈值的点会被忽略。
    - margin: 相对边长扩张比例（对 w 和 h 分别扩张）。
    - min_box: 退化情况下的最小宽高（单位与坐标一致）。
    """
    if kpts_xyc is None:
        return None
    arr = np.asarray(kpts_xyc)
    if arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.shape[-1] < 3:
        return None

    # 过滤无效（低置信或非有限值）
    conf = arr[:, 2]
    finite = np.all(np.isfinite(arr), axis=-1)
    valid_mask = (conf > conf_thr) & finite
    if valid_mask.sum() < 3:
        return None

    xs = arr[valid_mask, 0]
    ys = arr[valid_mask, 1]
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())

    # 处理退化框：若 w/h 非正，扩成最小盒
    w = xmax - xmin
    h = ymax - ymin
    if w < min_box:
        cx = 0.5 * (xmin + xmax)
        half = 0.5 * max(min_box, 1.0)
        xmin, xmax = cx - half, cx + half
        w = xmax - xmin
    if h < min_box:
        cy = 0.5 * (ymin + ymax)
        half = 0.5 * max(min_box, 1.0)
        ymin, ymax = cy - half, cy + half
        h = ymax - ymin

    # 扩张 margin
    xmin -= margin * w
    xmax += margin * w
    ymin -= margin * h
    ymax += margin * h

    # 保障顺序
    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin

    return (xmin, ymin, xmax, ymax)


def mirror_pose_x(
    pose_xyc: np.ndarray,
    img_w: int,
    swap_hands: bool = True,
    left_ids: Optional[List[int]] = None,
    right_ids: Optional[List[int]] = None
) -> np.ndarray:
    """
    沿 X 轴镜像关键点。
    - pose_xyc: [T, K, 3] 或 [K, 3]，x 为像素坐标（0..W-1）
    - img_w: 图像宽度（像素）
    - 当 swap_hands=True 且提供 left/right id 列表且长度相等时，交换左右手关键点顺序。
    """
    if pose_xyc is None:
        return None
    out = np.array(pose_xyc, copy=True)
    out[..., 0] = (img_w - 1) - out[..., 0]

    if swap_hands and left_ids and right_ids and len(left_ids) == len(right_ids):
        # 用临时副本避免覆盖
        tmp = out.copy()
        out[..., left_ids, :] = tmp[..., right_ids, :]
        out[..., right_ids, :] = tmp[..., left_ids, :]
    return out


def clamp_crop_window(
    xmin: float, ymin: float, xmax: float, ymax: float, W: int, H: int, min_size: int = 1
) -> Tuple[int, int, int, int]:
    """
    将 crop 窗口限制在图像内，并保证最小尺寸。
    返回整数像素坐标 (xmin, ymin, xmax, ymax)，满足 0<=xmin<xmax<W, 0<=ymin<ymax<H
    """
    # 先排序确保左上到右下
    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin

    # clamp 到图像内
    xmin = int(max(0, np.floor(xmin)))
    ymin = int(max(0, np.floor(ymin)))
    xmax = int(min(W - 1, np.ceil(xmax)))
    ymax = int(min(H - 1, np.ceil(ymax)))

    # 最小尺寸保证（闭区间像素坐标，宽高=坐标差+1）
    if xmax - xmin + 1 < min_size:
        need = min_size - (xmax - xmin + 1)
        # 优先向右/下扩
        xmax = min(W - 1, xmax + need)
        if xmax - xmin + 1 < min_size:
            # 不够再向左/上
            xmin = max(0, xmin - (min_size - (xmax - xmin + 1)))

    if ymax - ymin + 1 < min_size:
        need = min_size - (ymax - ymin + 1)
        ymax = min(H - 1, ymax + need)
        if ymax - ymin + 1 < min_size:
            ymin = max(0, ymin - (min_size - (ymax - ymin + 1)))

    # 再次兜底：若极端情况下仍退化，强行设为 1x1
    if xmax <= xmin:
        if xmin < W - 1:
            xmax = xmin + 1
        else:
            xmin = max(0, xmin - 1)
            xmax = xmin + 1
    if ymax <= ymin:
        if ymin < H - 1:
            ymax = ymin + 1
        else:
            ymin = max(0, ymin - 1)
            ymax = ymin + 1

    return xmin, ymin, xmax, ymax



# ========== 参数包（一次采样，多模态共享） ==========


class AugParams:
    def __init__(self):
        # 空间
        self.apply_hflip = False
        self.crop_xyxy = None  # (xmin, ymin, xmax, ymax) in pixels
        self.blur_sigma = None
        self.erase_rects = []  # list of (xmin,ymin,xmax,ymax) or sentinel
        self.color_ops = None

        # 时序
        self.frame_indices = None  # List[int]
        self.time_masks = []       # list of (kind, value) e.g., ("ratio", 0.1)

        # 文本
        self.text_params = {}      # 预留

# ========== 主调度器 ==========
# class MultiModalAugmentor.__init__ 的开头
class MultiModalAugmentor:
    def __init__(self, cfg, image_size=(224,224), left_ids=None, right_ids=None):

        cfg = _ns_to_dict(cfg)
        self.cfg = cfg
        self.im_w, self.im_h = image_size[0], image_size[1]
        self.left_ids = left_ids or []
        self.right_ids = right_ids or []




        # ---- RGB 相关
        rgb_cfg = cfg.get("rgb", {})
        # cj = rgb_cfg.get("color_jitter", {"b":0.2,"c":0.2,"s":0.1,"h":0.05})
        cj = cfg.get("rgb", {}).get("color_jitter", {"p": 0.0, "b": 0.2, "c": 0.2, "s": 0.1, "h": 0.05})
        self.t_color = T.ColorJitter(brightness=cj.get("b",0.2),
                                     contrast=cj.get("c",0.2),
                                     saturation=cj.get("s",0.1),
                                     hue=cj.get("h",0.05))
        self.p_color = rgb_cfg.get("color_p", 0.8)  # ★ 颜色增强概率
        self.p_hflip = rgb_cfg.get("hflip_p", 0.3)
        self.p_blur  = rgb_cfg.get("blur_p", 0.2)
        self.p_erase = rgb_cfg.get("erase_p", 0.1)
        self.erase_scale = rgb_cfg.get("erase_scale", (0.02, 0.1))

        # ---- 时序相关
        t_cfg = cfg.get("temporal", {})
        self.max_frames   = t_cfg.get("max_frames", 64)
        self.sampling_cfg = t_cfg.get("strategy", "random")  # "random"/"uniform"/"dense" 或 {"mode":"mixed","weights":{...}}
        self.p_time_mask  = t_cfg.get("time_mask_p", 0.0)
        self.time_mask_ratio = t_cfg.get("time_mask_ratio", 0.1)

        # ---- 文本
        txt_cfg = cfg.get("text", {})
        self.text_prob  = txt_cfg.get("prob", 0.0)
        self.text_modes = txt_cfg.get("modes", ["noise"])

    # -------- 采样参数包（给一个样本） --------
    def sample_params(self, rng: np.random.RandomState, pose_for_bbox: Optional[np.ndarray], W: int, H: int) -> AugParams:
        p = AugParams()

        # 1) 水平翻转
        p.apply_hflip = bool(rng.rand() < self.p_hflip)

        # 2) 裁剪（用关键点 bbox，退化时全图）
        if pose_for_bbox is not None:
            bb = bbox_from_keypoints(pose_for_bbox)
            if bb is None:
                xmin, ymin, xmax, ymax = 0, 0, W-1, H-1
            else:
                xmin, ymin, xmax, ymax = bb
        else:
            xmin, ymin, xmax, ymax = 0, 0, W-1, H-1
        p.crop_xyxy = clamp_crop_window(xmin, ymin, xmax, ymax, W, H)

        # 3) 颜色/模糊/擦除 参数
        # 颜色概率
        if rng.rand() < self.p_color:
            cj = self.cfg.get("rgb", {}).get("color_jitter", {"b": 0.2, "c": 0.2, "s": 0.1, "h": 0.05})
            b, c, s, h = cj.get("b", 0.2), cj.get("c", 0.2), cj.get("s", 0.1), cj.get("h", 0.05)
            ops = []
            if b and b > 0:
                # brightness in [max(0,1-b), 1+b]
                bf = float(rng.uniform(max(0.0, 1.0 - b), 1.0 + b))
                ops.append(("brightness", bf))
            if c and c > 0:
                cf = float(rng.uniform(max(0.0, 1.0 - c), 1.0 + c))
                ops.append(("contrast", cf))
            if s and s > 0:
                sf = float(rng.uniform(max(0.0, 1.0 - s), 1.0 + s))
                ops.append(("saturation", sf))
            if h and h > 0:
                # hue delta in [-h, h]
                hf = float(rng.uniform(-h, h))
                ops.append(("hue", hf))
            # 随机顺序（也由 rng 控制，保证可复现）
            if ops:
                order = rng.permutation(len(ops)).tolist()
                p.color_ops = [ops[i] for i in order]

        # 4) 时间 mask
        if self.p_time_mask > 0 and rng.rand() < self.p_time_mask:
            p.time_masks.append(("ratio", float(self.time_mask_ratio)))
        return p


    # -------- 采样策略（混合/单一） --------
    def _pick_strategy(self, rng: np.random.RandomState) -> str:
        strat = self.sampling_cfg
        if isinstance(strat, dict) and strat.get("mode") == "mixed":
            weights = strat.get("weights", {"random": 1.0})
            names, w = zip(*weights.items())
            w = np.asarray(w, dtype=float); w = w / w.sum()
            i = int(rng.choice(len(names), p=w))
            return names[i]
        if isinstance(strat, str):
            return strat
        return "random"

    # -------- 时序索引生成 --------
    def build_frame_indices(self, rng: np.random.RandomState, length: int) -> List[int]:
        M = min(self.max_frames, length) if self.max_frames > 0 else length
        name = self._pick_strategy(rng)
        if name == "uniform":
            return np.linspace(0, length-1, M).round().astype(int).tolist()
        elif name == "dense":
            start = int(rng.randint(0, max(1, length - M + 1)))
            idx = list(range(start, min(start + M, length)))
            while len(idx) < M:
                idx.append(idx[-1])
            return idx
        else:  # random
            return sorted(rng.choice(length, size=M, replace=(M > length)).tolist())

    # -------- 应用到 RGB --------

    def apply_rgb(self, frames, p: AugParams, rng):
        if p.frame_indices is not None:
            frames = [frames[i] for i in p.frame_indices]

        out = []
        for img in frames:
            was_tensor = torch.is_tensor(img)
            if was_tensor:
                img = T.ToPILImage()(img)

            xmin, ymin, xmax, ymax = p.crop_xyxy
            img = img.crop((int(xmin), int(ymin), int(xmax), int(ymax)))

            if p.apply_hflip:
                img = F.hflip(img)

            # 确定性颜色扰动：按 p.color_ops 的顺序执行
            if p.color_ops:
                for kind, val in p.color_ops:
                    if kind == "brightness":
                        img = F.adjust_brightness(img, val)
                    elif kind == "contrast":
                        img = F.adjust_contrast(img, val)
                    elif kind == "saturation":
                        img = F.adjust_saturation(img, val)
                    elif kind == "hue":
                        img = F.adjust_hue(img, val)

            if p.blur_sigma is not None:
                img = T.GaussianBlur(kernel_size=3, sigma=p.blur_sigma)(img)

            img = T.Resize((self.im_h, self.im_w))(img)
            img_t = T.ToTensor()(img)
            # TODO: 按 p.erase_rects 做确定性擦除（留空也不影响确定性）
            out.append(img_t)
        return torch.stack(out, dim=0)

    # -------- 应用到 Pose --------
    def apply_pose(self, pose_xyc: np.ndarray, p: AugParams, orig_W: int, orig_H: int, rng: np.random.RandomState):
        if p.frame_indices is not None:
            pose_xyc = pose_xyc[p.frame_indices]

        xmin, ymin, xmax, ymax = p.crop_xyxy
        crop_W = (xmax - xmin + 1); crop_H = (ymax - ymin + 1)
        sx = self.im_w / max(1, crop_W)
        sy = self.im_h / max(1, crop_H)

        out = pose_xyc.copy()
        out[..., 0] = (out[..., 0] - xmin) * sx
        out[..., 1] = (out[..., 1] - ymin) * sy

        if p.apply_hflip:

            out = mirror_pose_x(out, img_w=self.im_w, swap_hands=True,
                                left_ids=self.left_ids, right_ids=self.right_ids)

        noise_level = self.cfg.get("pose", {}).get("noise_level", 0.0)
        if noise_level > 0:
            valid = out[..., 2] > self.cfg.get("pose", {}).get("conf_thr", 0.3)
            noise = rng.normal(0, noise_level, size=out[..., :2].shape)
            out[..., :2][valid] += noise[valid]
        return out

    # -------- 时间掩码（统一用 rng） --------
    def apply_time_mask(self, seq: torch.Tensor, p: AugParams, rng: np.random.RandomState):
        out = seq
        for kind, val in p.time_masks:
            if kind == "ratio":
                L = out.shape[0]
                m = int(max(1, L * val))
                start = int(rng.randint(0, max(1, L - m + 1)))
                out[start:start+m] = 0
        return out

    # -------- 文本 --------
    def apply_text(self, text: str, p: AugParams, rng: np.random.RandomState) -> str:
        if rng.rand() > self.text_prob or not self.text_modes:
            return text
        mode = rng.choice(self.text_modes)
        if mode == "noise":
            if len(text) < 6:
                return text
            chars = list(text)
            n = max(1, int(0.01 * len(chars)))
            for _ in range(n):
                i = int(rng.randint(0, len(chars)))
                # 这里占位：不破坏语义，只做“无操作”示例
                chars[i] = chars[i]
            return "".join(chars)
        return text

    # -------- 样本级入口 --------
    def __call__(self, rgb_frames, pose_xyc: Optional[np.ndarray], text: str,
                 sample_index: int, global_seed: int, orig_size: Tuple[int,int]):
        W, H = orig_size
        rng = make_sample_rng(global_seed, sample_index)

        pose_for_bbox = pose_xyc if (pose_xyc is not None and getattr(pose_xyc, "size", 0) > 0) else None
        params = self.sample_params(rng, pose_for_bbox, W, H)

        length = len(rgb_frames)  # list 或 tensor 都支持
        params.frame_indices = self.build_frame_indices(rng, length)

        rgb_aug = self.apply_rgb(rgb_frames, params, rng)

        pose_aug = None
        if pose_xyc is not None:
            pose_aug = self.apply_pose(pose_xyc, params, orig_W=W, orig_H=H, rng=rng)

        text_aug = self.apply_text(text, params, rng)
        # 如需对 rgb_aug / pose_aug 再做时间 mask，可在此调用 apply_time_mask(...)
        return rgb_aug, pose_aug, text_aug, params


if __name__ == "__main__":
    def _smoke_test():
        rng = make_sample_rng(123, 7)
        assert isinstance(rng, np.random.RandomState)

        # bbox
        pts = np.array([
            [10, 10, 0.9],
            [20, 15, 0.8],
            [18, 30, 0.95],
            [np.nan, 0, 0.9],  # 无效
        ])
        bb = bbox_from_keypoints(pts, conf_thr=0.5, margin=0.1, min_box=2.0)
        assert bb is not None
        x1, y1, x2, y2 = bb
        assert x2 > x1 and y2 > y1

        # mirror + swap
        K = 6
        pose = np.zeros((K, 3))
        pose[:, 0] = np.arange(K)  # x = 0..5
        left_ids = [0, 1, 2]
        right_ids = [3, 4, 5]
        mirrored = mirror_pose_x(pose, img_w=100, swap_hands=True, left_ids=left_ids, right_ids=right_ids)
        assert np.allclose(mirrored[right_ids, 0], 99 - pose[left_ids, 0])

        # clamp
        cx1, cy1, cx2, cy2 = clamp_crop_window(-10, -10, 999, 999, W=100, H=80, min_size=8)
        assert 0 <= cx1 < cx2 < 100 and 0 <= cy1 < cy2 < 80

        print("utils OK")

    _smoke_test()


