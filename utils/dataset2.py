import os
import json
import copy
import random

import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from PIL import Image
from decord import VideoReader, cpu
from torchvision import transforms


# 1) 工具函数
# -------------------------
def load_rgb(path, frame_indices, size=(224, 224), transform=None):
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    vr = VideoReader(path, num_threads=1, ctx=cpu(0))
    if len(vr) == 0:
        # 空视频兜底
        return torch.zeros(len(frame_indices), 3, size[1], size[0])
    # 关键：夹紧索引，保持长度不变
    clamped = np.clip(frame_indices, 0, len(vr) - 1)
    buf = vr.get_batch(clamped).asnumpy()
    to_tensor = transforms.ToTensor()
    imgs = [(transform(Image.fromarray(im).convert('RGB').resize(size))
             if transform else to_tensor(Image.fromarray(im).convert('RGB').resize(size)))
            for im in buf]
    return torch.stack(imgs, 0)  # [T, 3, H, W]




def load_kpt(pkl_path, frame_indices, video_path=None):
    """
    返回:
      - skeletons: [T,1,133,2]  (通常 0~1 归一化坐标，保持与 pkl 原样一致)
      - confs:     [T,1,133]
      - meta: {'space':'norm01', 'img_hw': (H,W) 或 None}
    """
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    pose = pickle.load(open(pkl_path, 'rb'))

    total = len(pose['scores'])
    if total == 0:
        T = len(frame_indices)
        return {
            'skeletons': np.zeros((T,1,133,2)),
            'confs':     np.zeros((T,1,133)),
            'meta':      {'space':'norm01','img_hw': None}
        }

    clamped = np.clip(frame_indices, 0, total-1)
    kpts = np.array([pose['keypoints'][i] for i in clamped])  # 0~1
    conf = np.array([pose['scores'][i]    for i in clamped])

    img_hw = None
    if video_path is not None:
        vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
        if len(vr) > 0:
            f0 = vr.get_batch([0]).asnumpy()[0]  # [H,W,C]
            img_hw = (f0.shape[0], f0.shape[1])

    return {'skeletons': kpts, 'confs': conf, 'meta': {'space':'norm01','img_hw': img_hw}}



def load_part_kp(skeletons, confs):
    thr = 0.3
    kps_with_scores, scale, xs, ys, dx, dy = {}, None, None, None, None, None
    try:
        for part in ['body','left','right','face_all']:
            kps, confidences = [], []
            for skeleton, conf in zip(skeletons, confs):
                skeleton = skeleton[0]; conf = conf[0]
                if part == 'body':
                    kp = skeleton[[0]+[i for i in range(3,11)], :]
                    cf = conf   [[0]+[i for i in range(3,11)]]
                elif part == 'left':
                    kp = skeleton[91:112, :]; kp = kp - kp[0,:]; cf = conf[91:112]
                elif part == 'right':
                    kp = skeleton[112:133,:]; kp = kp - kp[0,:]; cf = conf[112:133]
                else:
                    idx = [i for i in list(range(23,40))[::2]] + [i for i in range(83,91)] + [53]
                    kp = skeleton[idx,:]; kp = kp - kp[-1,:]; cf = conf[idx]
                kps.append(kp); confidences.append(cf)

            kps = np.stack(kps,0); confidences = np.stack(confidences,0)
            if part == 'body':
                res, s, xy = crop_scale(np.concatenate([kps, confidences[...,None]], -1), thr)
                scale = s
                if xy is not None:
                    xs, ys, dx, dy = xy[0], xy[1], xy[2], xy[3]
                else:
                    xs = ys = dx = dy = 0.0
            else:
                if not scale:
                    res = np.zeros(np.concatenate([kps, confidences[...,None]], -1).shape)
                else:
                    res = np.concatenate([kps, confidences[...,None]], -1)
                    res[..., :2] = res[..., :2] / scale
                    res = np.clip(res, -1, 1)
                    res[res[...,2] <= thr] = 0
            kps_with_scores[part] = torch.tensor(res, dtype=torch.float32)

    except Exception:
        T = len(skeletons)
        return (
            {'body':torch.zeros((T,9,3)), 'left':torch.zeros((T,21,3)),
             'right':torch.zeros((T,21,3)), 'face_all':torch.zeros((T,18,3))},
            {'scale':0.0,'xs':0.0,'ys':0.0,'scale_x':0.0,'scale_y':0.0}
        )

    return kps_with_scores, {
        'scale': float(scale or 0.0),
        'xs': float(xs or 0.0), 'ys': float(ys or 0.0),
        'scale_x': float(dx or 0.0), 'scale_y': float(dy or 0.0),
    }





def crop_scale(motion, thr):
    """
    motion: [..., 17, 3] 或 [T, K, 3]，xy 为 0~1 归一化坐标，conf 在 motion[...,2]
    返回：
      result: 归一化到 [-1,1]（以正方形视野 s 为边长）
      meta:   {'scale': s, 'scale_x': dx, 'scale_y': dy, 'xs': xs, 'ys': ys}
              这里 s = max(dx, dy)，xs/ys 为 0~1 下的左上角
    """
    try:
        result = copy.deepcopy(motion)
        valid = motion[..., 2] > thr
        valid_coords = motion[valid][..., :2]
        if valid_coords.size < 8:  # 少于4个点
            return np.zeros_like(motion), 0.0, None

        xmin = float(np.min(valid_coords[:, 0])); xmax = float(np.max(valid_coords[:, 0]))
        ymin = float(np.min(valid_coords[:, 1])); ymax = float(np.max(valid_coords[:, 1]))

        dx = xmax - xmin
        dy = ymax - ymin
        if dx <= 0 or dy <= 0:
            return np.zeros_like(motion), 0.0, None

        s = max(dx, dy)              # 正方形边长（归一化单位）
        cx = (xmin + xmax) * 0.5
        cy = (ymin + ymax) * 0.5
        xs = cx - s * 0.5            # 左上角（归一化 0~1）
        ys = cy - s * 0.5

        # 归一化到 [-1, 1]（以正方形 s 为参照）
        result[..., :2] = (motion[..., :2] - [xs, ys]) / s        # → [0,1]
        result[..., :2] = (result[..., :2] - 0.5) * 2.0           # → [-1,1]
        result = np.clip(result, -1, 1)
        result[result[..., 2] <= thr] = 0

        # 用 meta 返回 dx/dy，供像素裁剪更准确
        return result, s, [xs, ys, dx, dy]
    except Exception as e:
        print(f"crop_scale失败: {e}")
        return np.zeros_like(motion), 0.0, None






def split_pose_parts(skeletons, confs, thr=0.3):
    T = skeletons.shape[0]
    sk_list = [skeletons[t] for t in range(T)]
    cf_list = [confs[t]     for t in range(T)]
    pose_sample, crop_meta = load_part_kp(sk_list, cf_list)
    return pose_sample, crop_meta



def load_rgb_with_crop(path, frame_indices, crop_meta_norm01, img_hw, size=(224,224), transform=None):
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    vr = VideoReader(path, num_threads=1, ctx=cpu(0))
    if len(vr) == 0:
        return torch.zeros(len(frame_indices), 3, size[1], size[0])

    clamped = np.clip(frame_indices, 0, len(vr)-1)
    buf = vr.get_batch(clamped).asnumpy()  # [T,H,W,C]

    H, W = (img_hw or (buf.shape[1], buf.shape[2]))
    s01  = float(crop_meta_norm01.get('scale', 0.0))
    xs01 = float(crop_meta_norm01.get('xs', 0.0))
    ys01 = float(crop_meta_norm01.get('ys', 0.0))
    dx01 = float(crop_meta_norm01.get('scale_x', 0.0))
    dy01 = float(crop_meta_norm01.get('scale_y', 0.0))

    # 像素边长：必须分轴再取 max
    side_px = int(round(max(dx01 * W, dy01 * H))) if (dx01>0 and dy01>0) else int(round(s01 * max(W, H)))
    side_px = max(1, side_px)  # 防 0

    # 左上角像素
    xs = int(round(xs01 * W))
    ys = int(round(ys01 * H))

    to_tensor = transforms.ToTensor()
    out = []
    for im in buf:
        x0,y0,x1,y1 = xs, ys, xs+side_px, ys+side_px
        H0,W0 = im.shape[:2]
        ix0,iy0,ix1,iy1 = max(0,x0), max(0,y0), min(W0,x1), min(H0,y1)
        crop = np.zeros((side_px, side_px, 3), dtype=np.uint8)
        if ix1>ix0 and iy1>iy0:
            cx0, cy0 = ix0-x0, iy0-y0
            crop[cy0:cy0+(iy1-iy0), cx0:cx0+(ix1-ix0)] = im[iy0:iy1, ix0:ix1]
        img = Image.fromarray(crop).convert('RGB').resize(size)
        out.append(transform(img) if transform else to_tensor(img))
    return torch.stack(out, 0)




# 如仍需这两个函数，可以保留（与 load_rgb 功能重复，保留以兼容旧代码）
def load_video_support_rgb(path, tmp):
    try:
        vr = VideoReader(path, num_threads=1, ctx=cpu(0))
        buffer = vr.get_batch(np.asarray(tmp, dtype=np.int64)).asnumpy()
        return buffer  # [T, H, W, C]
    except Exception as e:
        print(f"load_video_support_rgb失败 {path}: {e}")
        return np.zeros((len(tmp), 224, 224, 3), dtype=np.uint8)


def process_rgb_images(path, indices, data_transform, target_len):
    try:
        imgs = load_video_support_rgb(path, indices)  # [T, H, W, C]
        rgb_imgs = []
        for img in imgs:
            img = Image.fromarray(img).convert('RGB').resize((224, 224))
            img_tensor = data_transform(img)  # [C, H, W]
            rgb_imgs.append(img_tensor)
        img_tensor = torch.stack(rgb_imgs, dim=0)  # [T, C, H, W]
        T, C, H, W = img_tensor.shape
        if T < target_len:
            pad = img_tensor[-1:].expand(target_len - T, C, H, W)
            img_tensor = torch.cat([img_tensor, pad], dim=0)
        return img_tensor
    except Exception as e:
        print(f"process_rgb_images失败 {path}: {e}")
        return torch.zeros((target_len, 3, 224, 224))




# 2) 基类
# -------------------------
class Base_Dataset(Dataset):
    def __init__(self, args, cfg, phase):
        self.args = args
        self.cfg = cfg
        self.phase = phase

        self.max_length = getattr(args, "max_length", 128)
        self.rgb_support = getattr(args, "rgb_support", True)

        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        # 如果暂时不使用增强，建议先关掉，稳定后再开
        self.enable_aug = (phase == "train") and getattr(args, "use_aug", False)
        self.seed = getattr(cfg, "seed", getattr(args, "seed", 3407))

    def _sample_indices(self, duration):
        """按 max_length 采样时间索引 tmp。"""
        if duration > self.max_length:
            return np.array(sorted(random.sample(range(duration), k=self.max_length)))
        return np.arange(duration)

    def collate_fn(self, batch):
        """
        将样本列表打包成 batch：
        - 对 pose 的每个部位时间 pad 到 T_max
        - attention_mask
        - 对 rgb_img 时间 pad 到 T_rgb_max
        """
        # 解包
        name_batch, pose_list, text_batch, idx_list, support_list = [], [], [], [], []
        for name, pose_sample, text, tmp, support in batch:
            name_batch.append(name)
            pose_list.append(pose_sample)  # dict: part -> [T, K, 3]
            text_batch.append(text)
            idx_list.append(torch.as_tensor(tmp, dtype=torch.long))
            support_list.append(support)

        # 1) 对齐 pose
        src_input = {}
        if not pose_list or not pose_list[0]:
            B = len(batch);
            Tm = self.max_length
            src_input['body'] = torch.zeros((B, Tm, 9, 3))
            src_input['left'] = torch.zeros((B, Tm, 21, 3))
            src_input['right'] = torch.zeros((B, Tm, 21, 3))
            src_input['face_all'] = torch.zeros((B, Tm, 18, 3))
            src_input['attention_mask'] = torch.zeros((B, Tm), dtype=torch.long)
            src_input['src_length_batch'] = torch.zeros(B, dtype=torch.long)

        else:
            pose_parts = list(pose_list[0].keys())  # ['body','left','right','face_all']
            # 计算每个样本的 T（以 body 为准）
            lengths = torch.LongTensor([pose_list[i][pose_parts[0]].shape[0]
                                        for i in range(len(pose_list))])
            T_max = int(lengths.max().item())

            for part in pose_parts:
                padded = []
                for i in range(len(pose_list)):
                    x = pose_list[i][part]  # [T, K, 3] (torch.Tensor)
                    T = x.shape[0]
                    if T < T_max:
                        pad = x[-1:].expand(T_max - T, x.shape[1], x.shape[2])
                        x = torch.cat([x, pad], dim=0)
                    padded.append(x)
                src_input[part] = torch.stack(padded, dim=0)  # [B, T_max, K, 3]

            # attention_mask：1=有效, 0=pad
            attn_mask = pad_sequence(
                [torch.ones(L, dtype=torch.long) for L in lengths],
                batch_first=True, padding_value=0
            )  # [B, T_max]
            src_input['attention_mask'] = attn_mask
            src_input['src_length_batch'] = lengths

        src_input['name_batch'] = name_batch

        # 2) 对齐 RGB（如果启用）
        if self.rgb_support:
            rgb_list = [support.get('rgb_img', None) for support in support_list]
            # 容错：若某项缺失，放一个空帧占位
            for i in range(len(rgb_list)):
                if rgb_list[i] is None:
                    rgb_list[i] = torch.zeros(1, 3, 224, 224)

            T_rgb_list = [x.shape[0] for x in rgb_list]
            T_rgb_max = max(T_rgb_list) if T_rgb_list else 1

            rgb_padded = []
            for x in rgb_list:
                T, C, H, W = x.shape
                if T < T_rgb_max:
                    pad = x[-1:].expand(T_rgb_max - T, C, H, W)
                    x = torch.cat([x, pad], dim=0)
                rgb_padded.append(x)

            src_input['rgb_img'] = torch.stack(rgb_padded, dim=0)  # [B, T_rgb_max, 3, 224, 224]
            src_input['rgb_img_indices'] = [s.get('rgb_img_indices') for s in support_list]
            src_input['rgb_img_len'] = T_rgb_list

        # 3) 目标打包
        tgt_input = {
            'gt_sentence': text_batch,
        }

        return src_input, tgt_input


# 3) 子类（以 CSLNews / CSLDaily / BOBSL 为例）
# -------------------------
class CSLNewsDataset(Base_Dataset):
    def __init__(self, label_path, args, phase, cfg):
        super().__init__(args, cfg, phase)
        with open(label_path, "r", encoding="utf-8") as f:
            self.annotation = json.load(f)

        # 使用getattr安全获取配置
        self.pose_dir = getattr(cfg.data_path, 'pose_dirs', './pose')
        self.rgb_dir = getattr(cfg.data_path, 'rgb_dirs', './rgb')

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        try:
            rec = self.annotation[index]
            text = rec.get('text', '')
            vname = rec.get('video', '')
            pkl = rec.get('pose', '')

            if not vname or not pkl:
                raise ValueError(f"数据记录缺少视频或姿态文件信息: {rec}")

            video_path = os.path.join(self.rgb_dir, vname)
            pose_path = os.path.join(self.pose_dir, pkl)

            # 检查文件是否存在
            if not os.path.exists(pose_path):
                raise FileNotFoundError(f"姿态文件不存在: {pose_path}")

            # 仅加载一次 pkl，既拿 duration 也可用于后续抽帧
            pose_all = pickle.load(open(pose_path, 'rb'))
            duration = len(pose_all['scores'])
            tmp = self._sample_indices(duration)

            # pose：按 tmp 抽帧
            # pose
            kpt_raw = load_kpt(pose_path, tmp, video_path=video_path)
            pose_sample, crop_meta_norm01 = split_pose_parts(kpt_raw['skeletons'], kpt_raw['confs'])
            H, W = (kpt_raw['meta']['img_hw'] or (None, None))


            support = {'video_name': vname, 'rgb_img_indices': np.asarray(tmp, dtype=np.int64)}
            if self.rgb_support and os.path.exists(video_path):
                rgb_seq = load_rgb(
                    video_path, tmp,
                    size=(224, 224), transform=self.data_transform
                )
                support['rgb_img'] = rgb_seq
                # 这里不再使用 crop_meta_norm01，因为我们没做裁剪
                support['crop_meta'] = {'scale': 0.0, 'xs': 0.0, 'ys': 0.0,
                                        'scale_x': 0.0, 'scale_y': 0.0}
            else:
                support['rgb_img'] = torch.zeros(len(tmp), 3, 224, 224)
                support['crop_meta'] = {'scale': 0.0, 'xs': 0.0, 'ys': 0.0,
                                        'scale_x': 0.0, 'scale_y': 0.0}

            return vname, pose_sample, text, tmp, support



        except Exception as e:
            print(f"CSLNewsDataset加载数据失败 index={index}: {e}")
            # 占位样本（按真实 K 值）
            placeholder_pose = {
                'body': torch.zeros((self.max_length, 9, 3)),
                'left': torch.zeros((self.max_length, 21, 3)),
                'right': torch.zeros((self.max_length, 21, 3)),
                'face_all': torch.zeros((self.max_length, 18, 3)),
            }
            placeholder_indices = np.arange(self.max_length, dtype=np.int64)
            support = {
                'rgb_img': torch.zeros(self.max_length, 3, 224, 224),
                'rgb_img_indices': placeholder_indices,  # 便于一致性
                'video_name': f"error_video_{index}",
            }

            return f"error_{index}", placeholder_pose, "", placeholder_indices, support


class CSLDailyDataset(CSLNewsDataset):
    """CSLDaily数据集"""
    pass


class BOBSLDataset(CSLNewsDataset):
    """BOBSL数据集"""
    pass


class PhoenixDataset(CSLNewsDataset):
    """Phoenix数据集"""
    pass


# 4) 工厂
# -------------------------
def create_dataloader(args, cfg, phase='train'):
    label_paths = {
        'train': getattr(cfg.data_path, 'train_label_paths', None),
        'val': getattr(cfg.data_path, 'val_label_paths', None),
        'test': getattr(cfg.data_path, 'test_label_paths', None),
    }

    label_path = label_paths[phase]
    if label_path is None:
        raise ValueError(f"{phase}阶段的标签路径未配置")

    if args.dataset_name == 'CSL_News':
        dataset = CSLNewsDataset(label_path, args, phase, cfg)
    elif args.dataset_name == 'CSL_Daily':
        dataset = CSLDailyDataset(label_path, args, phase, cfg)
    elif args.dataset_name == 'BOBSL':
        dataset = BOBSLDataset(label_path, args, phase, cfg)
    elif args.dataset_name == 'Phoenix':
        dataset = PhoenixDataset(label_path, args, phase, cfg)
    else:
        raise NotImplementedError(f"不支持的数据集: {args.dataset_name}")

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(phase == 'train'),
        num_workers=getattr(args, 'num_workers', 4),
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        drop_last=(phase == 'train')  # 训练时丢弃最后一个不完整的batch
    )