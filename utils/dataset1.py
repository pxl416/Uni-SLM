import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from PIL import Image
import os
import random
import numpy as np
import copy
import pickle
from decord import VideoReader, cpu
import json
import pathlib
from torchvision import transforms
from typing import Optional
from utils.augment1 import MultiModalAugmentor, make_sample_rng
from PIL import Image



# load sub-pose
def load_part_kp(skeletons, confs, force_ok=False):
    thr = 0.3
    kps_with_scores = {}
    scale = None

    for part in ['body', 'left', 'right', 'face_all']:
        kps = []
        confidences = []

        for skeleton, conf in zip(skeletons, confs):
            skeleton = skeleton[0]
            conf = conf[0]

            if part == 'body':
                hand_kp2d = skeleton[[0] + [i for i in range(3, 11)], :]
                confidence = conf[[0] + [i for i in range(3, 11)]]
            elif part == 'left':
                hand_kp2d = skeleton[91:112, :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]
                confidence = conf[91:112]
            elif part == 'right':
                hand_kp2d = skeleton[112:133, :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]
                confidence = conf[112:133]
            elif part == 'face_all':
                hand_kp2d = skeleton[[i for i in list(range(23, 23 + 17))[::2]] + [i for i in range(83, 83 + 8)] + [53],
                            :]
                hand_kp2d = hand_kp2d - hand_kp2d[-1, :]
                confidence = conf[[i for i in list(range(23, 23 + 17))[::2]] + [i for i in range(83, 83 + 8)] + [53]]

            else:
                raise NotImplementedError

            kps.append(hand_kp2d)
            confidences.append(confidence)

        kps = np.stack(kps, axis=0)
        confidences = np.stack(confidences, axis=0)

        if part == 'body':
            if force_ok:
                result, scale, _ = crop_scale(np.concatenate([kps, confidences[..., None]], axis=-1), thr)

            else:
                result, scale, _ = crop_scale(np.concatenate([kps, confidences[..., None]], axis=-1), thr)
        else:
            assert not scale is None
            result = np.concatenate([kps, confidences[..., None]], axis=-1)
            if scale == 0:
                result = np.zeros(result.shape)
            else:
                result[..., :2] = (result[..., :2]) / scale
                result = np.clip(result, -1, 1)
                # mask useless kp
                result[result[..., 2] <= thr] = 0

        kps_with_scores[part] = torch.tensor(result)

    return kps_with_scores


def load_kpt(pkl_path, frame_indices, video_path=None):
    """ 读出关键点 + 置信度，并带上 img_hw 与坐标空间标记： - skeletons: [T,1,133,2]，通常为 0~1 归一化坐标（和你原始 pkl 一致） - confs: [T,1,133] - meta: {'space': 'norm01', 'img_hw': (H,W) or None} """
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    pose = pickle.load(open(pkl_path, 'rb'))
    total = len(pose['scores'])
    if total == 0:
        T = len(frame_indices)
        return {'skeletons': np.zeros((T,1,133,2)), 'confs': np.zeros((T,1,133)), 'meta': {'space':'norm01','img_hw': None}}
    clamped = np.clip(frame_indices, 0, total-1)
    kpts = np.array([pose['keypoints'][i] for i in clamped]) # [T,1,133,2] 0~1
    conf = np.array([pose['scores'][i] for i in clamped]) # [T,1,133] # 可选：如果给了视频路径，就顺手取一下 (H,W)，方便后续把 0~1 → 像素
    img_hw = None
    if video_path is not None:
        vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
        if len(vr) > 0:
            f0 = vr.get_batch([0]).asnumpy()[0] # [H,W,C]
            img_hw = (f0.shape[0], f0.shape[1]) # (H,W)
    return {'skeletons': kpts, 'confs': conf, 'meta': {'space':'norm01','img_hw': img_hw}}


# input: T, N, 3
# input is un-normed joints
def crop_scale(motion, thr):
    '''
        Motion: [(M), T, 17, 3].
        Normalize to [-1, 1]
    '''
    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 2]>thr][:,:2]
    if len(valid_coords) < 4:
        return np.zeros(motion.shape), 0, None
    xmin = min(valid_coords[:,0])
    xmax = max(valid_coords[:,0])
    ymin = min(valid_coords[:,1])
    ymax = max(valid_coords[:,1])
    # ratio = np.random.uniform(low=scale_range[0], high=scale_range[1], size=1)[0]
    ratio = 1
    scale = max(xmax-xmin, ymax-ymin) * ratio
    if scale==0:
        return np.zeros(motion.shape), 0, None
    xs = (xmin+xmax-scale) / 2
    ys = (ymin+ymax-scale) / 2
    result[...,:2] = (motion[..., :2] - [xs,ys]) / scale
    result[...,:2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    # mask useless kp
    result[result[...,2]<=thr] = 0
    return result, scale, [xs,ys]


def load_support_rgb_dict(tmp, skeletons, confs, full_path, data_transform):
    support_rgb_dict = {}

    confs = np.array(confs)
    skeletons = np.array(skeletons)

    # sample index of low scores
    left_confs_filter = confs[:, 0, 91:112].mean(-1)
    left_confs_filter_indices = np.where(left_confs_filter > 0.3)[0]

    if len(left_confs_filter_indices) == 0:
        left_sampled_indices = None
        left_skeletons = None
    else:

        left_confs = confs[left_confs_filter_indices]
        left_confs = left_confs[:, 0, [95, 99, 103, 107, 111]].min(-1)

        left_weights = np.max(left_confs) - left_confs + 1e-5
        left_probabilities = left_weights / np.sum(left_weights)

        left_sample_size = int(np.ceil(0.1 * len(left_confs_filter_indices)))

        left_sampled_indices = np.random.choice(left_confs_filter_indices.tolist(),
                                                size=left_sample_size,
                                                replace=False,
                                                p=left_probabilities)
        # left_sampled_indices: values: 0-255(0,max_len)
        # tmp: values: 0-(end-start)
        left_sampled_indices = np.sort(left_sampled_indices)

        left_skeletons = skeletons[left_sampled_indices, 0, 91:112]

    right_confs_filter = confs[:, 0, 112:].mean(-1)
    right_confs_filter_indices = np.where(right_confs_filter > 0.3)[0]
    if len(right_confs_filter_indices) == 0:
        right_sampled_indices = None
        right_skeletons = None

    else:
        right_confs = confs[right_confs_filter_indices]
        right_confs = right_confs[:, 0, [95 + 21, 99 + 21, 103 + 21, 107 + 21, 111 + 21]].min(-1)

        right_weights = np.max(right_confs) - right_confs + 1e-5
        right_probabilities = right_weights / np.sum(right_weights)

        right_sample_size = int(np.ceil(0.1 * len(right_confs_filter_indices)))

        right_sampled_indices = np.random.choice(right_confs_filter_indices.tolist(),
                                                 size=right_sample_size,
                                                 replace=False,
                                                 p=right_probabilities)
        right_sampled_indices = np.sort(right_sampled_indices)

        right_skeletons = skeletons[right_sampled_indices, 0, 112:133]

    image_size = 112
    all_indices = []
    if not left_sampled_indices is None:
        all_indices.append(left_sampled_indices)
    if not right_sampled_indices is None:
        all_indices.append(right_sampled_indices)
    if len(all_indices) == 0:
        support_rgb_dict['left_sampled_indices'] = torch.tensor([-1])
        support_rgb_dict['left_hands'] = torch.zeros(1, 3, image_size, image_size)
        support_rgb_dict['left_skeletons_norm'] = torch.zeros(1, 21, 2)

        support_rgb_dict['right_sampled_indices'] = torch.tensor([-1])
        support_rgb_dict['right_hands'] = torch.zeros(1, 3, image_size, image_size)
        support_rgb_dict['right_skeletons_norm'] = torch.zeros(1, 21, 2)

        return support_rgb_dict

    sampled_indices = np.concatenate(all_indices)
    sampled_indices = np.unique(sampled_indices)
    sampled_indices_real = tmp[sampled_indices]

    # load image sample
    imgs = load_video_support_rgb(full_path, sampled_indices_real)

    # get hand bbox
    left_new_box, right_new_box, box_hw = bbox_4hands(left_skeletons,
                                                      right_skeletons,
                                                      imgs[0].shape[:2])

    # crop left and right hand
    image_size = 112
    if box_hw == 0:
        support_rgb_dict['left_sampled_indices'] = torch.tensor([-1])
        support_rgb_dict['left_hands'] = torch.zeros(1, 3, image_size, image_size)
        support_rgb_dict['left_skeletons_norm'] = torch.zeros(1, 21, 2)

        support_rgb_dict['right_sampled_indices'] = torch.tensor([-1])
        support_rgb_dict['right_hands'] = torch.zeros(1, 3, image_size, image_size)
        support_rgb_dict['right_skeletons_norm'] = torch.zeros(1, 21, 2)

        return support_rgb_dict

    factor = image_size / box_hw

    if left_sampled_indices is None:
        left_hands = torch.zeros(1, 3, image_size, image_size)
        left_skeletons_norm = torch.zeros(1, 21, 2)

    else:
        left_hands = torch.zeros(len(left_sampled_indices), 3, image_size, image_size)

        left_skeletons_norm = left_skeletons * imgs[0].shape[:2][::-1] - left_new_box[:, None, [0, 1]]
        left_skeletons_norm = left_skeletons_norm / box_hw
        left_skeletons_norm = left_skeletons_norm.clip(0, 1)

    if right_sampled_indices is None:
        right_hands = torch.zeros(1, 3, image_size, image_size)
        right_skeletons_norm = torch.zeros(1, 21, 2)

    else:
        right_hands = torch.zeros(len(right_sampled_indices), 3, image_size, image_size)

        right_skeletons_norm = right_skeletons * imgs[0].shape[:2][::-1] - right_new_box[:, None, [0, 1]]
        right_skeletons_norm = right_skeletons_norm / box_hw
        right_skeletons_norm = right_skeletons_norm.clip(0, 1)
    left_idx = 0
    right_idx = 0

    for idx, img in enumerate(imgs):
        mapping_idx = sampled_indices[idx]
        if not left_sampled_indices is None and left_idx < len(left_sampled_indices) and mapping_idx == \
                left_sampled_indices[left_idx]:
            box = left_new_box[left_idx]

            img_draw = np.uint8(copy.deepcopy(img))[box[1]:box[3], box[0]:box[2], :]
            img_draw = np.pad(img_draw,
                              ((0, max(0, box_hw - img_draw.shape[0])), (0, max(0, box_hw - img_draw.shape[1])),
                               (0, 0)), mode='constant', constant_values=0)

            f_img = Image.fromarray(img_draw).convert('RGB').resize((image_size, image_size))
            f_img = data_transform(f_img).unsqueeze(0)
            left_hands[left_idx] = f_img
            left_idx += 1

        if not right_sampled_indices is None and right_idx < len(right_sampled_indices) and mapping_idx == \
                right_sampled_indices[right_idx]:
            box = right_new_box[right_idx]

            img_draw = np.uint8(copy.deepcopy(img))[box[1]:box[3], box[0]:box[2], :]
            img_draw = np.pad(img_draw,
                              ((0, max(0, box_hw - img_draw.shape[0])), (0, max(0, box_hw - img_draw.shape[1])),
                               (0, 0)), mode='constant', constant_values=0)

            f_img = Image.fromarray(img_draw).convert('RGB').resize((image_size, image_size))
            f_img = data_transform(f_img).unsqueeze(0)
            right_hands[right_idx] = f_img
            right_idx += 1

    if left_sampled_indices is None:
        left_sampled_indices = np.array([-1])

    if right_sampled_indices is None:
        right_sampled_indices = np.array([-1])

    # get index, images and keypoints priors
    support_rgb_dict['left_sampled_indices'] = torch.tensor(left_sampled_indices)
    support_rgb_dict['left_hands'] = left_hands
    support_rgb_dict['left_skeletons_norm'] = torch.tensor(left_skeletons_norm)

    support_rgb_dict['right_sampled_indices'] = torch.tensor(right_sampled_indices)
    support_rgb_dict['right_hands'] = right_hands
    support_rgb_dict['right_skeletons_norm'] = torch.tensor(right_skeletons_norm)

    return support_rgb_dict


# bbox of hands
def bbox_4hands(left_keypoints, right_keypoints, hw):
    # keypoints --> T,21,2
    # keypoints --> T,21,2

    def compute_bbox(keypoints):
        min_x = np.min(keypoints[..., 0], axis=1)
        min_y = np.min(keypoints[..., 1], axis=1)
        max_x = np.max(keypoints[..., 0], axis=1)
        max_y = np.max(keypoints[..., 1], axis=1)

        return (max_x + min_x) / 2, (max_y + min_y) / 2, (max_x - min_x), (max_y - min_y)

    H, W = hw

    if left_keypoints is None:
        left_keypoints = np.zeros([1, 21, 2])

    if right_keypoints is None:
        right_keypoints = np.zeros([1, 21, 2])
    # [T, 21, 2]
    left_mean_x, left_mean_y, left_diff_x, left_diff_y = compute_bbox(left_keypoints)
    left_mean_x = W * left_mean_x
    left_mean_y = H * left_mean_y

    left_diff_x = W * left_diff_x
    left_diff_y = H * left_diff_y

    left_diff_x = max(left_diff_x)
    left_diff_y = max(left_diff_y)
    left_box_hw = max(left_diff_x, left_diff_y)

    right_mean_x, right_mean_y, right_diff_x, right_diff_y = compute_bbox(right_keypoints)
    right_mean_x = W * right_mean_x
    right_mean_y = H * right_mean_y

    right_diff_x = W * right_diff_x
    right_diff_y = H * right_diff_y

    right_diff_x = max(right_diff_x)
    right_diff_y = max(right_diff_y)
    right_box_hw = max(right_diff_x, right_diff_y)

    box_hw = int(max(left_box_hw, right_box_hw) * 1.2 / 2) * 2
    box_hw = max(box_hw, 0)

    left_new_box = np.stack([left_mean_x - box_hw / 2, left_mean_y - box_hw / 2, left_mean_x + box_hw / 2,
                             left_mean_y + box_hw / 2]).astype(np.int16)
    right_new_box = np.stack([right_mean_x - box_hw / 2, right_mean_y - box_hw / 2, right_mean_x + box_hw / 2,
                              right_mean_y + box_hw / 2]).astype(np.int16)

    return left_new_box.transpose(1, 0), right_new_box.transpose(1, 0), box_hw


# use split rgb video for save time
def load_video_support_rgb(path, tmp):
    vr = VideoReader(path, num_threads=1, ctx=cpu(0))

    vr.seek(0)
    buffer = vr.get_batch(tmp).asnumpy()
    batch_image = buffer
    del vr

    return batch_image


def process_rgb_images(path, indices, data_transform, target_len):
    imgs = load_video_support_rgb(path, indices)  # [T, H, W, C]
    rgb_imgs = []

    for img in imgs:
        img = Image.fromarray(img).convert('RGB').resize((112, 112))
        img_tensor = data_transform(img)  # [C, H, W]
        rgb_imgs.append(img_tensor)

    img_tensor = torch.stack(rgb_imgs, dim=0)  # [T, C, H, W]

    # pad 到 target_len
    T, C, H, W = img_tensor.shape
    if T < target_len:
        pad = img_tensor[-1:].expand(target_len - T, C, H, W)
        img_tensor = torch.cat([img_tensor, pad], dim=0)

    return img_tensor




# build base dataset
class Base_Dataset(Dataset):
    def collate_fn(self, batch):
        tgt_batch ,src_length_batch ,name_batch ,pose_tmp ,gloss_batch = [] ,[] ,[] ,[] ,[]

        for name_sample, pose_sample, text, gloss, _ in batch:
            name_batch.append(name_sample)
            pose_tmp.append(pose_sample)
            tgt_batch.append(text)
            gloss_batch.append(gloss)

        src_input = {}

        keys = pose_tmp[0].keys()
        for key in keys:
            max_len = max([len(vid[key]) for vid in pose_tmp])
            video_length = torch.LongTensor([len(vid[key]) for vid in pose_tmp])

            padded_video = [torch.cat(
                (
                    vid[key],
                    vid[key][-1][None].expand(max_len - len(vid[key]), -1, -1),
                )
                , dim=0)
                for vid in pose_tmp]

            img_batch = torch.stack(padded_video ,0)

            src_input[key] = img_batch
            if 'attention_mask' not in src_input.keys():
                src_length_batch = video_length

                mask_gen = []
                for i in src_length_batch:
                    tmp = torch.ones([i]) + 7
                    mask_gen.append(tmp)
                mask_gen = pad_sequence(mask_gen, padding_value=0 ,batch_first=True)
                img_padding_mask = (mask_gen != 0).long()
                src_input['attention_mask'] = img_padding_mask

                src_input['name_batch'] = name_batch
                src_input['src_length_batch'] = src_length_batch

        if self.rgb_support:
            support_rgb_dicts = {key :[] for key in batch[0][-1].keys()}
            for _, _, _, _, support_rgb_dict in batch:
                for key in support_rgb_dict.keys():
                    support_rgb_dicts[key].append(support_rgb_dict[key])

            # Step 1: 获取视频路径和采样帧
            rgb_imgs = []

            rgb_img_indices_batch = []
            rgb_img_len_batch = []

            max_rgb_len = 0
            for i, (_, _, _, rgb_indices, support_rgb_dict) in enumerate(batch):
                video_name = support_rgb_dict['video_name']
                full_path = os.path.join(self.rgb_dir, video_name)
                rgb_tensor = process_rgb_images(full_path, rgb_indices, self.data_transform,
                                                target_len=len(rgb_indices))
                rgb_imgs.append(rgb_tensor)
                max_rgb_len = max(max_rgb_len, rgb_tensor.shape[0])

                # rgb_imgs.append(rgb_tensor)

                rgb_img_indices_batch.append(torch.tensor(rgb_indices))  # 记录索引
                rgb_img_len_batch.append(len(rgb_indices))  # 记录帧数

            # Step 2: pad 到 max_rgb_len 并堆叠
            padded_imgs = []
            for rgb_tensor in rgb_imgs:
                T, C, H, W = rgb_tensor.shape
                if T < max_rgb_len:
                    pad = rgb_tensor[-1:].expand(max_rgb_len - T, C, H, W)
                    rgb_tensor = torch.cat([rgb_tensor, pad], dim=0)
                padded_imgs.append(rgb_tensor)

            src_input['rgb_img'] = torch.stack(padded_imgs, dim=0)  # [B, T, 3, 112, 112]
            src_input['rgb_img_indices'] = rgb_img_indices_batch
            src_input['rgb_img_len'] = rgb_img_len_batch

            for part in ['left', 'right']:
                index_key = f'{part}_sampled_indices'
                skeletons_key = f'{part}_skeletons_norm'
                rgb_key = f'{part}_hands'
                len_key = f'{part}_rgb_len'

                index_batch = torch.cat(support_rgb_dicts[index_key], dim=0)
                skeletons_batch = torch.cat(support_rgb_dicts[skeletons_key], dim=0)
                # img_batch = torch.stack(support_rgb_dicts[rgb_key], dim=0)  # ✅ 正确拼接为 [B, T, C, H, W]
                # Step 1: 获取每个样本的帧数
                T_list = [x.shape[0] for x in support_rgb_dicts[rgb_key]]
                max_T = max(T_list)

                # Step 2: 对每个样本进行帧数对齐（时间维 pad）
                padded_imgs = []
                for x in support_rgb_dicts[rgb_key]:
                    T, C, H, W = x.shape
                    if T < max_T:
                        pad = x[-1].unsqueeze(0).expand(max_T - T, C, H, W)  # 补最后一帧
                        x = torch.cat([x, pad], dim=0)
                    padded_imgs.append(x)

                # Step 3: 堆叠为 [B, T, C, H, W]
                img_batch = torch.stack(padded_imgs, dim=0)

                src_input[index_key] = index_batch
                src_input[skeletons_key] = skeletons_batch
                src_input[rgb_key] = img_batch
                src_input[len_key] = [len(index) for index in support_rgb_dicts[index_key]]

        tgt_input = {}
        tgt_input['gt_sentence'] = tgt_batch
        tgt_input['gt_gloss'] = gloss_batch

        return src_input, tgt_input


def _validate_sample(self, video_path: str, pose_path: Optional[str], text: str):
    errs = []
    if not os.path.exists(video_path): errs.append(f"missing video: {video_path}")
    if pose_path and not os.path.exists(pose_path): errs.append(f"missing pose: {pose_path}")
    if not text or not str(text).strip(): errs.append("empty text")
    if errs:
        # 这里可选择记录 log 并 raise
        raise ValueError("; ".join(errs))




class CSLNewsDataset(Base_Dataset):
    def __init__(self, label_path, args, phase, cfg):
        super().__init__()

        # 保存基础信息
        self.args = args
        self.phase = phase
        self.max_length = getattr(args, "max_length", 128)
        self.rgb_support = getattr(args, "rgb_support", False)

        # 读取标注文件
        with open(label_path, "r", encoding="utf-8") as f:
            self.annotation = json.load(f)

        # 设置路径（通过 args 中的路径或 config 中的映射）
        if args.dataset_name == "CSL_News":
            self.pose_dir = cfg.data_path.pose_dirs
            self.rgb_dir = cfg.data_path.rgb_dirs

        else:
            raise NotImplementedError(f"Dataset {args.dataset_name} not supported.")

        self.data_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    ])
        # 是否启用增强（仅训练期且允许）
        self.enable_aug = (phase == "train") and getattr(args, "use_aug", True)

        # 输出尺寸（增强器内部使用）
        if hasattr(cfg, "augmentation") and hasattr(cfg.augmentation, "output") and hasattr(cfg.augmentation.output,
                                                                                            "size"):
            out_w, out_h = tuple(cfg.augmentation.output.size)
        else:
            out_w, out_h = 224, 224  # 默认

        # 左右手关键点 id（若没有就留空，flip 时不交换）
        LEFT_IDS = getattr(cfg, "left_kp_ids", [])
        RIGHT_IDS = getattr(cfg, "right_kp_ids", [])

        # 构造增强器
        aug_cfg = getattr(cfg, "augmentation", {})
        self.aug = MultiModalAugmentor(cfg=aug_cfg, image_size=(out_w, out_h),
                                       left_ids=LEFT_IDS, right_ids=RIGHT_IDS)

        # 全局种子（优先 cfg.seed，其次 args.seed，最后保底）
        self.global_seed = getattr(cfg, "seed", getattr(args, "seed", 3407))

        # 设置数据范围（默认 99% 用于训练，1% 用于验证/测试）
        total_samples = len(self.annotation)
        if phase == "train":
            self.start_idx = int(0.0 * total_samples)
            self.end_idx = int(0.99 * total_samples)
        else:
            self.start_idx = int(0.99 * total_samples)
            self.end_idx = total_samples


    def __len__(self):
        return self.end_idx - self.start_idx

    def __getitem__(self, index):
        num_retries = 10
        for _ in range(num_retries):
            sample = self.annotation[self.start_idx + index]
            text = sample['text']
            name_sample = sample['video']
            try:
                pose_sample, support_rgb_dict = self.load_pose(sample['pose'], sample['video'])
            except Exception:
                import traceback;
                traceback.print_exc()
                print(f"Failed to load examples with video: {name_sample}. Will randomly sample a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            break
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")

        # ====== 最小可用增强（pose-only） ======
        if self.enable_aug:
            # 用 body 的 T 作为 clip 长度（你的 load_part_kp 返回 'body','left','right','face_all'）
            T_len = pose_sample['body'].shape[0]

            # 原图尺寸：当前没有真实 W/H，就用增强器目标尺寸；后续若能从视频读取，替换这里即可
            orig_W, orig_H = self.aug.im_w, self.aug.im_h

            # 样本级 RNG（可复现）
            rng = make_sample_rng(self.global_seed, index)

            # 采样一次参数（先不启用 bbox 裁剪，等你调好再开）
            params = self.aug.sample_params(rng, pose_for_bbox=None, W=orig_W, H=orig_H)
            params.frame_indices = self.aug.build_frame_indices(rng, T_len)

            # 对每个部位同步几何与噪声（与 RGB 同参；RGB 以后再接）
            for part in list(pose_sample.keys()):
                tens = pose_sample[part]
                np_pose = tens.detach().cpu().numpy()
                np_pose_aug = self.aug.apply_pose(np_pose, params, orig_W=orig_W, orig_H=orig_H, rng=rng)
                pose_sample[part] = torch.from_numpy(np_pose_aug).type_as(tens)

        # ====== 增强结束 ======

        return name_sample, pose_sample, text, support_rgb_dict.get('rgb_img_indices', []), support_rgb_dict

    def load_pose(self, pose_name, rgb_name):
        pose = pickle.load(open(os.path.join(self.pose_dir, pose_name), 'rb'))
        full_path = os.path.join(self.rgb_dir, rgb_name)

        duration = len(pose['scores'])

        if duration > self.max_length:
            tmp = sorted(random.sample(range(duration), k=self.max_length))
        else:
            tmp = list(range(duration))

        tmp = np.array(tmp)

        # dict_keys(['keypoints', 'scores'])
        # keypoints (1, 133, 2)
        # scores (1, 133)

        skeletons = pose['keypoints']
        confs = pose['scores']
        skeletons_tmp = []
        confs_tmp = []

        for index in tmp:
            skeletons_tmp.append(skeletons[index])
            confs_tmp.append(confs[index])

        skeletons = skeletons_tmp
        confs = confs_tmp

        kps_with_scores = load_part_kp(skeletons, confs)

        support_rgb_dict = {}
        if self.rgb_support:
            # support_rgb_dict = load_support_rgb_dict(tmp, skeletons, confs, full_path, self.data_transform)

            support_rgb_dict = load_support_rgb_dict(tmp, skeletons, confs, full_path, self.data_transform)
            support_rgb_dict['video_name'] = rgb_name  # 💡添加 video_name
            support_rgb_dict['rgb_img_indices'] = tmp  # 💡保留采样帧 index（tmp 本身是 sorted index）

        return kps_with_scores, support_rgb_dict

    def __str__(self):
        return f'#total {len(self)}'




def create_dataloader(args, cfg, phase='train'):
    if args.dataset_name == 'CSL_News':
        label_path = {
            'train': cfg.data_path.train_label_paths,
            'val': cfg.data_path.val_label_paths,
            'test': cfg.data_path.test_label_paths
        }[phase]

        dataset = CSLNewsDataset(label_path=label_path, args=args, cfg=cfg, phase=phase)

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(phase == 'train'),
            num_workers=getattr(args, 'num_workers', 4),
            pin_memory=True,
            collate_fn=dataset.collate_fn
        )
        return dataloader
    else:
        raise NotImplementedError(f"Dataset {args.dataset_name} not supported.")


if __name__ == "__main__":
    import argparse
    import sys
    import tempfile
    import numpy as np
    from PIL import Image
    import pickle
    import json
    import random


    # 创建临时测试数据
    def create_test_data():
        """创建测试用的姿态和视频数据"""
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        print(f"创建测试数据在: {temp_dir}")

        # 创建测试姿态数据
        test_pose = {
            'keypoints': np.random.rand(30, 1, 133, 2).astype(np.float32),  # 30帧，133个关键点
            'scores': np.random.rand(30, 1, 133).astype(np.float32)  # 30帧，133个置信度
        }

        pose_path = f"{temp_dir}/test_pose.pkl"
        with open(pose_path, 'wb') as f:
            pickle.dump(test_pose, f)

        # 创建测试标注
        test_annotation = [{
            'text': '这是一个测试句子',
            'video': 'test_video.mp4',  # 注意：这里只是文件名，实际不会读取视频
            'pose': 'test_pose.pkl'
        } for _ in range(3)]  # 创建3个样本

        annotation_path = f"{temp_dir}/test_annotation.json"
        with open(annotation_path, 'w', encoding='utf-8') as f:
            json.dump(test_annotation, f, ensure_ascii=False)

        return temp_dir, annotation_path, pose_path


    def test_pose_functions():
        """测试姿态相关函数"""
        print("=== 测试姿态处理函数 ===")

        # 创建测试数据
        skeletons = [np.random.rand(1, 133, 2) for _ in range(10)]
        confs = [np.random.rand(1, 133) for _ in range(10)]

        try:
            # 测试 load_part_kp
            kps_with_scores = load_part_kp(skeletons, confs, force_ok=True)
            print(f"✓ load_part_kp 测试通过")
            print(f"  输出包含: {list(kps_with_scores.keys())}")
            for key, value in kps_with_scores.items():
                print(f"  {key}: {value.shape}")

            # 测试 crop_scale
            motion = np.random.rand(5, 17, 3)  # 5帧，17个关键点，3维(x,y,conf)
            result, scale, center = crop_scale(motion, thr=0.3)
            print(f"✓ crop_scale 测试通过")
            print(f"  输出形状: {result.shape}, 缩放比例: {scale}")

            return True
        except Exception as e:
            print(f"✗ 姿态函数测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False


    def test_dataset_loading_no_rgb():
        """测试数据集加载（无RGB支持）"""
        print("\n=== 测试数据集加载（无RGB支持） ===")

        try:
            # 创建模拟配置
            class MockArgs:
                dataset_name = "CSL_News"
                batch_size = 2
                max_length = 20
                rgb_support = False  # 关闭RGB支持
                num_workers = 0

            class MockConfig:
                class data_path:
                    pose_dirs = temp_dir
                    rgb_dirs = temp_dir  # 即使不用也要提供
                    train_label_paths = annotation_path

            args = MockArgs()
            cfg = MockConfig()

            # 测试数据集实例化
            dataset = CSLNewsDataset(
                label_path=annotation_path,
                args=args,
                cfg=cfg,
                phase='train'
            )
            print(f"✓ 数据集实例化成功")
            print(f"  数据集大小: {len(dataset)}")

            # 测试单个样本加载 - 需要修改CSLNewsDataset的__getitem__方法
            # 由于视频文件不存在，我们需要跳过视频加载部分
            # 这里我们直接测试load_pose方法
            pose_sample, support_rgb_dict = dataset.load_pose('test_pose.pkl', 'test_video.mp4')
            print(f"✓ load_pose 测试通过")
            print(f"  姿态数据键: {list(pose_sample.keys())}")
            print(f"  RGB支持数据: {support_rgb_dict}")  # 应该为空或包含默认值

            return True

        except Exception as e:
            print(f"✗ 数据集加载测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False


    def test_rgb_functions():
        """测试RGB相关函数"""
        print("\n=== 测试RGB处理函数 ===")

        try:
            # 创建测试图像
            test_img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)

            # 测试数据转换
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            img_tensor = data_transform(test_img)
            print(f"✓ 图像转换测试通过")
            print(f"  输出形状: {img_tensor.shape}")

            # 测试process_rgb_images函数（不实际读取视频）
            # 创建模拟的图像数据
            dummy_imgs = [np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8) for _ in range(5)]
            dummy_target_len = 10

            # 手动模拟process_rgb_images的功能
            rgb_imgs = []
            for img in dummy_imgs:
                img_pil = Image.fromarray(img).convert('RGB').resize((112, 112))
                img_tensor = data_transform(img_pil)
                rgb_imgs.append(img_tensor)

            img_tensor = torch.stack(rgb_imgs, dim=0)  # [T, C, H, W]

            # pad 到 target_len
            T, C, H, W = img_tensor.shape
            if T < dummy_target_len:
                pad = img_tensor[-1:].expand(dummy_target_len - T, C, H, W)
                img_tensor = torch.cat([img_tensor, pad], dim=0)

            print(f"✓ process_rgb_images 模拟测试通过")
            print(f"  输出形状: {img_tensor.shape}")

            return True

        except Exception as e:
            print(f"✗ RGB函数测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False


    def test_collate_fn():
        """测试collate函数"""
        print("\n=== 测试collate函数 ===")

        try:
            # 创建模拟批次数据
            batch = []
            for i in range(2):
                # 创建模拟姿态数据
                pose_sample = {
                    'body': torch.randn(random.randint(5, 8), 10, 3),
                    'left': torch.randn(random.randint(5, 8), 21, 3),
                    'right': torch.randn(random.randint(5, 8), 21, 3),
                    'face_all': torch.randn(random.randint(5, 8), 14, 3)
                }
                # 当rgb_support=False时，最后两个参数可以是任意值
                batch.append((
                    f'sample_{i}',  # name
                    pose_sample,  # pose
                    f'text_{i}',  # text
                    'gloss_{i}',  # gloss
                    {}  # support rgb dict
                ))

            # 创建数据集实例并测试collate_fn
            # 需要给Base_Dataset添加rgb_support属性
            dataset = Base_Dataset()
            dataset.rgb_support = False  # 手动添加属性

            src_input, tgt_input = dataset.collate_fn(batch)

            print(f"✓ collate_fn 测试通过")
            print(f"  输入数据键: {list(src_input.keys())}")
            print(f"  目标数据键: {list(tgt_input.keys())}")
            for key in ['body', 'left', 'right', 'face_all']:
                print(f"  {key}形状: {src_input[key].shape}")

            return True

        except Exception as e:
            print(f"✗ collate_fn测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False


    def test_simple_functions():
        """测试一些简单辅助函数"""
        print("\n=== 测试简单辅助函数 ===")

        try:
            # 测试bbox_4hands函数
            left_keypoints = np.random.rand(3, 21, 2)
            right_keypoints = np.random.rand(3, 21, 2)
            hw = (224, 224)

            left_boxes, right_boxes, box_hw = bbox_4hands(left_keypoints, right_keypoints, hw)
            print(f"✓ bbox_4hands 测试通过")
            print(f"  左手边界框形状: {left_boxes.shape}")
            print(f"  右手边界框形状: {right_boxes.shape}")
            print(f"  边界框大小: {box_hw}")

            return True

        except Exception as e:
            print(f"✗ 简单函数测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False


    # 运行测试
    print("开始单元测试...")

    # 创建测试数据
    temp_dir, annotation_path, pose_path = create_test_data()

    tests_passed = 0
    total_tests = 5

    # 运行测试
    if test_pose_functions():
        tests_passed += 1

    if test_rgb_functions():
        tests_passed += 1

    if test_dataset_loading_no_rgb():
        tests_passed += 1

    if test_collate_fn():
        tests_passed += 1

    if test_simple_functions():
        tests_passed += 1

    # 输出测试结果
    print(f"\n=== 测试结果 ===")
    print(f"通过测试: {tests_passed}/{total_tests}")

    if tests_passed == total_tests:
        print("🎉 所有测试通过！")
        sys.exit(0)
    else:
        print("❌ 部分测试失败")
        sys.exit(1)


"""
{
  "rgb_frames": List[PIL.Image] 或 torch.Tensor [T,C,H,W]（未缩放或已统一到视频原始尺寸）,
  "pose_xyc": np.ndarray [T,K,3]（像素坐标）,
  "text": str,
  "meta": {
    "W": 原始视频宽, "H": 原始视频高, "index": 样本全局索引（int）
  }
}

"""
