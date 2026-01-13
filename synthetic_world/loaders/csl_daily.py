# synthetic_world/loaders/csl_daily.py
import os
import pickle
from typing import List, Optional
import cv2
import numpy as np

from synthetic_world.assets import SignAsset


# -------- Helpers --------

def _load_text_and_gloss(pkl_path: str):
    """加载CSL-Daily的标注文件"""
    with open(pkl_path, "rb") as f:
        anno = pickle.load(f)

    info = anno["info"]
    gloss_map = anno["gloss_map"]

    anno_dict = {item["name"]: item for item in info}
    return anno_dict, gloss_map


def _count_frames_in_folder(folder: str) -> int:
    """统计文件夹中的图片数量"""
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    count = 0
    for f in os.listdir(folder):
        if f.lower().endswith(extensions):
            count += 1
    return count


def _get_resolution_from_folder(folder: str) -> tuple:
    """从文件夹第一张图片获取分辨率"""
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(extensions):
            img_path = os.path.join(folder, f)
            img = cv2.imread(img_path)
            if img is not None:
                return (img.shape[1], img.shape[0])  # (W, H)
    return (0, 0)


def _infer_category_from_text(text: str) -> str:
    """从文本推断手势类别"""
    text_lower = text.lower()

    # 简单的启发式规则
    greeting_words = ['你好', 'hello', 'hi', '早上好', '晚上好', '再见']
    question_words = ['什么', '哪里', '何时', '为什么', '怎么', '多少', '谁']
    number_words = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '百', '千', '万']
    family_words = ['爸爸', '妈妈', '哥哥', '姐姐', '弟弟', '妹妹', '家庭', '父母']

    for word in greeting_words:
        if word in text_lower:
            return "greeting"

    for word in question_words:
        if word in text_lower:
            return "question"

    for word in number_words:
        if word in text_lower:
            return "number"

    for word in family_words:
        if word in text_lower:
            return "family"

    return "general"


def _estimate_complexity_from_gloss(gloss: List[str]) -> float:
    """从词素估计手势复杂度"""
    if not gloss:
        return 0.5

    # 简单的启发式：词素越多越复杂
    num_gloss = len(gloss)
    complexity = min(0.3 + num_gloss * 0.1, 1.0)

    # 检查是否包含双手手势
    two_handed_indicators = ['双手', '对称', '交互']
    for gloss_item in gloss:
        if any(indicator in gloss_item for indicator in two_handed_indicators):
            complexity += 0.2

    return min(complexity, 1.0)


def _is_two_handed_from_gloss(gloss: List[str]) -> bool:
    """从词素判断是否双手手势"""
    if not gloss:
        return False

    two_handed_indicators = ['双手', '对称', '交互', '同时']
    for gloss_item in gloss:
        if any(indicator in gloss_item for indicator in two_handed_indicators):
            return True
    return False


# -------- Main Loader --------

def load_csl_daily_as_assets(
        root: str,
        rgb_dir: str,
        anno_pkl: str,
        split_file: Optional[str] = None,
        fps: int = 25,
        max_samples: Optional[int] = None,
        verbose: bool = True,
) -> List[SignAsset]:
    """
    Load CSL-Daily as SignAsset list (lazy loading).

    Args:
        root: 数据集根目录
        rgb_dir: RGB帧文件夹名（相对于root）
        anno_pkl: 标注文件路径（相对于root）
        split_file: 划分文件（可选）
        fps: 帧率
        max_samples: 最大加载样本数（用于测试）
        verbose: 是否打印进度

    Returns:
        List[SignAsset]: 手语资产列表
    """

    rgb_root = os.path.join(root, rgb_dir)
    anno_path = os.path.join(root, anno_pkl)

    if verbose:
        print(f"[CSL-Daily] Loading annotations from {anno_path}")

    anno_dict, _ = _load_text_and_gloss(anno_path)

    # 确定要加载的样本
    if split_file:
        split_path = os.path.join(root, split_file)
        if verbose:
            print(f"[CSL-Daily] Using split file: {split_path}")
        with open(split_path, "r", encoding='utf-8') as f:
            names = [l.strip() for l in f if l.strip()]
    else:
        if verbose:
            print(f"[CSL-Daily] Loading all samples from {rgb_root}")
        names = sorted(os.listdir(rgb_root))

    assets = []
    skipped = 0

    for idx, name in enumerate(names):
        if max_samples and len(assets) >= max_samples:
            if verbose:
                print(f"[CSL-Daily] Reached max_samples={max_samples}")
            break

        if name not in anno_dict:
            if verbose and skipped < 5:  # 只打印前几个跳过信息
                print(f"[CSL-Daily] Skip {name}: not in annotations")
            skipped += 1
            continue

        clip_dir = os.path.join(rgb_root, name)
        if not os.path.isdir(clip_dir):
            if verbose and skipped < 5:
                print(f"[CSL-Daily] Skip {name}: not a directory")
            skipped += 1
            continue

        # 统计帧数
        num_frames = _count_frames_in_folder(clip_dir)
        if num_frames == 0:
            if verbose and skipped < 5:
                print(f"[CSL-Daily] Skip {name}: no frames found")
            skipped += 1
            continue

        # 获取标注信息
        a = anno_dict[name]
        text = "".join(a["label_char"])
        gloss = a["label_gloss"]

        # 推断语义信息
        category = _infer_category_from_text(text)
        complexity = _estimate_complexity_from_gloss(gloss)
        two_handed = _is_two_handed_from_gloss(gloss)

        # 创建资产（懒加载模式）
        asset = SignAsset(
            asset_id=name,
            video_path=clip_dir,  # 文件夹路径
            text=text,
            gloss=gloss,
            fps=fps,
            num_frames=num_frames,  # 避免重复统计
            semantic_category=category,
            complexity=complexity,
            two_handed=two_handed,
        )

        assets.append(asset)

        if verbose and (idx + 1) % 100 == 0:
            print(f"[CSL-Daily] Processed {idx + 1}/{len(names)} samples")

    if verbose:
        print(f"[CSL-Daily] Indexed {len(assets)} SignAssets (skipped {skipped})")

    return assets


# -------- Test --------

if __name__ == "__main__":
    # 注意：需要修改为你的实际路径
    ROOT = "/home/pxl416/PeixiLiu/px_proj/px_data/csl-daily-frames-512x512"
    RGB = "sentence"
    ANNO = "sentence_label/csl2020ct_v2.pkl"
    SPLIT = "sentence_label/split_1_train.txt"

    print("=== Testing CSL-Daily Loader ===\n")

    try:
        # 测试加载少量样本
        assets = load_csl_daily_as_assets(
            root=ROOT,
            rgb_dir=RGB,
            anno_pkl=ANNO,
            split_file=SPLIT,
            max_samples=3,
            verbose=True,
        )

        if assets:
            print("\n" + "=" * 50)
            print("Sample asset details:")
            print("=" * 50)

            a = assets[0]
            print(f"Asset ID: {a.asset_id}")
            print(f"Text: {a.text}")
            print(f"Gloss (first 5): {a.gloss[:5]}")
            print(f"Category: {a.semantic_category}")
            print(f"Complexity: {a.complexity:.2f}")
            print(f"Two-handed: {a.two_handed}")
            print(f"Video path: {a.video_path}")
            print(f"Num frames: {a.num_frames}")
            print(f"Duration: {a.duration:.2f} sec")
            print(f"Resolution: {a.resolution}")

            # 测试懒加载
            print("\nTesting lazy loading...")
            frames = a.load_frames(max_frames=5)  # 只加载前5帧
            print(f"Loaded frames shape: {frames.shape}")
            print(f"Frames dtype: {frames.dtype}")
            print(f"Frames range: [{frames.min()}, {frames.max()}]")

            print("\nTest passed! ✔")
        else:
            print("No assets loaded. Please check paths.")

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please update the paths in the test code.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()