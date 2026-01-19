# datasets/syndatasets.py
from __future__ import annotations

import random
import numpy as np
from types import SimpleNamespace

from datasets.datasets import BaseDataset
from utils.config import load_yaml, dict_to_ns
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


print("[DEBUG] syndatasets.py loaded (correct version).")


class SyntheticDataset(BaseDataset):
    """
    Synthetic dataset for pretraining.

    Responsibility:
      synthetic_world.builder -> raw sample
      adapt to BaseDataset.get_item_data() contract

    This dataset MUST NOT:
      - normalize
      - resize
      - pad
      - collate
    """

    def __init__(self, args, cfg, phase: str):
        super().__init__(args, cfg, phase)

        # -------- sanity --------
        if not hasattr(cfg, "synthetic_method"):
            raise KeyError(
                "cfg.synthetic_method is required "
                "(e.g. config/synthetic/syn1.yaml)"
            )

        # -------- load synthetic cfg --------
        syn_cfg_path = cfg.synthetic_method
        self.syn_cfg_raw = load_yaml(syn_cfg_path)
        self.syn_cfg = dict_to_ns(self.syn_cfg_raw)

        # -------- init synthetic world --------
        try:
            from synthetic_world.builder import SyntheticWorldBuilder

            self.builder = SyntheticWorldBuilder(
                syn_cfg=self.syn_cfg,
                dataset_cfg=cfg,
                split=self.phase,
            )
            self.use_builder = True
            print("[INFO] SyntheticWorldBuilder loaded.")
        except Exception as e:
            print(f"[WARNING] SyntheticWorldBuilder unavailable: {e}")
            self.builder = None
            self.use_builder = False

        # virtual infinite-style dataset
        self.length = getattr(self.syn_cfg, "virtual_length", 1_000_000)

    def __len__(self):
        return self.length

    def get_item_data(self, idx):
        """
        Must return:
          name, pose_sample, text, support
        (BaseDataset contract)
        """

        # -------- generate raw synthetic sample --------
        if self.use_builder:
            sample = self.builder.generate(idx)
        else:
            sample = self._dummy_sample(idx)

        # -------- unpack --------
        name = sample.get("name", f"synth_{idx:06d}")
        text = sample.get("text", "")

        # pose is optional for synthetic
        pose_sample = {
            "keypoints": sample.get("keypoints", None)
        }

        # IMPORTANT:
        # rgb must be (T, H, W, C), uint8 or float
        # All processing happens in BaseDataset
        support = {
            "rgb_img": sample["rgb"],
            "gloss": sample.get("gloss", []),
            "segments": {
                # pretraining supervision lives here
                "temporal_mask": sample.get("temporal_mask", None)
            }
        }

        return name, pose_sample, text, support

    # -------- fallback dummy --------
    def _dummy_sample(self, idx):
        T = random.randint(16, 64)
        H, W = 112, 112

        rgb = np.random.randint(0, 255, (T, H, W, 3), dtype=np.uint8)

        temporal_mask = np.zeros((T,), dtype=np.float32)
        temporal_mask[T // 4: T * 3 // 4] = 1.0

        return {
            "name": f"dummy_{idx:06d}",
            "rgb": rgb,
            "temporal_mask": temporal_mask,
            "text": "",
            "gloss": [],
        }


# -------- dataset registry --------

DATASET_REGISTRY = {
    "Synthetic_Pretrain": "datasets.syndatasets:SyntheticDataset",
}


# -------- local test --------

if __name__ == "__main__":
    print("=== SyntheticDataset sanity test ===")

    class Args:
        pass

    args = Args()

    cfg = SimpleNamespace()
    cfg.synthetic_method = "config/synthetic/syn1.yaml"
    cfg.global_data = SimpleNamespace(channel=3, resize=[112, 112])
    cfg.seed = 3407

    ds = SyntheticDataset(args, cfg, phase="train")
    print("length =", len(ds))

    name, pose, text, support = ds.get_item_data(0)
    print("name:", name)
    print("rgb shape:", support["rgb_img"].shape)
    print("temporal_mask shape:",
          None if support["segments"]["temporal_mask"] is None
          else support["segments"]["temporal_mask"].shape)

    print("✔ SyntheticDataset OK")
