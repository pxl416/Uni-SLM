# datasets/SyntheticWorldDataset.py

from torch.utils.data import Dataset

class SyntheticWorldDataset(Dataset):
    def __init__(self, world_engine, length=1000000):
        self.engine = world_engine
        self.length = length   # fake length, infinite stream

    def __getitem__(self, idx):
        sample = self.engine.sample()   # <-- 关键

        return {
            "rgb": sample.video_tensor,
            "temporal_gt": sample.temporal_gt,
            "spatial_gt": sample.spatial_gt,
            "text": sample.text_align
        }

    def __len__(self):
        return self.length



