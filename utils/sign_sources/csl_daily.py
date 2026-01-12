# utils/sign_sources/csl_daily.py
import random
import numpy as np
from utils.offline_composer import VideoSample, load_video_avi

class CSLDailySignSource:
    def __init__(self, index_list):
        """
        index_list: List[dict]
        each dict contains:
          - video_path
          - start
          - end
          - pose_path (optional)
        """
        self.samples = index_list
        assert len(self.samples) > 0

    def sample(self, seed=None):
        if seed is not None:
            random.seed(seed)

        item = random.choice(self.samples)

        video = load_video_avi(item["video_path"])
        frames = video.frames[item["start"]:item["end"]]

        sign_video = VideoSample(frames, video.fps)
        T = sign_video.T

        temporal_mask = np.ones(T, dtype=np.int64)

        return {
            "video": sign_video,
            "temporal_mask": temporal_mask,
            "keypoints": None,
            "source": "CSL-Daily",
            "meta": {
                "video_path": item["video_path"],
                "start": item["start"],
                "end": item["end"]
            }
        }
