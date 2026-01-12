import numpy as np

class TemporalComposer:
    def __init__(self, mode: str, **kwargs):
        self.mode = mode
        self.cfg = kwargs

    def __call__(self, video, meta=None):
        if self.mode == "weak":
            return temporal_weak(video)
        elif self.mode == "hard":
            return temporal_hard(video, meta)
        elif self.mode == "soft":
            return temporal_soft(video, meta)
        elif self.mode == "synthetic":
            return temporal_synthetic(video, meta)
        else:
            raise ValueError(self.mode)


# ------------------- Methods -------------------

def temporal_weak(video):
    return np.ones(video.T, dtype=np.float32)


def temporal_hard(video, meta):
    raise NotImplementedError


def temporal_soft(video, meta):
    raise NotImplementedError


def temporal_synthetic(video, meta):
    """
    meta = {
        "bg1": VideoSample,
        "bg2": VideoSample,
    }
    """
    bg1 = meta["bg1"]
    bg2 = meta["bg2"]

    target_fps = video.fps

    bg1 = resample_video(bg1, target_fps)
    bg2 = resample_video(bg2, target_fps)

    merged_frames = np.concatenate(
        [bg1.frames, video.frames, bg2.frames],
        axis=0
    )

    t_label = np.concatenate([
        np.zeros(bg1.T, dtype=np.int64),
        np.ones(video.T, dtype=np.int64),
        np.zeros(bg2.T, dtype=np.int64),
    ])

    return merged_frames, target_fps, t_label


# ------------------- Utils -------------------

def resample_video(video, target_fps):
    if video.fps == target_fps:
        return video

    ratio = target_fps / video.fps
    T = video.T
    new_T = int(round(T * ratio))

    idx = np.linspace(0, T - 1, new_T).astype(np.int32)
    new_frames = video.frames[idx]

    # return a lightweight proxy
    class _V:
        pass
    v = _V()
    v.frames = new_frames
    v.fps = target_fps
    v.T = new_T

    return v
