# utils/spatial_composer.py

class SpatialComposer:
    def __init__(self, mode: str, **kwargs):
        self.mode = mode
        self.cfg = kwargs

    def __call__(self, video, kpts=None):
        if self.mode == "none":
            return spatial_none(video)
        elif self.mode == "keypoint":
            return spatial_from_kpts(video, kpts)
        elif self.mode == "motion":
            return spatial_from_motion(video)
        elif self.mode == "segment":
            return spatial_from_segmenter(video)
        else:
            raise ValueError(self.mode)

