# models/Head/recognition.py
import torch
import torch.nn as nn


class RecognitionHead(nn.Module):
    """
    Simple CTC Recognition Head.

    Input:
        video_feat: (B, T, D)
    Output:
        {"logits": (B, T, num_classes)}
    """

    def __init__(self, hidden_dim: int, num_classes: int, blank_id: int = 0):
        super().__init__()

        self.num_classes = num_classes
        self.blank_id = blank_id

        # Linear classifier for each timestep
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, video_feat):
        """
        video_feat: (B, T, D)
        return:
            { "logits": (B, T, num_classes) }
        """
        logits = self.fc(video_feat)  # (B, T, num_classes)
        return {"logits": logits}
