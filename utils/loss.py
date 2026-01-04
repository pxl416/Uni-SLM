# utils/loss.py
import torch
import torch.nn.functional as F


# 1. Temporal mask downsampling
def downsample_mask(
    mask: torch.Tensor,
    target_len: int,
    mode: str = "avg",
):
    """
    Downsample temporal mask to match encoder time resolution.

    Args:
        mask: (B, T_raw), bool or float
        target_len: int, encoder temporal length T_enc
        mode:
            - "avg": average pooling (soft proportion)
            - "max": max pooling (hard existence)

    Returns:
        mask_ds: (B, T_enc), float in [0, 1]
    """
    if mask.dim() != 2:
        raise ValueError(f"mask must be (B, T), got {mask.shape}")

    B, T_raw = mask.shape
    mask = mask.float()

    # No resampling needed
    if T_raw == target_len:
        return mask

    if T_raw < target_len:
        raise ValueError(
            f"Cannot downsample mask from T={T_raw} to larger T={target_len}"
        )

    # (B, T_raw) -> (B, 1, T_raw)
    mask = mask.unsqueeze(1)

    if mode == "avg":
        # Linear interpolation approximates average pooling
        mask_ds = F.interpolate(
            mask,
            size=target_len,
            mode="linear",
            align_corners=False,
        )

    elif mode == "max":
        # Hard downsampling: any active frame -> active
        stride = T_raw // target_len
        if stride < 1:
            raise ValueError("Invalid stride computed for max pooling")

        mask_ds = F.max_pool1d(
            mask,
            kernel_size=stride,
            stride=stride,
        )

        # In case of rounding mismatch
        if mask_ds.shape[-1] != target_len:
            mask_ds = F.interpolate(
                mask_ds,
                size=target_len,
                mode="nearest",
            )

    else:
        raise ValueError(f"Unknown downsample mode: {mode}")

    return mask_ds.squeeze(1)

# 2. Basic Temporal BCE Loss (recommended baseline)
def temporal_bce_loss(
    logits: torch.Tensor,        # (B, T_pred)
    rgb_mask: torch.Tensor,      # (B, T_src)
    reduction: str = "mean",
):
    """
    Temporal BCE loss with downsampled target.

    logits: raw logits (before sigmoid)
    rgb_mask: binary mask on original timeline
    """

    B, T_pred = logits.shape
    T_src = rgb_mask.shape[1]

    if T_src != T_pred:
        rgb_mask = torch.nn.functional.interpolate(
            rgb_mask.unsqueeze(1).float(),
            size=T_pred,
            mode="nearest",
        ).squeeze(1)

    target = rgb_mask.float()

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction=reduction
    )
    return loss

# 3. Optional: Temporal MSE Loss (soft regression variant)
def temporal_mse_loss(
    logits: torch.Tensor,
    gt_mask: torch.Tensor,
    downsample_mode: str = "avg",
):
    """
    Mean Squared Error loss on sigmoid outputs.

    Sometimes useful for smoother supervision.

    Args:
        logits: (B, T_enc)
        gt_mask: (B, T_raw)

    Returns:
        loss: scalar tensor
    """
    B, T_enc = logits.shape

    gt_ds = downsample_mask(
        gt_mask,
        target_len=T_enc,
        mode=downsample_mode,
    )

    pred = torch.sigmoid(logits)

    return F.mse_loss(pred, gt_ds)

# 4. Optional: Temporal Dice Loss (robust to imbalance)
def temporal_dice_loss(
    logits: torch.Tensor,
    gt_mask: torch.Tensor,
    downsample_mode: str = "avg",
    eps: float = 1e-6,
):
    """
    Dice loss for temporal segmentation.

    Useful when positive frames are rare.

    Args:
        logits: (B, T_enc)
        gt_mask: (B, T_raw)

    Returns:
        loss: scalar tensor
    """
    B, T_enc = logits.shape

    gt_ds = downsample_mask(
        gt_mask,
        target_len=T_enc,
        mode=downsample_mode,
    )

    pred = torch.sigmoid(logits)

    intersection = (pred * gt_ds).sum(dim=1)
    union = pred.sum(dim=1) + gt_ds.sum(dim=1)

    dice = (2 * intersection + eps) / (union + eps)

    return 1.0 - dice.mean()

# 5. Combined Loss (future-proof hook)
def temporal_combined_loss(
    logits: torch.Tensor,
    gt_mask: torch.Tensor,
    weights: dict | None = None,
):
    """
    Combine multiple temporal losses.

    Example:
        weights = {
            "bce": 1.0,
            "dice": 0.5,
        }

    Args:
        logits: (B, T_enc)
        gt_mask: (B, T_raw)
        weights: dict specifying loss weights

    Returns:
        loss: scalar tensor
    """
    if weights is None:
        weights = {"bce": 1.0}

    loss = 0.0

    if "bce" in weights:
        loss += weights["bce"] * temporal_bce_loss(logits, gt_mask)

    if "mse" in weights:
        loss += weights["mse"] * temporal_mse_loss(logits, gt_mask)

    if "dice" in weights:
        loss += weights["dice"] * temporal_dice_loss(logits, gt_mask)

    return loss
