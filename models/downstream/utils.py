from typing import Optional

import torch
import torch.nn.functional as F


def discretize_image(image, bins):
    # Assuming image pixel values are scaled between 0 and 1
    discretized = (image * (bins - 1)).long()
    return discretized


def calc_loc_error(predicted_images, real_model_image):
    real_lump = real_model_image == 2
    B, H, W = real_lump.shape
    mask = real_lump.to(torch.float32)

    # Create coordinate grids
    y_coords = torch.arange(H, dtype=torch.float32, device=mask.device)
    x_coords = torch.arange(W, dtype=torch.float32, device=mask.device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')  # shape [H, W]
    y_grid = y_grid.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
    x_grid = x_grid.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
    mass = mask.sum(dim=(1, 2)) + 1e-6  # prevent division by zero
    y_center = (mask * y_grid).sum(dim=(1, 2)) / mass
    x_center = (mask * x_grid).sum(dim=(1, 2)) / mass
    real_locations = torch.stack([y_center, x_center], dim=1)  # shape [B, 2]

    pred_lump = predicted_images == 2
    mask = pred_lump.to(torch.float32)
    mass = mask.sum(dim=(1, 2)) + 1e-6
    y_center = (mask * y_grid).sum(dim=(1, 2)) / mass
    x_center = (mask * x_grid).sum(dim=(1, 2)) / mass
    pred_locations = torch.stack([y_center, x_center], dim=1)  # shape [B, 2]

    return torch.norm(real_locations - pred_locations, dim=1)


def dice_loss(predicted_images, real_model_images, num_bins: int) -> torch.Tensor:
    """
    Compute the Dice loss for image segmentation.
    """
    smooth = 1e-6
    predicted_images = torch.softmax(predicted_images, dim=1)
    predicted_images = predicted_images.view(-1, num_bins, 128, 128)
    real_model_images = torch.nn.functional.one_hot(real_model_images, num_classes=num_bins).permute(0, 3, 1,
                                                                                                     2).float()
    real_model_images = real_model_images.view(-1, num_bins, 128, 128)
    intersection = (predicted_images * real_model_images).sum(dim=(2, 3))
    union = predicted_images.sum(dim=(2, 3)) + real_model_images.sum(dim=(2, 3))
    dice_score = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice_score.mean()  # Return the Dice loss as 1 - Dice score


def multiclass_dice_loss(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Multi-class Dice loss for 2D or 3D inputs.

    pred: [B, C, ...] logits, where ... = spatial dims (H,W) or (D,H,W)
    gt:   [B, ...] integer labels (0..C-1)
    """
    num_classes = pred.shape[1]

    # Convert gt to one-hot: [B, ..., C]
    gt_one_hot = F.one_hot(gt, num_classes=num_classes).permute(0, -1, *range(1, gt.dim())).float()
    # Now gt_one_hot: [B, C, ...]

    # Softmax over channel dim
    pred_softmax = F.softmax(pred, dim=1)

    # All spatial dims after channel
    spatial_dims = tuple(range(2, pred.dim()))

    intersection = (pred_softmax * gt_one_hot).sum(dim=(0, *spatial_dims))
    union = pred_softmax.sum(dim=(0, *spatial_dims)) + gt_one_hot.sum(dim=(0, *spatial_dims))

    dice_per_class = (2 * intersection + eps) / (union + eps)

    loss = 1 - dice_per_class.mean()
    return loss


def multiclass_focal_loss(pred: torch.Tensor, gt: torch.Tensor, alpha: Optional[torch.Tensor] = None, gamma: float = 2.,
                          eps: float = 1e-6) -> torch.Tensor:
    """
    Multi-class Focal Loss for logits input.

    pred: logits
    gt:     [B, D, H, W] integer labels (0..C-1)
    alpha:  Tensor or list of class weights [C] (default: all 1)
    gamma:  Focusing parameter
    """
    num_classes = pred.shape[1]

    # Handle alpha
    if alpha is None:
        alpha = torch.ones(num_classes, device=pred.device)
    else:
        if alpha.numel() == 1:
            alpha = alpha.repeat(num_classes)

    # Softmax and log-softmax
    log_pred = F.log_softmax(pred, dim=1)
    pred_softmax = torch.exp(log_pred + eps)

    # Gather the probabilities of the true class
    gt = gt.long()
    p_t = pred_softmax.gather(1, gt.unsqueeze(1))  # [B, 1, D, H, W]
    log_p_t = log_pred.gather(1, gt.unsqueeze(1))

    # Gather alpha for each voxel
    alpha_t = alpha[gt]

    # Compute focal loss
    loss = -alpha_t * ((1 - p_t) ** gamma) * log_p_t
    loss = loss.squeeze(1)  # [B, D, H, W]

    return loss.mean()


def focal_loss(pred: torch.Tensor, gt: torch.Tensor, num_bins: int, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0,
               eps: float = 1e-6) -> torch.Tensor:

    count = torch.bincount(gt.view(-1), minlength=num_bins)
    if alpha is None:
        alpha = 1.0 / (count.float() + 1e-6)
        alpha = alpha / alpha.max()
    # Move to same device and dtype as predictions
    alpha = alpha.to(pred.device).type(pred.dtype)

    loss = multiclass_focal_loss(pred, gt, alpha, gamma, eps)
    return loss


def is_power_of_2(x):
    return (x != 0) and (x & (x - 1) == 0)
