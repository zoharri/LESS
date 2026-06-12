import torch


def undiscretize_image(softmax_image, bins=8):
    # Use argmax to find the bin index with the maximum probability
    discretized_image = torch.argmax(softmax_image, dim=softmax_image.dim() - 3)
    # Scale the pixel values back to [0,1]
    undiscretized = discretized_image.float() / (bins - 1)
    return undiscretized


def images_to_logits(imgs, num_bins):
    """
    Convert batch of images [B, H, W] with values in (-inf, inf)
    into logits [B, num_bins, H, W].

    - Values clamped to [-1, 1]
    - Discretized into num_bins bins
    - Logits are one-hot in log-space (0 for the assigned bin, -inf otherwise)
    """
    vmin, vmax = -1.0, 1.0
    B, H, W = imgs.shape

    # Clamp
    imgs_clamped = imgs.clamp(vmin, vmax)

    # Bin width
    bin_width = (vmax - vmin) / num_bins  # = 2 / num_bins

    # Compute bin indices [B,H,W]
    bin_indices = ((imgs_clamped - vmin) / bin_width).long()
    bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)

    # Start with all -inf logits
    logits = torch.full((B, num_bins, H, W), -float("inf"), device=imgs.device)

    # Scatter: put 0 at the correct bin index
    logits.scatter_(1, bin_indices.unsqueeze(1), 0.0)

    return logits
