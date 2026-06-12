import torch

from models.downstream.predictors.quantity_regression.mlp_quantity_reg import MLPQuantityReg


class MLPLumpCenterReg(MLPQuantityReg):
    """
    A model that predicts the center of a lump from the representations.
    """

    def __init__(self, config_path: str, representation_size: int):
        super().__init__(config_path, representation_size)

    def calc_quantity(self, image: torch.Tensor) -> torch.Tensor:
        lump_mask = (image == 2).float()  # Create a mask for the lump class
        B, H, W = lump_mask.shape  # Get batch size, height, and width
        y_coords = torch.arange(H, device=lump_mask.device).view(1, H, 1).expand(B, H, W)
        x_coords = torch.arange(W, device=lump_mask.device).view(1, 1, W).expand(B, H, W)

        # Calculate total number of masked pixels per sample
        total = lump_mask.sum(dim=(1, 2)) + 1e-6  # avoid division by zero

        # Compute weighted sum of coordinates
        mean_y = (lump_mask * y_coords).sum(dim=(1, 2)) / total
        mean_x = (lump_mask * x_coords).sum(dim=(1, 2)) / total

        # Stack the results
        mean_indices = torch.stack([mean_y, mean_x], dim=1)  # [B, 2]

        return mean_indices

    @property
    def quantity_name(self):
        return "lump_center"

    @property
    def quantity_size(self) -> int:
        return 2

    def normalize_quantity(self, quantity: torch.Tensor) -> torch.Tensor:
        return (quantity - self.config.image_size / 2) / (self.config.image_size / 2)

    def denormalize_quantity(self, quantity: torch.Tensor) -> torch.Tensor:
        return quantity * (self.config.image_size / 2) + self.config.image_size / 2
