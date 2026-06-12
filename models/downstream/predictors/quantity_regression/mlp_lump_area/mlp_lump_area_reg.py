import torch

from models.downstream.predictors.quantity_regression.mlp_quantity_reg import MLPQuantityReg


class MLPLumpAreaReg(MLPQuantityReg):
    """
    A model that predicts the area of a lump from the representations.
    """

    def __init__(self, config_path: str, representation_size: int):
        super().__init__(config_path, representation_size)
        self._nominal_area = self.config.image_size * self.config.image_size / 50

    def calc_quantity(self, image: torch.Tensor) -> torch.Tensor:
        # Calculate the area by counting the number of pixels classified as lump
        return (image == 2).float().sum(dim=(1, 2)).unsqueeze(1)

    @property
    def quantity_name(self):
        return "lump_area"

    @property
    def quantity_size(self) -> int:
        return 1

    def normalize_quantity(self, quantity: torch.Tensor) -> torch.Tensor:
        return quantity / self._nominal_area

    def denormalize_quantity(self, quantity: torch.Tensor) -> torch.Tensor:
        return quantity * self._nominal_area
