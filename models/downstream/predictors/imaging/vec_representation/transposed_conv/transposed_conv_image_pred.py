from typing import Dict, Tuple

from torch import nn

from models.downstream.predictors.imaging.image_pred import ImagePred
from models.downstream.utils import *


class TransposedConvImagePred(ImagePred):
    """
    A model that predicts images from a representation using transposed convolution layers.
    """

    def __init__(self, config_path: str, representation_size: int):
        super().__init__(config_path, representation_size)

        assert self.config.image_size == 128, "TransposedConvImagePred is designed for 128x128 images right now."

        self.image_predictor = nn.Sequential(
            nn.Linear(self.representation_size, self.config.num_channels * 4 * 4),
            # Upscale to an intermediary size
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Unflatten(1, (self.config.num_channels, 4, 4)),
            nn.ConvTranspose2d(self.config.num_channels, self.config.num_channels,
                               kernel_size=4, stride=2,
                               padding=1),  # Upscale to 8x8
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.ConvTranspose2d(self.config.num_channels, self.config.num_channels,
                               kernel_size=4, stride=2,
                               padding=1),  # Upscale to 16x16
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.ConvTranspose2d(self.config.num_channels, self.config.num_channels, kernel_size=4, stride=2,
                               padding=1),  # Upscale to 32x32
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.ConvTranspose2d(self.config.num_channels, self.config.num_bins, kernel_size=4, stride=2,
                               padding=1),  # Upscale to 64x64
            nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False)
        )

    def calculate_canvas_uncertainty(self, canvas: torch.Tensor) -> torch.Tensor:
        """
        Calculates uncertainty map from the canvas by computing entropy over the bins.
        Args:
            canvas: [B, NumBins, H, W]
        Returns:
            uncertainty_map: [B, H, W] numpy array
        """
        with torch.no_grad():
            probs = torch.softmax(canvas, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)  # [B, H, W]
        return entropy

    def inference(self, representation: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        predicted_images = self.image_predictor(representation)
        return predicted_images, {"predicted_images": predicted_images,
                                  "canvas_uncertainty": self.calculate_canvas_uncertainty(predicted_images)}
