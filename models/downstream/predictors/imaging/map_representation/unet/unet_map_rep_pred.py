from typing import Dict, Tuple

import torch
from torch import nn

from models.downstream.predictors.imaging.image_pred import ImagePred


class SimpleUNet(nn.Module):
    """
    A simple U-Net architecture for image-to-image prediction.
    """

    def __init__(self, input_channels: int, output_channels: int, base_channels: int = 64):
        super(SimpleUNet, self).__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)  # 128 -> 64

        self.encoder2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)  # 64 -> 32

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Output layer
        self.output = nn.Conv2d(base_channels, output_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool2(enc2))

        # Decoder
        dec2 = self.upconv2(bottleneck)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)

        # Output
        return self.output(dec1)


class UNetMapRepPred(ImagePred):
    """
    A representation map predictor using U-Net architecture.
    
    This model takes a 2D representation (e.g., force map) and predicts an image
    using a U-Net architecture that preserves spatial relationships through
    skip connections.
    """

    def __init__(self, config_path: str, representation_size: int):
        super().__init__(config_path, representation_size)
        self.image_predictor = self._build_image_predictor()

    def _build_image_predictor(self) -> nn.Module:
        """
        Build a U-Net architecture for image prediction.
        
        Returns:
            SimpleUNet: U-Net model for image prediction
        """
        return SimpleUNet(
            input_channels=1,  # Single channel input (grayscale representation)
            output_channels=self.config.num_bins,  # Multi-class output
            base_channels=self.config.hidden_size
        )

    def inference(self, representation: torch.Tensor, **kwargs) -> Tuple[
        torch.Tensor, Dict[str, torch.Tensor]]:
        pred = self.image_predictor(representation)
        return pred, {"predicted_images": pred}
