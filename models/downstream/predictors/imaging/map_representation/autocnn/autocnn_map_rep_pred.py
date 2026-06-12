from typing import Dict, Tuple

import torch
from torch import nn

from models.downstream.predictors.imaging.image_pred import ImagePred


class ShallowEncoder(nn.Module):
    """
    A very shallow CNN encoder that takes an input image and produces a flat latent vector.
    """

    def __init__(self, in_channels: int, latent_dim: int, base_channels: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.fc = nn.Linear(base_channels * 8 * 8, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W)
        x = self.conv(x)  # (N, base_channels, 8, 8)
        N = x.size(0)
        x = x.view(N, -1)  # (N, base_channels*8*8)
        return self.fc(x)  # (N, latent_dim)


class InverseDecoder(nn.Module):
    """
    Decoder (inverse CNN) that takes a latent vector and produces a multi-class 128×128 image.
    """

    def __init__(self, latent_dim: int, output_channels: int, base_channels: int = 256, dropout: float = 0.0):
        super().__init__()
        self.output_channels = output_channels
        self.fc = nn.Linear(latent_dim, base_channels * 8 * 8)

        layers = []
        ch = base_channels

        # Progressive upsampling layers
        for out_ch in [ch // 2, ch // 4, ch // 8, ch // 16]:
            layers.append(nn.ConvTranspose2d(ch, out_ch, kernel_size=4, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            ch = out_ch

        # Final output layer
        layers.append(nn.Conv2d(ch, output_channels, kernel_size=3, padding=1))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, latent_dim)
        N = x.size(0)
        x = self.fc(x)  # (N, base_channels*8*8)
        x = x.view(N, -1, 8, 8)  # (N, base_channels, 8, 8)
        return self.decoder(x)  # (N, output_channels, 128, 128)


class ShallowAutoCNN(nn.Module):
    """
    Full autoencoder model: shallow encoder + inverse decoder.
    
    This architecture compresses the input representation through a bottleneck
    and then reconstructs it to the target image space.
    """

    def __init__(self, in_channels: int, output_channels: int, latent_dim: int,
                 enc_channels: int = 64, dec_channels: int = 256,
                 dropout: float = 0.0):
        super().__init__()
        self.encoder = ShallowEncoder(in_channels, latent_dim, enc_channels)
        self.decoder = InverseDecoder(latent_dim, output_channels, dec_channels, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


class AutoCNNMapRepPred(ImagePred):
    """
    A representation map predictor using AutoCNN (Autoencoder CNN) architecture.
    
    This model takes a 2D representation (e.g., force map) and predicts an image
    using an encoder-decoder architecture that learns a compressed latent
    representation before reconstructing the target image.
    """

    def __init__(self, config_path: str, representation_size: int):
        super().__init__(config_path, representation_size)
        self.image_predictor = self._build_image_predictor()

    def _build_image_predictor(self) -> nn.Module:
        """
        Build an AutoCNN architecture for image prediction.
        
        Returns:
            ShallowAutoCNN: Autoencoder CNN model for image prediction
        """
        return ShallowAutoCNN(
            in_channels=1,  # Single channel input (grayscale representation)
            output_channels=self.config.num_bins,  # Multi-class output
            latent_dim=self.config.num_channels // 4,
            enc_channels=self.config.num_channels // 4,
            dec_channels=self.config.num_channels,
            dropout=self.config.dropout
        )

    def inference(self, representation: torch.Tensor, **kwargs) -> Tuple[
        torch.Tensor, Dict[str, torch.Tensor]]:
        pred = self.image_predictor(representation)
        return pred, {"predicted_images": pred}
