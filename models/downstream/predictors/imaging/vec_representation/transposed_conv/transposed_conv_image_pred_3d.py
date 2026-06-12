from typing import Dict, Tuple

import torch
import torch.nn as nn

from models.downstream.predictors.imaging.image_pred_3d import ImagePred3D


class TransposedConvImagePred3D(ImagePred3D):
    """
       A model that predicts 3D images from a representation using transposed convolution layers.
    """

    def __init__(self, config_path: str, representation_size: int):
        """
        Parameters
        ----------
        config_path (str): Path to the model configuration file.
        representation_size (int): Dimensionality of the input representation vector.
        """
        super().__init__(config_path, representation_size)

        assert self.config.amount_of_slices == 26, ("TransposedConvImagePred3D is designed for images with"
                                                    " only 26 slices right now.")
        assert self.config.image_size == 128, "TransposedConvImagePred is designed for 128x128 images right now."

        self.fc = nn.Linear(representation_size,
                            int(self.config.num_channels * (
                                    self.config.amount_of_slices // 2) * 4 * 4))  # [B, 512*13*4*4]

        # out = (in - 1) * stride - 2 * padding + kernel + output_padding
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(self.config.num_channels, (self.config.num_channels // 2), kernel_size=(3, 4, 4),
                               stride=(1, 2, 2), padding=(1, 1, 1)),  # [B,256,13,8,8]
            nn.BatchNorm3d((self.config.num_channels // 2)),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=self.config.dropout),

            nn.ConvTranspose3d((self.config.num_channels // 2), (self.config.num_channels // 4), kernel_size=(3, 4, 4),
                               stride=(1, 2, 2), padding=(1, 1, 1)),  # [B,128,13,16,16]
            nn.BatchNorm3d((self.config.num_channels // 4)),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=self.config.dropout),

            nn.ConvTranspose3d((self.config.num_channels // 4), (self.config.num_channels // 8), kernel_size=(3, 4, 4),
                               stride=(1, 2, 2), padding=(1, 1, 1)),  # [B,64,13,32,32]
            nn.BatchNorm3d((self.config.num_channels // 8)),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=self.config.dropout),

            nn.ConvTranspose3d((self.config.num_channels // 8), (self.config.num_channels // 16), kernel_size=(3, 4, 4),
                               stride=(1, 2, 2), padding=(1, 1, 1)),  # [B,32,13,64,64]
            nn.BatchNorm3d((self.config.num_channels // 16)),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=self.config.dropout),

            nn.ConvTranspose3d((self.config.num_channels // 16), self.config.num_bins, kernel_size=(4, 4, 4),
                               stride=(2, 2, 2), padding=(1, 1, 1)),  # [B,4,26,128,128]
        )

    def calculate_canvas_uncertainty(self, canvas: torch.Tensor) -> torch.Tensor:
        """
        Calculates uncertainty map from the canvas by computing entropy over the bins.
        Args:
            canvas: [B, NumBins, D, H, W]
        Returns:
            uncertainty_map: [B, H, W] numpy array
        """
        with torch.no_grad():
            probs = torch.softmax(canvas, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)  # [B, D, H, W]
            entropy = entropy[:, self.metrics.SLICE_IDX]  # Select slice SLICE_IDX -> [B, H, W]
        return entropy

    def inference(self, representation: torch.Tensor, **kwargs) -> Tuple[
        torch.Tensor, Dict[str, torch.Tensor]]:
        """

        Parameters:
            representation (torch.Tensor): Input representation tensor of shape [B, representation_size].

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                - Predicted class labels per voxel, shape [B, D, H, W].
                - A dictionary containing intermediate inference results, including
                  logits of shape [B, num_classes, D, H, W].

        """
        x = self.fc(representation)  # [B, 512*13*4*4]
        x = x.view(-1, self.config.num_channels, self.config.amount_of_slices // 2, 4, 4)  # [B, 512, 13, 4, 4]

        predicted_images_logit = self.decoder(x)  # [B, 4, 26, 128, 128]

        inference_results = {
            "predicted_images": predicted_images_logit,
            "canvas_uncertainty": self.calculate_canvas_uncertainty(predicted_images_logit)
        }

        return predicted_images_logit, inference_results
