import random
from typing import Dict, Tuple

import torch
import torchvision.transforms.functional as TF
from torch import nn

from models.downstream.predictors.imaging.image_pred import ImagePred


def apply_same_augmentations(a, b):
    # random angle
    angle = random.uniform(-30, 30)

    # random flips
    flip_x = random.random() < 0.5
    flip_y = random.random() < 0.5

    def apply(img):
        out = TF.rotate(img, angle)
        if flip_x:
            out = TF.hflip(out)
        if flip_y:
            out = TF.vflip(out)
        return out

    return apply(a), apply(b)


class LocalCNNMapRepPred(ImagePred):
    def __init__(self, config_path: str, representation_size: int):
        super().__init__(config_path, representation_size)
        self.image_predictor = self._build_image_predictor()

    def _build_image_predictor(self) -> nn.Module:
        """
        Build a CNN architecture for image prediction.

        Returns:
            CNNImagePredictor: CNN model for image prediction
        """
        layers = []
        in_channels = self.representation_size
        layer_num = 0
        while in_channels > 16:
            out_channels = max(16, self.representation_size // (self.config.cnn_down_multiplier * 2 ** (layer_num + 1)))
            if self.config.use1x1:
                layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=1))
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=self.config.cnn_kernel_size,
                                    padding=self.config.cnn_kernel_size // 2))
            layers.append(nn.ReLU())
            if self.config.dropout > 0:
                layers.append(nn.Dropout(self.config.dropout))
            if self.config.cnn_usebn:
                layers.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels
            layer_num += 1
        layers.append(nn.Conv2d(in_channels, self.config.num_bins, kernel_size=self.config.cnn_kernel_size,
                                padding=self.config.cnn_kernel_size // 2))

        return nn.Sequential(*layers)

    def inference(self, representation: torch.Tensor, **kwargs) -> Tuple[
        torch.Tensor, Dict[str, torch.Tensor]]:
        pred = self.image_predictor(representation)
        return pred, {"predicted_images": pred}

    def forward(self, representation: torch.Tensor, target: torch.Tensor, is_train: bool, **kwargs) -> Tuple[
        torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        # augmentations
        if self.config.local_cnn_use_aug and is_train:
            representation, target = apply_same_augmentations(representation, target)
        return super().forward(representation, target, is_train, **kwargs)
