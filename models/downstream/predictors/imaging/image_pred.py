from abc import ABC
from typing import Dict, Tuple, Optional

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from image_utils import undiscretize_image
from models.downstream.metrics.imaging import ImagingMetrics
from models.downstream.metrics.metrics import Metrics
from models.downstream.predictors.downstream_model import DownstreamModel

matplotlib.use('Agg')

import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss
import torch.nn.functional as F
from models.downstream.utils import *


class ImagePred(DownstreamModel, ABC):
    """
    Abstract base class for image prediction models.
    
    This class provides the common interface and functionality for models that predict 
    images from learned representations. It includes loss computation, metrics initialization,
    and visualization capabilities for image prediction tasks.
    """

    def resize_image_to_target_size(self, image: torch.Tensor) -> torch.Tensor:
        """
        Resize image to target size using bilinear interpolation.

        Args:
            image: Input image tensor of shape (B, C, H, W) or (B, H, W)

        Returns:
            Resized image tensor
        """
        should_squeeze = False
        if image.dim() == 3:
            should_squeeze = True
            # Add channel dimension if missing
            image = image.unsqueeze(1)

        current_size = image.shape[-1]
        if current_size != self.config.image_size:
            image = F.interpolate(
                image,
                size=(self.config.image_size, self.config.image_size),
                mode='bilinear',
                align_corners=False
            )
        if should_squeeze:
            image = image.squeeze(1)

        return image

    def forward(self, representation: torch.Tensor, target: torch.Tensor, is_train: bool, **kwargs) -> Tuple[
        torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        target = self.resize_image_to_target_size(target)
        return super().forward(representation, target, is_train, **kwargs)

    def compute_loss(self, inference_results: Dict[str, torch.Tensor], gt: torch.Tensor, **kwargs) -> Tuple[
        torch.Tensor, Dict[str, torch.Tensor]]:
        predicted_images = inference_results["predicted_images"]
        real_model_image = gt.long()
        if self.config.loss == "cross_entropy":
            if self.config.balance_image_classes:
                with torch.no_grad():
                    flat_labels = real_model_image.view(-1)
                    class_counts = torch.bincount(flat_labels, minlength=self.config.num_bins).float()
                    class_weights = 1.0 / (class_counts + 1e-6)
                    class_weights = class_weights * (self.config.num_bins / class_weights.sum())  # Normalize

                loss = nn.CrossEntropyLoss(weight=class_weights.to(predicted_images.device))(predicted_images,
                                                                                             real_model_image)
            else:
                loss = nn.CrossEntropyLoss()(predicted_images, real_model_image)
        elif self.config.loss == "focal":
            loss = focal_loss(predicted_images, real_model_image, self.config.num_bins)
        elif self.config.loss == "dice":
            loss = multiclass_dice_loss(predicted_images, real_model_image, self.config.num_bins)
        elif self.config.loss == "dice_focal":
            loss_dice = multiclass_dice_loss(predicted_images, real_model_image, self.config.num_bins)

            loss_focal = focal_loss(predicted_images, real_model_image, self.config.num_bins)
            scale = loss_dice.detach() / (loss_focal.detach() + 1e-8)
            loss_focal_scaled = loss_focal * scale

            lam = 0.5
            loss = lam * loss_dice + (1 - lam) * loss_focal_scaled
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss}")
        return loss, {}

    def init_metrics(self) -> Metrics:
        return ImagingMetrics(self.config.num_bins, self.config.image_size, self.config.inference_on_big_phantom, self.config.non_background_threshold)

    def visualize_predictions(self, inference_results: Dict[str, torch.Tensor], gt: torch.Tensor,
                              **kwargs) -> Optional[plt.Figure]:
        """
            Visualize model predictions alongside their ground truth images.

            Parameters
            ----------
            inference_results : Dict[str, torch.Tensor]
                Dictionary containing model outputs. Must include:
                - "predicted_images": Tensor of predicted images (shape: [N, H, W] or [N, 1, H, W]).
            gt : torch.Tensor
                Ground truth tensor of corresponding target images (shape: [N, H, W]).
            **kwargs :
                Supported keys:
                - horizontal : bool, optional
                    If True, displays predictions and ground truths horizontally
                    (two rows: model images on top, predictions below).
                    If False (default), displays vertically (two columns per sample).
                - exp_names : List[str], optional
                    List of experiment or phantom names to display as titles.

            Returns
            -------
            matplotlib.figure.Figure or None
                Matplotlib Figure object containing the visualization.
                Returns None if no figure is created (e.g., empty predictions).
        """
        predictions = inference_results["predicted_images"]
        num_images_to_plot = min(8, len(predictions))

        horizontal = kwargs.pop("horizontal", False)
        exp_names = kwargs.pop("exp_names", None)

        # Determine subplot layout
        if horizontal:
            fig, axes = plt.subplots(2, num_images_to_plot, figsize=(4 * num_images_to_plot, 8))
        else:
            fig, axes = plt.subplots(num_images_to_plot, 2, figsize=(12, 4 * num_images_to_plot))

        # Ensure axes is 2D for consistent indexing
        axes = np.atleast_2d(axes)

        for i, (model_image, predicted_image) in enumerate(
                zip(gt[:num_images_to_plot], predictions[:num_images_to_plot])):

            model_image = model_image.float() / (self.config.num_bins - 1)
            predicted_image = undiscretize_image(predicted_image, bins=self.config.num_bins)

            if horizontal:
                ax_model = axes[0, i]
                ax_pred = axes[1, i]
            else:
                ax_model = axes[i, 0]
                ax_pred = axes[i, 1]

            # Plot model image
            ax_model.imshow(model_image.cpu().detach().numpy(), cmap='gray', vmin=0, vmax=1)
            if exp_names is not None:
                ax_model.set_title(f'Model Image {i} - {exp_names[i]}')
            else:
                ax_model.set_title(f'Model Image {i}')
            ax_model.axis('off')

            # Plot predicted image
            ax_pred.imshow(predicted_image.cpu().detach().numpy(), cmap='gray', vmin=0, vmax=1)
            ax_pred.set_title(f'Predicted Image')
            ax_pred.axis('off')

        plt.tight_layout()
        return fig
