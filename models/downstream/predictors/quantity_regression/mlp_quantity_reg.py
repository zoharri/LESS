from abc import abstractmethod, ABC
from typing import Tuple, Dict, Optional

import torch
from matplotlib import pyplot as plt
from torch import nn

from models.downstream.metrics.metrics import Metrics
from models.downstream.metrics.quantity_regression import QuantityRegressionMetrics
from models.downstream.predictors.downstream_model import DownstreamModel


class MLPQuantityReg(DownstreamModel, ABC):
    """
    A model that predicts scalars from the representations
    """

    def __init__(self, config_path: str, representation_size: int):
        super().__init__(config_path, representation_size)
        in_dim = self.representation_size
        layers = []
        if self.config.num_layers > 0:
            hidden_sizes = [int(max(self.config.num_channels / (2 ** i), 2)) for i in range(0, self.config.num_layers)]
            for h in hidden_sizes:
                layers += [nn.Linear(in_dim, h), nn.ReLU()]
                if self.config.dropout > 0:
                    layers.append(nn.Dropout(self.config.dropout))
                in_dim = h
        layers.append(nn.Linear(in_dim, self.quantity_size))
        self._model = nn.Sequential(*layers)

    def forward(self, representation: torch.Tensor, target: torch.Tensor, is_train: bool, **kwargs) -> Tuple[
        torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        real_model_image = target.contiguous().view(-1, self.config.image_size, self.config.image_size)
        real_quantity = self.calc_quantity(real_model_image)
        return super().forward(representation, real_quantity, is_train, **kwargs)

    def inference(self, representation: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        predicted_quantity = self._model(representation)
        denormalized_quantity = self.denormalize_quantity(predicted_quantity)
        return denormalized_quantity, {
            f"normalized_predicted_{self.quantity_name}": predicted_quantity,
            f"denormalized_predicted_{self.quantity_name}": denormalized_quantity}

    def compute_loss(self, inference_results: Dict[str, torch.Tensor], target: torch.Tensor, **kwargs) -> Tuple[
        torch.Tensor, Dict[str, torch.Tensor]]:
        predicted_quantity = inference_results[f"normalized_predicted_{self.quantity_name}"]
        real_quantity = self.normalize_quantity(target)
        criterion = nn.MSELoss()
        loss = criterion(predicted_quantity, real_quantity)
        return loss, {}

    def init_metrics(self) -> Metrics:
        return QuantityRegressionMetrics(self.quantity_name)

    @abstractmethod
    def calc_quantity(self, image: torch.Tensor) -> torch.Tensor:
        """
        Calculate a quantity from the predicted image.
        This method should be overridden in subclasses to implement specific quantity calculations.
        """
        raise NotImplementedError("This method should be overridden in subclasses to calculate a specific quantity.")

    @abstractmethod
    def normalize_quantity(self, quantity: torch.Tensor) -> torch.Tensor:
        """
        Normalize the predicted quantity.
        This method should be overridden in subclasses to implement specific normalization logic.
        """
        raise NotImplementedError("This method should be overridden in subclasses to normalize the quantity.")

    @abstractmethod
    def denormalize_quantity(self, quantity: torch.Tensor) -> torch.Tensor:
        """
        Denormalize the predicted quantity.
        This method should be overridden in subclasses to implement specific denormalization logic.
        """
        raise NotImplementedError("This method should be overridden in subclasses to denormalize the quantity.")

    @property
    @abstractmethod
    def quantity_name(self):
        raise NotImplementedError("This method should be overridden in subclasses to return the quantity name.")

    @property
    @abstractmethod
    def quantity_size(self) -> int:
        raise NotImplementedError("This method should be overridden in subclasses to return the size of the quantity.")

    def visualize_predictions(self, inference_results: Dict[str, torch.Tensor], target: torch.Tensor,
                              **kwargs) -> Optional[plt.Figure]:
        predictions = inference_results[f"denormalized_predicted_{self.quantity_name}"]
        target_quantities = self.calc_quantity(
            target.contiguous().view(-1, self.config.image_size, self.config.image_size))
        num_images_to_plot = min(8, len(predictions))
        fig, axes = plt.subplots(num_images_to_plot, 1, figsize=(6, 4 * num_images_to_plot))
        for i, (model_image, predicted_quantity, target_quantity) in enumerate(
                zip(target[:num_images_to_plot], predictions[:num_images_to_plot],
                    target_quantities[:num_images_to_plot])):
            model_image = model_image.float() / (self.config.num_bins - 1)
            axes[i].imshow(model_image.cpu().detach().numpy(), cmap='gray', vmin=0, vmax=1)
            axes[i].set_title(
                f'{self.quantity_name}: Real {target_quantity.item():.2f}, Pred {predicted_quantity.item():.2f}')
        plt.tight_layout()
        return fig
