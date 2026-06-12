from abc import abstractmethod, ABC
from typing import Dict, Tuple, Optional

import matplotlib.pyplot as plt
from torch import nn

from models.abstract_model import AbstractModel
from models.downstream.configs import DownstreamModelConfig
from models.downstream.metrics.metrics import Metrics
from models.downstream.utils import *


class DownstreamModel(AbstractModel, ABC):
    """
    Abstract base class for downstream models that take a representation as input and predicts something.
    """

    def __init__(self, config_path: str, representation_size: int):
        super().__init__(config_path)
        self.main_config = self.config
        self.config = self.main_config.get_active_config()
        self.representation_size = representation_size
        self.input_dropout = nn.Dropout(self.config.input_dropout)
        self.metrics = self.init_metrics()

    def forward(self, representation: torch.Tensor, gt: torch.Tensor, is_train: bool, **kwargs) -> Tuple[
        torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        representation = self.apply_input_dropout(representation)
        artifact, inference_results = self.inference(representation, **kwargs)
        loss, loss_info = self.compute_loss(inference_results, gt, **kwargs)
        if not is_train:
            self.metrics.update(artifact, gt)
        return artifact, loss, loss_info, inference_results

    def apply_input_dropout(self, representation: torch.Tensor) -> torch.Tensor:
        """
        Apply input dropout to the representation tensor.
        
        Args:
            representation (torch.Tensor): Input representation tensor
            
        Returns:
            torch.Tensor: Representation tensor with input dropout applied
        """
        return self.input_dropout(representation)

    @abstractmethod
    def inference(self, representation: torch.Tensor, **kwargs) -> Tuple[
        torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Perform inference using the model.
        This method should be overridden in subclasses to implement specific inference logic.
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_loss(self, inference_results: Dict[str, torch.Tensor], target: torch.Tensor, **kwargs) -> Tuple[
        torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the loss for the model.
        This method should be overridden in subclasses to implement specific loss computation logic.
        """
        raise NotImplementedError()

    def reset_metrics(self):
        """
        Reset the metrics for the model.
        This method should be overridden in subclasses to reset specific metrics.
        """
        self.metrics.reset()

    def get_avg_metrics(self):
        """
        Get the average metrics for the model.
        This method should be overridden in subclasses to compute specific average metrics.
        """
        return self.metrics.average_metrics()

    @abstractmethod
    def init_metrics(self) -> Metrics:
        """
        Initialize the metrics for the model.
        This method should be overridden in subclasses to initialize specific metrics.
        """
        raise NotImplementedError()

    @property
    def config_type(self):
        """
        Returns the type of configuration for a downstream model.
        """
        return DownstreamModelConfig

    @property
    def name(self) -> str:
        return self.main_config.name

    def visualize_predictions(self, inference_results: Dict[str, torch.Tensor], gt: torch.Tensor,
                              **kwargs) -> Optional[plt.Figure]:
        """
        Visualize predictions and gt images.
        No vis on default
        """
        return None
