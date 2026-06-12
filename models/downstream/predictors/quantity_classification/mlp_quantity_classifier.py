from abc import abstractmethod, ABC
from typing import Tuple, Dict

import torch
from torch import nn

from models.downstream.metrics.metrics import Metrics
from models.downstream.metrics.quantity_classification import QuantityClassificationMetrics
from models.downstream.predictors.downstream_model import DownstreamModel


class MLPQuantityClassifier(DownstreamModel, ABC):
    """
    A model that predicts a class from the representations
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
        layers.append(nn.Linear(in_dim, self.num_classes))
        self._model = nn.Sequential(*layers)

    def forward(self, representation: torch.Tensor, target: torch.Tensor, is_train: bool, **kwargs) -> Tuple[
        torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        real_class, kwargs = self.calc_class(kwargs)
        return super().forward(representation, real_class, is_train, **kwargs)

    def inference(self, representation: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        predicted_logits = self._model(representation)
        predicted_class = torch.argmax(predicted_logits, dim=1)
        return predicted_class, {"predicted_logits": predicted_logits, "predicted_class": predicted_class}

    def compute_loss(self, inference_results: Dict[str, torch.Tensor], target: torch.Tensor, **kwargs) -> Tuple[
        torch.Tensor, Dict[str, torch.Tensor]]:
        predicted_logits = inference_results["predicted_logits"]
        criterion = nn.CrossEntropyLoss()
        loss = criterion(predicted_logits, target)
        return loss, {"loss": loss}

    def init_metrics(self) -> Metrics:
        return QuantityClassificationMetrics(self.classifier_name,
                                             num_classes=self.num_classes)

    @abstractmethod
    def calc_class(self, kwargs: dict) -> Tuple[torch.Tensor, dict]:
        """
        Calculate the class from the kwargs and return it and the kwargs without it
        """
        raise NotImplementedError("This method should be overridden in subclasses to calculate a specific class.")

    @property
    @abstractmethod
    def classifier_name(self):
        raise NotImplementedError("This method should be overridden in subclasses to return the classifier name.")

    @property
    @abstractmethod
    def num_classes(self) -> int:
        raise NotImplementedError("This method should be overridden in subclasses to return the amount of classes.")
