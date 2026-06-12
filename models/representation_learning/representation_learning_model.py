from abc import ABC, abstractmethod
from typing import Tuple, Dict

import torch
import torch.nn as nn

from models.abstract_model import AbstractModel
from models.representation_learning.configs import RepresentationModelConfig


class L2SP_Loss(nn.Module):
    def __init__(self, model):
        super().__init__()
        # 1. Save a frozen copy of the pre-trained weights
        self.pretrained_weights = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.pretrained_weights[name] = param.detach().clone()

    def forward(self, model):
        # 2. Calculate L2 distance between current and pretrained weights
        l2_sp_loss = 0.0
        for name, param in model.named_parameters():
            if name in self.pretrained_weights:
                target = self.pretrained_weights[name]
                l2_sp_loss += torch.norm(param - target, p=2) ** 2

        return l2_sp_loss


class RepresentationLearningModel(AbstractModel, ABC):
    """
    Abstract base class for representation learning models.
    
    This class defines the interface that all representation learning models must implement.
    Representation learning models take tactile data (forces and locations) as input and
    learn to produce meaningful representations that can be used by downstream tasks.
    
    Subclasses should implement:
    - inference(): Perform inference to generate representations
    - compute_loss(): Compute the loss for training
    - representation_size: Property returning the size of the learned representation
    """

    @abstractmethod
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.main_config = self.config
        self.config = self.main_config.get_active_config()
        self.on_first_forward_pass = True
        self.l2_sp_loss_module = None

    @abstractmethod
    def inference(self, forces: torch.Tensor, locations: torch.Tensor, **kwargs) -> Tuple[
        torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Perform inference using the model.
        This method should be overridden in subclasses to implement specific inference logic.
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_loss(self, inference_results: Dict[str, torch.Tensor], forces: torch.Tensor, locations: torch.Tensor,
                     **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the loss for the model.
        This method should be overridden in subclasses to implement specific loss computation logic.
        """
        raise NotImplementedError()

    def forward(self, forces: torch.Tensor, locations: torch.Tensor, **kwargs) -> Tuple[
        torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if self.on_first_forward_pass:
            if self.config.l2_sp_regularization_weight > 0.0:
                self.l2_sp_loss_module = L2SP_Loss(self)
        representation, inference_results = self.inference(forces, locations, **kwargs)
        loss, loss_info = self.compute_loss(inference_results, forces, locations, **kwargs)
        if self.l2_sp_loss_module is not None:
            l2_sp_loss = self.l2_sp_loss_module(self)
            loss += self.config.l2_sp_regularization_weight * l2_sp_loss
            loss_info['l2_sp_loss'] = l2_sp_loss.detach()
        self.on_first_forward_pass = False
        return representation, loss, loss_info, inference_results

    @property
    @abstractmethod
    def representation_size(self) -> int:
        """
        Returns the size of the representation produced by the model.
        This should be overridden in subclasses to return the specific representation size.
        """
        raise NotImplementedError()

    @property
    def name(self) -> str:
        return self.main_config.name

    @property
    def config_type(self):
        """
        Returns the type of configuration for this model.
        """
        return RepresentationModelConfig
