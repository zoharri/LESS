from abc import abstractmethod, ABC

import pyrallis
import torch
from torch import nn


class AbstractModel(nn.Module, ABC):
    """
    Abstract base class for models that take a representation as input and predicts something.
    """

    def __init__(self, config_path: str):
        super().__init__()
        self.config = pyrallis.parse(self.config_type, config_path, args=[])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    @abstractmethod
    def config_type(self):
        """
        Returns the type of configuration for this model.
        This should be overridden in subclasses to return the specific configuration type.
        """
        raise NotImplementedError()

    @property
    def config_dict(self):
        """
        Returns the configuration of the model as a dictionary.
        This is useful for logging and saving the model configuration.
        """
        return pyrallis.encode(self.config)
