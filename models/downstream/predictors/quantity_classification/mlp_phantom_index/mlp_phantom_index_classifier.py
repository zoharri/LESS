from typing import Tuple

import torch

from models.downstream.predictors.quantity_classification.mlp_quantity_classifier import MLPQuantityClassifier


class MLPPhantomIndexClassifier(MLPQuantityClassifier):
    """
    A model that predicts the index of the phantom from the representations using MLP.
    """

    def __init__(self, config_path: str, representation_size: int):
        super().__init__(config_path, representation_size)

    def calc_class(self, kwargs: dict) -> Tuple[torch.Tensor, dict]:
        """
        Extract the phantom index from the experiment name in kwargs.
        
        Args:
            kwargs: Dictionary containing experiment metadata, must include 'exp_name'
            
        Returns:
            Tuple containing:
                - phantom_index: Tensor of phantom indices (0-based)
                - kwargs: Updated kwargs with exp_name removed
        """
        # Extract the phantom index from the kwargs
        exp_name = kwargs.pop("exp_name", None)
        if exp_name is None:
            raise ValueError("exp_name must be provided in kwargs")

        # Extract phantom index from experiment name (assumes format where 2nd character is phantom number)
        phantom_index = torch.tensor([int(curr_exp_name[1]) - 1 for curr_exp_name in exp_name], device=self.device)
        return phantom_index, kwargs

    @property
    def classifier_name(self):
        """Name of the classifier for metrics and logging."""
        return "phantom_index"

    @property
    def num_classes(self) -> int:
        """Number of phantom classes (4 different phantoms)."""
        return 4
