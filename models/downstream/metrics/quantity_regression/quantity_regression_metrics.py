import numpy as np
import torch

from models.downstream.metrics.metrics import Metrics


class QuantityRegressionMetrics(Metrics):
    """
    Comprehensive metrics for quantity regression tasks.
    
    This class provides metrics for evaluating continuous quantity predictions such as
    lump area, lump center coordinates, and other measurable properties. It tracks
    both absolute and relative errors to provide insights into model performance
    across different scales of quantities.
    
    Key Features:
    - Absolute error tracking (L2 norm)
    - Relative error computation (normalized by ground truth magnitude)  
    - Sample-wise accumulation for batch processing
    - Named quantity tracking for interpretable results
    
    Args:
        quantity_name (str): Name of the quantity being predicted (e.g., "lump_area", 
                           "lump_center"). Used for generating named metric outputs.
    """

    def __init__(self, quantity_name):
        self._quantity_name = quantity_name
        super().__init__()

    def reset(self):
        """
        Reset all accumulated metrics to their initial state.
        
        Clears all error accumulations and sample counts, preparing the metrics
        object for a new evaluation cycle.
        """
        self.metrics = {
            f"quantity_error": np.array([]),
            f"rel_quantity_error": np.array([]),
            "num_samples": np.array([]),
        }

    def _update_metric(self, metric_type, value):
        if metric_type in self.metrics:
            self.metrics[metric_type] = np.append(self.metrics[metric_type], value, axis=0)
        else:
            raise ValueError(f"Invalid metric type: {metric_type}")

    def average_quantity_error(self):
        """
        Calculate the average absolute error across all samples.
        
        Returns:
            float: Mean absolute error (L2 norm) between predictions and ground truth.
        """
        return self.metrics["quantity_error"].sum() / self.metrics["num_samples"].sum()

    def average_rel_quantity_error(self):
        """
        Calculate the average relative error across all samples.
        
        Relative error is computed as absolute_error / ground_truth_magnitude,
        providing scale-invariant evaluation.
        
        Returns:
            float: Mean relative error normalized by ground truth magnitude.
        """
        return self.metrics["rel_quantity_error"].sum() / self.metrics["num_samples"].sum()

    def average_metrics(self):
        """
        Generate a comprehensive summary of all computed metrics.
        
        Returns:
            dict: Dictionary containing named metrics with keys:
                - "{quantity_name}_error": Average absolute error
                - "{quantity_name}_rel_error": Average relative error
        """
        return {
            f"{self._quantity_name}_error": self.average_quantity_error(),
            f"{self._quantity_name}_rel_error": self.average_rel_quantity_error(),
        }

    def update(self, predicted: torch.Tensor, gt: torch.Tensor):
        """
        Update metrics with a batch of predictions and ground truth values.
        
        Computes both absolute and relative errors for the batch and accumulates
        them into the running metrics. Handles multi-dimensional quantities by
        computing L2 norm across the last dimension.
        
        Args:
            predicted (torch.Tensor): Predicted quantity values of shape [B, D] where
                                     B is batch size and D is quantity dimensionality
            gt (torch.Tensor): Ground truth quantity values of shape [B, D]
        
        Note:
            For scalar quantities, D=1. For vector quantities like coordinates, D>1.
            The L2 norm is computed across dimension D to get per-sample errors.
        """
        quantity_error = torch.linalg.norm(gt - predicted, dim=1)
        rel_quantity_error = (quantity_error / torch.linalg.norm(gt, dim=1))

        self._update_metric("quantity_error", quantity_error.cpu())
        self._update_metric("rel_quantity_error", rel_quantity_error.cpu())
        self._update_metric("num_samples", np.expand_dims(predicted.size(0), axis=0))
