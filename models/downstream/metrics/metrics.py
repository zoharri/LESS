from abc import ABC, abstractmethod

import numpy as np
import torch


class Metrics(ABC):
    """
    Abstract base class for all model evaluation metrics.
    
    This class defines the common interface and functionality that all metrics classes
    must implement. It provides a standardized framework for accumulating, updating,
    and summarizing evaluation metrics across different model types and tasks.
    
    The metrics system is designed to:
    - Accumulate results across multiple batches during evaluation
    - Provide interpretable named outputs for logging and analysis
    - Handle different data types (images, scalars, classifications)
    
    Subclasses must implement:
    - reset(): Clear accumulated metrics for new evaluation cycle
    - update(): Process new batch of predictions and ground truth
    - average_metrics(): Compute final averaged results
    """

    def __init__(self):
        """
        Initialize the metrics object.
        
        Sets up the internal metrics storage dictionary and calls reset()
        to prepare for the first evaluation cycle.
        """
        self.metrics = {}
        self.reset()

    @abstractmethod
    def reset(self):
        """
        Reset all accumulated metrics to their initial state.
        
        This method should clear all metric accumulators and prepare the object
        for a new evaluation cycle. Must be implemented by subclasses.
        
        Called automatically during initialization and should be called manually
        at the start of each evaluation epoch.
        """
        pass

    def _update_metric(self, metric_type, value):
        """
        Internal helper to accumulate a metric value.
        
        Appends new values to the existing metric array. This is a utility method
        used by subclasses to maintain consistent metric accumulation.
        
        Args:
            metric_type (str): Name of the metric to update. Must exist in self.metrics.
            value: New metric value(s) to append. Can be scalar or array.
            
        Raises:
            ValueError: If metric_type is not initialized in self.metrics.
        """
        if metric_type in self.metrics:
            self.metrics[metric_type] = np.append(self.metrics[metric_type], value, axis=0)
        else:
            raise ValueError(f"Invalid metric type: {metric_type}")

    @abstractmethod
    def average_metrics(self):
        """
        Compute and return final averaged metrics.
        
        This method should process all accumulated metric values and return
        a dictionary of final results suitable for logging, plotting, or analysis.
        
        Returns:
            dict: Dictionary mapping metric names to their computed values.
                 Keys should be descriptive strings, values should be numeric.
        """
        pass

    @abstractmethod
    def update(self, predicted: torch.Tensor, gt: torch.Tensor):
        """
        Update metrics with a batch of predictions and ground truth.
        
        This method processes new model outputs and corresponding ground truth
        to compute and accumulate relevant metrics. The specific metrics computed
        depend on the task type (imaging, regression, classification).
        
        Args:
            predicted (torch.Tensor): Model predictions. Shape and content depend on task type.
            gt (torch.Tensor): Ground truth targets. Shape should be compatible with predicted.
        """
        pass
