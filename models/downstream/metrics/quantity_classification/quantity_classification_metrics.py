import numpy as np
import torch

from models.downstream.metrics.metrics import Metrics


class QuantityClassificationMetrics(Metrics):
    """
    Comprehensive metrics for discrete classification tasks.
    
    This class provides a complete evaluation framework for classification models
    that predict discrete categories or classes, such as phantom index prediction,
    material classification, or object type identification.
    
    Key Features:
    - Accuracy tracking (per-sample and overall)
    - Confusion matrix computation for detailed error analysis
    - Support for multi-class classification scenarios
    - Named classifier tracking for interpretable results
    
    Args:
        classifier_name (str): Name of the classifier being evaluated (e.g., "phantom_index",
                              "material_type"). Used for generating named metric outputs.
        num_classes (int): Number of classes in the classification task.
    """

    def __init__(self, classifier_name: str, num_classes: int):
        self._classifier_name = classifier_name
        self._num_classes = num_classes
        super().__init__()

    def reset(self) -> None:
        """
        Reset all accumulated metrics to their initial state.
        
        Clears accuracy tracking, confusion matrix, and sample counts,
        preparing the metrics object for a new evaluation cycle.
        """
        self.metrics = {
            "confusion_matrix": np.zeros((0, self._num_classes, self._num_classes)),
            "accuracy": np.array([]),
            "num_samples": np.array([]),
        }

    def _update_metric(self, metric_type: str, value: np.ndarray) -> None:
        if metric_type in self.metrics:
            self.metrics[metric_type] = np.append(self.metrics[metric_type], value, axis=0)
        else:
            raise ValueError(f"Invalid metric type: {metric_type}")

    def average_accuracy(self) -> float:
        """
        Calculate the overall classification accuracy.
        
        Returns:
            float: Accuracy in range [0, 1], where 1.0 represents perfect classification.
        """
        return self.metrics["accuracy"].sum() / self.metrics["num_samples"].sum()

    def confusion_matrix(self) -> np.ndarray:
        """
        Generate the accumulated confusion matrix.
        
        The confusion matrix provides detailed insights into classification errors,
        showing which classes are commonly confused with each other.
        
        Returns:
            np.ndarray: Confusion matrix of shape [num_classes, num_classes] where
                       entry [i,j] represents the number of samples of true class j
                       that were predicted as class i.
        """
        return np.sum(self.metrics["confusion_matrix"], axis=0)

    def average_metrics(self) -> dict:
        """
        Generate a comprehensive summary of all computed metrics.
        
        Returns:
            dict: Dictionary containing named metrics with keys:
                - "{classifier_name}_accuracy": Overall classification accuracy
                - "{classifier_name}_confusion_matrix": Accumulated confusion matrix
        """
        return {
            f"{self._classifier_name}_accuracy": self.average_accuracy(),
            f"{self._classifier_name}_confusion_matrix": self.confusion_matrix(),
        }

    def update(self, predicted: torch.Tensor, gt: torch.Tensor) -> None:
        """
        Update metrics with a batch of predictions and ground truth labels.
        
        Processes a batch of predictions to compute accuracy and update the confusion
        matrix. Handles both individual predictions and batched evaluation.
        
        Args:
            predicted (torch.Tensor): Predicted class indices of shape [B] where
                                     B is batch size. Values should be in range [0, num_classes-1].
            gt (torch.Tensor): Ground truth class indices of shape [B] with same range as predicted.
        
        Note:
            Both predicted and gt should contain integer class indices, not one-hot encodings
            or probability distributions.
        """
        confusion_matrix = torch.zeros(1, self._num_classes, self._num_classes)
        for i in range(self._num_classes):
            for j in range(self._num_classes):
                confusion_matrix[0, i, j] = torch.sum((predicted == i) & (gt == j)).item()

        accuracy = predicted == gt

        self._update_metric("num_samples", np.expand_dims(gt.size(0), axis=0))
        self._update_metric("accuracy", accuracy.cpu())
        self._update_metric("confusion_matrix", confusion_matrix.cpu())
