import numpy as np
import torch

from models.downstream.metrics.metrics import Metrics
from .imaging_metrics_utils import *


class ImagingMetrics(Metrics):
    """
    Comprehensive metrics for evaluating image prediction models.
    
    This class provides a complete suite of metrics specifically designed for evaluating
    image prediction models in medical/tactile imaging contexts, particularly for tasks
    involving lump detection and segmentation.
    
    Metrics include:
    - Classification accuracy metrics (pixel-wise, overall)
    - Segmentation metrics (precision, recall, F1-score)
    - Geometric metrics (location error, center of mass error, size error)
    - Distribution metrics (Earth Mover's Distance)
    
    Args:
        num_bins (int): Number of intensity/class bins used in image discretization.
                       Typically 3 for background/tissue/lump classification.
    
    """

    def __init__(self, num_bins, image_size, inference_on_big_phantom, non_background_threshold):
        self.num_bins = num_bins
        self.image_size = image_size
        self.inference_on_big_phantom = inference_on_big_phantom
        self.non_background_threshold = non_background_threshold
        super().__init__()

    def reset(self):
        """
        Reset all accumulated metrics to their initial state.
        
        Initializes all metric arrays to empty and prepares the metrics object
        for a new evaluation cycle. Should be called at the beginning of each
        evaluation epoch.
        """
        self.metrics = {
            "true_positives": np.array([]),
            "false_positives": np.array([]),
            "true_negatives": np.array([]),
            "false_negatives": np.array([]),
            "total_positives": np.array([]),
            "total_predicted_positives": np.array([]),
            "error_rate": np.array([]),
            "size_error": np.array([]),
            "rel_size_error": np.array([]),
            "loc_error": np.array([]),
            "num_images": np.array([]),
            "classification_error": np.array([]),
            "lump_diameter_error": np.array([]),
            "rel_lump_diameter_error": np.array([]),
            "confusion_matrix": np.zeros((0, self.num_bins, self.num_bins))
        }

    def _update_metric(self, metric_type: str, value: np.ndarray) -> None:
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

    def false_negative_rate(self) -> float:
        """
        Calculate the false negative rate.
        
        Returns:
            float: False negative rate in range [0, 1]
        """
        if self.metrics["total_positives"].sum() == 0:
            return 0
        return self.metrics["false_negatives"].sum() / self.metrics["total_positives"].sum()

    def false_positive_rate(self) -> float:
        """
        Calculate the false positive rate.

        Returns:
            float: False positive rate in range [0, 1]
        """
        if self.metrics["total_predicted_positives"].sum() == 0:
            return 0
        return self.metrics["false_positives"].sum() / self.metrics["total_predicted_positives"].sum()

    def average_size_error(self) -> float:
        """
        Calculate the average lump size error.

        Returns:
            float: Average size error
        """
        return self.metrics["size_error"].sum() / self.metrics["num_images"].sum()

    def average_rel_size_error(self) -> float:
        """
        Calculate the average relative lump size error.

        Returns:
            float: Average relative size error
        """
        return self.metrics["rel_size_error"].sum() / self.metrics["num_images"].sum()

    def average_loc_error(self) -> float:
        """
        Calculate the average location error.

        Returns:
            float: Average location error
        """
        return self.metrics["loc_error"].sum() / self.metrics["num_images"].sum()

    def classification_error(self) -> float:
        return self.metrics["classification_error"].sum() / self.metrics["num_images"].sum()

    def error_rate(self) -> float:
        return self.metrics["error_rate"].mean()

    def precision(self) -> float:
        """
        Calculate precision for lump detection.
        
        Precision measures the fraction of predicted lump pixels that are actually lumps.
        Higher precision means fewer false positive detections.
        
        Returns:
            float: Precision score in range [0, 1], where 1.0 is perfect precision.
        """
        if (self.metrics["true_positives"].sum() + self.metrics["false_positives"].sum()) == 0:
            return 0
        return self.metrics["true_positives"].sum() / (
                self.metrics["true_positives"].sum() + self.metrics["false_positives"].sum())

    def recall(self) -> float:
        """
        Calculate recall (sensitivity) for lump detection.
        
        Recall measures the fraction of actual lump pixels that are correctly detected.
        Higher recall means fewer missed lumps (false negatives).
        
        Returns:
            float: Recall score in range [0, 1], where 1.0 is perfect recall.
        """
        if (self.metrics["true_positives"].sum() + self.metrics["false_negatives"].sum()) == 0:
            return 0
        return self.metrics["true_positives"].sum() / (
                self.metrics["true_positives"].sum() + self.metrics["false_negatives"].sum())

    def f1_score(self) -> float:
        """
        Calculate F1-score for lump detection.
        
        Returns:
            float: F1-score in range [0, 1]
        """
        precision = self.precision()
        recall = self.recall()
        if (precision + recall) == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)

    def average_lump_diameter_error(self) -> float:
        return self.metrics["lump_diameter_error"].sum() / self.metrics["num_images"].sum()

    def average_rel_lump_diameter_error(self) -> float:
        return self.metrics["rel_lump_diameter_error"].sum() / self.metrics["num_images"].sum()

    def confusion_matrix(self) -> np.ndarray:
        """
        Calculate the confusion matrix

        Returns:
            np.ndarray: Confusion matrix of shape [num_bins, num_bins]
        """
        return np.mean(self.metrics["confusion_matrix"], axis=0)

    def average_metrics(self) -> dict:
        """
        Generate a comprehensive summary of all computed metrics.

        Returns:
            dict: Dictionary containing named metrics with keys:
                - "error_rate": Average error rate
                - "size_error": Average size error
                - "loc_error": Average location error
                - "rel_size_error": Average relative size error
                - "classification_error": Average classification error
                - "false_negative_rate": Average false negative rate
                - "false_positive_rate": Average false positive rate
                - "precision": Average precision
                - "recall": Average recall
                - "f1_score": Average F1-score
                - "gt_size": Average ground truth size
                - "confusion_matrix": Average confusion matrix
        """
        return {
            "error_rate": self.error_rate(),
            "size_error": self.average_size_error(),
            "loc_error": self.average_loc_error(),
            "rel_size_error": self.average_rel_size_error(),
            "classification_error": self.classification_error(),
            "false_negative_rate": self.false_negative_rate(),
            "false_positive_rate": self.false_positive_rate(),
            "precision": self.precision(),
            "recall": self.recall(),
            "f1_score": self.f1_score(),
            "gt_size": self.metrics["total_positives"].sum() / self.metrics["num_images"].sum(),
            "lump_diameter_error": self.average_lump_diameter_error(),
            "rel_lump_diameter_error": self.average_rel_lump_diameter_error(),
            "confusion_matrix": self.confusion_matrix()
        }

    def _update_all_metrics(
            self,
            tp: torch.Tensor,
            fp: torch.Tensor,
            tn: torch.Tensor,
            fn: torch.Tensor,
            total_positives: torch.Tensor,
            total_predicted_positives: torch.Tensor,
            error_rate: torch.Tensor,
            size_error: torch.Tensor,
            rel_size_error: torch.Tensor,
            num_images: int,
            loc_error: torch.Tensor,
            classification_error: torch.Tensor,
            lump_diameter_error: torch.Tensor,
            rel_lump_diameter_error: torch.Tensor,
            confusion_matrix: torch.Tensor
    ) -> None:
        """
        Helper function to update all metrics at once.
        All tensor inputs can be on GPU or CPU.
        """

        # Binary classification / segmentation metrics
        self._update_metric("true_positives", np.expand_dims(tp.cpu(), axis=0))
        self._update_metric("false_positives", np.expand_dims(fp.cpu(), axis=0))
        self._update_metric("true_negatives", np.expand_dims(tn.cpu(), axis=0))
        self._update_metric("false_negatives", np.expand_dims(fn.cpu(), axis=0))
        self._update_metric("total_positives", np.expand_dims(total_positives.cpu(), axis=0))
        self._update_metric("total_predicted_positives", np.expand_dims(total_predicted_positives.cpu(), axis=0))

        # Error metrics
        self._update_metric("error_rate", np.expand_dims(error_rate.cpu(), axis=0))
        self._update_metric("size_error", np.expand_dims(size_error.cpu(), axis=0))
        self._update_metric("rel_size_error", np.expand_dims(rel_size_error.cpu(), axis=0))
        self._update_metric("num_images", np.expand_dims(num_images, axis=0))
        self._update_metric("loc_error", np.expand_dims(loc_error.cpu().numpy(), axis=0))

        self._update_metric("lump_diameter_error", np.expand_dims(lump_diameter_error.cpu(), axis=0))
        self._update_metric("rel_lump_diameter_error", np.expand_dims(rel_lump_diameter_error.cpu(), axis=0))

        # Classification error and confusion matrix
        self._update_metric("classification_error", np.expand_dims(classification_error.cpu(), axis=0))
        self._update_metric("confusion_matrix", confusion_matrix.cpu().numpy())

    def update(self, predicted: torch.Tensor, gt: torch.Tensor):
        """
        Update metrics with a batch of predictions and ground truth.
        
        Processes a batch of predicted and ground truth images to compute and accumulate
        all supported metrics including segmentation metrics, geometric errors, and
        angular measurements.
        
        Args:
            predicted (torch.Tensor): Predicted images of shape [B, C, H, W] where C is num_bins
            gt (torch.Tensor): Ground truth images of shape [B, H, W] with class indices
        """
        # image sizes are [B, C, H, W]
        predicted_images = predicted.view(-1, self.num_bins, self.image_size, self.image_size)
        real_model_image = gt.contiguous().view(-1, self.image_size, self.image_size)
        scores = torch.softmax(predicted_images, 1)
        scores_01 = scores[:, :2]  # [B, 2, H, W]
        scores_23 = scores[:, 2:4]  # [B, 2, H, W]

        # max over groups
        max_all, cls_all = scores.max(dim=1)
        max_01, cls_01 = scores_01.max(dim=1)  # cls_23 in {0,1} (local)
        max_23, cls_23 = scores_23.max(dim=1)  # cls_23 in {0,1} (local)

        # check threshold for classes 2/3
        use_23 = max_23 > self.non_background_threshold

        # final class map
        predicted_labels = torch.where(
            use_23,
            cls_all,  # map {0,1} -> {2,3}
            cls_01  # {0,1}
        )


        # Calculate the error: a value of 1 for incorrect predictions and 0 for correct predictions
        incorrect_predictions = (predicted_labels != real_model_image).float()
        # Calculate the average error across all images and pixels
        error_rate = torch.mean(incorrect_predictions)

        # count the number of true lumps and predicted lumps
        true_lumps = torch.sum(real_model_image == 2, dim=(1, 2))
        predicted_lumps = torch.sum(predicted_labels == 2, dim=(1, 2))
        # size error is the absolute difference between the number of true lumps and predicted lumps
        size_error = torch.abs(true_lumps - predicted_lumps).sum()

        rel_size_error = (torch.abs(true_lumps - predicted_lumps) / (0.5 * (true_lumps + predicted_lumps))).sum()
        rel_size_error[torch.isnan(rel_size_error)] = 0.0

        # classification error is the number of samples where there is at least on true lump and no predicted lumps, or no true lumps and at least one predicted lump
        classification_error = torch.sum(
            (true_lumps > 0) & (predicted_lumps == 0) | (true_lumps == 0) & (predicted_lumps > 0)).float()

        # compute the f1 score for predicting the lump correctly for each pixel in the images
        true_positive = torch.sum((predicted_labels == 2) & (real_model_image == 2))
        false_positive = torch.sum((predicted_labels == 2) & (real_model_image != 2))
        false_negative = torch.sum((predicted_labels != 2) & (real_model_image == 2))
        true_negative = (torch.sum((predicted_labels != 2) & (real_model_image != 2))).float()
        total_positive = true_positive + false_negative
        total_predicted_positives = true_positive + false_positive

        confusion_matrix = compute_confusion_matrix(predicted_labels, real_model_image, self.num_bins)

        sum_loc_errors = compute_loc_error(predicted_labels, real_model_image, predicted_lumps, true_lumps,
                                           self.inference_on_big_phantom, lump_value=2)

        lump_diameter_error, rel_lump_error = diameter_error(predicted_labels, gt,
                                                             self.inference_on_big_phantom,2)
        lump_diameter_error = lump_diameter_error.sum()
        rel_lump_diameter_error = rel_lump_error.sum()

        self._update_all_metrics(
            tp=true_positive, fp=false_positive, tn=true_negative, fn=false_negative,
            total_positives=total_positive, total_predicted_positives=total_predicted_positives,
            error_rate=error_rate, size_error=size_error, rel_size_error=rel_size_error,
            num_images=real_model_image.size(0), classification_error=classification_error,
            confusion_matrix=confusion_matrix,
            loc_error=sum_loc_errors, lump_diameter_error=lump_diameter_error,
            rel_lump_diameter_error=rel_lump_diameter_error)
