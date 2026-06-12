import torch
from models.downstream.metrics.imaging.imaging_metrics import ImagingMetrics
from .imaging_metrics_utils import *


class ImagingMetrics3d(ImagingMetrics):
    """
        An extended version of ImagingMetrics in which we can compute metrics for 3D images
    """

    def __init__(self, num_bins: int, image_size: int, inference_on_big_phantom: bool, non_background_threshold:float):
        """

        Parameters
        ----------
        num_bins (int): Number of intensity/class bins used in image discretization.
                       Typically, 4 for background/insert/pillar/lump classification.
        image_size (int): The width or height of the images in the dataset (in our case Width == Height).
        """
        # minus 1 because there is no pillar in the 2d case
        self.metrics_2d = ImagingMetrics(num_bins - 1, image_size, inference_on_big_phantom, non_background_threshold)

        super().__init__(num_bins, image_size, inference_on_big_phantom, non_background_threshold)

        # The slice with the biggest area of lump
        self.SLICE_IDX = 14

        # special 3d metrics
        self.extra_metrics = {
            "pillar_diameter_error": np.array([]),
            "predicted_lump_size": np.array([]),
            "non_abs_rel_size_error": np.array([]),
            "rel_pillar_diameter_error": np.array([]),
        }

    def reset(self):
        super().reset()
        self.metrics_2d.reset()
        self.extra_metrics = {
            "pillar_diameter_error": np.array([]),
            "predicted_lump_size": np.array([]),
            "non_abs_rel_size_error": np.array([]),
            "rel_pillar_diameter_error": np.array([]),
        }

    def _update_metric(self, metric_type: str, value: np.ndarray) -> None:
        if metric_type in self.extra_metrics:
            self.extra_metrics[metric_type] = np.append(self.extra_metrics[metric_type], value, axis=0)
        elif metric_type in self.metrics:
            self.metrics[metric_type] = np.append(self.metrics[metric_type], value, axis=0)
        else:
            raise ValueError(f"Invalid metric type: {metric_type}")

    def average_pillar_diameter_error(self) -> float:
        return self.extra_metrics["pillar_diameter_error"].sum() / self.metrics["num_images"].sum()

    def average_rel_pillar_diameter_error(self) -> float:
        return self.extra_metrics["rel_pillar_diameter_error"].sum() / self.metrics["num_images"].sum()

    def average_predicted_lump_size(self) -> float:
        return self.extra_metrics["predicted_lump_size"].sum() / self.metrics["num_images"].sum()

    def average_non_abs_rel_size_error(self) -> float:
        return self.extra_metrics["non_abs_rel_size_error"].sum() / self.metrics["num_images"].sum()

    def _update_all_metrics_3d(
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
            confusion_matrix: torch.Tensor,
            pillar_diameter_error: torch.Tensor,
            lump_diameter_error: torch.Tensor,
            predicted_lump_size: torch.Tensor,
            non_abs_rel_size_error: torch.Tensor,
            rel_pillar_diameter_error: torch.Tensor,
            rel_lump_diameter_error: torch.Tensor,
    ) -> None:

        super()._update_all_metrics(
            tp, fp, tn, fn,
            total_positives, total_predicted_positives,
            error_rate, size_error, rel_size_error,
            num_images, loc_error, classification_error,
            lump_diameter_error, rel_lump_diameter_error,
            confusion_matrix
        )

        self._update_metric("pillar_diameter_error", np.expand_dims(pillar_diameter_error.cpu(), axis=0))
        self._update_metric("predicted_lump_size", np.expand_dims(predicted_lump_size.cpu(), axis=0))
        self._update_metric("non_abs_rel_size_error", np.expand_dims(non_abs_rel_size_error.cpu(), axis=0))
        self._update_metric("rel_pillar_diameter_error", np.expand_dims(rel_pillar_diameter_error.cpu(), axis=0))

    def update(self, predicted: torch.Tensor, gt: torch.Tensor):
        """
        Update metrics with a batch of predictions and ground truth.

        Processes a batch of predicted and ground truth images to compute and accumulate
        all supported metrics including segmentation metrics, geometric errors, and
        angular measurements.

        Args:
            predicted (torch.Tensor): Predicted images of shape [B, C, D, H, W]
            gt (torch.Tensor): Ground truth images of shape [B, D, H, W] with class indices
        """
        PILLAR_VALUE = 2
        LUMP_VALUE = 3

        # get the right slice that has been used in the 2d case as well
        pred_2d = predicted[:, :, self.SLICE_IDX, :, :].clone()
        gt_2d = gt[:, self.SLICE_IDX, :, :].clone()

        # lump = 2 in the 2d case
        pred_2d[pred_2d == 3] = 2
        gt_2d[gt_2d == 3] = 2

        # remove the pillar layer as there is no pillar in the 2d case
        pred_2d = torch.cat([pred_2d[:, :2], pred_2d[:, 3:4]], dim=1)

        self.metrics_2d.update(pred_2d, gt_2d)

        scores = torch.softmax(predicted, 1)
        scores_01 = scores[:, :2]  # [B, 2, H, W]
        scores_23 = scores[:, 2:4]  # [B, 2, H, W]

        # max over groups
        max_all, cls_all = scores.max(dim=1)
        max_01, cls_01 = scores_01.max(dim=1)  # cls_23 in {0,1} (local)
        max_23, cls_23 = scores_23.max(dim=1)  # cls_23 in {0,1} (local)

        # check threshold for classes 2/3
        use_23 = max_23 > self.non_background_threshold

        # final class map
        predicted = torch.where(
            use_23,
            cls_all,  # map {0,1} -> {2,3}
            cls_01  # {0,1}
        )

        tp = (torch.sum((predicted == LUMP_VALUE) & (gt == LUMP_VALUE))).float()
        fp = (torch.sum((predicted == LUMP_VALUE) & (gt != LUMP_VALUE))).float()
        fn = (torch.sum((predicted != LUMP_VALUE) & (gt == LUMP_VALUE))).float()
        tn = (torch.sum((predicted != LUMP_VALUE) & (gt != LUMP_VALUE))).float()
        total_positives = tp + fn
        total_predicted_positives = tp + fp

        error_rate = torch.mean((predicted != gt).float())

        # find the amount of lumps in gt and pred
        true_lumps = torch.sum(gt == LUMP_VALUE, dim=(1, 2, 3)).float()  # [B]
        predicted_lumps = torch.sum(predicted == LUMP_VALUE, dim=(1, 2, 3)).float()  # [B]

        # size error is the absolute difference between the number of true lumps and predicted lumps
        size_error = torch.abs(true_lumps - predicted_lumps).sum()
        rel_size_error = (torch.abs(true_lumps - predicted_lumps) / (0.5 * (true_lumps + predicted_lumps))).sum()
        non_abs_rel_size_error = ((true_lumps - predicted_lumps) / (0.5 * (true_lumps + predicted_lumps))).sum()
        rel_size_error[torch.isnan(rel_size_error)] = 0.0
        non_abs_rel_size_error[torch.isnan(non_abs_rel_size_error)] = 0.0

        num_images = gt.shape[0]  # The amount of items per batch

        # classification error is the number of samples where there is at least one true lump and no predicted lumps,
        # or no true lumps and at least one predicted lump
        classification_error = torch.sum(
            (true_lumps > 0) & (predicted_lumps == 0) | (true_lumps == 0) & (predicted_lumps > 0)).float()

        confusion_matrix = compute_confusion_matrix(predicted, gt, num_classes=self.num_bins)

        sum_loc_error = compute_loc_error(predicted, gt, predicted_lumps, true_lumps, lump_value=LUMP_VALUE, inference_on_big_phantom=self.inference_on_big_phantom)

        pillar_diameter_error, rel_pillar_error = diameter_error(predicted, gt, self.inference_on_big_phantom, PILLAR_VALUE)
        lump_diameter_error, rel_lump_error = diameter_error(predicted, gt, self.inference_on_big_phantom, LUMP_VALUE)

        pillar_diameter_error = pillar_diameter_error.sum()
        rel_pillar_diameter_error = rel_pillar_error.sum()

        lump_diameter_error = lump_diameter_error.sum()
        rel_lump_diameter_error = rel_lump_error.sum()

        predicted_lump_size = predicted_lumps.sum()

        self._update_all_metrics_3d(
            tp, fp, tn, fn,
            total_positives, total_predicted_positives,
            error_rate, size_error, rel_size_error,
            num_images, sum_loc_error, classification_error,
            confusion_matrix, pillar_diameter_error, lump_diameter_error, predicted_lump_size,
            non_abs_rel_size_error, rel_pillar_diameter_error, rel_lump_diameter_error
        )

    def average_metrics(self) -> dict:
        """
        Generate a comprehensive summary of all computed metrics.
        """
        average_metrics_2d = self.metrics_2d.average_metrics()
        average_metrics_2d.pop("confusion_matrix")
        average_metrics_2d = {"2D/" + k: v for k, v in average_metrics_2d.items()}

        average_metrics = {
                              "3D/error_rate_3d": self.error_rate(),
                              "3D/size_error_3d": self.average_size_error(),
                              "3D/loc_error_3d": self.average_loc_error(),
                              "3D/rel_size_error_3d": self.average_rel_size_error(),
                              "3D/classification_error_3d": self.classification_error(),
                              "3D/false_negative_rate_3d": self.false_negative_rate(),
                              "3D/false_positive_rate_3d": self.false_positive_rate(),
                              "3D/precision_3d": self.precision(),
                              "3D/recall_3d": self.recall(),
                              "3D/f1_score_3d": self.f1_score(),
                              "3D/gt_size_3d": self.metrics["total_positives"].sum() / self.metrics["num_images"].sum(),
                              "confusion_matrix": self.confusion_matrix(),
                              "3D/pillar_diameter_error": self.average_pillar_diameter_error(),
                              "3D/lump_diameter_error_3d": self.average_lump_diameter_error(),
                              "3D/predicted_lump_size": self.average_predicted_lump_size(),
                              "3D/non_abs_rel_size_error": self.average_non_abs_rel_size_error(),
                              "3D/rel_pillar_diameter_error": self.average_rel_pillar_diameter_error(),
                              "3D/rel_lump_diameter_error_3d": self.average_rel_lump_diameter_error(),
                          } | average_metrics_2d

        return average_metrics
