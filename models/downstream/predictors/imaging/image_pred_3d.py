from abc import ABC
from typing import Dict, Optional

from matplotlib import pyplot as plt

from models.downstream.metrics import ImagingMetrics3d
from models.downstream.metrics.metrics import Metrics
from models.downstream.predictors.imaging.image_pred import ImagePred

from models.downstream.utils import *
from visualize_inserts_3d import visualize_inserts_3d

class ImagePred3D(ImagePred, ABC):

    def __init__(self, config_path: str, representation_size: int):
        super().__init__(config_path, representation_size)

        self.vis = visualize_inserts_3d(debug=False, show_text=False)

    def init_metrics(self) -> Metrics:
        return ImagingMetrics3d(self.config.num_bins, self.config.image_size, self.config.inference_on_big_phantom, self.config.non_background_threshold)

    def visualize_predictions(self, inference_results: Dict[str, torch.Tensor], gt: torch.Tensor,
                              **kwargs) -> Optional[plt.Figure]:
        """
        Visualize ground truth and predicted 3D images and save as screenshots.

        Parameters
        ----------
        inference_results (Dict[str, torch.Tensor]): Dictionary containing model outputs,
         [B, num_classes, D, H, W], must include "predicted_images".
        gt (torch.Tensor): Ground truth class labels of shape [B, D, H, W].
        **kwargs: Additional keyword arguments. Supports:
        - horizontal (bool): Whether to display results horizontally (default: False).
        - exp_names [str]: the names of the phantoms.

        Returns
        -------
        Optional[plt.Figure]: A Matplotlib figure showing side-by-side GT and predictions,
                               or None if visualization fails.
        """

        self.vis.reset_history()
        high_res = kwargs.pop("high_res", False)
        B = gt.shape[0]

        if B > 10:
            B = 10


        # split scores
        scores = torch.softmax(inference_results["predicted_images"], 1)
        scores_01 = scores[:, :2]  # [B, 2, H, W]
        scores_23 = scores[:, 2:4]  # [B, 2, H, W]

        # max over groups
        max_all, cls_all = scores.max(dim=1)
        max_01, cls_01 = scores_01.max(dim=1)  # cls_23 in {0,1} (local)
        max_23, cls_23 = scores_23.max(dim=1)  # cls_23 in {0,1} (local)

        # check threshold for classes 2/3
        use_23 = max_23 > self.config.non_background_threshold

        # final class map
        pred = torch.where(
            use_23,
            cls_all,  # map {0,1} -> {2,3}
            cls_01  # {0,1}
        )

        for i in range(B):
            self.vis.load_array(gt[i].detach().cpu(), is_big_phantom=self.config.inference_on_big_phantom)
            self.vis.load_array(pred[i].detach().cpu(), is_big_phantom=self.config.inference_on_big_phantom)

        imgs = self.vis.show()

        # load the showcase screenshots
        gts = []
        preds = []
        for i in range(B):
            gts.append(imgs[2 * i])
            preds.append(imgs[2 * i + 1])

        horizontal = kwargs.pop("horizontal", False)
        exp_names = kwargs.pop("exp_names", None)
        multi = 1 if not high_res else 10
        if horizontal:
            fig, axes = plt.subplots(2, B, figsize=(3 * B * multi, 6 * multi))  # two rows
            if B == 1:
                axes = axes.reshape(2, 1)
        else:
            fig, axes = plt.subplots(B, 2, figsize=(6 * multi, 3 * B * multi))  # two columns
            if B == 1:
                axes = axes.reshape(1, 2)

        for i in range(B):
            if horizontal:
                axes[0, i].imshow(gts[i])

                if exp_names is not None:
                    axes[0, i].set_title(f"GT {i} - {exp_names[i]}")
                else:
                    axes[0, i].set_title(f"GT {i}")

                axes[0, i].axis("off")

                axes[1, i].imshow(preds[i])
                axes[1, i].set_title(f"Pred {i}")
                axes[1, i].axis("off")
            else:
                axes[i, 0].imshow(gts[i])

                if exp_names is not None:
                    axes[i, 0].set_title(f"GT {i} - {exp_names[i]}")
                else:
                    axes[i, 0].set_title(f"GT {i}")

                axes[i, 0].axis("off")

                axes[i, 1].imshow(preds[i])
                axes[i, 1].set_title(f"Pred {i}")
                axes[i, 1].axis("off")

        plt.tight_layout()
        return fig
