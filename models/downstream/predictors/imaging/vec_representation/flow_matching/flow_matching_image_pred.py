from typing import Dict, Tuple

from torch import nn

from models.downstream.predictors.imaging.image_pred import ImagePred
from models.downstream.predictors.imaging.vec_representation.flow_matching.unet import UNetWithCondition
from models.downstream.utils import *


class FlowMatchingImagePred(ImagePred):
    """
    Predicts semantic images from a representation using guided flow matching "inspired" architecture.
    Full details and motivation can be found in the paper.
    """

    def __init__(self, config_path: str, representation_size: int):
        super().__init__(config_path, representation_size)
        assert self.config.image_size == 128, "Only supports 128x128 images"

        self.latent_dim = self.config.num_channels  # typically something like 8 or 16
        self.hidden_size = self.config.num_channels * 4
        self.condition_dim = self.representation_size

        # Flow field prediction: input is (x, t, representation), output is dx/dt
        self.flow_predictor = UNetWithCondition(
            in_channels=self.latent_dim + 1,  # plus 1 for time t
            cond_dim=self.condition_dim,
            out_channels=self.latent_dim,
            hidden_size=self.hidden_size,
            dropout=self.config.dropout,
            use_residual=self.config.use_residual_in_unet_flowmatching,
        )

        # Decoder from latent image to logits
        self.decoder = nn.Sequential(
            nn.Conv2d(self.latent_dim, self.hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_size, self.config.num_bins, kernel_size=1)
        )

    def _sample_latents(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        return torch.randn(batch_size, self.latent_dim, height, width, device=device)

    def inference(self, representation: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B = representation.shape[0]
        z = self._sample_latents(B, 128, 128, self.device)
        t_vals = torch.linspace(1, 0, steps=self.config.flow_steps, device=self.device)

        for t in t_vals:
            t_tensor = t.expand(B, 1, 128, 128)
            z = z + self.flow_predictor(torch.cat([z, t_tensor], dim=1), representation)

        logits = self.decoder(z)
        return logits, {"predicted_images": logits}

    def _inference_all_intermidiate(self, representation: torch.Tensor, **kwargs) -> Tuple[
        torch.Tensor, Dict[str, torch.Tensor]]:
        B = representation.shape[0]
        z = self._sample_latents(B, 128, 128, self.device)
        t_vals = torch.linspace(1, 0, steps=self.config.flow_steps, device=self.device)
        all_logits = []
        for t in t_vals:
            t_tensor = t.expand(B, 1, 128, 128)
            z = z + self.flow_predictor(torch.cat([z, t_tensor], dim=1), representation)
            all_logits.append(self.decoder(z))
        return torch.stack(all_logits), {}
