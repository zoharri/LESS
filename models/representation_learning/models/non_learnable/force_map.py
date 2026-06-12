"""
Non-learnable force map model for representation learning.

This module contains the ForceMapModel, which generates force maps using
Kernel Density Estimation (KDE). This is a non-learnable model that doesn't
require training and uses statistical methods to create representations
from tactile data.
"""

from typing import Tuple, Dict

import numpy as np
import torch
from scipy.stats import gaussian_kde

from models.representation_learning.representation_learning_model import RepresentationLearningModel


def kde_force_heatmap(last_pos, forces, grid_res=100, bandwidth_scale=0.3):
    """
    Generate a force heatmap using Kernel Density Estimation (KDE).
    
    This function creates a 2D heatmap representing the distribution of forces
    at different spatial locations using KDE. The heatmap can be used as a
    representation of the tactile interaction.
    
    Args:
        last_pos: Last positions of the tactile interaction
        forces: Force values corresponding to the positions
        grid_res: Resolution of the output grid
        bandwidth_scale: Scaling factor for the KDE bandwidth
        
    Returns:
        A 2D numpy array representing the force heatmap
    """
    positions = last_pos.T  # (2, N)

    # Apply KDE with bandwidth scaling
    kde = gaussian_kde(positions, weights=forces, bw_method=bandwidth_scale)

    x_min, x_max = last_pos[:, 0].min(), last_pos[:, 0].max()
    y_min, y_max = last_pos[:, 1].min(), last_pos[:, 1].max()

    x_grid, y_grid = np.meshgrid(
        np.linspace(x_min, x_max, grid_res),
        np.linspace(y_min, y_max, grid_res)
    )

    grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()])
    z = kde(grid_coords).reshape(grid_res, grid_res)

    return z


class ForceMapModel(RepresentationLearningModel):
    """
    A non-learnable model that generates force maps using Kernel Density Estimation.
    
    This model doesn't require training and uses statistical methods to create
    representations from tactile data. It generates 2D force heatmaps that
    represent the distribution of forces at different spatial locations.
    
    The model is useful for creating baseline representations or for cases where
    training a neural network is not feasible or desired.
    """

    def __init__(self, config_path: str):
        super(ForceMapModel, self).__init__(config_path)

    def inference(self, forces: torch.Tensor, locations: torch.Tensor, **kwargs) -> Tuple[
        torch.Tensor, Dict[str, torch.Tensor]]:
        padding_mask = kwargs.get("padding_mask", None)  # (B, L) True = valid
        B = locations.size(0)

        # Compute per-batch-item last valid position and aggregated force
        if padding_mask is not None:
            seq_lens = padding_mask.sum(dim=1).clamp(min=1).long()  # (B,)
            last_pos = torch.stack([
                locations[b, :seq_lens[b]].mean(dim=1)[:, :2].mean(dim=0)
                for b in range(B)
            ])  # (B, 2) — use mean position as a proxy for last position
        else:
            last_pos = locations.mean(dim=1).squeeze(1)[:, :2]  # (B, 2)

        if self.config.forcemap_use_norm:
            forces_agg = torch.norm(forces, dim=-1)
        else:
            forces_agg = forces[:, :, :, -1]

        # Average over sensors
        forces_agg = forces_agg.mean(dim=-1)  # (B, L)

        # Mask padded positions
        if padding_mask is not None:
            forces_agg = forces_agg * padding_mask.float().to(forces_agg.device)

        # Mean over valid time steps
        if padding_mask is not None:
            valid_counts = padding_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
            forces_scalar = forces_agg.sum(dim=1) / valid_counts.squeeze(1)
        else:
            forces_scalar = forces_agg.mean(dim=1)  # (B,)

        # Build per-step locations for KDE (use valid steps per batch item)
        force_maps = []
        for b in range(B):
            if padding_mask is not None:
                valid_len = seq_lens[b].item()
                pos_b = locations[b, :valid_len].mean(dim=1)[:, :2].cpu().numpy()  # (valid_len, 2)
                f_b = forces_agg[b, :valid_len].cpu().numpy()                       # (valid_len,)
            else:
                pos_b = locations[b].mean(dim=1)[:, :2].cpu().numpy()
                f_b = forces_agg[b].cpu().numpy()
            if len(pos_b) < 2:
                force_maps.append(torch.zeros(self.config.image_size, self.config.image_size,
                                              device=self.device))
                continue
            force_map = kde_force_heatmap(pos_b, f_b,
                                          grid_res=self.config.image_size,
                                          bandwidth_scale=self.config.bandwidth_scale)
            force_maps.append(torch.flip(torch.tensor(force_map, device=self.device), dims=(-1,)).float())
        force_map = torch.stack(force_maps, dim=0)
        return force_map, {"force_map": force_map}

    def compute_loss(self, inference_results: Dict[str, torch.Tensor], forces: torch.Tensor,
                     locations: torch.Tensor,
                     **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # In this case, we don't compute a loss as this is not learnable
        return torch.tensor(0.0, device=self.device), {}

    @property
    def representation_size(self) -> int:
        """
        Returns the size of the representation produced by the model.
        In this case, it is the size of the force map.
        """
        return self.config.image_size * self.config.image_size  # Assuming square grid
