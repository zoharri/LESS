"""
ParticlesAggregation Non Learnable model
"""

from typing import Tuple, Dict

import numpy as np
import torch
from torch import nn

from models.representation_learning.representation_learning_model import RepresentationLearningModel
from models.representation_learning.models.force_reconstruction.vector_location_encoder import VectorLocationEncoder


class ParticlesAggregation(RepresentationLearningModel):
    """
    Non-learnable baseline that aggregates force measurements onto a fixed particle grid via EMA.

    Each particle accumulates the forces of nearby sensor readings. compute_loss always returns 0.
    """

    def __init__(self, config_path: str):
        super(ParticlesAggregation, self).__init__(config_path)

    def _build_static_particles_grid(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.config.particles_res_x <= 0 or self.config.particles_res_y <= 0:
            raise ValueError(
                f"Particle resolutions must be positive, got "
                f"({self.config.particles_res_x}, {self.config.particles_res_y})."
            )
        if self.config.particles_max_x <= self.config.particles_min_x or self.config.particles_max_y <= self.config.particles_min_y:
            raise ValueError(
                f"Invalid particle bounds: "
                f"x[{self.config.particles_min_x}, {self.config.particles_max_x}], "
                f"y[{self.config.particles_min_y}, {self.config.particles_max_y}]"
            )

        x_coords = torch.arange(
            self.config.particles_min_x,
            self.config.particles_max_x + 0.5 * self.config.particles_res_x,
            self.config.particles_res_x,
            device=device,
            dtype=dtype,
        )
        y_coords = torch.arange(
            self.config.particles_min_y,
            self.config.particles_max_y + 0.5 * self.config.particles_res_y,
            self.config.particles_res_y,
            device=device,
            dtype=dtype,
        )
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing="xy")
        return torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)


    def get_relative_particle_inputs(self, input_forces, input_locations, mean_input_locations, max_distance, particles_locations=None):
        """
        Args:
            input_forces: [B, L, D, D']
            input_locations: [B, L, K, K']
            mean_input_locations: [B, L, 2]
            max_distance: float
            particles_locations: [B, P, 2] (optional)

        Returns:
            filtered_locations: [N, L', K, K']
            filtered_forces:    [N, L', D, D']
            neighbor_mask:      [N, L']
            batch_indices:      [N]
            particles_locations:[B, P, 2]
            active_mask:        [B, P] (Bool mask indicating valid particles)
        """
        B, L, _ = mean_input_locations.shape
        device = mean_input_locations.device

        # 1. Generate shared static particles grid (replicated for all batch items)
        if particles_locations is None:
            base_grid = self._build_static_particles_grid(device=device, dtype=mean_input_locations.dtype)
            particles_locations = base_grid.unsqueeze(0).expand(B, -1, -1)
        P = particles_locations.shape[1]

        # 2. Compute Distances & Mask
        dists = torch.cdist(particles_locations, mean_input_locations)
        valid_neighbor_mask = dists <= max_distance

        # [B, P] - This is the mask we need for simple assignment later
        particle_has_neighbors = valid_neighbor_mask.any(dim=-1)

        if not particle_has_neighbors.any():
            K, Kp = input_locations.shape[-2:]
            D, Dp = input_forces.shape[-2:]
            return (torch.empty(0, 0, K, Kp, device=device),
                    torch.empty(0, 0, D, Dp, device=device),
                    torch.empty(0, 0, device=device, dtype=torch.bool),
                    torch.empty(0, device=device, dtype=torch.long),
                    particles_locations,
                    particle_has_neighbors)

        # 3. Flatten and Select
        batch_ids_grid = torch.arange(B, device=device).unsqueeze(1).expand(B, P)

        batch_indices = batch_ids_grid[particle_has_neighbors]
        active_particles = particles_locations[particle_has_neighbors]
        active_neighbor_mask = valid_neighbor_mask[particle_has_neighbors]

        # 4. Sort and Crop
        sorted_mask, sorted_indices = torch.sort(active_neighbor_mask.int(), dim=-1, descending=True, stable=True)

        num_neighbors = sorted_mask.sum(dim=-1)
        L_prime = num_neighbors.max().item()
        if L_prime == 0: L_prime = 1

        neighbor_indices = sorted_indices[:, :L_prime]
        neighbor_mask = sorted_mask[:, :L_prime].bool()

        # 5. Gather Data
        batch_indices_expanded = batch_indices.unsqueeze(1).expand(-1, L_prime)

        filtered_locations = input_locations[batch_indices_expanded, neighbor_indices]
        filtered_forces = input_forces[batch_indices_expanded, neighbor_indices]

        # 6. Make Relative & Mask
        p_x = active_particles[:, 0].view(-1, 1, 1)
        p_y = active_particles[:, 1].view(-1, 1, 1)

        filtered_locations[..., 0] = filtered_locations[..., 0] - p_x
        filtered_locations[..., 1] = filtered_locations[..., 1] - p_y

        mask_expanded_loc = neighbor_mask.unsqueeze(-1).unsqueeze(-1)
        filtered_locations = filtered_locations * mask_expanded_loc
        filtered_forces = filtered_forces * mask_expanded_loc

        return filtered_locations, filtered_forces, neighbor_mask, batch_indices, particles_locations, particle_has_neighbors

    def inference(self, forces: torch.Tensor, locations: torch.Tensor, **kwargs) -> Tuple[
        torch.Tensor, Dict[str, torch.Tensor]]:
        h_0 = kwargs.get("h_0", None)

        input_forces = forces.to(self.device)
        input_locations = locations.to(self.device)

        input_locations, input_forces, padding_mask, batch_indices, particles_locations, active_mask = self.get_relative_particle_inputs(
            input_forces, input_locations, input_locations.mean(dim=2)[:, :, :2],
            max_distance=self.config.max_particle_distance)

        representations, h_n = self.encode(input_locations, input_forces, h_0=h_0)

        # Reshape the representations back
        representations_map = torch.zeros(active_mask.shape[0], active_mask.shape[1], representations.shape[-1],
                                          device=representations.device)
        representations_map[active_mask] = representations[torch.arange(representations.shape[0]),
                                           padding_mask.sum(dim=1) - 1, :]

        return representations_map, {"all_outputs": representations.clone(),
                                     "padding_mask": padding_mask, "input_forces": input_forces,
                                     "particles_locations": particles_locations,
                                     "active_particles_mask": active_mask}

    def encode_online(self, forces: torch.Tensor, locations: torch.Tensor, **kwargs) -> Tuple[
        torch.Tensor, Dict[str, torch.Tensor]]:
        h_0 = kwargs.get("h_0", None)
        particles_locations = kwargs.get("particles_locations", None)

        input_forces = forces.to(self.device)
        input_locations = locations.to(self.device)

        input_locations, input_forces, padding_mask, batch_indices, particles_locations, active_mask = self.get_relative_particle_inputs(
            input_forces, input_locations, input_locations.mean(dim=2)[:, :, :2],
            max_distance=self.config.max_particle_distance,
            particles_locations=particles_locations)

        h_0_active = h_0[active_mask].unsqueeze(0) if h_0 is not None else None
        representations, _ = self.encode(input_locations, input_forces, h_0=h_0_active)

        # Reshape the representations back
        representations_map = torch.zeros(active_mask.shape[0], active_mask.shape[1], representations.shape[-1],
                                          device=representations.device) if h_0 is None else h_0
        # take last position in representations based on padding mask
        representations_map[active_mask] = representations[torch.arange(representations.shape[0]), padding_mask.sum(dim=1) - 1, :]

        return representations_map, {"h_n": representations_map,
                                     "padding_mask": padding_mask, "input_forces": input_forces,
                                     "particles_locations": particles_locations,
                                     "active_particles_mask": active_mask}

    def encode(self, locations: torch.Tensor, forces: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        h_0 = kwargs.get("h_0", None)
        batch_size, seq_len, num_sensors, dim_location = locations.shape
        forces_ = forces.clone() # B, T, S, 3
        B, T, S, D = forces.shape  # D=3

        # Reshape h_0 to separate sensor stats
        # [B, 1, S, 9] -> Splitting last dim into 3 chunks of 3
        h_0_reshaped = h_0.view(B, 1, S, -1) if h_0 is not None else torch.zeros(B, 1, S, 9, device=forces.device)

        prev_mean = h_0_reshaped[..., 0:3]  # [B, 1, S, 3]
        prev_max = h_0_reshaped[..., 3:6]
        prev_std = h_0_reshaped[..., 6:9]

        # Initialize running variance from std (Var = Std^2)
        prev_var = prev_std.pow(2)

        # We remove the time dimension for the loop state
        curr_mean = prev_mean.squeeze(1)  # [B, S, 3]
        curr_max = prev_max.squeeze(1)
        curr_var = prev_var.squeeze(1)

        features_list = []

        # Causal Loop over Time
        for t in range(T):
            x_t = forces[:, t, :, :]  # [B, S, 3]

            curr_max = torch.max(curr_max, x_t)

            old_mean = curr_mean
            curr_mean = 0.99 * curr_mean + (1 - 0.99) * x_t

            diff_term = (x_t - old_mean) * (x_t - curr_mean)
            curr_var = 0.99 * curr_var + (1 - 0.99) * diff_term

            # Compute Std
            curr_std = torch.sqrt(torch.clamp(curr_var, min=1e-6))

            # Concatenate per sensor: [Mean, Max, Std] -> [B, S, 9]
            feats_t = torch.cat([curr_mean, curr_max, curr_std], dim=-1)
            features_list.append(feats_t)

        # Stack time back: [B, T, S, 9]
        features_stacked = torch.stack(features_list, dim=1)

        # Flatten S and Features into F: [B, T, S*9]
        features = features_stacked.view(B, T, -1)

        return features, None

    def compute_loss(self, inference_results: Dict[str, torch.Tensor], forces: torch.Tensor,
                     locations: torch.Tensor,
                     **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # In this case, we don't compute a loss as this is not learnable
        return torch.tensor(0.0, device=self.device), {}
    @property
    def representation_size(self) -> int:
        """
        Returns the size of the representation produced by the model.
        This should be overridden in subclasses to return the specific representation size.
        """
        return 30*9
