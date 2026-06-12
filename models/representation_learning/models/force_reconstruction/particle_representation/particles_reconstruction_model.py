"""
ParticlesReconstructionModel
"""

from typing import Tuple, Dict

import numpy as np
import torch
from torch import nn

from models.representation_learning.representation_learning_model import RepresentationLearningModel
from models.representation_learning.models.force_reconstruction.vector_location_encoder import VectorLocationEncoder


class ParticlesReconstructionModel(RepresentationLearningModel):
    """
    A configurable model for reconstructing forces from locations.

    This model supports multiple architectures:
    - GRU: Gated Recurrent Unit for sequential processing
    - Transformer: Attention-based architecture for parallel processing

    The model takes force and location data as input and learns to reconstruct
    forces from the learned representations. It uses positional encoding to
    capture spatial information and can be configured for different types of
    force reconstruction tasks.
    """

    def __init__(self, config_path: str):
        super(ParticlesReconstructionModel, self).__init__(config_path)

        self.force_location_encoder = VectorLocationEncoder(self.config.locations_size, self.config.pe_max_freq_log2,
                                                            self.config.gru_input_embed_dim,
                                                            self.config.force_size)
        self.latent_location_encoder = VectorLocationEncoder(self.config.locations_size, self.config.pe_max_freq_log2,
                                                             self.config.decoder_input_embed_dim,
                                                             self.config.representation_size)

        arch = self.config.arch.lower()

        if arch == "gru":
            self.encoder = nn.GRU(input_size=self.force_location_encoder.output_size(),
                                  hidden_size=self.config.representation_size,
                                  batch_first=True,
                                  dropout=self.config.dropout)
        elif arch == "transformer":
            self.input_encoder_layer = nn.Linear(self.force_location_encoder.output_size(),
                                                 self.config.representation_size)
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.representation_size,
                                                       nhead=self.config.transformer_nhead,
                                                       dim_feedforward=self.config.transformer_ff_dim,
                                                       batch_first=True,
                                                       dropout=self.config.dropout)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.config.transformer_num_layers)
        else:
            raise ValueError(f"Unknown arch: {arch}")

        self.force_predictor = nn.Sequential(
            nn.Linear(self.latent_location_encoder.output_size(), self.config.force_predictor_hidden),
            nn.ReLU(),
            nn.Linear(self.config.force_predictor_hidden, self.config.force_predictor_hidden // 2),
            nn.ReLU(),
            nn.Linear(self.config.force_predictor_hidden // 2, self.config.force_size)
        )

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

    def get_relative_particle_inputs(self, input_forces, input_locations, mean_input_locations, max_distance, particles_locations=None, padding_mask=None):
        """
        Args:
            input_forces: [B, L, D, D']
            input_locations: [B, L, K, K']
            mean_input_locations: [B, L, 2]
            max_distance: float
            particles_locations: [B, P, 2] (optional)
            padding_mask: [B, L] bool, True = valid position (optional). Padded positions
                          are excluded from the neighbor search so zero-filled padding
                          entries don't spuriously attract nearby particles.

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

        # Exclude padding positions so zero-filled entries don't count as neighbors
        if padding_mask is not None:
            valid_neighbor_mask = valid_neighbor_mask & padding_mask.unsqueeze(1)  # (B, P, L)

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
        predict_all = kwargs.get("predict_all", False)
        particles_locations = kwargs.get("particles_locations", None)
        active_mask = kwargs.get("active_mask", None)
        padding_mask = kwargs.get("padding_mask", None)  # (B, L) True = valid

        representation, encode_results = self.encode(
            forces,
            locations,
            h_0=h_0,
            particles_locations=particles_locations,
            active_mask=active_mask,
            padding_mask=padding_mask,
        )
        representations = encode_results["all_outputs"]
        output_locations = encode_results["output_locations"]
        padding_mask = encode_results["padding_mask"]

        predicted_forces, input_steps, reconstruction_steps = self.decode(
            representations,
            output_locations,
            encode_results["padding_mask"],
            predict_all=predict_all,
        )

        return representation, {
            "predicted_forces": predicted_forces,
            "h_n": encode_results["h_n"],
            "input_steps": input_steps,
            "reconstruction_steps": reconstruction_steps,
            "all_outputs": representations.clone(),
            "padding_mask": padding_mask,
            "input_forces": encode_results["input_forces"],
            "particles_locations": encode_results["particles_locations"],
            "active_particles_mask": encode_results["active_particles_mask"],
        }

    def get_masked_input_output(self, forces: torch.Tensor, locations: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates masked input and output tensors based on the configured mask percentage.
        """
        if self.config.mask_percentage > 0:
            batch_size, num_steps, _, _ = locations.shape
            if self.config.trajectory_level_mask:
                traj_len = self.config.trajectory_length
                num_trajs = num_steps // traj_len
                num_input_trajs = int(num_trajs * (1 - self.config.mask_percentage))

                rand = torch.rand(batch_size, num_trajs, device=self.device)
                permuted_trajs = rand.argsort(dim=1)

                input_trajs = permuted_trajs[:, :num_input_trajs].sort(dim=1)[0]
                output_trajs = permuted_trajs[:, num_input_trajs:].sort(dim=1)[0]

                base = torch.arange(traj_len, device=self.device).view(1, 1, traj_len)
                input_indices = (input_trajs.unsqueeze(-1) * traj_len + base).reshape(batch_size, -1)
                output_indices = (output_trajs.unsqueeze(-1) * traj_len + base).reshape(batch_size, -1)
            else:
                rand = torch.rand(batch_size, num_steps, device=self.device)
                permuted = rand.argsort(dim=1)

                num_input_steps = int(num_steps * self.config.mask_percentage)
                input_indices = permuted[:, :num_input_steps].sort(dim=1)[0]
                output_indices = permuted[:, num_input_steps:].sort(dim=1)[0]
            batch_indices = torch.arange(batch_size).unsqueeze(-1)
            input_locations = locations[batch_indices, input_indices, :]
            input_forces = forces[batch_indices, input_indices, :]
            output_locations = locations[batch_indices, output_indices, :]
        else:
            input_indices = torch.arange(locations.size(1)).unsqueeze(0).expand(locations.size(0), -1).to(self.device)
            output_indices = input_indices
            input_locations = locations
            input_forces = forces
            output_locations = locations

        return input_forces, input_locations, output_locations, input_indices, output_indices

    def _encode_sequence(self, locations: torch.Tensor, forces: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        h_0 = kwargs.get("h_0", None)
        batch_size, seq_len, num_sensors, dim_location = locations.shape
        combined = self.force_location_encoder(forces.reshape(batch_size * seq_len, -1),
                                               locations.reshape(batch_size * seq_len, num_sensors, dim_location))
        combined = combined.reshape(batch_size, seq_len, -1)

        arch = self.config.arch.lower()
        if arch == "gru":
            if self.config.gru_tbptt_step_size is not None:
                output = []
                hidden_state = h_0
                for i in range(int(np.ceil(combined.shape[1] / self.gru_tbptt_step_size))):
                    curr_input = combined[:,
                                 i * self.gru_tbptt_step_size:i * self.gru_tbptt_step_size + self.gru_tbptt_step_size]
                    curr_output, hidden_state = self.encoder(curr_input, hidden_state)
                    output.append(curr_output)
                    hidden_state = hidden_state.detach()
                output = torch.cat(output, dim=1)
                h_n = hidden_state
            else:
                output, h_n = self.encoder(combined, h_0)
            return output, h_n

        elif arch == "transformer":
            combined = self.input_encoder_layer(combined)
            mask = torch.triu(torch.ones((seq_len, seq_len), device=self.device) * float('-inf'),
                              diagonal=1) if self.config.transformer_use_causal_mask else None
            output = self.encoder(combined, mask=mask, is_causal=self.config.transformer_use_causal_mask)
            h_n = output[:, -1, :].unsqueeze(0)
            return output, h_n

    def encode(self, forces: torch.Tensor, locations: torch.Tensor, **kwargs) -> Tuple[
        torch.Tensor, Dict[str, torch.Tensor]]:
        h_0 = kwargs.get("h_0", None)
        particles_locations = kwargs.get("particles_locations", None)
        active_mask = kwargs.get("active_mask", None)
        batch_padding_mask = kwargs.get("padding_mask", None)  # (B, L) from pad_collate

        input_forces = forces.to(self.device)
        input_locations = locations.to(self.device)
        if batch_padding_mask is not None:
            batch_padding_mask = batch_padding_mask.to(self.device)

        output_locations, input_forces, padding_mask, _, particles_locations, current_active_mask = (
            self.get_relative_particle_inputs(
                input_forces,
                input_locations,
                input_locations.mean(dim=2)[:, :, :2],
                max_distance=self.config.max_particle_distance,
                particles_locations=particles_locations,
                padding_mask=batch_padding_mask,
            )
        )

        if active_mask is not None:
            active_mask = active_mask.to(self.device, dtype=torch.bool)
            if active_mask.shape != current_active_mask.shape:
                raise ValueError(
                    f"`active_mask` shape {tuple(active_mask.shape)} does not match "
                    f"current mask shape {tuple(current_active_mask.shape)}."
                )
            merged_active_mask = active_mask | current_active_mask
        else:
            merged_active_mask = current_active_mask

        if h_0 is None:
            representations_map = torch.zeros(
                current_active_mask.shape[0],
                current_active_mask.shape[1],
                self.config.representation_size,
                device=self.device,
                dtype=input_forces.dtype,
            )
        else:
            representations_map = h_0.clone()

        if output_locations.size(0) == 0:
            all_outputs = torch.empty(
                0,
                0,
                self.config.representation_size,
                device=self.device,
                dtype=representations_map.dtype,
            )
        else:
            h_0_active = h_0[current_active_mask].unsqueeze(0) if h_0 is not None else None
            all_outputs, _ = self._encode_sequence(output_locations, input_forces, h_0=h_0_active)
            last_steps = padding_mask.sum(dim=1) - 1
            active_indices = torch.arange(all_outputs.shape[0], device=all_outputs.device)
            representations_map[current_active_mask] = all_outputs[active_indices, last_steps, :]

        encode_results: Dict[str, torch.Tensor] = {
            "h_n": representations_map,
            "padding_mask": padding_mask,
            "input_forces": input_forces,
            "particles_locations": particles_locations,
            "active_particles_mask": merged_active_mask,
            "all_outputs": all_outputs,
            "output_locations": output_locations,
        }

        return representations_map, encode_results

    def decode(self, representations: torch.Tensor, locations: torch.Tensor, padding_mask: torch.Tensor,
               predict_all: bool = False):
        locations_mean = locations.mean(dim=2)
        batch_size, loc_seq_len, _ = locations_mean.shape
        _, rep_seq_len, _ = representations.shape

        # Ensure mask is float for sampling
        # valid=1.0, padding=0.0
        mask_weights = padding_mask.float()

        # --- 1. Sample Input Steps ---
        if self.config.input_num_random_samples != -1 and predict_all is False:
            # Sample indices where mask is True using multinomial
            # replacement=True allows picking the same index twice (independent sampling)
            input_steps = torch.multinomial(
                mask_weights,
                self.config.input_num_random_samples,
                replacement=True
            ).to(self.device)
        else:
            # For predict_all, we take the full sequence (0 to L)
            input_steps = torch.arange(rep_seq_len).unsqueeze(0).expand(batch_size, rep_seq_len).to(self.device)

        # --- 2. Sample Reconstruction Steps ---
        if self.config.reconstruction_num_random_samples != -1 and predict_all is False:
            # We need independent reconstruction targets for every input sample.
            # Shape needed: [B, input_num_samples, reconstruction_num_samples]

            total_samples_needed = self.config.input_num_random_samples * self.config.reconstruction_num_random_samples

            # Sample flattened list of valid indices
            flat_recon_steps = torch.multinomial(
                mask_weights,
                total_samples_needed,
                replacement=True
            ).to(self.device)

            # Reshape to match the required dimensions
            reconstruction_steps = flat_recon_steps.view(
                batch_size,
                self.config.input_num_random_samples,
                self.config.reconstruction_num_random_samples
            )
        else:
            # Handling the non-random cases (taking full sequence or expanding input steps)
            if self.config.input_num_random_samples != -1 and predict_all is False:
                reconstruction_steps = (
                    torch.arange(0, loc_seq_len).unsqueeze(0).unsqueeze(0).expand(batch_size,
                                                                                  self.config.input_num_random_samples,
                                                                                  -1).to(self.device))
            else:
                reconstruction_steps = torch.arange(0, loc_seq_len).unsqueeze(0).unsqueeze(0).expand(batch_size,
                                                                                                     rep_seq_len,
                                                                                                     -1).to(self.device)

        # --- 3. Gather and Combine (Unchanged Logic) ---
        batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(-1)

        # Gather representations based on input_steps
        expanded_output = representations[batch_indices, input_steps].view(batch_size, input_steps.size(1), -1)
        expanded_output = expanded_output.unsqueeze(2).expand(-1, -1, reconstruction_steps.size(2), -1)

        # Gather locations based on reconstruction_steps
        # Note: We expand locations first to handle the gather
        expanded_locations = locations.unsqueeze(1).expand(-1, rep_seq_len, -1, -1, -1)

        # Since reconstruction_steps might have rank 3 [B, In, Rec], we need careful indexing
        # If predict_all is True, input_steps is [B, L], recon is [B, L, L] (potentially)
        # If predict_all is False, input_steps is [B, S1], recon is [B, S1, S2]

        # We broadcast batch_indices and input_steps to match reconstruction_steps shape
        # batch_indices: [B, 1, 1]
        # input_steps:   [B, S1, 1]
        # recon_steps:   [B, S1, S2]
        expanded_locations = expanded_locations[
            batch_indices.unsqueeze(-1),
            input_steps.unsqueeze(-1),
            reconstruction_steps
        ]

        combined_features = self.latent_location_encoder(
            expanded_output.reshape(-1, expanded_output.size(-1)),
            expanded_locations.reshape(-1, expanded_locations.size(-2), expanded_locations.size(-1))
        )
        combined_features = combined_features.view(batch_size, input_steps.size(1), reconstruction_steps.size(2), -1)

        predicted_forces = self.force_predictor(combined_features)

        return predicted_forces, input_steps, reconstruction_steps

    def compute_loss(self, inference_results: Dict[str, torch.Tensor], forces: torch.Tensor,
                     locations: torch.Tensor,
                     **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        forces_target = inference_results["input_forces"]
        batch_indices = torch.arange(forces_target.size(0)).unsqueeze(-1).unsqueeze(-1)
        forces_target = forces_target.view(forces_target.size(0), forces_target.size(1), -1)
        forces_target = forces_target.unsqueeze(1).expand(-1, forces_target.size(1), -1,
                                                          -1)  # expand to match predicted shape
        forces_target = forces_target[
            batch_indices, inference_results["input_steps"].unsqueeze(-1), inference_results[
                "reconstruction_steps"]]
        predicted_vectors = inference_results["predicted_forces"]
        vector_prediction_criterion = nn.MSELoss()
        all_recon_loss = vector_prediction_criterion(predicted_vectors, forces_target)

        consistency_loss = torch.mean(
            torch.norm(inference_results["all_outputs"][:, 1:] - inference_results["all_outputs"][:, :-1], dim=-1))

        total_loss = all_recon_loss

        if self.config.consistency_reg_weight > 0:
            total_loss += self.config.consistency_reg_weight * consistency_loss

        return total_loss, {"all_forces_loss": all_recon_loss, "consistency_hidden_loss": consistency_loss}

    @property
    def representation_size(self) -> int:
        """
        Returns the size of the representation produced by the model.
        This should be overridden in subclasses to return the specific representation size.
        """
        return self.config.representation_size
