"""
Vector-based reconstruction model for representation learning.

This module contains the ReconstructionModel, which is a configurable neural network
model for reconstructing forces from locations. It supports multiple architectures
including GRU and Transformer.

The model uses positional encoding to encode spatial information and can be configured
for different types of force reconstruction tasks.
"""

from typing import Tuple, Dict

import numpy as np
import torch
from torch import nn

from models.representation_learning.representation_learning_model import RepresentationLearningModel
from models.representation_learning.models.force_reconstruction.vector_location_encoder import VectorLocationEncoder


class ReconstructionModel(RepresentationLearningModel):
    """
    A configurable model for reconstructing forces from locations.
    
    This model supports multiple architectures:
    - GRU: Gated Recurrent Unit for sequential processing
    - Transformer: Attention-based architecture for parallel processing
    
    The model takes force and location data as input and learns to reconstruct
    forces from the learned representations. It uses positional encoding to
    capture spatial information and can be configured for different types of
    force reconstruction tasks.
    
    This model replaces the old GRUReconstructionModel and provides a more
    flexible and extensible architecture.
    """

    def __init__(self, config_path: str):
        super(ReconstructionModel, self).__init__(config_path)

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

    def inference(self, forces: torch.Tensor, locations: torch.Tensor, **kwargs) -> Tuple[
        torch.Tensor, Dict[str, torch.Tensor]]:
        h_0 = kwargs.get("h_0", None)
        predict_all = kwargs.get("predict_all", False)
        padding_mask = kwargs.get("padding_mask", None)  # (B, L) True = valid
        input_forces, input_locations, output_locations, mask_input_indices, mask_output_indices = (
            self.get_masked_input_output(forces, locations, padding_mask))
        representation, encode_results = self.encode(input_forces, input_locations, h_0=h_0,
                                                     padding_mask=padding_mask)
        representations = encode_results["all_outputs"]
        h_n = encode_results["h_n"]
        predicted_forces, input_steps, reconstruction_steps = self.decode(representations, output_locations,
                                                                          padding_mask=padding_mask,
                                                                          predict_all=predict_all)

        batch_size = mask_output_indices.size(0)
        mask_input_indices  = mask_input_indices.to(self.device)
        mask_output_indices = mask_output_indices.to(self.device)
        input_steps = mask_input_indices[torch.arange(batch_size, device=self.device).unsqueeze(-1), input_steps]
        reconstruction_steps = mask_output_indices[
            torch.arange(batch_size, device=self.device).unsqueeze(-1).unsqueeze(-1), reconstruction_steps]

        return representation, {"predicted_forces": predicted_forces, "h_n": h_n, "input_steps": input_steps,
                                "reconstruction_steps": reconstruction_steps, "all_outputs": representations.clone()}

    def get_masked_input_output(self, forces: torch.Tensor, locations: torch.Tensor,
                                padding_mask: torch.Tensor = None) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates masked input and output tensors based on the configured mask percentage.
        When padding_mask is provided, sampling is restricted to valid (non-padded) positions.
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
                if padding_mask is not None:
                    # Sample only from valid positions using padding_mask as weights
                    weights = padding_mask.float().to(self.device)
                    valid_counts = weights.sum(dim=1, keepdim=True).clamp(min=1)
                    num_input_steps = max(1, int(weights.sum(dim=1).float().mean().item() * self.config.mask_percentage))
                    input_indices = torch.multinomial(weights, num_input_steps, replacement=False).sort(dim=1)[0]
                    # For output, use all valid positions
                    output_indices = torch.arange(num_steps, device=self.device).unsqueeze(0).expand(batch_size, -1)
                    output_indices = output_indices * padding_mask.long().to(self.device)
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
        padding_mask = kwargs.get("padding_mask", None)  # (B, L) True = valid
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
            causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=self.device) * float('-inf'),
                                     diagonal=1) if self.config.transformer_use_causal_mask else None
            # True in src_key_padding_mask means "ignore this position" (PyTorch convention)
            key_padding_mask = ~padding_mask.to(self.device) if padding_mask is not None else None
            output = self.encoder(combined, mask=causal_mask,
                                  src_key_padding_mask=key_padding_mask,
                                  is_causal=self.config.transformer_use_causal_mask)
            h_n = output[:, -1, :].unsqueeze(0)
            return output, h_n

    def encode(self, forces: torch.Tensor, locations: torch.Tensor, **kwargs) -> Tuple[
        torch.Tensor, Dict[str, torch.Tensor]]:
        h_0 = kwargs.get("h_0", None)
        padding_mask = kwargs.get("padding_mask", None)  # (B, L) True = valid
        _ = kwargs.get("active_mask", None)

        input_forces = forces.to(self.device)
        input_locations = locations.to(self.device)
        all_outputs, h_n = self._encode_sequence(input_locations, input_forces, h_0=h_0,
                                                  padding_mask=padding_mask)

        # Use the hidden state at the last valid position instead of after padding
        if padding_mask is not None:
            pm = padding_mask.to(self.device)
            seq_lens = pm.sum(dim=1).clamp(min=1)  # (B,)
            last_valid = (seq_lens - 1).long()      # (B,)
            representation = all_outputs[torch.arange(all_outputs.size(0), device=self.device), last_valid]
        else:
            representation = h_n[-1]

        encode_results: Dict[str, torch.Tensor] = {"h_n": h_n, "all_outputs": all_outputs}
        return representation, encode_results

    def decode(self, representations: torch.Tensor, locations: torch.Tensor, padding_mask: torch.Tensor = None,
               predict_all: bool = False):
        locations_mean = locations.mean(dim=2)
        batch_size, loc_seq_len, _ = locations_mean.shape
        _, rep_seq_len, _ = representations.shape

        # Build sampling weights: uniform over valid positions, zero over padding
        if padding_mask is not None:
            sample_weights = padding_mask.float().to(self.device)  # (B, L)
        else:
            sample_weights = None

        if self.config.input_num_random_samples != -1 and predict_all is False:
            if sample_weights is not None:
                input_steps = torch.multinomial(sample_weights, self.config.input_num_random_samples,
                                                replacement=True).to(self.device)
            else:
                input_steps = torch.randint(0, rep_seq_len,
                                            (batch_size, self.config.input_num_random_samples)).to(self.device)
        else:
            input_steps = torch.arange(rep_seq_len).unsqueeze(0).expand(batch_size, rep_seq_len).to(self.device)
        if self.config.reconstruction_num_random_samples != -1 and predict_all is False:
            if sample_weights is not None:
                flat = torch.multinomial(sample_weights,
                                         self.config.input_num_random_samples * self.config.reconstruction_num_random_samples,
                                         replacement=True).to(self.device)
                reconstruction_steps = flat.view(batch_size, self.config.input_num_random_samples,
                                                  self.config.reconstruction_num_random_samples)
            else:
                reconstruction_steps = torch.randint(0, loc_seq_len, (
                    batch_size, self.config.input_num_random_samples, self.config.reconstruction_num_random_samples)).to(
                    self.device)
        else:
            if self.config.input_num_random_samples != -1 and predict_all is False:
                reconstruction_steps = (
                    torch.arange(0, loc_seq_len).unsqueeze(0).unsqueeze(0).expand(batch_size,
                                                                                  self.config.input_num_random_samples,
                                                                                  -1).to(self.device))
            else:
                reconstruction_steps = torch.arange(0, loc_seq_len).unsqueeze(0).unsqueeze(0).expand(batch_size,
                                                                                                     rep_seq_len,
                                                                                                     -1).to(self.device)

        batch_indices = torch.arange(batch_size).unsqueeze(-1)
        expanded_output = representations[batch_indices, input_steps].view(batch_size, input_steps.size(1), -1)
        expanded_output = expanded_output.unsqueeze(2).expand(-1, -1, reconstruction_steps.size(2), -1)

        expanded_locations = locations.unsqueeze(1).expand(-1, rep_seq_len, -1, -1, -1)
        expanded_locations = expanded_locations[
            batch_indices.unsqueeze(-1), input_steps.unsqueeze(-1), reconstruction_steps]

        combined_features = self.latent_location_encoder(expanded_output.reshape(-1, expanded_output.size(-1)),
                                                         expanded_locations.reshape(-1, expanded_locations.size(-2),
                                                                                    expanded_locations.size(-1)))
        combined_features = combined_features.view(batch_size, input_steps.size(1), reconstruction_steps.size(2), -1)

        predicted_forces = self.force_predictor(combined_features)
        return predicted_forces, input_steps, reconstruction_steps

    def compute_loss(self, inference_results: Dict[str, torch.Tensor], forces: torch.Tensor,
                     locations: torch.Tensor,
                     **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_indices = torch.arange(forces.size(0)).unsqueeze(-1).unsqueeze(-1)
        forces_target = forces.to(self.device)
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
