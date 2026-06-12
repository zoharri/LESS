"""
Vector-based reconstruction model for representation learning.

This module contains the ReconstructionModel, which is a configurable neural network
model for reconstructing forces from locations. It supports multiple architectures
including GRU and Transformer.

The model uses positional encoding to encode spatial information and can be configured
for different types of force reconstruction tasks.
"""

import math
from typing import Tuple, Dict

import torch
import torch.nn.functional as F
from torch import nn

from models.representation_learning.representation_learning_model import RepresentationLearningModel
from models.representation_learning.models.force_reconstruction.vector_location_encoder import VectorLocationEncoder

class CustomMultiheadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        """
        x: (batch, seq_len, embed_dim)
        attn_mask: additive mask of shape (batch, seq_len, seq_len) with -inf for masked positions
        """
        B, N, _ = x.size()

        # Project Q, K, V
        Q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        K = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, N, N)

        if attn_mask is not None:
            # Expand for heads: (B, 1, N, N)
            attn_scores = attn_scores + attn_mask.unsqueeze(1)

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        out = torch.matmul(attn_probs, V)  # (B, H, N, D)
        out = out.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        out = self.out_proj(out)
        return out


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = CustomMultiheadSelfAttention(embed_dim, num_heads, dropout)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, attn_mask=None):
        attn_output = self.self_attn(src, attn_mask=attn_mask)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)

        ff = self.linear2(F.relu(self.linear1(src)))
        src = src + self.dropout(ff)
        src = self.norm2(src)
        return src


class CustomTransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(embed_dim, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, src, attn_mask=None):
        for layer in self.layers:
            src = layer(src, attn_mask=attn_mask)
        return src


class LocalReconstructionModel(RepresentationLearningModel):
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
        super(LocalReconstructionModel, self).__init__(config_path)

        assert self.config.arch in ["gru", "transformer"], "Unsupported architecture specified."

        self.force_location_encoder = VectorLocationEncoder(self.config.locations_size, self.config.pe_max_freq_log2,
                                                            self.config.gru_input_embed_dim,
                                                            self.config.force_size)
        self.latent_location_encoder = VectorLocationEncoder(self.config.locations_size, self.config.pe_max_freq_log2,
                                                             self.config.decoder_input_embed_dim,
                                                             self.config.representation_size)

        self.input_encoder_layer = nn.Linear(self.force_location_encoder.output_size(),
                                             self.config.representation_size)
        self.encoder = CustomTransformerEncoder(num_layers=self.config.transformer_num_layers,
                                                embed_dim=self.config.representation_size,
                                                num_heads=self.config.transformer_nhead,
                                                dim_feedforward=self.config.transformer_ff_dim,
                                                dropout=self.config.dropout)

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
            self.get_masked_input_output(
                forces, locations))
        representations, h_n = self.encode(input_locations, input_forces, h_0=h_0,
                                           padding_mask=padding_mask)
        predicted_forces, reconstruction_steps = self.decode(representations, input_locations, output_locations,
                                                             predict_all=predict_all,
                                                             padding_mask=padding_mask)

        batch_size = mask_output_indices.size(0)
        input_steps = mask_input_indices
        reconstruction_steps = mask_output_indices[
            torch.arange(batch_size).unsqueeze(-1), reconstruction_steps]

        input_locations_mean = input_locations.mean(dim=2)[:, :, :2]
        x_pixels_locaitons = torch.linspace(input_locations_mean[:, :, 0].min(), input_locations_mean[:, :, 0].max(),
                                            steps=128)
        y_pixels_locations = torch.linspace(input_locations_mean[:, :, 1].min(), input_locations_mean[:, :, 1].max(),
                                            steps=128)
        xv, yv = torch.meshgrid(x_pixels_locaitons, y_pixels_locations, indexing='xy')
        pixel_locations = torch.stack([xv, yv], dim=-1).to(self.device)
        pixel_locations = pixel_locations.unsqueeze(0).expand(batch_size, -1, -1, -1)
        pixel_locations = pixel_locations.reshape(batch_size, -1, 2)
        local_mask = self.get_distance_attn_mask(input_locations_mean, pixel_locations,
                                                 max_distance=self.config.transformer_max_distance_for_pixel_rep)
        local_mask = F.softmax(local_mask, dim=1)
        # replace nan with 0
        local_mask = torch.nan_to_num(local_mask, nan=0.0)
        aggregated_representations = torch.einsum("bif,bin->bnf", representations, local_mask)
        aggregated_representations = aggregated_representations.view(batch_size, 128, 128, -1)
        aggregated_representations = aggregated_representations.permute(0, 3, 1, 2)
        return aggregated_representations, {"predicted_forces": predicted_forces, "h_n": h_n,
                                            "input_steps": input_steps,
                                            "reconstruction_steps": reconstruction_steps,
                                            "all_outputs": representations.clone()}

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

    def encode(self, locations: torch.Tensor, forces: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        h_0 = kwargs.get("h_0", None)
        padding_mask = kwargs.get("padding_mask", None)  # (B, L) True = valid
        batch_size, seq_len, num_sensors, dim_location = locations.shape
        combined = self.force_location_encoder(forces.reshape(batch_size * seq_len, -1),
                                               locations.reshape(batch_size * seq_len, num_sensors, dim_location))
        combined = combined.reshape(batch_size, seq_len, -1)

        combined = self.input_encoder_layer(combined)
        attn_mask = self.get_distance_attn_mask(locations[:, :, :, :2].mean(dim=2),
                                                locations[:, :, :, :2].mean(dim=2),
                                                causal=self.config.transformer_use_causal_mask,
                                                max_distance=self.config.transformer_max_distance_for_local_attention)
        # Mask padded positions so they don't attend to or from valid positions
        if padding_mask is not None:
            # padding_mask: (B, L) True=valid → padded positions get -inf in attn_mask
            pad_attn = (~padding_mask).float().to(self.device) * float('-inf')  # (B, L)
            pad_attn = pad_attn.unsqueeze(1).expand(-1, seq_len, -1)            # (B, L, L)
            attn_mask = attn_mask + pad_attn
        output = self.encoder(combined, attn_mask=attn_mask)

        return output, output

    def get_distance_attn_mask(self, input_locations: torch.Tensor, output_locations: torch.Tensor,
                               max_distance: float = 2, causal: bool = False) -> torch.Tensor:
        """
        Compute additive attention mask based on distances.
        Returns -inf where attention should be blocked.

        Args:
            input_locations:  [B, N1, 2]
            output_locations: [B, N2, 2]
            causal: whether to apply a causal mask
        Returns:
            attn_mask: [B, N1, N2] additive mask
        """
        attn_mask = torch.zeros(input_locations.size(0), input_locations.size(1),
                                output_locations.size(1), device=input_locations.device)
        if max_distance != -1:
            diff = input_locations.unsqueeze(2) - output_locations.unsqueeze(1)  # [B, N1, N2, 2]
            distances = torch.norm(diff, dim=-1)  # [B, N1, N2]
            distances[distances > max_distance] = torch.inf
            attn_mask = -distances
            attn_mask = attn_mask.to(input_locations.device)

        if causal:
            # Create causal mask: block future positions
            N1, N2 = input_locations.shape[1], output_locations.shape[1]
            causal_mask = torch.triu(torch.ones(N1, N2, device=input_locations.device), diagonal=1).bool()
            attn_mask = attn_mask.masked_fill(causal_mask.unsqueeze(0), float('-inf'))

        return attn_mask

    def decode(self, representations: torch.Tensor, input_locations: torch.Tensor, pred_locations: torch.Tensor,
               predict_all: bool = False, padding_mask: torch.Tensor = None):
        input_locations_mean = input_locations.mean(dim=2)
        pred_locations_mean = pred_locations.mean(dim=2)
        batch_size, loc_seq_len, _ = pred_locations_mean.shape
        _, rep_seq_len, _ = representations.shape
        local_mask = self.get_distance_attn_mask(input_locations_mean, pred_locations_mean,
                                                 max_distance=self.config.transformer_max_distance_for_decode)
        aggregated_representations = torch.einsum("bif,bin->bnf", representations, F.softmax(local_mask, dim=1))
        if self.config.reconstruction_num_random_samples != -1 and predict_all is False:
            if padding_mask is not None:
                sample_weights = padding_mask.float().to(self.device)
                reconstruction_steps = torch.multinomial(sample_weights,
                                                          self.config.reconstruction_num_random_samples,
                                                          replacement=True).to(self.device)
            else:
                reconstruction_steps = torch.randint(0, loc_seq_len, (
                    batch_size, self.config.reconstruction_num_random_samples)).to(self.device)
        else:
            reconstruction_steps = torch.arange(0, loc_seq_len).unsqueeze(0).expand(batch_size, rep_seq_len).to(
                self.device)

        batch_indices = torch.arange(batch_size).unsqueeze(-1)

        combined_features = self.latent_location_encoder(
            aggregated_representations[batch_indices, reconstruction_steps].reshape(-1, aggregated_representations.size(
                -1)),
            pred_locations[batch_indices, reconstruction_steps].reshape(-1, pred_locations.size(-2),
                                                                        pred_locations.size(-1)))
        combined_features = combined_features.view(batch_size, 1, reconstruction_steps.size(1), -1)
        predicted_forces = self.force_predictor(combined_features)
        return predicted_forces, reconstruction_steps

    def compute_loss(self, inference_results: Dict[str, torch.Tensor], forces: torch.Tensor,
                     locations: torch.Tensor,
                     **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_indices = torch.arange(forces.size(0)).unsqueeze(-1)
        forces_target = forces.to(self.device)
        forces_target = forces_target.view(forces_target.size(0), forces_target.size(1), -1)
        forces_target = forces_target[batch_indices, inference_results["reconstruction_steps"]].unsqueeze(1)
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
