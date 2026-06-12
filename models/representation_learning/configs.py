"""
Configuration classes for representation learning models.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class BaseRepModelConfig:
    l2_sp_regularization_weight: float = 0.0


@dataclass
class ReconstructionRepModelConfig(BaseRepModelConfig):
    force_size: int = 90
    locations_size: int = 6
    representation_size: int = 1024
    gru_tbptt_step_size: Optional[int] = None
    input_num_random_samples: int = 128
    reconstruction_num_random_samples: int = 128
    pe_max_freq_log2: int = 4
    force_predictor_hidden: int = 2048
    decoder_input_embed_dim: int = 1020
    gru_input_embed_dim: int = 252
    consistency_reg_weight: float = 0.0
    mask_percentage: float = 0.2
    trajectory_level_mask: bool = True
    dropout: float = 0.0
    trajectory_length: int = -1
    arch: str = "gru"
    transformer_num_layers: int = 1
    transformer_nhead: int = 1
    transformer_ff_dim: int = 128
    transformer_use_causal_mask: bool = False
    use_local_attention_mask: bool = False
    transformer_max_distance_for_local_attention: float = 1.25
    transformer_max_distance_for_decode: float = 10.0
    transformer_max_distance_for_pixel_rep: float = 0.3


@dataclass
class LocalReconstructionRepModelConfig(ReconstructionRepModelConfig):
    pass


@dataclass
class ParticlesReconstructionRepModelConfig(ReconstructionRepModelConfig):
    particles_min_x: float = -3.0
    particles_min_y: float = -3.0
    particles_max_x: float = 3.0
    particles_max_y: float = 3.0
    particles_res_x: float = 0.35
    particles_res_y: float = 0.35
    max_particle_distance: float = 0.6


@dataclass
class ForceMapRepModelConfig(BaseRepModelConfig):
    trajectory_length: int = -1
    forcemap_use_last_force: Optional[int] = None
    bandwidth_scale: float = 0.2
    image_size: int = 128
    forcemap_use_norm: bool = False


@dataclass
class ParticlesAggregationRepModelConfig(BaseRepModelConfig):
    particles_min_x: float = -3.0
    particles_min_y: float = -3.0
    particles_max_x: float = 3.0
    particles_max_y: float = 3.0
    particles_res_x: float = 0.35
    particles_res_y: float = 0.35
    max_particle_distance: float = 0.6


@dataclass
class RepresentationModelConfig:
    name: str = "ReconstructionModel"
    l2_sp_regularization_weight: float = 0.0
    reconstruction_model: ReconstructionRepModelConfig = field(default_factory=ReconstructionRepModelConfig)
    local_reconstruction_model: LocalReconstructionRepModelConfig = field(default_factory=LocalReconstructionRepModelConfig)
    particles_reconstruction_model: ParticlesReconstructionRepModelConfig = field(default_factory=ParticlesReconstructionRepModelConfig)
    force_map_model: ForceMapRepModelConfig = field(default_factory=ForceMapRepModelConfig)
    particles_aggregation_model: ParticlesAggregationRepModelConfig = field(default_factory=ParticlesAggregationRepModelConfig)

    def _name_to_attr(self) -> Dict[str, str]:
        return {
            "ReconstructionModel": "reconstruction_model",
            "LocalReconstructionModel": "local_reconstruction_model",
            "ParticlesReconstructionModel": "particles_reconstruction_model",
            "ForceMapModel": "force_map_model",
            "ParticlesAggregation": "particles_aggregation_model",
        }

    def get_active_config(self) -> BaseRepModelConfig:
        mapping = self._name_to_attr()
        if self.name not in mapping:
            raise ValueError(
                f"Unknown representation model name: {self.name}. "
                f"Supported: {list(mapping.keys())}"
            )
        active_cfg = deepcopy(getattr(self, mapping[self.name]))
        active_cfg.l2_sp_regularization_weight = self.l2_sp_regularization_weight
        return active_cfg
