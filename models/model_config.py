from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str  # Name of the model
    params_path: str  # Path to the model parameters
