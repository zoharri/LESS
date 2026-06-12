"""
Representation Learning Package

This package provides models and utilities for learning representations from tactile data.
It includes:
- Learnable models for force reconstruction (vector, map, and particle-based representations)
- Non-learnable models (e.g., force maps)
- Utilities for positional encoding and configuration management

Structure:
- models/: Implementation of force reconstruction and non-learnable models
- positional_encoding/: Positional encoding utilities
- configs.py: Configuration classes for representation learning models
- representation_learning_model.py: Abstract base class for representation learning models
"""

from .configs import RepresentationModelConfig
from .models import ReconstructionModel, LocalReconstructionModel, ParticlesReconstructionModel, ForceMapModel, ParticlesAggregation
from .representation_learning_model import RepresentationLearningModel

__all__ = [
    'RepresentationLearningModel',
    'RepresentationModelConfig',
    'ReconstructionModel',
    'LocalReconstructionModel',
    'ParticlesReconstructionModel',
    'ForceMapModel',
    'ParticlesAggregation'
]

