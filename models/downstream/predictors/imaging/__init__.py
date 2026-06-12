"""
Imaging Predictors

This module contains models that predict images from representations using various architectures.

Available modules:
- image_pred: Abstract base class for image prediction models
- vec_representation: Vector representation based predictors (ADM, Flow Matching, Transposed Conv)
- map_representation: 2D map representation based predictors (UNet, CNN, AutoCNN, LocalCNNRepPred)
- particles_representation: Particle representation based predictors (Transposed Conv ParRep, Transposed Conv ParRep 3D)
"""

from .image_pred import ImagePred
from .particles_representation import TransposedConvParRepPred, TransposedConvParRepPred3D
from .map_representation import UNetMapRepPred, CNNMapRepPred, AutoCNNMapRepPred, LocalCNNMapRepPred
from .vec_representation import FlowMatchingImagePred, TransposedConvImagePred, TransposedConvImagePred3D
from .image_pred_3d import ImagePred3D

__all__ = [
    'ImagePred',
    'ImagePred3D',
    'FlowMatchingImagePred',
    'TransposedConvImagePred',
    'TransposedConvImagePred3D',
    'UNetMapRepPred',
    'CNNMapRepPred',
    'LocalCNNMapRepPred',
    'AutoCNNMapRepPred',
    'TransposedConvParRepPred',
    'TransposedConvParRepPred3D'
]
