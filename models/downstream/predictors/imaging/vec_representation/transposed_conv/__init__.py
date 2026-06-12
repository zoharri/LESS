"""
Transposed Convolution Predictor

This module contains a model that uses transposed convolution for image prediction.
"""

from .transposed_conv_image_pred import TransposedConvImagePred
from .transposed_conv_image_pred_3d import TransposedConvImagePred3D

__all__ = [
    'TransposedConvImagePred',
    'TransposedConvImagePred3D'
]
