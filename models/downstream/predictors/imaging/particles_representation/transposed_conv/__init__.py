"""
Transposed Convolution Predictor

This module contains a model that uses transposed convolution for image prediction.
"""

from .transposed_conv_par_rep_pred import TransposedConvParRepPred
from .transposed_conv_par_rep_pred_3d import TransposedConvParRepPred3D

__all__ = [
    'TransposedConvParRepPred',
    'TransposedConvParRepPred3D'
]
