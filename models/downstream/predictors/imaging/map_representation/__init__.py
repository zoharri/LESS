"""
Map Representation Predictors

This module contains predictors that work with 2D map representations (e.g., force maps)
and predict images using various architectures.

Available models:
- MapRepresentationPred: Abstract base class
- UNetMapRepPred: U-Net based predictor with skip connections
- CNNMapRepPred: CNN based predictor with convolutional layers
- AutoCNNMapRepPred: Autoencoder CNN based predictor with latent compression
"""

from .autocnn import AutoCNNMapRepPred
from .cnn import CNNMapRepPred
from .unet import UNetMapRepPred
from .localcnn import LocalCNNMapRepPred

__all__ = [
    'UNetMapRepPred',
    'CNNMapRepPred',
    'AutoCNNMapRepPred',
    'LocalCNNMapRepPred'
]
