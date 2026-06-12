"""
Downstream Models

This module contains all downstream models that take representations as input
and predict various outputs.

The models are organized in predictors/ subdirectories by task type:
- imaging: Models that predict images
- quantity: Models that predict scalar/vector quantities

Available predictors:
- FlowMatchingImagePred: Flow Matching for image prediction
- TransposedConvImagePred: Transposed convolution for image prediction
- UNetMapRepPred: U-Net for 2D map representation to image prediction
- CNNMapRepPred: CNN for 2D map representation to image prediction
- AutoCNNMapRepPred: AutoCNN for 2D map representation to image prediction
- MLPQuantityPred: MLP for quantity prediction
- LumpAreaPred: Predictor for lump area
- LumpCenterPred: Predictor for lump center coordinates
"""

# Import all downstream models from predictors
from .predictors.imaging import (
    ImagePred,
    FlowMatchingImagePred,
    TransposedConvImagePred,
    TransposedConvImagePred3D,
    UNetMapRepPred,
    CNNMapRepPred,
    AutoCNNMapRepPred,
    LocalCNNMapRepPred,
    TransposedConvParRepPred,
    TransposedConvParRepPred3D,
)
from .predictors.quantity_classification.mlp_phantom_index import (
    MLPPhantomIndexClassifier
)
from .predictors.quantity_regression.mlp_lump_area import (
    MLPLumpAreaReg
)
from .predictors.quantity_regression.mlp_lump_center import (
    MLPLumpCenterReg
)

__all__ = [
    'ImagePred',
    'FlowMatchingImagePred',
    'TransposedConvImagePred',
    'TransposedConvImagePred3D',
    'UNetMapRepPred',
    'CNNMapRepPred',
    'LocalCNNMapRepPred',
    'AutoCNNMapRepPred',
    'TransposedConvParRepPred',
    'TransposedConvParRepPred3D',
    'MLPLumpAreaReg',
    'MLPLumpCenterReg',
    'MLPPhantomIndexClassifier'
]
