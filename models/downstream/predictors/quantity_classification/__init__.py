"""
Quantity Classification Predictors

This module contains models that perform classification tasks on quantities 
derived from representations.
"""

from .mlp_phantom_index import MLPPhantomIndexClassifier

__all__ = [
    'MLPPhantomIndexClassifier'
]
