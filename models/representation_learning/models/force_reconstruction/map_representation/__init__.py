"""
Map Representation Models

This directory contains models that work with map-based representations
for force reconstruction tasks. These models typically encode force and
location information into map/grid representations that can be processed
by neural networks.
"""

from .reconstruction_model import LocalReconstructionModel

__all__ = [
    'LocalReconstructionModel',
]
