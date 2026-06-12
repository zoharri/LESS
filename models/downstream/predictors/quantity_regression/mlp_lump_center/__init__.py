"""
Lump Center Predictors

This module contains models that predict lump center coordinates from representations.
"""

from .mlp_lump_center_reg import MLPLumpCenterReg

__all__ = [
    'MLPLumpCenterReg'
]
