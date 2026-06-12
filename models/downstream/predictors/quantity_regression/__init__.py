"""
Quantity Predictors

This module contains models that predict various continuous quantities from representations.

Available modules:
- lump_area: Lump area prediction models
- lump_center: Lump center prediction models
"""

from .mlp_lump_area import MLPLumpAreaReg
from .mlp_lump_center import MLPLumpCenterReg

__all__ = [
    'MLPLumpAreaReg',
    'MLPLumpCenterReg'
]
