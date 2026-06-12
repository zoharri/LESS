"""
Models subdirectory for representation learning.

This directory contains different types of representation learning models:
- force_reconstruction/: Models that learn to reconstruct forces from locations
- non_learnable/: Models that don't require training (e.g., force maps)
"""

from .force_reconstruction import *
from .non_learnable import *
