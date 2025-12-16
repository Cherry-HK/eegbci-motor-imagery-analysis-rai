"""
EEG Motor Imagery Analysis Package

This package provides tools for analyzing EEG data from the PhysioNet
EEG Motor Movement/Imagery Dataset.
"""

from . import preprocessing
from . import features
from . import classification
from . import visualization

__version__ = '1.0.0'

__all__ = [
    'preprocessing',
    'features',
    'classification',
    'visualization'
]
