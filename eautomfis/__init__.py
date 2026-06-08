"""
e-AutoMFIS: Ensemble Automatic TSK Fuzzy Inference System for Time Series Forecasting.

This package implements the e-AutoMFIS model for multivariate time series forecasting
using Takagi-Sugeno-Kang (TSK) fuzzy inference systems with ensemble learning.

Main components:
- EAutoMFIS: Main forecaster class
- EAutoMFISConfig: Configuration dataclass
- Partition methods: uniform, percentile, clustering
- Mining: Apriori-style premise formulation
- TSK inference: Weighted average with any-order consequents
"""

from .config import EAutoMFISConfig
from .forecaster import EAutoMFIS

__version__ = "0.1.0"
__all__ = ["EAutoMFIS", "EAutoMFISConfig"]
