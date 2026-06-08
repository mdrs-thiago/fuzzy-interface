"""
Configuration module for e-AutoMFIS.

Contains the main configuration dataclass with all hyperparameters
and serialization utilities for reproducibility.
"""

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional, List
import json
import yaml
from pathlib import Path


@dataclass
class EAutoMFISConfig:
    """
    Configuration for the e-AutoMFIS model.
    
    Attributes:
        # Data parameters
        max_lag: Maximum lag for autoregressive features
        forecast_horizon: Number of steps ahead to forecast
        target_indices: Indices of target variables (None = all)
        
        # Partitioning
        num_terms: Number of fuzzy terms per variable
        partition_method: Method for creating fuzzy partitions
        
        # Mining (premise formulation)
        min_support: Minimum fuzzy support for premises
        max_antecedents: Maximum number of antecedents per rule
        max_candidates_per_level: Budget cap per mining level
        max_total_candidates: Total budget cap for mining
        activation_method: How to compute premise activation
        
        # TSK Consequent
        tsk_order: Order of TSK consequent (0=constant, 1=linear, etc.)
        tsk_regularization: Ridge regularization for TSK-1+
        
        # Filtering
        similarity_threshold: Threshold for redundancy detection
        alpha_complexity: Penalty weight for rule complexity
        top_k_rules: Maximum rules per member after filtering
        
        # Ensemble
        num_members: Number of ensemble members
        feature_subset_size: Features per member (excluding core)
        core_features: Indices of features always included
        temporal_folds: Number of temporal folds for subsampling
        min_diversity: Minimum Jaccard distance between members
        ensemble_weight_tau: Temperature for ensemble weighting
        
        # Training
        n_jobs: Number of parallel jobs (-1 = all CPUs)
        device: Device for computation ('auto', 'cpu', 'cuda')
        batch_size: Batch size for GPU operations
        
        # Reproducibility
        random_seed: Random seed for reproducibility
        
        # Logging
        verbose: Verbosity level (0=silent, 1=progress, 2=debug)
        log_timing: Whether to log timing information
    """
    
    # Data parameters
    max_lag: int = 5
    forecast_horizon: int = 1
    target_indices: Optional[List[int]] = None
    
    # Partitioning
    num_terms: int = 5
    partition_method: Literal["uniform", "percentile", "clustering"] = "percentile"
    
    # Mining
    min_support: float = 0.05
    max_antecedents: int = 4
    max_candidates_per_level: int = 1000
    max_total_candidates: int = 10000
    activation_method: Literal["cardinality", "cardinality_nonnull"] = "cardinality_nonnull"
    min_frequency: int = 5  # Minimum non-null activations for cardinality_nonnull
    
    # TSK Consequent
    tsk_order: int = 0
    tsk_regularization: float = 1e-3
    
    # Filtering
    similarity_threshold: float = 0.8
    alpha_complexity: float = 0.1
    top_k_rules: int = 50
    gamma_complexity: float = 0.01  # For validation-based selection
    
    # Ensemble
    num_members: int = 10
    feature_subset_size: int = 5
    core_features: Optional[List[int]] = None
    temporal_folds: int = 5
    min_diversity: float = 0.3
    ensemble_weight_tau: float = 1.0
    
    # Training
    n_jobs: int = -1
    device: Literal["auto", "cpu", "cuda"] = "auto"
    batch_size: int = 1024
    
    # Reproducibility
    random_seed: int = 42
    
    # Logging
    verbose: int = 1
    log_timing: bool = True
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)
    
    def to_json(self, path: Optional[Path] = None) -> str:
        """Serialize config to JSON string, optionally saving to file."""
        json_str = json.dumps(self.to_dict(), indent=2)
        if path:
            Path(path).write_text(json_str)
        return json_str
    
    def to_yaml(self, path: Optional[Path] = None) -> str:
        """Serialize config to YAML string, optionally saving to file."""
        yaml_str = yaml.dump(self.to_dict(), default_flow_style=False)
        if path:
            Path(path).write_text(yaml_str)
        return yaml_str
    
    @classmethod
    def from_dict(cls, d: dict) -> "EAutoMFISConfig":
        """Create config from dictionary."""
        return cls(**d)
    
    @classmethod
    def from_json(cls, path_or_str: str) -> "EAutoMFISConfig":
        """Load config from JSON file or string."""
        path = Path(path_or_str)
        if path.exists():
            data = json.loads(path.read_text())
        else:
            data = json.loads(path_or_str)
        return cls.from_dict(data)
    
    @classmethod
    def from_yaml(cls, path_or_str: str) -> "EAutoMFISConfig":
        """Load config from YAML file or string."""
        path = Path(path_or_str)
        if path.exists():
            data = yaml.safe_load(path.read_text())
        else:
            data = yaml.safe_load(path_or_str)
        return cls.from_dict(data)
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.max_lag >= 1, "max_lag must be >= 1"
        assert self.forecast_horizon >= 1, "forecast_horizon must be >= 1"
        assert self.num_terms >= 2, "num_terms must be >= 2"
        assert 0 < self.min_support <= 1, "min_support must be in (0, 1]"
        assert self.max_antecedents >= 1, "max_antecedents must be >= 1"
        assert self.tsk_order >= 0, "tsk_order must be >= 0"
        assert self.num_members >= 1, "num_members must be >= 1"
        assert 0 < self.similarity_threshold <= 1, "similarity_threshold must be in (0, 1]"
    
    def __post_init__(self):
        """Validate after initialization."""
        self.validate()
