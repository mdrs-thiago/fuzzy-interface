"""
Main e-AutoMFIS forecaster module.

Provides the main EAutoMFIS class that orchestrates:
- Ensemble training
- Rule aggregation
- TSK inference
- Multi-step forecasting
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union, Tuple
import numpy as np
import torch

from .config import EAutoMFISConfig
from .utils import set_seeds, Timer, Logger, compute_metrics, ForecastMetrics
from .validation import create_lagged_features, temporal_train_val_test_split
from .partition import (
    create_partitions_for_dataset, 
    compute_membership_matrix, 
    FuzzyPartition
)
from .mining import (
    mine_premises, 
    premises_to_indices, 
    compute_firing_strengths_vectorized,
    Premise
)
from .association import create_rules_from_premises, FuzzyRule
from .filtering import filter_rules_internal, aggregate_rules_external
from .inference import TSKInferenceEngine, create_inference_engine
from .ensemble import (
    EnsembleTrainer,
    EnsembleMember,
    generate_member_configs,
    compute_ensemble_weights,
)


class EAutoMFIS:
    """
    Ensemble Automatic TSK Fuzzy Inference System for time series forecasting.
    
    Main features:
    - Automatic rule generation via Apriori-style mining
    - TSK consequents of any order (0=constant, 1=linear, etc.)
    - Ensemble learning with temporal and feature subsampling
    - GPU-accelerated computation
    - Interpretable rule-based predictions
    
    Example:
        >>> model = EAutoMFIS(EAutoMFISConfig(num_members=5, tsk_order=0))
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
        >>> print(model.explain(X_test[0]))
    """
    
    def __init__(
        self, 
        config: Optional[EAutoMFISConfig] = None,
        n_jobs: int = 1,
        device: str = "auto",
        verbose: bool = True,
    ):
        """
        Initialize the e-AutoMFIS model.
        
        Args:
            config: Model configuration. If None, uses default config.
            n_jobs: Number of parallel jobs for training (-1 = all CPUs)
            device: Computation device ('auto', 'cpu', 'cuda')
            verbose: Whether to print training progress
        """
        self.config = config or EAutoMFISConfig()
        self.n_jobs = n_jobs if n_jobs > 0 else 1
        self.verbose = verbose
        
        # Device selection
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.logger = Logger(verbose=1 if verbose else 0)
        
        # Will be populated during training
        self.is_fitted = False
        self.partitions_: Optional[List[FuzzyPartition]] = None
        self.members_: Optional[List[EnsembleMember]] = None
        self.member_weights_: Optional[np.ndarray] = None
        self.rules_: Optional[List[FuzzyRule]] = None
        self.inference_engine_: Optional[TSKInferenceEngine] = None
        
        # Training metadata
        self.n_features_: int = 0
        self.n_samples_: int = 0
        self.train_time_: float = 0.0
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "EAutoMFIS":
        """
        Train the e-AutoMFIS model.
        
        Pipeline:
        1. Set seeds for reproducibility
        2. Create fuzzy partitions for all features
        3. Train ensemble members in parallel
        4. Compute ensemble weights via validation
        5. Aggregate rules across members
        6. Create inference engine
        
        Args:
            X: Feature matrix [N, D]
            y: Target values [N] or [N, H] for multi-horizon
            X_val: Optional validation features (for ensemble weighting)
            y_val: Optional validation targets
            
        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        self.n_samples_, self.n_features_ = X.shape
        
        self.logger.info(f"Training e-AutoMFIS on {self.n_samples_} samples, {self.n_features_} features")
        self.logger.info(f"Config: {self.config.num_members} members, {self.config.num_terms} terms, TSK-{self.config.tsk_order}")
        
        # 1. Set seeds
        set_seeds(self.config.random_seed)
        
        with Timer("total_training", verbose=False) as total_timer:
            
            # 2. Create partitions for full feature space
            with Timer("partitioning", verbose=False):
                self.partitions_ = create_partitions_for_dataset(
                    X,
                    num_terms=self.config.num_terms,
                    method=self.config.partition_method,
                )
            self.logger.info(f"Created {len(self.partitions_)} partitions")
            
            # 3. Train ensemble members
            with Timer("ensemble_training", verbose=False):
                trainer = EnsembleTrainer(
                    config=self.config,
                    n_jobs=self.n_jobs,
                    device=self.device,
                    verbose=self.verbose,
                )
                self.members_ = trainer.train(X, y)
            
            total_rules = sum(m.num_rules for m in self.members_)
            self.logger.info(f"Trained {len(self.members_)} members with {total_rules} total rules")
            
            # 4. Compute ensemble weights
            if X_val is not None and y_val is not None:
                with Timer("ensemble_weighting", verbose=False):
                    self.members_ = compute_ensemble_weights(
                        self.members_,
                        X_val, y_val,
                        tau=self.config.ensemble_weight_tau,
                    )
                self.logger.info("Computed ensemble weights from validation")
            else:
                # Uniform weights
                self.member_weights_ = np.ones(len(self.members_)) / len(self.members_)
                for m in self.members_:
                    m.weight = 1.0 / len(self.members_)
            
            self.member_weights_ = np.array([m.weight for m in self.members_])
            
            # 5. Collect all rules (simplified aggregation for now)
            # Full aggregation would require remapping to full feature space
            self.rules_ = []
            for member in self.members_:
                self.rules_.extend(member.rules)
            
            self.logger.info(f"Collected {len(self.rules_)} total rules")
            
            # 6. Create inference engine
            # For now, we'll use the first member's structure
            # A proper implementation would aggregate properly
            if self.members_ and self.members_[0].rules:
                self.inference_engine_ = create_inference_engine(
                    self.members_[0].rules,
                    self.members_[0].partitions,
                    max_antecedents=self.config.max_antecedents,
                    device=self.device,
                )
        
        self.train_time_ = total_timer.elapsed
        self.is_fitted = True
        
        self.logger.info(f"Training complete in {self.train_time_:.2f}s")
        
        return self
    
    def predict(
        self, 
        X: np.ndarray, 
        horizon: int = 1,
        return_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate forecasts using ensemble of TSK systems.
        
        Args:
            X: Feature matrix [N, D]
            horizon: Forecast horizon (for multi-horizon models)
            return_std: Whether to return prediction std across members
            
        Returns:
            Predictions [N] or [N, H] for multi-horizon
            Optionally also returns std [N] or [N, H]
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = np.asarray(X, dtype=np.float32)
        
        # Ensemble prediction: weighted average across members
        member_preds = []
        
        for member in self.members_:
            if not member.rules:
                continue
            
            # Map to member's feature space
            X_member = X[:, member.config.feature_indices]
            
            # Predict
            engine = create_inference_engine(
                member.rules,
                member.partitions,
                max_antecedents=self.config.max_antecedents,
                device=self.device,
            )
            
            pred = engine.infer(X_member).detach().cpu().numpy()
            member_preds.append((member.weight, pred))
        
        if not member_preds:
            return np.zeros(X.shape[0])
        
        # Weighted average
        weights = np.array([w for w, _ in member_preds])
        preds = np.stack([p for _, p in member_preds], axis=0)  # [M, N]
        
        weights = weights / weights.sum()
        y_pred = np.average(preds, axis=0, weights=weights)
        
        if return_std:
            y_std = np.sqrt(np.average((preds - y_pred) ** 2, axis=0, weights=weights))
            return y_pred, y_std
        
        return y_pred
    
    def score(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
    ) -> ForecastMetrics:
        """
        Compute forecast metrics on test data.
        
        Args:
            X: Feature matrix
            y: True target values
            
        Returns:
            ForecastMetrics object
        """
        y_pred = self.predict(X)
        
        return compute_metrics(
            y_true=y,
            y_pred=y_pred,
            num_rules=len(self.rules_) if self.rules_ else 0,
            avg_antecedents=self._avg_antecedents(),
            train_time=self.train_time_,
        )
    
    def explain(
        self, 
        x: np.ndarray,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Explain prediction for a single sample.
        
        Args:
            x: Single input [D] or batch [N, D]
            top_k: Number of top rules to show per member
            
        Returns:
            Explanation dictionary
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        explanation = {
            "prediction": self.predict(x)[0],
            "member_contributions": [],
        }
        
        for member in self.members_:
            if not member.rules:
                continue
            
            x_member = x[:, member.config.feature_indices]
            
            engine = create_inference_engine(
                member.rules,
                member.partitions,
                max_antecedents=self.config.max_antecedents,
            )
            
            member_exp = engine.explain(x_member, top_k=top_k)
            member_exp["member_id"] = member.config.member_id
            member_exp["member_weight"] = member.weight
            
            explanation["member_contributions"].append(member_exp)
        
        return explanation
    
    def get_rules(self) -> List[str]:
        """
        Get human-readable rules from the trained model.
        
        Returns:
            List of rule strings
        """
        if not self.is_fitted or not self.rules_:
            return []
        
        return [str(rule) for rule in self.rules_]
    
    def _avg_antecedents(self) -> float:
        """Compute average number of antecedents per rule."""
        if not self.rules_:
            return 0.0
        return np.mean([r.premise.size for r in self.rules_])
    
    def get_config(self) -> dict:
        """Get model configuration as dictionary."""
        return self.config.to_dict()
    
    def summary(self) -> str:
        """Get model summary string."""
        if not self.is_fitted:
            return "EAutoMFIS (not fitted)"
        
        lines = [
            "=" * 50,
            "e-AutoMFIS Model Summary",
            "=" * 50,
            f"Status: Fitted",
            f"Samples: {self.n_samples_}, Features: {self.n_features_}",
            f"Members: {len(self.members_)}",
            f"Total Rules: {len(self.rules_)}",
            f"Avg Antecedents: {self._avg_antecedents():.2f}",
            f"TSK Order: {self.config.tsk_order}",
            f"Training Time: {self.train_time_:.2f}s",
            "=" * 50,
        ]
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        if self.is_fitted:
            return f"EAutoMFIS(fitted, members={len(self.members_)}, rules={len(self.rules_)})"
        return "EAutoMFIS(not fitted)"
