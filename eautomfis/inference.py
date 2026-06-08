"""
TSK inference module for e-AutoMFIS.

Implements the Takagi-Sugeno-Kang inference:
- Compute firing strengths for all rules
- Weighted average of consequent outputs
- Multi-horizon support
"""

from typing import List, Optional, Union
import torch
import numpy as np

from .mining import Premise, compute_premise_activation, premises_to_indices, compute_firing_strengths_vectorized
from .association import FuzzyRule, TSKConsequent
from .partition import FuzzyPartition, compute_membership_matrix


class TSKInferenceEngine:
    """
    TSK inference engine for fuzzy rule-based prediction.
    
    Computes:
        y_hat = Σ_r (w_r * f_r(x)) / Σ_r w_r
    
    where:
        w_r = firing_strength_r * rule_weight_r
        f_r(x) = TSK consequent function
    """
    
    def __init__(
        self,
        rules: List[FuzzyRule],
        partitions: List[FuzzyPartition],
        max_antecedents: int = 4,
        device: str = "cpu",
    ):
        """
        Initialize inference engine.
        
        Args:
            rules: List of fitted FuzzyRule objects
            partitions: List of FuzzyPartition objects (one per feature)
            max_antecedents: Maximum antecedents per rule (for padding)
            device: Computation device
        """
        self.rules = rules
        self.partitions = partitions
        self.max_antecedents = max_antecedents
        self.device = torch.device(device)
        
        # Pre-compute indexed representation
        premises = [r.premise for r in rules]
        self.feat_idx, self.term_idx, self.mask = premises_to_indices(premises, max_antecedents)
        self.feat_idx = self.feat_idx.to(self.device)
        self.term_idx = self.term_idx.to(self.device)
        self.mask = self.mask.to(self.device)
        
        # Rule weights
        self.rule_weights = torch.tensor([r.weight for r in rules], dtype=torch.float32, device=self.device)
        
        # Check if multi-horizon
        self.is_multi_horizon = (
            len(rules) > 0 and 
            isinstance(rules[0].consequent, list)
        )
        
        if self.is_multi_horizon:
            self.num_horizons = len(rules[0].consequent)
        else:
            self.num_horizons = 1
    
    def compute_memberships(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Compute membership degrees for input.
        
        Args:
            X: Input features [N, D]
            
        Returns:
            Memberships [N, D, T]
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        X = X.to(self.device)
        
        memberships = compute_membership_matrix(X.cpu().numpy(), self.partitions)
        return memberships.to(self.device)
    
    def compute_firing_strengths(self, memberships: torch.Tensor) -> torch.Tensor:
        """
        Compute firing strengths for all rules.
        
        Args:
            memberships: [N, D, T] membership tensor
            
        Returns:
            Firing strengths [N, R]
        """
        return compute_firing_strengths_vectorized(
            memberships, self.feat_idx, self.term_idx, self.mask
        )
    
    def infer(
        self,
        X: Union[np.ndarray, torch.Tensor],
        memberships: Optional[torch.Tensor] = None,
        horizon: int = 1,
    ) -> torch.Tensor:
        """
        Perform TSK inference.
        
        Args:
            X: Input features [N, D]
            memberships: Pre-computed memberships (optional)
            horizon: Forecast horizon (for multi-horizon models)
            
        Returns:
            Predictions [N] or [N, H]
        """
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X).float()
            X = X.to(self.device)
            
            # Compute memberships if not provided
            if memberships is None:
                memberships = self.compute_memberships(X)
            else:
                memberships = memberships.to(self.device)
            
            # Firing strengths
            firing = self.compute_firing_strengths(memberships)  # [N, R]
            
            # Apply rule weights
            weighted_firing = firing * self.rule_weights.unsqueeze(0)  # [N, R]
            
            # Normalize
            denominator = weighted_firing.sum(dim=1, keepdim=True) + 1e-8  # [N, 1]
            
            if self.is_multi_horizon:
                # Multi-horizon prediction
                H = min(horizon, self.num_horizons)
                predictions = torch.zeros(X.shape[0], H, device=self.device)
                
                for h in range(H):
                    # Compute consequent outputs for horizon h
                    conseq_outputs = torch.zeros(X.shape[0], len(self.rules), device=self.device)
                    
                    for r, rule in enumerate(self.rules):
                        conseq = rule.consequent[h]
                        conseq_outputs[:, r] = conseq(X)
                    
                    # Weighted average
                    predictions[:, h] = (weighted_firing * conseq_outputs).sum(dim=1) / denominator.squeeze()
                
                return predictions
            
            else:
                # Single horizon
                conseq_outputs = torch.zeros(X.shape[0], len(self.rules), device=self.device)
                
                for r, rule in enumerate(self.rules):
                    conseq = rule.consequent
                    conseq_outputs[:, r] = conseq(X)
                
                predictions = (weighted_firing * conseq_outputs).sum(dim=1) / denominator.squeeze()
                return predictions
    
    def explain(
        self,
        x: Union[np.ndarray, torch.Tensor],
        top_k: int = 5,
    ) -> dict:
        """
        Explain prediction for a single sample.
        
        Args:
            x: Single input [D] or [1, D]
            top_k: Number of top rules to show
            
        Returns:
            Explanation dictionary
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        
        # Compute memberships and firing
        memberships = self.compute_memberships(x)
        firing = self.compute_firing_strengths(memberships).squeeze()  # [R]
        
        # Sort by firing strength
        sorted_idx = firing.argsort(descending=True)
        
        explanation = {
            "prediction": self.infer(x).item(),
            "top_rules": [],
        }
        
        for i in range(min(top_k, len(self.rules))):
            idx = sorted_idx[i].item()
            rule = self.rules[idx]
            strength = firing[idx].item()
            
            if strength > 1e-6:
                rule_info = {
                    "rule_id": idx,
                    "premise": str(rule.premise),
                    "firing_strength": strength,
                    "weight": rule.weight,
                    "contribution": strength * rule.weight,
                }
                
                if isinstance(rule.consequent, TSKConsequent):
                    if rule.consequent.order == 0:
                        rule_info["consequent"] = rule.consequent.coefficients.item()
                    else:
                        rule_info["consequent"] = rule.consequent(x).item()
                
                explanation["top_rules"].append(rule_info)
        
        return explanation


def create_inference_engine(
    rules: List[FuzzyRule],
    partitions: List[FuzzyPartition],
    max_antecedents: int = 4,
    device: str = "cpu",
) -> TSKInferenceEngine:
    """
    Factory function to create an inference engine.
    
    Args:
        rules: List of FuzzyRule objects
        partitions: List of FuzzyPartition objects
        max_antecedents: Max antecedents per rule
        device: Computation device
        
    Returns:
        TSKInferenceEngine
    """
    return TSKInferenceEngine(rules, partitions, max_antecedents, device)


def predict_multistep_direct(
    engine: TSKInferenceEngine,
    X: Union[np.ndarray, torch.Tensor],
    horizon: int,
) -> torch.Tensor:
    """
    Multi-step direct prediction.
    
    Each horizon has its own consequent parameters.
    
    Args:
        engine: TSK inference engine
        X: Input features [N, D]
        horizon: Forecast horizon
        
    Returns:
        Predictions [N, H]
    """
    return engine.infer(X, horizon=horizon)


def predict_multistep_recursive(
    engine: TSKInferenceEngine,
    X: Union[np.ndarray, torch.Tensor],
    horizon: int,
    target_col: int = 0,
) -> torch.Tensor:
    """
    Multi-step recursive prediction.
    
    Uses previous predictions as input for future steps.
    
    Args:
        engine: TSK inference engine (single-horizon)
        X: Initial input features [N, D]
        horizon: Forecast horizon
        target_col: Column index for target variable
        
    Returns:
        Predictions [N, H]
    """
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    
    N, D = X.shape
    predictions = torch.zeros(N, horizon)
    
    current_X = X.clone()
    
    for h in range(horizon):
        # Predict one step
        y_h = engine.infer(current_X)
        predictions[:, h] = y_h
        
        # Update input for next step (shift and insert prediction)
        # This assumes a specific lag structure
        if h < horizon - 1:
            # Shift lagged values and insert new prediction
            # This is a simplified version - actual implementation depends on feature structure
            current_X = current_X.roll(-1, dims=1)
            current_X[:, target_col] = y_h
    
    return predictions
