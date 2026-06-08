"""
TSK consequent association module for e-AutoMFIS.

Implements Takagi-Sugeno-Kang (TSK) consequent computation:
- TSK-0: Constant consequent (weighted mean)
- TSK-1+: Linear/polynomial consequent (weighted least squares with ridge)

Also includes classical association metrics:
- Mean Compatibility (CM)
- Fuzzy Confidence Degree (GCF)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn


@dataclass
class TSKConsequent:
    """
    TSK consequent of configurable order.
    
    - Order 0: f(x) = c (constant)
    - Order 1: f(x) = c0 + c1*x1 + c2*x2 + ... (linear)
    - Order n: polynomial (not typically used beyond 1)
    
    Attributes:
        order: TSK order (0, 1, ...)
        coefficients: Coefficient tensor [1] for order 0, [D+1] for order 1
        regularization: Ridge regularization strength
        feature_indices: Which features this consequent uses (for TSK-1+)
    """
    order: int
    coefficients: torch.Tensor
    regularization: float = 1e-3
    feature_indices: Optional[List[int]] = None
    
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate consequent function.
        
        Args:
            X: Input features [N, D] (only used for order > 0)
            
        Returns:
            Output values [N]
        """
        if self.order == 0:
            # Constant
            return self.coefficients.expand(X.shape[0])
        else:
            # Linear: c0 + c1*x1 + c2*x2 + ...
            if self.feature_indices is not None:
                X = X[:, self.feature_indices]
            
            # Add bias column
            X_aug = torch.cat([torch.ones(X.shape[0], 1, device=X.device), X], dim=1)
            return X_aug @ self.coefficients


def fit_tsk_consequent(
    X: torch.Tensor,
    y: torch.Tensor,
    weights: torch.Tensor,
    order: int = 0,
    regularization: float = 1e-3,
    feature_indices: Optional[List[int]] = None,
) -> TSKConsequent:
    """
    Fit a TSK consequent using weighted least squares.
    
    Args:
        X: Input features [N, D]
        y: Target values [N] or [N, 1]
        weights: Sample weights (firing strengths) [N]
        order: TSK order (0 or 1)
        regularization: Ridge regularization for order > 0
        feature_indices: Which features to use for TSK-1+
        
    Returns:
        Fitted TSKConsequent
    """
    y = y.squeeze()
    weights = weights.squeeze()
    
    # Filter to samples with non-zero weight
    mask = weights > 1e-8
    if mask.sum() == 0:
        # No active samples, return zero consequent
        if order == 0:
            coef = torch.tensor([0.0])
        else:
            n_features = len(feature_indices) if feature_indices else X.shape[1]
            coef = torch.zeros(n_features + 1)
        return TSKConsequent(order=order, coefficients=coef, 
                             regularization=regularization, feature_indices=feature_indices)
    
    X_active = X[mask]
    y_active = y[mask]
    w_active = weights[mask]
    
    if order == 0:
        # TSK-0: Weighted mean
        # c = sum(w * y) / sum(w)
        c = (w_active * y_active).sum() / (w_active.sum() + 1e-8)
        coef = c.unsqueeze(0)
        
    else:
        # TSK-1+: Weighted ridge regression
        if feature_indices is not None:
            X_active = X_active[:, feature_indices]
        
        # Augment with bias
        X_aug = torch.cat([torch.ones(X_active.shape[0], 1, device=X_active.device), X_active], dim=1)
        
        # Weighted normal equations with ridge
        # (X'WX + λI)β = X'Wy
        W = torch.diag(w_active)
        XtWX = X_aug.T @ W @ X_aug
        XtWy = X_aug.T @ W @ y_active
        
        # Add ridge regularization (not to bias term)
        reg_matrix = regularization * torch.eye(XtWX.shape[0], device=XtWX.device)
        reg_matrix[0, 0] = 0  # Don't regularize bias
        
        try:
            coef = torch.linalg.solve(XtWX + reg_matrix, XtWy)
        except RuntimeError:
            # Singular matrix, fallback to pseudo-inverse
            coef = torch.linalg.lstsq(XtWX + reg_matrix, XtWy).solution
    
    return TSKConsequent(
        order=order,
        coefficients=coef,
        regularization=regularization,
        feature_indices=feature_indices,
    )


def fit_tsk_consequents_batch(
    X: torch.Tensor,
    y: torch.Tensor,
    firing_strengths: torch.Tensor,
    order: int = 0,
    regularization: float = 1e-3,
    feature_indices_per_rule: Optional[List[List[int]]] = None,
) -> List[TSKConsequent]:
    """
    Fit TSK consequents for multiple rules.
    
    Args:
        X: Input features [N, D]
        y: Target values [N] or [N, H] for multi-horizon
        firing_strengths: Firing strengths [N, R] (R rules)
        order: TSK order
        regularization: Ridge regularization
        feature_indices_per_rule: Optional per-rule feature selection
        
    Returns:
        List of R TSKConsequent objects
    """
    N, R = firing_strengths.shape
    y = y.squeeze() if y.dim() > 1 and y.shape[1] == 1 else y
    
    consequents = []
    for r in range(R):
        weights = firing_strengths[:, r]
        feat_idx = feature_indices_per_rule[r] if feature_indices_per_rule else None
        
        # Handle multi-horizon
        if y.dim() == 2:
            # Fit separate consequent for each horizon
            horizon_conseqs = []
            for h in range(y.shape[1]):
                conseq = fit_tsk_consequent(X, y[:, h], weights, order, regularization, feat_idx)
                horizon_conseqs.append(conseq)
            consequents.append(horizon_conseqs)
        else:
            conseq = fit_tsk_consequent(X, y, weights, order, regularization, feat_idx)
            consequents.append(conseq)
    
    return consequents


# =============================================================================
# Classical Association Metrics (for comparison/interpretability)
# =============================================================================

def mean_compatibility(
    premise_activations: torch.Tensor,
    consequent_memberships: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Mean Compatibility (CM) between premises and consequent terms.
    
    CM = (1/N) * Σ μ_premise(x_i) * μ_consequent(y_i)
    
    Args:
        premise_activations: [N] or [N, R] premise firing strengths
        consequent_memberships: [N, T] memberships for each consequent term
        
    Returns:
        CM values [T] or [R, T]
    """
    if premise_activations.dim() == 1:
        premise_activations = premise_activations.unsqueeze(1)
    
    # CM for each premise-consequent pair: [R, T]
    # (N, R).T @ (N, T) / N = (R, N) @ (N, T) / N = (R, T)
    N = premise_activations.shape[0]
    cm = (premise_activations.T @ consequent_memberships) / N
    
    return cm


def fuzzy_confidence_degree(
    premise_activations: torch.Tensor,
    consequent_memberships: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Fuzzy Confidence Degree (GCF) using cosine similarity.
    
    GCF = (μ_P · μ_C) / (||μ_P|| × ||μ_C||)
    
    Args:
        premise_activations: [N] or [N, R]
        consequent_memberships: [N, T]
        
    Returns:
        GCF values [T] or [R, T]
    """
    if premise_activations.dim() == 1:
        premise_activations = premise_activations.unsqueeze(1)
    
    # Normalize
    p_norm = premise_activations / (premise_activations.norm(dim=0, keepdim=True) + 1e-8)
    c_norm = consequent_memberships / (consequent_memberships.norm(dim=0, keepdim=True) + 1e-8)
    
    # Cosine similarity: [R, T]
    gcf = p_norm.T @ c_norm
    
    return gcf


def select_best_consequent_term(
    premise_activations: torch.Tensor,
    consequent_memberships: torch.Tensor,
    method: str = "cm",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select the best consequent term for each premise.
    
    Args:
        premise_activations: [N, R] premise firing strengths
        consequent_memberships: [N, T] memberships for each term
        method: 'cm' or 'gcf'
        
    Returns:
        (best_term_indices, scores) - both [R]
    """
    if method == "cm":
        scores = mean_compatibility(premise_activations, consequent_memberships)
    else:
        scores = fuzzy_confidence_degree(premise_activations, consequent_memberships)
    
    best_idx = scores.argmax(dim=-1)
    best_scores = scores.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)
    
    return best_idx, best_scores


# =============================================================================
# Rule Representation
# =============================================================================

@dataclass
class FuzzyRule:
    """
    Complete fuzzy rule with premise and consequent.
    
    For TSK systems:
        IF (x1 IS A1) AND (x2 IS A2) THEN y = f(x)
    
    Attributes:
        premise: Premise object (antecedents)
        consequent: TSKConsequent or term index (for Mamdani)
        weight: Rule weight/importance
        quality_score: Quality metric (support, coverage, etc.)
    """
    premise: "Premise"  # Forward reference
    consequent: Union[TSKConsequent, int, List[TSKConsequent]]
    weight: float = 1.0
    quality_score: float = 0.0
    
    def __repr__(self) -> str:
        if isinstance(self.consequent, TSKConsequent):
            if self.consequent.order == 0:
                conseq_str = f"y = {self.consequent.coefficients.item():.4f}"
            else:
                conseq_str = f"y = f(x) [order={self.consequent.order}]"
        elif isinstance(self.consequent, list):
            conseq_str = f"[{len(self.consequent)} horizons]"
        else:
            conseq_str = f"T{self.consequent}"
        
        return f"Rule({self.premise} => {conseq_str}, w={self.weight:.3f})"


def create_rules_from_premises(
    premises: List["Premise"],
    X: torch.Tensor,
    y: torch.Tensor,
    firing_strengths: torch.Tensor,
    tsk_order: int = 0,
    regularization: float = 1e-3,
) -> List[FuzzyRule]:
    """
    Create complete fuzzy rules by fitting TSK consequents.
    
    Args:
        premises: List of Premise objects
        X: Input features [N, D]
        y: Target values [N] or [N, H]
        firing_strengths: [N, R] firing strengths
        tsk_order: TSK order
        regularization: Ridge regularization
        
    Returns:
        List of FuzzyRule objects
    """
    consequents = fit_tsk_consequents_batch(
        X, y, firing_strengths, 
        order=tsk_order, 
        regularization=regularization
    )
    
    rules = []
    for i, (premise, consequent) in enumerate(zip(premises, consequents)):
        rule = FuzzyRule(
            premise=premise,
            consequent=consequent,
            weight=1.0,
            quality_score=premise.support,
        )
        rules.append(rule)
    
    return rules
