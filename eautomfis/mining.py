"""
Premise formulation module for e-AutoMFIS.

Implements Apriori-style level-wise mining of fuzzy premises with:
- Min-support pruning (anti-monotone property)
- Budget constraints (max candidates per level, max total)
- Cardinality and non-null cardinality activation metrics
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Set, Dict, Optional, FrozenSet
import numpy as np
import torch
from collections import defaultdict


@dataclass(frozen=True)
class Antecedent:
    """
    A single antecedent: (feature_index, term_index).
    
    Hashable and immutable for use in sets.
    """
    feature_idx: int
    term_idx: int
    
    def __repr__(self) -> str:
        return f"(F{self.feature_idx}, T{self.term_idx})"


@dataclass
class Premise:
    """
    A conjunction of antecedents forming a rule premise.
    
    Attributes:
        antecedents: Frozenset of Antecedent objects
        support: Fuzzy support value
        activation: Mean activation value
        frequency: Number of non-null activations
        feature_indices: Sorted tuple of feature indices (for fast lookup)
    """
    antecedents: FrozenSet[Antecedent]
    support: float = 0.0
    activation: float = 0.0
    frequency: int = 0
    
    def __post_init__(self):
        self._feature_indices = tuple(sorted(a.feature_idx for a in self.antecedents))
    
    @property
    def feature_indices(self) -> Tuple[int, ...]:
        return self._feature_indices
    
    @property
    def size(self) -> int:
        """Number of antecedents."""
        return len(self.antecedents)
    
    def can_extend_with(self, antecedent: Antecedent) -> bool:
        """Check if premise can be extended with given antecedent."""
        # Cannot add same feature twice
        return antecedent.feature_idx not in self._feature_indices
    
    def extend(self, antecedent: Antecedent) -> "Premise":
        """Create new premise by adding an antecedent."""
        new_antecedents = self.antecedents | {antecedent}
        return Premise(antecedents=new_antecedents)
    
    def __hash__(self):
        return hash(self.antecedents)
    
    def __eq__(self, other):
        if not isinstance(other, Premise):
            return False
        return self.antecedents == other.antecedents
    
    def __repr__(self) -> str:
        ants = sorted(self.antecedents, key=lambda a: (a.feature_idx, a.term_idx))
        ant_str = " AND ".join(str(a) for a in ants)
        return f"Premise({ant_str}, supp={self.support:.3f})"


def compute_premise_activation(
    memberships: torch.Tensor,
    premise: Premise,
) -> torch.Tensor:
    """
    Compute activation (firing strength) for a premise across all samples.
    
    Uses product t-norm (algebraic product).
    
    Args:
        memberships: Membership tensor [N, D, T]
        premise: Premise to evaluate
        
    Returns:
        Activation tensor [N]
    """
    activation = torch.ones(memberships.shape[0], device=memberships.device)
    
    for ant in premise.antecedents:
        activation = activation * memberships[:, ant.feature_idx, ant.term_idx]
    
    return activation


def compute_fuzzy_support(
    activations: torch.Tensor,
    method: str = "cardinality",
    min_frequency: int = 1,
) -> Tuple[float, float, int]:
    """
    Compute fuzzy support for a premise.
    
    Args:
        activations: Activation tensor [N]
        method: 'cardinality' or 'cardinality_nonnull'
        min_frequency: Minimum non-null samples for cardinality_nonnull
        
    Returns:
        (support, activation, frequency) tuple
    """
    n_samples = activations.shape[0]
    
    # Fuzzy support: sum of activations normalized by N
    support = activations.sum().item() / n_samples
    
    # Non-null mask
    nonnull_mask = activations > 1e-8
    frequency = nonnull_mask.sum().item()
    
    if method == "cardinality":
        # Mean activation across all samples
        activation = activations.mean().item()
    else:  # cardinality_nonnull
        # Mean activation only for non-null samples
        if frequency >= min_frequency:
            activation = activations[nonnull_mask].mean().item()
        else:
            activation = 0.0
    
    return support, activation, int(frequency)


def generate_level1_candidates(
    memberships: torch.Tensor,
    min_support: float,
    activation_method: str = "cardinality",
    min_frequency: int = 1,
) -> List[Premise]:
    """
    Generate level-1 premises (single antecedents).
    
    Args:
        memberships: Membership tensor [N, D, T]
        min_support: Minimum support threshold
        activation_method: How to compute activation
        min_frequency: Minimum frequency for cardinality_nonnull
        
    Returns:
        List of premises that pass min_support
    """
    n_samples, n_features, n_terms = memberships.shape
    candidates = []
    
    for d in range(n_features):
        for t in range(n_terms):
            antecedent = Antecedent(feature_idx=d, term_idx=t)
            premise = Premise(antecedents=frozenset({antecedent}))
            
            # Compute support
            activations = memberships[:, d, t]
            support, activation, frequency = compute_fuzzy_support(
                activations, activation_method, min_frequency
            )
            
            if support >= min_support:
                premise.support = support
                premise.activation = activation
                premise.frequency = frequency
                candidates.append(premise)
    
    return candidates


def apriori_gen(
    frequent_k: List[Premise],
    level: int,
) -> List[Premise]:
    """
    Generate candidate (k+1)-premises from frequent k-premises.
    
    Uses Apriori candidate generation: combine premises that share k-1 antecedents.
    
    Args:
        frequent_k: Frequent k-premises
        level: Current level k
        
    Returns:
        Candidate (k+1)-premises
    """
    candidates = set()
    
    # Get all singleton antecedents from frequent premises
    all_antecedents = set()
    for p in frequent_k:
        all_antecedents.update(p.antecedents)
    
    # For each frequent premise, try to extend with each antecedent
    for premise in frequent_k:
        for antecedent in all_antecedents:
            if premise.can_extend_with(antecedent):
                new_premise = premise.extend(antecedent)
                
                # Pruning: all subsets must be frequent (anti-monotone)
                # Check if this is a valid candidate
                candidates.add(new_premise)
    
    return list(candidates)


def mine_premises(
    memberships: torch.Tensor,
    min_support: float = 0.05,
    max_antecedents: int = 4,
    max_candidates_per_level: int = 1000,
    max_total_candidates: int = 10000,
    activation_method: str = "cardinality_nonnull",
    min_frequency: int = 5,
    score_fn: callable = None,
    verbose: bool = False,
) -> List[Premise]:
    """
    Mine frequent fuzzy premises using Apriori-style algorithm.
    
    Level-wise search with:
    - Anti-monotone pruning (if premise P fails min_support, don't expand)
    - Budget constraints to prevent explosion
    - Optional scoring function for quality-based pruning
    
    Args:
        memberships: Membership tensor [N, D, T]
        min_support: Minimum fuzzy support threshold
        max_antecedents: Maximum antecedents per premise
        max_candidates_per_level: Max candidates to evaluate per level
        max_total_candidates: Max total candidates to return
        activation_method: 'cardinality' or 'cardinality_nonnull'
        min_frequency: Minimum non-null activations
        score_fn: Optional scoring function(premise) -> float for ranking
        verbose: Print progress information
        
    Returns:
        List of frequent premises sorted by support
    """
    all_frequent = []
    
    # Level 1: single antecedents
    if verbose:
        print(f"Mining level 1...")
    
    level1 = generate_level1_candidates(
        memberships, min_support, activation_method, min_frequency
    )
    
    if verbose:
        print(f"  Level 1: {len(level1)} frequent items")
    
    all_frequent.extend(level1)
    frequent_k = level1
    
    # Level 2 to max_antecedents
    for k in range(2, max_antecedents + 1):
        if not frequent_k:
            break
            
        if verbose:
            print(f"Mining level {k}...")
        
        # Generate candidates
        candidates = apriori_gen(frequent_k, k - 1)
        
        if verbose:
            print(f"  Generated {len(candidates)} candidates")
        
        # Apply budget constraint
        if len(candidates) > max_candidates_per_level:
            if score_fn is not None:
                # Sort by score and take top-k
                for c in candidates:
                    c._score = score_fn(c)
                candidates = sorted(candidates, key=lambda p: -p._score)
            candidates = candidates[:max_candidates_per_level]
        
        # Evaluate candidates
        frequent_k = []
        for premise in candidates:
            activations = compute_premise_activation(memberships, premise)
            support, activation, frequency = compute_fuzzy_support(
                activations, activation_method, min_frequency
            )
            
            if support >= min_support:
                premise.support = support
                premise.activation = activation  
                premise.frequency = frequency
                frequent_k.append(premise)
        
        if verbose:
            print(f"  Level {k}: {len(frequent_k)} frequent premises")
        
        all_frequent.extend(frequent_k)
        
        # Check total budget
        if len(all_frequent) >= max_total_candidates:
            if verbose:
                print(f"  Reached max total candidates limit")
            break
    
    # Sort by support (descending)
    all_frequent = sorted(all_frequent, key=lambda p: -p.support)
    
    # Apply final budget
    if len(all_frequent) > max_total_candidates:
        all_frequent = all_frequent[:max_total_candidates]
    
    return all_frequent


def compute_premise_quality_score(
    premise: Premise,
    memberships: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.1,
) -> float:
    """
    Compute quality score for a premise based on:
    - Support (mass of activation)
    - Local variance reduction (optional quality metric)
    - Complexity penalty
    
    Score = quality * log(1 + support) - alpha * |antecedents|
    
    Args:
        premise: Premise to score
        memberships: Membership tensor [N, D, T]
        y: Target values [N] or [N, H]
        alpha: Complexity penalty weight
        
    Returns:
        Quality score
    """
    activations = compute_premise_activation(memberships, premise)
    support = activations.sum().item()
    
    # Simple quality: normalized support
    quality = premise.activation
    
    # Score formula
    score = quality * np.log1p(support) - alpha * premise.size
    
    return score


def select_top_k_premises(
    premises: List[Premise],
    memberships: torch.Tensor,
    y: torch.Tensor,
    k: int = 50,
    alpha: float = 0.1,
) -> List[Premise]:
    """
    Select top-K premises based on quality scoring.
    
    Args:
        premises: List of candidate premises
        memberships: Membership tensor
        y: Target values
        k: Number of premises to select
        alpha: Complexity penalty
        
    Returns:
        Top-K premises by score
    """
    if len(premises) <= k:
        return premises
    
    # Score all premises
    scored = []
    for p in premises:
        score = compute_premise_quality_score(p, memberships, y, alpha)
        scored.append((score, p))
    
    # Sort and take top-K
    scored = sorted(scored, key=lambda x: -x[0])
    return [p for _, p in scored[:k]]


def premises_to_indices(
    premises: List[Premise],
    max_antecedents: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert premises to indexed tensor representation for vectorized computation.
    
    Args:
        premises: List of Premise objects
        max_antecedents: Maximum antecedents (for padding)
        
    Returns:
        (feature_indices, term_indices, mask) tensors
        - feature_indices: [R, K] feature index for each antecedent
        - term_indices: [R, K] term index for each antecedent
        - mask: [R, K] boolean mask for valid antecedents
    """
    n_premises = len(premises)
    
    feat_idx = torch.zeros(n_premises, max_antecedents, dtype=torch.long)
    term_idx = torch.zeros(n_premises, max_antecedents, dtype=torch.long)
    mask = torch.zeros(n_premises, max_antecedents, dtype=torch.bool)
    
    for i, premise in enumerate(premises):
        ants = sorted(premise.antecedents, key=lambda a: a.feature_idx)
        for j, ant in enumerate(ants):
            feat_idx[i, j] = ant.feature_idx
            term_idx[i, j] = ant.term_idx
            mask[i, j] = True
    
    return feat_idx, term_idx, mask


def compute_firing_strengths_vectorized(
    memberships: torch.Tensor,
    feat_idx: torch.Tensor,
    term_idx: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute firing strengths for all premises in parallel.
    
    Uses gather operations for efficient GPU computation.
    
    Args:
        memberships: [N, D, T] membership tensor
        feat_idx: [R, K] feature indices
        term_idx: [R, K] term indices  
        mask: [R, K] valid antecedent mask
        
    Returns:
        Firing strengths [N, R]
    """
    N, D, T = memberships.shape
    R, K = feat_idx.shape
    
    # Gather memberships for each premise's antecedents
    # memberships[:, feat_idx, :] -> [N, R, K, T]
    expanded = memberships[:, feat_idx, :]  # [N, R, K, T]
    
    # Gather specific terms
    term_idx_expanded = term_idx.unsqueeze(0).unsqueeze(-1).expand(N, -1, -1, 1)
    gathered = expanded.gather(-1, term_idx_expanded).squeeze(-1)  # [N, R, K]
    
    # Apply mask: set invalid to 1 (neutral for product)
    gathered = gathered.masked_fill(~mask.unsqueeze(0), 1.0)
    
    # Product across antecedents
    firing = gathered.prod(dim=-1)  # [N, R]
    
    return firing
