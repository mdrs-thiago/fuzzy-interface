"""
Rule filtering module for e-AutoMFIS.

Implements:
- Similarity-based redundancy detection (Jaccard-like)
- Conflict detection (similar premises, different consequents)
- Evidence-based one-shot scoring
- Top-K selection with complexity penalty
"""

from dataclasses import dataclass
from typing import List, Tuple, Set, Optional, Dict
import numpy as np
import torch

from .mining import Premise, Antecedent, compute_premise_activation
from .association import FuzzyRule, TSKConsequent


def premise_jaccard_similarity(p1: Premise, p2: Premise) -> float:
    """
    Compute Jaccard similarity between two premises based on antecedents.
    
    J(P1, P2) = |A1 ∩ A2| / |A1 ∪ A2|
    
    Args:
        p1, p2: Premises to compare
        
    Returns:
        Jaccard similarity [0, 1]
    """
    a1 = p1.antecedents
    a2 = p2.antecedents
    
    intersection = len(a1 & a2)
    union = len(a1 | a2)
    
    return intersection / union if union > 0 else 0.0


def activation_similarity(
    memberships: torch.Tensor,
    p1: Premise,
    p2: Premise,
) -> float:
    """
    Compute activation-based similarity between premises.
    
    S(P, Q) = Σ min(μ_P(x), μ_Q(x)) / Σ max(μ_P(x), μ_Q(x))
    
    Args:
        memberships: [N, D, T] membership tensor
        p1, p2: Premises to compare
        
    Returns:
        Similarity value [0, 1]
    """
    act1 = compute_premise_activation(memberships, p1)
    act2 = compute_premise_activation(memberships, p2)
    
    min_sum = torch.min(act1, act2).sum()
    max_sum = torch.max(act1, act2).sum()
    
    return (min_sum / (max_sum + 1e-8)).item()


def filter_redundant_premises(
    premises: List[Premise],
    memberships: torch.Tensor,
    similarity_threshold: float = 0.8,
    use_activation: bool = True,
) -> List[Premise]:
    """
    Filter redundant premises based on similarity.
    
    When two premises are too similar, keep the one with higher support.
    
    Args:
        premises: List of premises
        memberships: [N, D, T] membership tensor
        similarity_threshold: Max allowed similarity
        use_activation: Use activation-based similarity (else Jaccard)
        
    Returns:
        Filtered list of premises
    """
    if not premises:
        return []
    
    # Sort by support (descending) to prefer higher support
    sorted_premises = sorted(premises, key=lambda p: -p.support)
    
    kept = []
    for premise in sorted_premises:
        is_redundant = False
        
        for kept_premise in kept:
            if use_activation:
                sim = activation_similarity(memberships, premise, kept_premise)
            else:
                sim = premise_jaccard_similarity(premise, kept_premise)
            
            if sim >= similarity_threshold:
                is_redundant = True
                break
        
        if not is_redundant:
            kept.append(premise)
    
    return kept


def detect_conflicts(
    rules: List[FuzzyRule],
    memberships: torch.Tensor,
    similarity_threshold: float = 0.8,
    consequent_diff_threshold: float = 0.5,
) -> List[Tuple[int, int]]:
    """
    Detect conflicting rules: similar premises but very different consequents.
    
    Args:
        rules: List of rules
        memberships: [N, D, T] membership tensor
        similarity_threshold: Min premise similarity to consider conflict
        consequent_diff_threshold: Min consequent difference to flag conflict
        
    Returns:
        List of (rule_i, rule_j) conflict pairs
    """
    conflicts = []
    
    for i in range(len(rules)):
        for j in range(i + 1, len(rules)):
            # Check premise similarity
            sim = activation_similarity(memberships, rules[i].premise, rules[j].premise)
            
            if sim >= similarity_threshold:
                # Check consequent difference
                c1 = rules[i].consequent
                c2 = rules[j].consequent
                
                if isinstance(c1, TSKConsequent) and isinstance(c2, TSKConsequent):
                    if c1.order == 0 and c2.order == 0:
                        diff = abs(c1.coefficients.item() - c2.coefficients.item())
                        if diff >= consequent_diff_threshold:
                            conflicts.append((i, j))
    
    return conflicts


def compute_evidence_score(
    premise: Premise,
    memberships: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.1,
) -> float:
    """
    Compute evidence-based score for a premise.
    
    score = Q * log(1 + S) - α * |antecedents|
    
    where:
    - Q = quality (activation or local variance reduction)
    - S = support (sum of activations)
    - α = complexity penalty
    
    Args:
        premise: Premise to score
        memberships: [N, D, T] tensor
        y: Target values [N]
        alpha: Complexity penalty weight
        
    Returns:
        Evidence score
    """
    activations = compute_premise_activation(memberships, premise)
    
    support = activations.sum().item()
    quality = premise.activation
    complexity = premise.size
    
    score = quality * np.log1p(support) - alpha * complexity
    return score


def score_and_weight_rules(
    rules: List[FuzzyRule],
    memberships: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.1,
) -> List[FuzzyRule]:
    """
    Score rules and compute normalized weights.
    
    weight = relu(score) / sum(relu(scores))
    
    Args:
        rules: List of rules
        memberships: [N, D, T] tensor
        y: Target values
        alpha: Complexity penalty
        
    Returns:
        Rules with updated weights and quality scores
    """
    scores = []
    for rule in rules:
        score = compute_evidence_score(rule.premise, memberships, y, alpha)
        rule.quality_score = score
        scores.append(score)
    
    # ReLU and normalize
    scores = np.array(scores)
    scores = np.maximum(scores, 0)
    
    total = scores.sum()
    if total > 0:
        weights = scores / total
    else:
        weights = np.ones(len(rules)) / len(rules)
    
    for i, rule in enumerate(rules):
        rule.weight = float(weights[i])
    
    return rules


def select_top_k_rules(
    rules: List[FuzzyRule],
    k: int,
    memberships: Optional[torch.Tensor] = None,
    y_val: Optional[torch.Tensor] = None,
    gamma: float = 0.01,
) -> List[FuzzyRule]:
    """
    Select top-K rules based on quality score.
    
    If validation data provided, uses validation-based selection:
    k* = argmin_k val_error(k) + γ * k
    
    Otherwise, simply takes top-K by score.
    
    Args:
        rules: List of rules
        k: Maximum number of rules
        memberships: Optional validation memberships
        y_val: Optional validation targets
        gamma: Complexity penalty for validation selection
        
    Returns:
        Selected rules
    """
    if len(rules) <= k:
        return rules
    
    # Sort by quality score
    sorted_rules = sorted(rules, key=lambda r: -r.quality_score)
    
    if memberships is None or y_val is None:
        # Simple top-K
        return sorted_rules[:k]
    
    # Validation-based selection
    # TODO: Implement proper validation error computation
    # For now, just use quality score
    return sorted_rules[:k]


def filter_rules_internal(
    rules: List[FuzzyRule],
    memberships: torch.Tensor,
    y: torch.Tensor,
    similarity_threshold: float = 0.8,
    alpha: float = 0.1,
    top_k: int = 50,
) -> List[FuzzyRule]:
    """
    Apply internal filtering pipeline:
    1. Remove redundant rules
    2. Score and weight rules
    3. Select top-K
    
    Args:
        rules: Input rules
        memberships: [N, D, T] tensor
        y: Target values
        similarity_threshold: For redundancy detection
        alpha: Complexity penalty
        top_k: Maximum rules to keep
        
    Returns:
        Filtered rules
    """
    if not rules:
        return []
    
    # 1. Filter redundant premises
    premises = [r.premise for r in rules]
    filtered_premises = filter_redundant_premises(
        premises, memberships, similarity_threshold
    )
    
    # Map back to rules
    premise_set = set(filtered_premises)
    rules = [r for r in rules if r.premise in premise_set]
    
    # 2. Score and weight
    rules = score_and_weight_rules(rules, memberships, y, alpha)
    
    # 3. Select top-K
    rules = select_top_k_rules(rules, top_k)
    
    return rules


def aggregate_rules_external(
    member_rules: List[List[FuzzyRule]],
    memberships: torch.Tensor,
    similarity_threshold: float = 0.8,
) -> List[FuzzyRule]:
    """
    Aggregate rules from multiple ensemble members.
    
    - Deduplicate similar rules (keep highest quality)
    - Merge weights for identical premises
    
    Args:
        member_rules: List of rule lists from each member
        memberships: [N, D, T] tensor for full training set
        similarity_threshold: For deduplication
        
    Returns:
        Aggregated rules
    """
    # Flatten all rules
    all_rules = []
    for rules in member_rules:
        all_rules.extend(rules)
    
    if not all_rules:
        return []
    
    # Sort by quality (descending)
    sorted_rules = sorted(all_rules, key=lambda r: -r.quality_score)
    
    # Deduplicate based on premise similarity
    kept_rules = []
    kept_premises = []
    
    for rule in sorted_rules:
        is_duplicate = False
        
        for kept_premise in kept_premises:
            sim = premise_jaccard_similarity(rule.premise, kept_premise)
            if sim >= similarity_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            kept_rules.append(rule)
            kept_premises.append(rule.premise)
    
    return kept_rules
