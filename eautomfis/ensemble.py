"""
Ensemble module for e-AutoMFIS.

Implements:
- Temporal and feature subsampling
- Core + random feature selection per member
- Diversity enforcement (Jaccard constraint)
- Parallel member training (CPU multiprocessing + GPU batching)
- One-shot ensemble weighting (OOF error-based)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict, Any
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import numpy as np
import torch

from .config import EAutoMFISConfig
from .partition import create_partitions_for_dataset, compute_membership_matrix, FuzzyPartition
from .mining import mine_premises, premises_to_indices, compute_firing_strengths_vectorized
from .association import create_rules_from_premises, FuzzyRule
from .filtering import filter_rules_internal, aggregate_rules_external
from .utils import set_seeds, Timer, Logger


@dataclass
class MemberConfig:
    """Configuration for a single ensemble member."""
    member_id: int
    feature_indices: List[int]
    temporal_indices: np.ndarray
    random_seed: int
    
    def __repr__(self) -> str:
        return (f"Member(id={self.member_id}, features={len(self.feature_indices)}, "
                f"samples={len(self.temporal_indices)})")


@dataclass 
class EnsembleMember:
    """
    A single trained ensemble member.
    
    Contains rules learned from a subset of features and data.
    """
    config: MemberConfig
    partitions: List[FuzzyPartition]
    rules: List[FuzzyRule]
    oof_error: float = 0.0
    weight: float = 1.0
    
    @property
    def num_rules(self) -> int:
        return len(self.rules)


def compute_feature_jaccard(f1: List[int], f2: List[int]) -> float:
    """Compute Jaccard similarity between two feature sets."""
    s1, s2 = set(f1), set(f2)
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    return intersection / union if union > 0 else 0.0


def generate_diverse_feature_subsets(
    n_features: int,
    n_members: int,
    subset_size: int,
    core_features: Optional[List[int]] = None,
    min_diversity: float = 0.3,
    max_attempts: int = 100,
    random_state: int = 42,
) -> List[List[int]]:
    """
    Generate diverse feature subsets for ensemble members.
    
    Each subset contains:
    - Core features (always included, e.g., target lags)
    - Random features (sampled ensuring diversity)
    
    Args:
        n_features: Total number of features
        n_members: Number of ensemble members
        subset_size: Size of each subset (including core)
        core_features: Features always included
        min_diversity: Minimum Jaccard distance between members (1 - similarity)
        max_attempts: Max attempts to find diverse subset
        random_state: Random seed
        
    Returns:
        List of feature index lists
    """
    rng = np.random.RandomState(random_state)
    
    core_features = core_features or []
    core_set = set(core_features)
    non_core = [i for i in range(n_features) if i not in core_set]
    
    additional_needed = max(0, subset_size - len(core_features))
    
    if additional_needed > len(non_core):
        additional_needed = len(non_core)
    
    subsets = []
    
    for m in range(n_members):
        for attempt in range(max_attempts):
            # Sample random features
            if additional_needed > 0 and len(non_core) > 0:
                random_features = rng.choice(non_core, size=min(additional_needed, len(non_core)), replace=False).tolist()
            else:
                random_features = []
            
            subset = list(core_features) + random_features
            
            # Check diversity against existing subsets
            is_diverse = True
            for existing in subsets:
                similarity = compute_feature_jaccard(subset, existing)
                if similarity > (1 - min_diversity):
                    is_diverse = False
                    break
            
            if is_diverse or attempt == max_attempts - 1:
                subsets.append(sorted(subset))
                break
    
    return subsets


def generate_temporal_folds(
    n_samples: int,
    n_folds: int,
    overlap_ratio: float = 0.2,
) -> List[np.ndarray]:
    """
    Generate overlapping temporal folds.
    
    Args:
        n_samples: Total number of samples
        n_folds: Number of folds
        overlap_ratio: Overlap between consecutive folds
        
    Returns:
        List of sample index arrays
    """
    folds = []
    
    effective_folds = n_folds - (n_folds - 1) * overlap_ratio
    fold_size = int(n_samples / effective_folds)
    step_size = int(fold_size * (1 - overlap_ratio))
    
    for i in range(n_folds):
        start = i * step_size
        end = min(start + fold_size, n_samples)
        
        if end > start:
            folds.append(np.arange(start, end))
    
    return folds


def generate_member_configs(
    n_samples: int,
    n_features: int,
    config: EAutoMFISConfig,
) -> List[MemberConfig]:
    """
    Generate configurations for all ensemble members.
    
    Args:
        n_samples: Number of training samples
        n_features: Number of features
        config: e-AutoMFIS configuration
        
    Returns:
        List of MemberConfig objects
    """
    # Generate diverse feature subsets
    feature_subsets = generate_diverse_feature_subsets(
        n_features=n_features,
        n_members=config.num_members,
        subset_size=config.feature_subset_size,
        core_features=config.core_features,
        min_diversity=config.min_diversity,
        random_state=config.random_seed,
    )
    
    # Generate temporal folds
    temporal_folds = generate_temporal_folds(
        n_samples=n_samples,
        n_folds=config.temporal_folds,
        overlap_ratio=0.2,
    )
    
    configs = []
    for m in range(config.num_members):
        # Cycle through temporal folds
        fold_idx = m % len(temporal_folds)
        
        configs.append(MemberConfig(
            member_id=m,
            feature_indices=feature_subsets[m],
            temporal_indices=temporal_folds[fold_idx],
            random_seed=config.random_seed + m,
        ))
    
    return configs


def train_single_member(
    X: np.ndarray,
    y: np.ndarray,
    member_config: MemberConfig,
    model_config: EAutoMFISConfig,
) -> EnsembleMember:
    """
    Train a single ensemble member.
    
    Args:
        X: Full feature matrix [N, D]
        y: Full target vector [N] or [N, H]
        member_config: Member configuration
        model_config: Model configuration
        
    Returns:
        Trained EnsembleMember
    """
    # Set seed for reproducibility
    set_seeds(member_config.random_seed)
    
    # Extract subset
    X_sub = X[member_config.temporal_indices][:, member_config.feature_indices]
    y_sub = y[member_config.temporal_indices]
    
    X_t = torch.from_numpy(X_sub).float()
    y_t = torch.from_numpy(y_sub).float()
    
    # Create partitions for this member's features
    partitions = create_partitions_for_dataset(
        X_sub,
        num_terms=model_config.num_terms,
        method=model_config.partition_method,
    )
    
    # Compute memberships
    memberships = compute_membership_matrix(X_sub, partitions)
    
    # Mine premises
    premises = mine_premises(
        memberships,
        min_support=model_config.min_support,
        max_antecedents=model_config.max_antecedents,
        max_candidates_per_level=model_config.max_candidates_per_level,
        max_total_candidates=model_config.max_total_candidates,
        activation_method=model_config.activation_method,
        min_frequency=model_config.min_frequency,
        verbose=False,
    )
    
    if not premises:
        return EnsembleMember(
            config=member_config,
            partitions=partitions,
            rules=[],
            oof_error=float('inf'),
            weight=0.0,
        )
    
    # Compute firing strengths
    feat_idx, term_idx, mask = premises_to_indices(premises, model_config.max_antecedents)
    firing_strengths = compute_firing_strengths_vectorized(memberships, feat_idx, term_idx, mask)
    
    # Create rules
    rules = create_rules_from_premises(
        premises, X_t, y_t, firing_strengths,
        tsk_order=model_config.tsk_order,
        regularization=model_config.tsk_regularization,
    )
    
    # Filter rules
    rules = filter_rules_internal(
        rules, memberships, y_t,
        similarity_threshold=model_config.similarity_threshold,
        alpha=model_config.alpha_complexity,
        top_k=model_config.top_k_rules,
    )
    
    return EnsembleMember(
        config=member_config,
        partitions=partitions,
        rules=rules,
        oof_error=0.0,  # Will be computed later
        weight=1.0 / model_config.num_members,
    )


class EnsembleTrainer:
    """
    Trainer for e-AutoMFIS ensemble.
    
    Supports:
    - Sequential training
    - Parallel CPU training (multiprocessing)
    - Batched GPU training
    """
    
    def __init__(
        self,
        config: EAutoMFISConfig,
        n_jobs: int = 1,
        device: str = "cpu",
        verbose: bool = True,
    ):
        self.config = config
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        self.device = device
        self.logger = Logger(verbose=1 if verbose else 0)
    
    def train_sequential(
        self,
        X: np.ndarray,
        y: np.ndarray,
        member_configs: List[MemberConfig],
    ) -> List[EnsembleMember]:
        """Train members sequentially."""
        members = []
        
        for i, cfg in enumerate(member_configs):
            self.logger.progress(i + 1, len(member_configs), "Training members")
            
            with Timer(f"member_{i}", verbose=False):
                member = train_single_member(X, y, cfg, self.config)
            
            members.append(member)
        
        return members
    
    def train_parallel_cpu(
        self,
        X: np.ndarray,
        y: np.ndarray,
        member_configs: List[MemberConfig],
    ) -> List[EnsembleMember]:
        """Train members in parallel using multiprocessing."""
        self.logger.info(f"Training {len(member_configs)} members in parallel ({self.n_jobs} workers)")
        
        # Use ThreadPoolExecutor for now (safer with PyTorch)
        # ProcessPoolExecutor requires pickling which can be problematic
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [
                executor.submit(train_single_member, X, y, cfg, self.config)
                for cfg in member_configs
            ]
            members = [f.result() for f in futures]
        
        return members
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        member_configs: Optional[List[MemberConfig]] = None,
    ) -> List[EnsembleMember]:
        """
        Train all ensemble members.
        
        Args:
            X: Feature matrix [N, D]
            y: Target vector [N] or [N, H]
            member_configs: Optional pre-generated configs
            
        Returns:
            List of trained EnsembleMember objects
        """
        if member_configs is None:
            member_configs = generate_member_configs(
                n_samples=X.shape[0],
                n_features=X.shape[1],
                config=self.config,
            )
        
        self.logger.info(f"Training ensemble with {len(member_configs)} members")
        
        with Timer("ensemble_training", verbose=False):
            if self.n_jobs == 1:
                members = self.train_sequential(X, y, member_configs)
            else:
                members = self.train_parallel_cpu(X, y, member_configs)
        
        self.logger.info(f"Training complete. Total rules: {sum(m.num_rules for m in members)}")
        
        return members


def compute_ensemble_weights(
    members: List[EnsembleMember],
    X_val: np.ndarray,
    y_val: np.ndarray,
    tau: float = 1.0,
) -> List[EnsembleMember]:
    """
    Compute ensemble weights based on validation error.
    
    weight_m = softmax(-τ * error_m)
    
    Args:
        members: List of trained members
        X_val: Validation features
        y_val: Validation targets
        tau: Temperature parameter
        
    Returns:
        Members with updated weights
    """
    from .inference import create_inference_engine
    
    errors = []
    
    for member in members:
        if not member.rules:
            errors.append(float('inf'))
            continue
        
        # Map validation data to member's feature space
        X_member = X_val[:, member.config.feature_indices]
        
        # Create inference engine for this member
        engine = create_inference_engine(
            member.rules,
            member.partitions,
            max_antecedents=len(member.config.feature_indices),
        )
        
        # Predict
        y_pred = engine.infer(X_member).detach().numpy()
        
        # Compute error
        mse = np.mean((y_pred - y_val) ** 2)
        errors.append(mse)
        member.oof_error = mse
    
    # Softmax weighting
    errors = np.array(errors)
    finite_mask = np.isfinite(errors)
    
    if finite_mask.sum() == 0:
        # All infinite, use uniform weights
        weights = np.ones(len(members)) / len(members)
    else:
        # Replace inf with max finite error + 1
        max_error = errors[finite_mask].max() if finite_mask.any() else 1.0
        errors = np.where(finite_mask, errors, max_error + 1)
        
        # Softmax
        exp_neg = np.exp(-tau * errors)
        weights = exp_neg / exp_neg.sum()
    
    for i, member in enumerate(members):
        member.weight = weights[i]
    
    return members


def aggregate_ensemble(
    members: List[EnsembleMember],
    full_partitions: List[FuzzyPartition],
    memberships: torch.Tensor,
    similarity_threshold: float = 0.8,
) -> Tuple[List[FuzzyRule], np.ndarray]:
    """
    Aggregate rules from all ensemble members.
    
    Args:
        members: Trained ensemble members
        full_partitions: Partitions for full feature space
        memberships: Full membership matrix
        similarity_threshold: For deduplication
        
    Returns:
        (aggregated_rules, member_weights)
    """
    # Collect all rules weighted by member weight
    all_rules = []
    for member in members:
        for rule in member.rules:
            # Scale rule quality by member weight
            rule.weight *= member.weight
            all_rules.append(rule)
    
    # Deduplicate
    aggregated = aggregate_rules_external(
        [[r] for r in all_rules],  # Flatten
        memberships,
        similarity_threshold,
    )
    
    member_weights = np.array([m.weight for m in members])
    
    return aggregated, member_weights
