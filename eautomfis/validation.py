"""
Validation utilities for e-AutoMFIS.

Contains:
- Temporal train/validation/test splitting
- Walk-forward validation framework
- Cross-validation utilities for time series
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterator, Union
import numpy as np
import torch


@dataclass
class TemporalSplit:
    """
    Container for a single temporal split.
    
    Attributes:
        train_idx: Indices for training data
        val_idx: Indices for validation data  
        test_idx: Indices for test data (optional)
        fold_id: Identifier for this fold
    """
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: Optional[np.ndarray] = None
    fold_id: int = 0
    
    def __repr__(self) -> str:
        test_info = f", test={len(self.test_idx)}" if self.test_idx is not None else ""
        return f"TemporalSplit(fold={self.fold_id}, train={len(self.train_idx)}, val={len(self.val_idx)}{test_info})"


def temporal_train_val_test_split(
    n_samples: int,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    min_train_samples: int = 10,
) -> TemporalSplit:
    """
    Split time series data into train/validation/test sets chronologically.
    
    Args:
        n_samples: Total number of samples
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for test
        min_train_samples: Minimum number of training samples
        
    Returns:
        TemporalSplit with train/val/test indices
    """
    train_ratio = 1 - val_ratio - test_ratio
    assert train_ratio > 0, "Train ratio must be positive"
    
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    train_end = max(train_end, min_train_samples)
    val_end = max(val_end, train_end + 1)
    
    return TemporalSplit(
        train_idx=np.arange(0, train_end),
        val_idx=np.arange(train_end, val_end),
        test_idx=np.arange(val_end, n_samples) if val_end < n_samples else None,
        fold_id=0,
    )


def walk_forward_split(
    n_samples: int,
    n_folds: int = 5,
    val_ratio: float = 0.2,
    min_train_samples: int = 10,
    expanding: bool = True,
) -> List[TemporalSplit]:
    """
    Generate walk-forward validation splits for time series.
    
    In walk-forward validation, the training window either expands or slides,
    and validation is always on the next chronological segment.
    
    Args:
        n_samples: Total number of samples
        n_folds: Number of validation folds
        val_ratio: Fraction of each fold for validation
        min_train_samples: Minimum training samples per fold
        expanding: If True, training window expands; if False, it slides
        
    Returns:
        List of TemporalSplit objects
        
    Example (n=100, n_folds=5, expanding=True):
        Fold 1: train=[0:60], val=[60:80]
        Fold 2: train=[0:64], val=[64:84]
        Fold 3: train=[0:68], val=[68:88]
        Fold 4: train=[0:72], val=[72:92]
        Fold 5: train=[0:76], val=[76:96]
    """
    splits = []
    
    # Calculate fold boundaries
    # Reserve last portion for test set (not used in walk-forward)
    usable_samples = int(n_samples * 0.95)  # Keep 5% for final test
    
    fold_size = usable_samples // (n_folds + 1)
    val_size = max(int(fold_size * val_ratio), 1)
    
    for fold_id in range(n_folds):
        # Validation window
        val_start = min_train_samples + (fold_id + 1) * fold_size
        val_end = min(val_start + val_size, usable_samples)
        
        if val_end > usable_samples:
            break
            
        # Training window
        if expanding:
            train_start = 0
        else:
            # Sliding window: keep fixed training size
            train_size = fold_size * (n_folds - fold_id)
            train_start = max(0, val_start - train_size)
        
        train_end = val_start
        
        if train_end - train_start < min_train_samples:
            continue
            
        splits.append(TemporalSplit(
            train_idx=np.arange(train_start, train_end),
            val_idx=np.arange(val_start, val_end),
            test_idx=None,
            fold_id=fold_id,
        ))
    
    return splits


def temporal_ensemble_splits(
    n_samples: int,
    n_members: int,
    overlap_ratio: float = 0.2,
) -> List[TemporalSplit]:
    """
    Generate temporal subsampling splits for ensemble members.
    
    Divides the time series into overlapping segments, one per ensemble member.
    
    Args:
        n_samples: Total number of samples
        n_members: Number of ensemble members
        overlap_ratio: Overlap between consecutive segments (0-1)
        
    Returns:
        List of TemporalSplit objects (train indices only)
    """
    splits = []
    
    # Calculate segment sizes with overlap
    effective_segments = n_members - (n_members - 1) * overlap_ratio
    segment_size = int(n_samples / effective_segments)
    step_size = int(segment_size * (1 - overlap_ratio))
    
    for member_id in range(n_members):
        start = member_id * step_size
        end = min(start + segment_size, n_samples)
        
        if end <= start:
            break
            
        splits.append(TemporalSplit(
            train_idx=np.arange(start, end),
            val_idx=np.array([], dtype=int),  # No validation for ensemble splits
            test_idx=None,
            fold_id=member_id,
        ))
    
    return splits


class WalkForwardValidator:
    """
    Walk-forward cross-validation for time series models.
    
    Usage:
        validator = WalkForwardValidator(X, y, n_folds=5)
        for fold, (X_train, y_train, X_val, y_val) in validator:
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            validator.record_score(fold, score)
        
        print(validator.get_results())
    """
    
    def __init__(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        n_folds: int = 5,
        expanding: bool = True,
        val_ratio: float = 0.2,
    ):
        self.X = X
        self.y = y
        self.n_folds = n_folds
        self.expanding = expanding
        self.val_ratio = val_ratio
        
        n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
        self.splits = walk_forward_split(
            n_samples=n_samples,
            n_folds=n_folds,
            val_ratio=val_ratio,
            expanding=expanding,
        )
        
        self.scores: List[dict] = []
        self.predictions: List[np.ndarray] = []
    
    def __iter__(self) -> Iterator[Tuple[int, Tuple]]:
        """Iterate over folds, yielding (fold_id, (X_train, y_train, X_val, y_val))."""
        for split in self.splits:
            X_train = self._index(self.X, split.train_idx)
            y_train = self._index(self.y, split.train_idx)
            X_val = self._index(self.X, split.val_idx)
            y_val = self._index(self.y, split.val_idx)
            
            yield split.fold_id, (X_train, y_train, X_val, y_val)
    
    def __len__(self) -> int:
        return len(self.splits)
    
    def _index(self, arr, idx):
        """Index array or tensor."""
        if isinstance(arr, torch.Tensor):
            return arr[idx]
        return arr[idx]
    
    def record_score(self, fold_id: int, metrics: dict) -> None:
        """Record validation metrics for a fold."""
        self.scores.append({"fold": fold_id, **metrics})
    
    def record_predictions(self, fold_id: int, y_pred: np.ndarray) -> None:
        """Record validation predictions for a fold."""
        self.predictions.append({"fold": fold_id, "y_pred": y_pred})
    
    def get_oof_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get out-of-fold predictions and corresponding true values.
        
        Returns:
            (y_true_oof, y_pred_oof) arrays
        """
        y_true_parts = []
        y_pred_parts = []
        
        for i, split in enumerate(self.splits):
            y_true_parts.append(self._index(self.y, split.val_idx))
            if i < len(self.predictions):
                y_pred_parts.append(self.predictions[i]["y_pred"])
        
        return np.concatenate(y_true_parts), np.concatenate(y_pred_parts)
    
    def get_results(self) -> dict:
        """Get aggregated cross-validation results."""
        if not self.scores:
            return {}
        
        # Aggregate numeric metrics across folds
        all_metrics = {}
        for score in self.scores:
            for key, value in score.items():
                if key == "fold":
                    continue
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)
        
        results = {
            "n_folds": len(self.scores),
            "per_fold": self.scores,
        }
        
        for key, values in all_metrics.items():
            results[f"{key}_mean"] = np.mean(values)
            results[f"{key}_std"] = np.std(values)
            results[f"{key}_min"] = np.min(values)
            results[f"{key}_max"] = np.max(values)
        
        return results


def create_lagged_features(
    data: np.ndarray,
    max_lag: int,
    target_col: int = 0,
    include_target_lags: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create lagged features for time series forecasting.
    
    Args:
        data: Input data of shape [T, D] (time x features)
        max_lag: Maximum lag to include
        target_col: Index of target column
        include_target_lags: Whether to include lags of the target
        
    Returns:
        X: Feature matrix [N, D*max_lag] 
        y: Target vector [N]
    """
    T, D = data.shape
    N = T - max_lag
    
    if N <= 0:
        raise ValueError(f"Not enough samples: T={T}, max_lag={max_lag}")
    
    # Build lagged feature matrix
    feature_cols = []
    for lag in range(1, max_lag + 1):
        for d in range(D):
            if d == target_col and not include_target_lags:
                continue
            feature_cols.append(data[max_lag - lag : T - lag, d])
    
    X = np.column_stack(feature_cols)
    y = data[max_lag:, target_col]
    
    return X, y


def create_multistep_targets(
    y: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """
    Create multi-step forecast targets.
    
    Args:
        y: Target vector [N]
        horizon: Forecast horizon (H)
        
    Returns:
        Y: Target matrix [N-H+1, H] where Y[t, h] = y[t+h]
    """
    N = len(y)
    if N < horizon:
        raise ValueError(f"Not enough samples: N={N}, horizon={horizon}")
    
    Y = np.zeros((N - horizon + 1, horizon))
    for h in range(horizon):
        Y[:, h] = y[h : N - horizon + 1 + h]
    
    return Y
