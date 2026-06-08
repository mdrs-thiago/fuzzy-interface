"""
Fuzzy partitioning module for e-AutoMFIS.

Implements Strong Fuzzy Partitions (where sum of memberships = 1 for any point)
using three methods:
- Uniform: evenly spaced terms
- Percentile (Tukey): terms at data percentiles
- Clustering: k-means centers as term peaks

All partitions use triangular MFs (internal) and trapezoidal MFs (extremes)
to ensure interpretability and coverage.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal, Union
import numpy as np
import torch
import torch.nn as nn

# Import from existing fuzzy_torch if available, otherwise define locally
try:
    from fuzzy_torch import TriangularMF, TrapezoidalMF
except ImportError:
    # Fallback definitions matching fuzzy_torch interface
    class TriangularMF(nn.Module):
        def __init__(self, a: float, b: float, c: float):
            super().__init__()
            self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
            self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))  
            self.c = nn.Parameter(torch.tensor(c, dtype=torch.float32))
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            left = (x - self.a) / (self.b - self.a + 1e-8)
            right = (self.c - x) / (self.c - self.b + 1e-8)
            return torch.clamp(torch.min(left, right), 0, 1)
    
    class TrapezoidalMF(nn.Module):
        def __init__(self, a: float, b: float, c: float, d: float):
            super().__init__()
            self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
            self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))
            self.c = nn.Parameter(torch.tensor(c, dtype=torch.float32))
            self.d = nn.Parameter(torch.tensor(d, dtype=torch.float32))
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            left = (x - self.a) / (self.b - self.a + 1e-8)
            right = (self.d - x) / (self.d - self.c + 1e-8)
            return torch.clamp(torch.min(torch.min(left, torch.ones_like(x)), right), 0, 1)


@dataclass
class FuzzyPartition:
    """
    A strong fuzzy partition for a single variable.
    
    Attributes:
        name: Variable name
        num_terms: Number of fuzzy terms
        centers: Peak values for each term [num_terms]
        term_names: Names for each term
        membership_functions: List of MF objects
        universe_min: Minimum value in universe of discourse
        universe_max: Maximum value in universe of discourse
    """
    name: str
    num_terms: int
    centers: np.ndarray
    term_names: List[str]
    membership_functions: List[nn.Module]
    universe_min: float
    universe_max: float
    
    def fuzzify(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Compute membership degrees for all terms.
        
        Args:
            x: Input values [N] or [N, 1]
            
        Returns:
            Membership matrix [N, num_terms]
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        x = x.squeeze()
        memberships = torch.stack([mf(x) for mf in self.membership_functions], dim=-1)
        return memberships
    
    def get_term_by_name(self, name: str) -> nn.Module:
        """Get membership function by term name."""
        idx = self.term_names.index(name)
        return self.membership_functions[idx]
    
    def __repr__(self) -> str:
        return (f"FuzzyPartition(name='{self.name}', terms={self.num_terms}, "
                f"range=[{self.universe_min:.2f}, {self.universe_max:.2f}])")


def generate_term_names(num_terms: int) -> List[str]:
    """
    Generate default linguistic term names.
    
    Uses standard naming convention:
    - 3 terms: Low, Medium, High
    - 5 terms: Very Low, Low, Medium, High, Very High
    - 7 terms: adds Extremely Low/High
    - etc.
    """
    if num_terms == 2:
        return ["Low", "High"]
    elif num_terms == 3:
        return ["Low", "Medium", "High"]
    elif num_terms == 4:
        return ["Low", "Medium-Low", "Medium-High", "High"]
    elif num_terms == 5:
        return ["Very Low", "Low", "Medium", "High", "Very High"]
    elif num_terms == 6:
        return ["Very Low", "Low", "Medium-Low", "Medium-High", "High", "Very High"]
    elif num_terms == 7:
        return ["Extremely Low", "Very Low", "Low", "Medium", "High", "Very High", "Extremely High"]
    else:
        # Generic naming for larger numbers
        return [f"T{i+1}" for i in range(num_terms)]


def _compute_uniform_centers(
    data_min: float, 
    data_max: float, 
    num_terms: int,
    margin: float = 0.05,
) -> np.ndarray:
    """
    Compute uniformly spaced centers across the data range.
    
    Args:
        data_min: Minimum data value
        data_max: Maximum data value
        num_terms: Number of terms
        margin: Margin beyond data range (fraction)
        
    Returns:
        Array of center values [num_terms]
    """
    range_span = data_max - data_min
    universe_min = data_min - margin * range_span
    universe_max = data_max + margin * range_span
    
    centers = np.linspace(universe_min, universe_max, num_terms)
    return centers


def _compute_percentile_centers(
    data: np.ndarray,
    num_terms: int,
) -> np.ndarray:
    """
    Compute centers at data percentiles (Tukey method).
    
    Centers are placed at evenly spaced percentiles:
    - For 5 terms: 0%, 25%, 50%, 75%, 100%
    
    Args:
        data: Data array [N]
        num_terms: Number of terms
        
    Returns:
        Array of center values [num_terms]
    """
    percentiles = np.linspace(0, 100, num_terms)
    centers = np.percentile(data, percentiles)
    return centers


def _compute_clustering_centers(
    data: np.ndarray,
    num_terms: int,
    max_iter: int = 100,
    random_state: int = 42,
) -> np.ndarray:
    """
    Compute centers using k-means clustering.
    
    Args:
        data: Data array [N]
        num_terms: Number of terms (clusters)
        max_iter: Maximum iterations for k-means
        random_state: Random seed
        
    Returns:
        Sorted array of center values [num_terms]
    """
    try:
        from sklearn.cluster import KMeans
        
        data_2d = data.reshape(-1, 1)
        kmeans = KMeans(n_clusters=num_terms, max_iter=max_iter, 
                        random_state=random_state, n_init=10)
        kmeans.fit(data_2d)
        centers = kmeans.cluster_centers_.flatten()
        
    except ImportError:
        # Fallback: simple k-means implementation
        np.random.seed(random_state)
        
        # Initialize centers randomly from data
        idx = np.random.choice(len(data), num_terms, replace=False)
        centers = data[idx].copy()
        
        for _ in range(max_iter):
            # Assign points to nearest center
            distances = np.abs(data[:, None] - centers[None, :])
            assignments = np.argmin(distances, axis=1)
            
            # Update centers
            new_centers = np.array([
                data[assignments == k].mean() if (assignments == k).any() else centers[k]
                for k in range(num_terms)
            ])
            
            if np.allclose(centers, new_centers):
                break
            centers = new_centers
    
    # Sort centers for proper ordering
    return np.sort(centers)


def create_strong_fuzzy_partition(
    centers: np.ndarray,
    term_names: List[str],
    name: str = "var",
    universe_min: float = None,
    universe_max: float = None,
) -> FuzzyPartition:
    """
    Create a strong fuzzy partition from center values.
    
    Strong fuzzy partition properties:
    - Sum of memberships = 1 for any point in universe
    - Uses triangular MFs for internal terms
    - Uses trapezoidal MFs for extreme terms (open-ended)
    
    Args:
        centers: Peak values for each term [num_terms]
        term_names: Names for each term
        name: Variable name
        universe_min: Minimum universe value (if None, uses smallest center)
        universe_max: Maximum universe value (if None, uses largest center)
        
    Returns:
        FuzzyPartition object
    """
    num_terms = len(centers)
    centers = np.sort(centers)  # Ensure sorted
    
    # Determine universe bounds
    if universe_min is None:
        universe_min = centers[0]
    if universe_max is None:
        universe_max = centers[-1]
    
    # Extend universe slightly for open-ended trapezoids
    range_span = universe_max - universe_min
    extended_min = universe_min - 0.1 * range_span
    extended_max = universe_max + 0.1 * range_span
    
    mfs = []
    
    for i in range(num_terms):
        if i == 0:
            # First term: left-open trapezoidal
            # Shape: full membership from -inf to center, then slopes down
            a = extended_min - range_span  # Far left (always 1)
            b = centers[0]  # Start of slope down
            c = centers[0]  # Peak
            d = centers[1] if num_terms > 1 else extended_max
            mfs.append(TrapezoidalMF(a, b, c, d))
            
        elif i == num_terms - 1:
            # Last term: right-open trapezoidal
            a = centers[-2] if num_terms > 1 else extended_min
            b = centers[-1]  # Peak
            c = centers[-1]  # Start of flat top
            d = extended_max + range_span  # Far right (always 1)
            mfs.append(TrapezoidalMF(a, b, c, d))
            
        else:
            # Internal terms: triangular
            a = centers[i - 1]  # Left foot
            b = centers[i]      # Peak
            c = centers[i + 1]  # Right foot
            mfs.append(TriangularMF(a, b, c))
    
    return FuzzyPartition(
        name=name,
        num_terms=num_terms,
        centers=centers,
        term_names=term_names,
        membership_functions=mfs,
        universe_min=universe_min,
        universe_max=universe_max,
    )


def create_partition(
    data: np.ndarray,
    num_terms: int = 5,
    method: Literal["uniform", "percentile", "clustering"] = "percentile",
    name: str = "var",
    term_names: Optional[List[str]] = None,
    **kwargs,
) -> FuzzyPartition:
    """
    Create a fuzzy partition for a variable.
    
    This is the main entry point for partition creation.
    
    Args:
        data: Data array [N] for fitting the partition
        num_terms: Number of fuzzy terms
        method: Partitioning method ('uniform', 'percentile', 'clustering')
        name: Variable name
        term_names: Optional custom term names (auto-generated if None)
        **kwargs: Additional arguments passed to center computation
        
    Returns:
        FuzzyPartition object
    """
    data = np.asarray(data).flatten()
    
    if term_names is None:
        term_names = generate_term_names(num_terms)
    
    if len(term_names) != num_terms:
        raise ValueError(f"term_names length ({len(term_names)}) must match num_terms ({num_terms})")
    
    # Compute centers based on method
    if method == "uniform":
        centers = _compute_uniform_centers(data.min(), data.max(), num_terms, **kwargs)
    elif method == "percentile":
        centers = _compute_percentile_centers(data, num_terms)
    elif method == "clustering":
        centers = _compute_clustering_centers(data, num_terms, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return create_strong_fuzzy_partition(
        centers=centers,
        term_names=term_names,
        name=name,
        universe_min=data.min(),
        universe_max=data.max(),
    )


def create_partitions_for_dataset(
    X: np.ndarray,
    num_terms: int = 5,
    method: Literal["uniform", "percentile", "clustering"] = "percentile",
    feature_names: Optional[List[str]] = None,
) -> List[FuzzyPartition]:
    """
    Create fuzzy partitions for all features in a dataset.
    
    Args:
        X: Data matrix [N, D]
        num_terms: Number of fuzzy terms per variable
        method: Partitioning method
        feature_names: Optional names for each feature
        
    Returns:
        List of FuzzyPartition objects, one per feature
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_samples, n_features = X.shape
    
    if feature_names is None:
        feature_names = [f"X{i}" for i in range(n_features)]
    
    partitions = []
    for i in range(n_features):
        partition = create_partition(
            data=X[:, i],
            num_terms=num_terms,
            method=method,
            name=feature_names[i],
        )
        partitions.append(partition)
    
    return partitions


def compute_membership_matrix(
    X: np.ndarray,
    partitions: List[FuzzyPartition],
) -> torch.Tensor:
    """
    Compute membership degrees for all samples, features, and terms.
    
    Args:
        X: Data matrix [N, D]
        partitions: List of partitions (one per feature)
        
    Returns:
        Membership tensor [N, D, T] where T is num_terms
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_samples, n_features = X.shape
    
    if len(partitions) != n_features:
        raise ValueError(f"Number of partitions ({len(partitions)}) must match features ({n_features})")
    
    num_terms = partitions[0].num_terms
    
    # Pre-allocate tensor
    memberships = torch.zeros(n_samples, n_features, num_terms)
    
    for d, partition in enumerate(partitions):
        memberships[:, d, :] = partition.fuzzify(X[:, d])
    
    return memberships


def verify_strong_partition(partition: FuzzyPartition, n_points: int = 100) -> Tuple[bool, float]:
    """
    Verify that a partition satisfies strong fuzzy partition property.
    
    Args:
        partition: Partition to verify
        n_points: Number of test points
        
    Returns:
        (is_valid, max_deviation) tuple
    """
    # Test points across universe
    x = np.linspace(partition.universe_min, partition.universe_max, n_points)
    memberships = partition.fuzzify(x)
    
    # Sum should be 1 for strong partition
    sums = memberships.sum(dim=-1)
    max_deviation = (sums - 1.0).abs().max().item()
    
    is_valid = max_deviation < 0.01  # Allow small numerical errors
    
    return is_valid, max_deviation
