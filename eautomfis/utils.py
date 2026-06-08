"""
Utility functions for e-AutoMFIS.

Contains:
- Reproducibility utilities (seed setting)
- Timing utilities (profiling context manager)
- Metrics (MAE, RMSE, etc.)
- Logging helpers
"""

import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from functools import wraps

import numpy as np
import torch


# =============================================================================
# Reproducibility
# =============================================================================

def set_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: str = "auto") -> torch.device:
    """
    Get the appropriate torch device.
    
    Args:
        device: 'auto', 'cpu', or 'cuda'
        
    Returns:
        torch.device object
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# =============================================================================
# Timing Utilities
# =============================================================================

@dataclass
class TimingStats:
    """Container for timing statistics."""
    name: str
    elapsed_seconds: float
    start_time: float
    end_time: float


class Timer:
    """
    Context manager and decorator for timing code execution.
    
    Usage as context manager:
        with Timer("my_operation") as t:
            do_something()
        print(t.elapsed)
    
    Usage as decorator:
        @Timer.decorator("function_name")
        def my_function():
            pass
    """
    
    _timings: Dict[str, List[float]] = {}
    
    def __init__(self, name: str = "timer", verbose: bool = False):
        self.name = name
        self.verbose = verbose
        self.start_time: float = 0
        self.end_time: float = 0
        self.elapsed: float = 0
    
    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args) -> None:
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        
        # Store timing
        if self.name not in Timer._timings:
            Timer._timings[self.name] = []
        Timer._timings[self.name].append(self.elapsed)
        
        if self.verbose:
            print(f"[{self.name}] Elapsed: {self.elapsed:.4f}s")
    
    @classmethod
    def decorator(cls, name: str = None, verbose: bool = False):
        """Decorator for timing function execution."""
        def wrapper(func: Callable) -> Callable:
            timer_name = name or func.__name__
            
            @wraps(func)
            def wrapped(*args, **kwargs):
                with cls(timer_name, verbose):
                    return func(*args, **kwargs)
            return wrapped
        return wrapper
    
    @classmethod
    def get_stats(cls, name: str) -> Optional[Dict[str, float]]:
        """Get timing statistics for a named timer."""
        if name not in cls._timings or not cls._timings[name]:
            return None
        
        times = cls._timings[name]
        return {
            "count": len(times),
            "total": sum(times),
            "mean": np.mean(times),
            "std": np.std(times),
            "min": min(times),
            "max": max(times),
        }
    
    @classmethod
    def get_all_stats(cls) -> Dict[str, Dict[str, float]]:
        """Get timing statistics for all timers."""
        return {name: cls.get_stats(name) for name in cls._timings}
    
    @classmethod
    def reset(cls) -> None:
        """Reset all timing statistics."""
        cls._timings.clear()
    
    @classmethod
    def report(cls) -> str:
        """Generate a formatted timing report."""
        lines = ["=" * 60, "Timing Report", "=" * 60]
        
        for name, stats in cls.get_all_stats().items():
            if stats:
                lines.append(
                    f"{name:30s} | count={stats['count']:4d} | "
                    f"total={stats['total']:8.3f}s | mean={stats['mean']:8.4f}s"
                )
        
        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# Metrics
# =============================================================================

def mae(y_true: np.ndarray, y_pred: np.ndarray, axis: int = None) -> np.ndarray:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred), axis=axis)


def rmse(y_true: np.ndarray, y_pred: np.ndarray, axis: int = None) -> np.ndarray:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=axis))


def mape(y_true: np.ndarray, y_pred: np.ndarray, axis: int = None, epsilon: float = 1e-8) -> np.ndarray:
    """Mean Absolute Percentage Error."""
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon)), axis=axis) * 100


def smape(y_true: np.ndarray, y_pred: np.ndarray, axis: int = None) -> np.ndarray:
    """Symmetric Mean Absolute Percentage Error."""
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / (denominator + 1e-8), axis=axis) * 100


@dataclass 
class ForecastMetrics:
    """Container for forecast evaluation metrics."""
    mae: float
    rmse: float
    mape: float = None
    smape: float = None
    
    # Per-horizon metrics (for multi-step)
    mae_per_horizon: np.ndarray = None
    rmse_per_horizon: np.ndarray = None
    
    # Model complexity metrics
    num_rules: int = None
    avg_antecedents: float = None
    
    # Timing
    train_time_seconds: float = None
    inference_time_seconds: float = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary (excluding None values)."""
        result = {}
        for k, v in self.__dict__.items():
            if v is not None:
                if isinstance(v, np.ndarray):
                    result[k] = v.tolist()
                else:
                    result[k] = v
        return result
    
    def __repr__(self) -> str:
        lines = ["ForecastMetrics:"]
        lines.append(f"  MAE:  {self.mae:.4f}")
        lines.append(f"  RMSE: {self.rmse:.4f}")
        if self.mape is not None:
            lines.append(f"  MAPE: {self.mape:.2f}%")
        if self.num_rules is not None:
            lines.append(f"  Rules: {self.num_rules}")
        if self.train_time_seconds is not None:
            lines.append(f"  Train time: {self.train_time_seconds:.2f}s")
        return "\n".join(lines)


def compute_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    num_rules: int = None,
    avg_antecedents: float = None,
    train_time: float = None,
    inference_time: float = None,
) -> ForecastMetrics:
    """
    Compute comprehensive forecast metrics.
    
    Args:
        y_true: Ground truth values [N] or [N, H] for multi-horizon
        y_pred: Predicted values [N] or [N, H]
        num_rules: Number of rules in the model
        avg_antecedents: Average antecedents per rule
        train_time: Training time in seconds
        inference_time: Inference time in seconds
        
    Returns:
        ForecastMetrics object
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    metrics = ForecastMetrics(
        mae=mae(y_true, y_pred),
        rmse=rmse(y_true, y_pred),
        mape=mape(y_true, y_pred),
        smape=smape(y_true, y_pred),
        num_rules=num_rules,
        avg_antecedents=avg_antecedents,
        train_time_seconds=train_time,
        inference_time_seconds=inference_time,
    )
    
    # Per-horizon metrics for multi-step forecasting
    if y_true.ndim == 2 and y_true.shape[1] > 1:
        metrics.mae_per_horizon = mae(y_true, y_pred, axis=0)
        metrics.rmse_per_horizon = rmse(y_true, y_pred, axis=0)
    
    return metrics


# =============================================================================
# Logging
# =============================================================================

class Logger:
    """Simple logger with verbosity control."""
    
    def __init__(self, verbose: int = 1, name: str = "e-AutoMFIS"):
        self.verbose = verbose
        self.name = name
    
    def info(self, msg: str, level: int = 1) -> None:
        """Log info message if verbosity >= level."""
        if self.verbose >= level:
            print(f"[{self.name}] {msg}")
    
    def debug(self, msg: str) -> None:
        """Log debug message (requires verbose >= 2)."""
        self.info(msg, level=2)
    
    def progress(self, current: int, total: int, prefix: str = "") -> None:
        """Display progress bar."""
        if self.verbose >= 1:
            pct = current / total * 100
            bar_len = 30
            filled = int(bar_len * current / total)
            bar = "#" * filled + "-" * (bar_len - filled)
            print(f"\r[{self.name}] {prefix} |{bar}| {pct:.1f}%", end="", flush=True)
            if current == total:
                print()  # Newline when complete
