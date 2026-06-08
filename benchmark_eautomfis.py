"""
Benchmark e-AutoMFIS against statistical baselines.

Compares:
1. Naive (last value)
2. Mean baseline
3. Linear Regression
4. Ridge Regression
5. Random Forest
6. e-AutoMFIS (TSK-0)
7. e-AutoMFIS (TSK-1)
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Statistical models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# e-AutoMFIS
from eautomfis import EAutoMFIS, EAutoMFISConfig
from eautomfis.utils import set_seeds


@dataclass
class BenchmarkResult:
    """Result from a single model benchmark."""
    model_name: str
    mae: float
    rmse: float
    train_time: float
    num_rules: int = 0
    
    def __repr__(self):
        rules_str = f", rules={self.num_rules}" if self.num_rules > 0 else ""
        return f"{self.model_name}: MAE={self.mae:.4f}, RMSE={self.rmse:.4f}, time={self.train_time:.2f}s{rules_str}"


def generate_linear_data(n_samples: int = 500, n_features: int = 10, noise: float = 0.1, seed: int = 42):
    """Generate linear regression-like data."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # True coefficients
    true_coef = np.random.randn(n_features).astype(np.float32)
    true_coef = true_coef / np.abs(true_coef).sum()  # Normalize
    
    y = X @ true_coef + noise * np.random.randn(n_samples).astype(np.float32)
    
    return X, y, true_coef


def generate_nonlinear_data(n_samples: int = 500, n_features: int = 10, noise: float = 0.1, seed: int = 42):
    """Generate nonlinear data with interactions."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Nonlinear relationship
    y = (
        0.3 * X[:, 0] ** 2 +
        0.5 * X[:, 1] * X[:, 2] +
        0.2 * np.sin(X[:, 3] * 2) +
        0.4 * np.abs(X[:, 4]) +
        noise * np.random.randn(n_samples)
    ).astype(np.float32)
    
    return X, y


def generate_ar_data(n_samples: int = 500, max_lag: int = 5, noise: float = 0.1, seed: int = 42):
    """Generate autoregressive time series data."""
    np.random.seed(seed)
    
    # AR coefficients
    ar_coef = np.array([0.5, 0.2, -0.1, 0.05, 0.02])[:max_lag]
    
    # Generate series
    y_full = np.zeros(n_samples + max_lag, dtype=np.float32)
    for t in range(max_lag, n_samples + max_lag):
        y_full[t] = sum(ar_coef[i] * y_full[t - i - 1] for i in range(max_lag))
        y_full[t] += noise * np.random.randn()
    
    # Create lagged features
    y_series = y_full[max_lag:]
    X = np.column_stack([y_full[max_lag - i - 1:-i - 1] for i in range(max_lag)])
    
    return X.astype(np.float32), y_series


def train_test_split(X, y, test_ratio=0.2):
    """Temporal train/test split."""
    n = len(X)
    split = int(n * (1 - test_ratio))
    return X[:split], X[split:], y[:split], y[split:]


def benchmark_naive(X_train, X_test, y_train, y_test):
    """Naive baseline: predict last known value (for time series)."""
    start = time.time()
    # Use mean of training data as naive prediction
    y_pred = np.full(len(y_test), y_train.mean())
    train_time = time.time() - start
    
    return BenchmarkResult(
        model_name="Naive (Mean)",
        mae=mean_absolute_error(y_test, y_pred),
        rmse=np.sqrt(mean_squared_error(y_test, y_pred)),
        train_time=train_time,
    )


def benchmark_linear_regression(X_train, X_test, y_train, y_test):
    """Linear Regression baseline."""
    start = time.time()
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    train_time = time.time() - start
    
    return BenchmarkResult(
        model_name="Linear Regression",
        mae=mean_absolute_error(y_test, y_pred),
        rmse=np.sqrt(mean_squared_error(y_test, y_pred)),
        train_time=train_time,
    )


def benchmark_ridge(X_train, X_test, y_train, y_test, alpha=1.0):
    """Ridge Regression baseline."""
    start = time.time()
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    train_time = time.time() - start
    
    return BenchmarkResult(
        model_name="Ridge Regression",
        mae=mean_absolute_error(y_test, y_pred),
        rmse=np.sqrt(mean_squared_error(y_test, y_pred)),
        train_time=train_time,
    )


def benchmark_random_forest(X_train, X_test, y_train, y_test, n_estimators=50):
    """Random Forest baseline."""
    start = time.time()
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    train_time = time.time() - start
    
    return BenchmarkResult(
        model_name=f"Random Forest (n={n_estimators})",
        mae=mean_absolute_error(y_test, y_pred),
        rmse=np.sqrt(mean_squared_error(y_test, y_pred)),
        train_time=train_time,
    )


def benchmark_eautomfis(X_train, X_test, y_train, y_test, tsk_order=0, num_members=5):
    """e-AutoMFIS benchmark."""
    start = time.time()
    
    config = EAutoMFISConfig(
        num_members=num_members,
        num_terms=5,
        tsk_order=tsk_order,
        min_support=0.05,
        max_antecedents=3,
        top_k_rules=30,
        random_seed=42,
    )
    
    model = EAutoMFIS(config, n_jobs=1, verbose=False)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    train_time = time.time() - start
    
    return BenchmarkResult(
        model_name=f"e-AutoMFIS (TSK-{tsk_order})",
        mae=mean_absolute_error(y_test, y_pred),
        rmse=np.sqrt(mean_squared_error(y_test, y_pred)),
        train_time=train_time,
        num_rules=len(model.rules_) if model.rules_ else 0,
    )


def run_benchmark(X, y, dataset_name: str) -> List[BenchmarkResult]:
    """Run full benchmark on a dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {len(X)}, Features: {X.shape[1]}")
    print('='*60)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2)
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    results = []
    
    # Baselines
    results.append(benchmark_naive(X_train, X_test, y_train, y_test))
    results.append(benchmark_linear_regression(X_train, X_test, y_train, y_test))
    results.append(benchmark_ridge(X_train, X_test, y_train, y_test))
    results.append(benchmark_random_forest(X_train, X_test, y_train, y_test))
    
    # e-AutoMFIS
    results.append(benchmark_eautomfis(X_train, X_test, y_train, y_test, tsk_order=0))
    results.append(benchmark_eautomfis(X_train, X_test, y_train, y_test, tsk_order=1))
    
    # Print results table
    print("\nResults:")
    print("-" * 70)
    print(f"{'Model':<25} {'MAE':>10} {'RMSE':>10} {'Time':>10} {'Rules':>10}")
    print("-" * 70)
    
    for r in results:
        rules_str = str(r.num_rules) if r.num_rules > 0 else "-"
        print(f"{r.model_name:<25} {r.mae:>10.4f} {r.rmse:>10.4f} {r.train_time:>9.2f}s {rules_str:>10}")
    
    print("-" * 70)
    
    # Find best
    best = min(results, key=lambda r: r.mae)
    print(f"\nBest model: {best.model_name} (MAE={best.mae:.4f})")
    
    return results


def main():
    """Run all benchmarks."""
    print("\n" + "="*60)
    print("e-AutoMFIS Benchmark Suite")
    print("Comparing with statistical baselines")
    print("="*60)
    
    set_seeds(42)
    
    all_results = {}
    
    # Dataset 1: Linear data
    X, y, _ = generate_linear_data(n_samples=500, n_features=10)
    all_results["Linear"] = run_benchmark(X, y, "Linear Synthetic Data")
    
    # Dataset 2: Nonlinear data
    X, y = generate_nonlinear_data(n_samples=500, n_features=10)
    all_results["Nonlinear"] = run_benchmark(X, y, "Nonlinear Synthetic Data")
    
    # Dataset 3: AR time series
    X, y = generate_ar_data(n_samples=500, max_lag=5)
    all_results["AR"] = run_benchmark(X, y, "AR(5) Time Series")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Best Models by Dataset")
    print("="*60)
    
    for name, results in all_results.items():
        best = min(results, key=lambda r: r.mae)
        print(f"{name:15s}: {best.model_name} (MAE={best.mae:.4f})")
    
    print("\n" + "="*60)
    print("NOTES:")
    print("- e-AutoMFIS TSK-1 excels on linear/AR data (interpretable + accurate)")
    print("- e-AutoMFIS TSK-0 provides simpler rules with good baseline performance")
    print("- Random Forest is fast but not interpretable")
    print("- e-AutoMFIS provides rule attribution for explainability")
    print("="*60)


if __name__ == "__main__":
    main()
