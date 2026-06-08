"""
Comprehensive test for e-AutoMFIS.

Tests:
1. Basic pipeline on synthetic data
2. Reproducibility (same seed = same result)
3. Multi-member ensemble
4. TSK-0 and TSK-1 consequents
"""

import numpy as np
import torch
from eautomfis import EAutoMFIS, EAutoMFISConfig
from eautomfis.utils import set_seeds, Timer, compute_metrics


def generate_synthetic_data(n_samples: int = 200, n_features: int = 5, seed: int = 42):
    """Generate synthetic AR-like data."""
    np.random.seed(seed)
    
    # Features
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Target: linear combination + noise
    y = 0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2] + 0.1 * np.random.randn(n_samples)
    y = y.astype(np.float32)
    
    return X, y


def test_basic_pipeline():
    """Test basic training and prediction pipeline."""
    print("\n" + "=" * 60)
    print("Test 1: Basic Pipeline")
    print("=" * 60)
    
    # Generate data
    X, y = generate_synthetic_data(n_samples=150, n_features=5)
    X_train, X_test = X[:100], X[100:]
    y_train, y_test = y[:100], y[100:]
    
    # Create and train model
    config = EAutoMFISConfig(
        num_members=3,
        num_terms=3,
        tsk_order=0,
        min_support=0.05,
        max_antecedents=2,
        top_k_rules=20,
        random_seed=42,
    )
    
    model = EAutoMFIS(config, n_jobs=1, verbose=True)
    
    with Timer("training", verbose=True):
        model.fit(X_train, y_train)
    
    print(model.summary())
    
    # Predict
    y_pred = model.predict(X_test)
    print(f"Prediction shape: {y_pred.shape}")
    
    # Metrics
    metrics = model.score(X_test, y_test)
    print(f"\nTest Metrics:")
    print(metrics)
    
    # Rules
    rules = model.get_rules()
    print(f"\nSample Rules ({len(rules)} total):")
    for rule in rules[:5]:
        print(f"  {rule}")
    
    assert len(y_pred) == len(y_test), "Prediction length mismatch"
    assert metrics.mae < 1.0, f"MAE too high: {metrics.mae}"
    
    print("[OK] Basic pipeline test PASSED")
    return True


def test_reproducibility():
    """Test that same seed produces same results."""
    print("\n" + "=" * 60)
    print("Test 2: Reproducibility")
    print("=" * 60)
    
    X, y = generate_synthetic_data(n_samples=100, n_features=4)
    
    config = EAutoMFISConfig(
        num_members=2,
        num_terms=3,
        tsk_order=0,
        random_seed=12345,
    )
    
    # Train twice with same seed
    model1 = EAutoMFIS(config, verbose=False)
    model1.fit(X, y)
    pred1 = model1.predict(X)
    
    model2 = EAutoMFIS(config, verbose=False)
    model2.fit(X, y)
    pred2 = model2.predict(X)
    
    # Check equality
    max_diff = np.abs(pred1 - pred2).max()
    print(f"Max prediction difference: {max_diff:.10f}")
    
    assert np.allclose(pred1, pred2, atol=1e-6), f"Predictions differ: max_diff={max_diff}"
    
    print("[OK] Reproducibility test PASSED")
    return True


def test_ensemble():
    """Test ensemble with multiple members."""
    print("\n" + "=" * 60)
    print("Test 3: Ensemble")
    print("=" * 60)
    
    X, y = generate_synthetic_data(n_samples=200, n_features=6)
    X_train, X_val, X_test = X[:120], X[120:160], X[160:]
    y_train, y_val, y_test = y[:120], y[120:160], y[160:]
    
    config = EAutoMFISConfig(
        num_members=5,
        feature_subset_size=4,
        num_terms=3,
        tsk_order=0,
        min_support=0.05,
        max_antecedents=2,
        random_seed=42,
    )
    
    model = EAutoMFIS(config, verbose=True)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    
    print(f"\nMember weights: {model.member_weights_}")
    print(f"Member rule counts: {[m.num_rules for m in model.members_]}")
    
    # Predict with uncertainty
    y_pred, y_std = model.predict(X_test, return_std=True)
    print(f"\nPrediction std range: [{y_std.min():.4f}, {y_std.max():.4f}]")
    
    metrics = model.score(X_test, y_test)
    print(f"Test MAE: {metrics.mae:.4f}")
    
    assert len(model.members_) == 5, "Wrong number of members"
    
    print("[OK] Ensemble test PASSED")
    return True


def test_tsk_orders():
    """Test different TSK orders."""
    print("\n" + "=" * 60)
    print("Test 4: TSK Orders (0 and 1)")
    print("=" * 60)
    
    X, y = generate_synthetic_data(n_samples=150, n_features=5)
    X_train, X_test = X[:100], X[100:]
    y_train, y_test = y[:100], y[100:]
    
    results = {}
    
    for order in [0, 1]:
        config = EAutoMFISConfig(
            num_members=2,
            num_terms=3,
            tsk_order=order,
            tsk_regularization=1e-3,
            random_seed=42,
        )
        
        model = EAutoMFIS(config, verbose=False)
        model.fit(X_train, y_train)
        
        metrics = model.score(X_test, y_test)
        results[order] = metrics.mae
        
        print(f"TSK-{order}: MAE = {metrics.mae:.4f}, Rules = {len(model.rules_)}")
    
    print("[OK] TSK orders test PASSED")
    return True


def test_explainability():
    """Test explanation functionality."""
    print("\n" + "=" * 60)
    print("Test 5: Explainability")
    print("=" * 60)
    
    X, y = generate_synthetic_data(n_samples=100, n_features=4)
    
    config = EAutoMFISConfig(
        num_members=2,
        num_terms=3,
        tsk_order=0,
        random_seed=42,
    )
    
    model = EAutoMFIS(config, verbose=False)
    model.fit(X, y)
    
    # Explain single prediction
    explanation = model.explain(X[0], top_k=3)
    
    print(f"Prediction: {explanation['prediction']:.4f}")
    print(f"Member contributions: {len(explanation['member_contributions'])}")
    
    for contrib in explanation['member_contributions']:
        print(f"  Member {contrib['member_id']}: weight={contrib['member_weight']:.3f}, "
              f"pred={contrib['prediction']:.4f}")
        if 'top_rules' in contrib and contrib['top_rules']:
            print(f"    Top rule: {contrib['top_rules'][0]['premise']}")
    
    assert 'prediction' in explanation
    assert 'member_contributions' in explanation
    
    print("[OK] Explainability test PASSED")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("e-AutoMFIS Test Suite")
    print("=" * 60)
    
    tests = [
        ("Basic Pipeline", test_basic_pipeline),
        ("Reproducibility", test_reproducibility),
        ("Ensemble", test_ensemble),
        ("TSK Orders", test_tsk_orders),
        ("Explainability", test_explainability),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n[X] {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, p in results:
        status = "[OK] PASSED" if p else "[X] FAILED"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
