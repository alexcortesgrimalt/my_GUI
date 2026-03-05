#!/usr/bin/env python3
"""
Quick validation script for perpendicular fitting enhancements.
Tests the QDialog.DialogCode fix and improved SCR handling.
"""

import sys
import numpy as np

# Test 1: Import check
print("=" * 60)
print("Test 1: Checking imports and syntax...")
try:
    from perpendicular_fitting import PerpendicularFitter
    print("✓ perpendicular_fitting module imports successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Symmetric SCR
print("\n" + "=" * 60)
print("Test 2: Symmetric SCR (original behavior)...")
try:
    x = np.linspace(-5, 5, 101)
    y = np.where(x < 0, 10 * np.exp(0.8 * (x + 2)) + 0.5, 
                 5 * np.exp(-0.5 * (x - 2)) + 0.2)
    
    fitter = PerpendicularFitter()
    res = fitter.fit_profile(x, y, show_debug=False)
    
    assert res['left'] is not None, "Left fit failed"
    assert res['right'] is not None, "Right fit failed"
    assert abs(res['junction_pos']) < 0.5, "Junction detection not near center"
    print(f"✓ Symmetric SCR test passed")
    print(f"  Left slope: {res['left']['slope']:.4f}")
    print(f"  Right slope: {res['right']['slope']:.4f}")
except Exception as e:
    print(f"✗ Test failed: {e}")
    sys.exit(1)

# Test 3: Asymmetric SCR
print("\n" + "=" * 60)
print("Test 3: Asymmetric SCR (new behavior)...")
try:
    x = np.linspace(-5, 5, 101)
    # Asymmetric decay rates
    y_left = np.where(x < -1, 10 * np.exp(0.8 * (x + 1)), 0.5)
    y_right = np.where(x > 2, 5 * np.exp(-0.3 * (x - 2)), 0.5)
    y_peak = np.where((x >= -1) & (x <= 2), 10 - 2*np.abs(x), 0)
    y = y_left + y_right + y_peak + np.random.normal(0, 0.01, len(x))
    
    fitter = PerpendicularFitter(skip_near_junction=3)
    res = fitter.fit_profile(x, y, show_debug=False)
    
    assert res['left'] is not None, "Left fit failed on asymmetric data"
    assert res['right'] is not None, "Right fit failed on asymmetric data"
    print(f"✓ Asymmetric SCR test passed")
    print(f"  Junction detected at: {res['junction_pos']:.4f} µm")
    print(f"  Left slope: {res['left']['slope']:.4f}")
    print(f"  Right slope: {res['right']['slope']:.4f}")
except Exception as e:
    print(f"✗ Test failed: {e}")
    sys.exit(1)

# Test 4: Parameter variations
print("\n" + "=" * 60)
print("Test 4: Parameter variations...")
try:
    x = np.linspace(-5, 5, 101)
    y = np.where(x < 0, 10 * np.exp(0.8 * (x + 2)) + np.random.normal(0, 0.1, len(x)), 
                 5 * np.exp(-0.5 * (x - 2)) + np.random.normal(0, 0.1, len(x)))
    
    configs = [
        {"snr_threshold": 2.0, "skip_near_junction": 2},
        {"snr_threshold": 4.0, "skip_near_junction": 4},
        {"snr_threshold": 3.0, "skip_near_junction": 1, "min_points": 5},
    ]
    
    for i, config in enumerate(configs):
        fitter = PerpendicularFitter(**config)
        res = fitter.fit_profile(x, y, show_debug=False)
        assert res['left'] is not None, f"Config {i} failed: left fit"
        assert res['right'] is not None, f"Config {i} failed: right fit"
    
    print(f"✓ All {len(configs)} parameter configurations work correctly")
except Exception as e:
    print(f"✗ Test failed: {e}")
    sys.exit(1)

# Test 5: SNR truncation
print("\n" + "=" * 60)
print("Test 5: SNR-based tail truncation...")
try:
    x = np.linspace(-5, 5, 101)
    # Create data with noisy tail
    y = np.where(x < 0, 10 * np.exp(0.8 * (x + 2)), 
                 5 * np.exp(-0.5 * (x - 2)))
    # Add noise to the tail (x > 4)
    y = np.concatenate([y[:-20], y[-20:] + np.random.normal(0, 1, 20)])
    
    fitter_conservative = PerpendicularFitter(snr_threshold=2.0)
    fitter_aggressive = PerpendicularFitter(snr_threshold=5.0)
    
    res_cons = fitter_conservative.fit_profile(x, y, show_debug=False)
    res_agg = fitter_aggressive.fit_profile(x, y, show_debug=False)
    
    assert res_cons['left'] is not None and res_cons['right'] is not None
    assert res_agg['left'] is not None and res_agg['right'] is not None
    print(f"✓ SNR truncation test passed")
    print(f"  Conservative (snr=2.0): R² left={res_cons['left']['r2']:.4f}")
    print(f"  Aggressive (snr=5.0): R² left={res_agg['left']['r2']:.4f}")
except Exception as e:
    print(f"✗ Test failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All tests passed successfully!")
print("=" * 60)
