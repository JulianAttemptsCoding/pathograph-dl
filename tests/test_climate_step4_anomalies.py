"""Unit test for Climate Step 4 anomaly computation."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest


def test_climate_step4_anomalies_synthetic():
    """Test Climate Step 4 with synthetic data to verify climatology and anomaly computation."""
    
    try:
        import zarr  # type: ignore
    except ImportError:
        pytest.skip("zarr not available")
    
    try:
        import yaml  # type: ignore
    except ImportError:
        pytest.skip("pyyaml not available")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create synthetic climate tensor
        # Small test: N=3, F=2, time covering baseline years 1991-1992 (24 months)
        # month_index for 1991-01 is (1991-1950)*12 = 492
        # month_index for 1992-12 is (1992-1950)*12 + 11 = 515
        
        N = 3
        F = 2
        baseline_start_mi = 492  # 1991-01
        baseline_end_mi = 515    # 1992-12
        
        # Add some months before and after baseline for testing
        time_indices = list(range(490, 518))  # 1990-11 to 1993-01
        T = len(time_indices)
        time_index = np.array(time_indices, dtype=np.int32)
        
        # Create known values
        # For each (month_of_year, n, f), set a baseline pattern
        # Month 0 (Jan): node 0 feature 0 = 10.0, node 1 feature 0 = 20.0, etc.
        # We'll create a simple pattern
        
        climate = np.full((T, N, F), np.nan, dtype=np.float32)
        mask = np.zeros((T, N, F), dtype=np.uint8)
        
        # Fill baseline period (1991-1992) with known values
        for t, mi in enumerate(time_indices):
            year = 1950 + (mi // 12)
            month = (mi % 12) + 1
            
            if 1991 <= year <= 1992:
                # Set known baseline values
                # Use a simple formula: value = 100 + month + n*10 + f
                for n in range(N):
                    for f in range(F):
                        val = 100.0 + month + n*10 + f
                        climate[t, n, f] = val
                        mask[t, n, f] = 1
        
        # Also fill some non-baseline values
        for t, mi in enumerate(time_indices):
            year = 1950 + (mi // 12)
            if year == 1990 or year == 1993:
                month = (mi % 12) + 1
                for n in range(N):
                    for f in range(F):
                        # Add +5 to baseline mean for testing anomalies
                        val = 100.0 + month + n*10 + f + 5.0
                        climate[t, n, f] = val
                        mask[t, n, f] = 1
        
        # Write input Zarr
        input_zarr = tmp_path / "climate_input.zarr"
        g_in = zarr.open_group(str(input_zarr), mode="w")
        g_in.create_array("climate", data=climate, chunks=(10, N, F))
        g_in.create_array("mask", data=mask, chunks=(10, N, F))
        g_in.create_array("time_index", data=time_index, chunks=(T,))
        
        # Write time_index_master
        time_index_master = tmp_path / "time_index_master.npy"
        np.save(time_index_master, time_index)
        
        # Create config
        config_path = tmp_path / "step4_config.yaml"
        config = {
            "baseline": {
                "year_start": 1991,
                "year_end": 1992,
            },
            "compute_zscore": True,
            "paths": {
                "time_index_master": str(time_index_master).replace("\\", "/"),
                "climate_tensor_zarr": str(input_zarr).replace("\\", "/"),
                "output_dir": str(tmp_path / "output").replace("\\", "/"),
                "output_zarr": "anomalies.zarr",
            },
            "arrays": {
                "anomaly": "anomaly",
                "zscore": "zscore",
                "mask": "mask",
                "time_index": "time_index",
                "climatology_mean": "climo_mean",
                "climatology_std": "climo_std",
                "climatology_count": "climo_count",
            },
            "chunking": {
                "time": 10,
                "nodes": N,
            },
        }
        
        with config_path.open("w", encoding="utf-8") as f:
            yaml.dump(config, f)
        
        # Run Step 4 script via subprocess
        script_path = Path("tools/climate_step4_compute_anomalies.py")
        if not script_path.exists():
            pytest.skip(f"Script not found: {script_path}")
        
        result = subprocess.run(
            [sys.executable, str(script_path), "--config", str(config_path)],
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            pytest.fail(f"Step 4 script failed with exit code {result.returncode}")
        
        # Read output
        output_zarr = tmp_path / "output" / "anomalies.zarr"
        assert output_zarr.exists(), "Output Zarr not created"
        
        g_out = zarr.open_group(str(output_zarr), mode="r")
        
        anomaly = np.asarray(g_out["anomaly"][:])
        zscore = np.asarray(g_out["zscore"][:])
        anom_mask = np.asarray(g_out["mask"][:])
        climo_mean = np.asarray(g_out["climo_mean"][:])
        climo_count = np.asarray(g_out["climo_count"][:])
        
        # Verify climatology
        # For each month in baseline (1991-1992), we have 2 samples per month
        # (one from 1991, one from 1992)
        # Mean should equal the baseline value: 100 + month + n*10 + f
        
        for month_idx in range(12):
            month = month_idx + 1
            for n in range(N):
                for f in range(F):
                    expected_mean = 100.0 + month + n*10 + f
                    count = climo_count[month_idx, n, f]
                    
                    # We filled all months for 1991-1992
                    assert count == 2, f"Expected count=2 for month={month}, n={n}, f={f}, got {count}"
                    
                    actual_mean = climo_mean[month_idx, n, f]
                    assert np.isclose(actual_mean, expected_mean, atol=1e-5), \
                        f"Mean mismatch for month={month}, n={n}, f={f}: expected {expected_mean}, got {actual_mean}"
        
        # Verify anomalies for non-baseline values
        # For 1990 and 1993, we added +5 to the baseline mean
        # So anomalies should be +5
        
        for t, mi in enumerate(time_indices):
            year = 1950 + (mi // 12)
            month = (mi % 12) + 1
            month_idx = month - 1
            
            if year == 1990 or year == 1993:
                for n in range(N):
                    for f in range(F):
                        if anom_mask[t, n, f] == 1:
                            anom_val = anomaly[t, n, f]
                            # Expected anomaly is +5
                            assert np.isclose(anom_val, 5.0, atol=1e-5), \
                                f"Anomaly mismatch at t={t}, n={n}, f={f}: expected 5.0, got {anom_val}"
        
        # Verify z-score masking when std == 0
        # Since all baseline values are identical for each (month, n, f), std should be 0
        # So z-scores should be masked (NaN)
        
        # Actually, we have 2 samples with identical values, so std = 0
        # Z-scores should be masked where std == 0
        
        zscore_mask = np.asarray(g_out["zscore_mask"][:])
        
        # Since std is 0 for all (month, n, f), all z-scores should be masked
        assert np.all(zscore_mask == 0), "Expected all z-scores to be masked when std=0"
        
        print("[OK] Synthetic anomaly test passed")


def test_climate_step4_with_variance():
    """Test with variance to verify z-score computation."""
    
    try:
        import zarr  # type: ignore
    except ImportError:
        pytest.skip("zarr not available")
    
    try:
        import yaml  # type: ignore
    except ImportError:
        pytest.skip("pyyaml not available")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        N = 2
        F = 1
        
        # 3 years of baseline: 1991-1993 (36 months)
        baseline_start_mi = 492  # 1991-01
        baseline_end_mi = 527    # 1993-12
        
        time_indices = list(range(492, 540))  # 1991-01 to 1994-11
        T = len(time_indices)
        time_index = np.array(time_indices, dtype=np.int32)
        
        climate = np.full((T, N, F), np.nan, dtype=np.float32)
        mask = np.zeros((T, N, F), dtype=np.uint8)
        
        # Fill baseline with varying values to create std > 0
        for t, mi in enumerate(time_indices):
            year = 1950 + (mi // 12)
            month = (mi % 12) + 1
            
            if 1991 <= year <= 1993:
                for n in range(N):
                    for f in range(F):
                        # Add year-based variation
                        val = 100.0 + month + n*10 + f + (year - 1991) * 2.0
                        climate[t, n, f] = val
                        mask[t, n, f] = 1
            elif year == 1994:
                # Set to baseline mean + 3*std for testing
                for n in range(N):
                    for f in range(F):
                        # Mean over 1991-1993 for this (month, n, f):
                        # 100 + month + n*10 + f + (0+2+4)/3 = 100 + month + n*10 + f + 2
                        mean_val = 100.0 + month + n*10 + f + 2.0
                        # Std of [0, 2, 4] is 1.633 (ddof=0)
                        std_val = np.std([0.0, 2.0, 4.0], ddof=0)
                        val = mean_val + 3.0 * std_val
                        climate[t, n, f] = val
                        mask[t, n, f] = 1
        
        # Write input Zarr
        input_zarr = tmp_path / "climate_input.zarr"
        g_in = zarr.open_group(str(input_zarr), mode="w")
        g_in.create_array("climate", data=climate, chunks=(10, N, F))
        g_in.create_array("mask", data=mask, chunks=(10, N, F))
        g_in.create_array("time_index", data=time_index, chunks=(T,))
        
        time_index_master = tmp_path / "time_index_master.npy"
        np.save(time_index_master, time_index)
        
        config_path = tmp_path / "step4_config.yaml"
        config = {
            "baseline": {
                "year_start": 1991,
                "year_end": 1993,
            },
            "compute_zscore": True,
            "paths": {
                "time_index_master": str(time_index_master).replace("\\", "/"),
                "climate_tensor_zarr": str(input_zarr).replace("\\", "/"),
                "output_dir": str(tmp_path / "output").replace("\\", "/"),
                "output_zarr": "anomalies.zarr",
            },
            "arrays": {
                "anomaly": "anomaly",
                "zscore": "zscore",
                "mask": "mask",
                "time_index": "time_index",
                "climatology_mean": "climo_mean",
                "climatology_std": "climo_std",
                "climatology_count": "climo_count",
            },
            "chunking": {
                "time": 10,
                "nodes": N,
            },
        }
        
        with config_path.open("w", encoding="utf-8") as f:
            yaml.dump(config, f)
        
        script_path = Path("tools/climate_step4_compute_anomalies.py")
        if not script_path.exists():
            pytest.skip(f"Script not found: {script_path}")
        
        result = subprocess.run(
            [sys.executable, str(script_path), "--config", str(config_path)],
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            pytest.fail(f"Step 4 script failed: {result.stderr}")
        
        output_zarr = tmp_path / "output" / "anomalies.zarr"
        g_out = zarr.open_group(str(output_zarr), mode="r")
        
        zscore = np.asarray(g_out["zscore"][:])
        zscore_mask = np.asarray(g_out["zscore_mask"][:])
        climo_std = np.asarray(g_out["climo_std"][:])
        
        # Verify that z-scores are computed where std > 0 and count >= 2
        for month_idx in range(12):
            for n in range(N):
                for f in range(F):
                    std_val = climo_std[month_idx, n, f]
                    assert std_val > 0, f"Expected std > 0 for month={month_idx+1}, n={n}, f={f}"
        
        # For 1994, z-score should be approximately 3.0
        for t, mi in enumerate(time_indices):
            year = 1950 + (mi // 12)
            
            if year == 1994:
                for n in range(N):
                    for f in range(F):
                        if zscore_mask[t, n, f] == 1:
                            z_val = zscore[t, n, f]
                            assert np.isclose(z_val, 3.0, atol=0.1), \
                                f"Z-score mismatch at t={t}, n={n}, f={f}: expected ~3.0, got {z_val}"
        
        print("[OK] Z-score test passed")
