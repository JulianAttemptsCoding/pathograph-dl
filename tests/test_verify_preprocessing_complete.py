"""Unit test for preprocessing acceptance verifier."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest


def test_verify_preprocessing_complete_all_pass():
    """Test acceptance verifier with valid synthetic artifacts."""
    
    try:
        import zarr  # type: ignore
    except ImportError:
        pytest.skip("zarr not available")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create synthetic artifacts
        T = 50  # Small time window
        N = 194
        K = 5   # Risk products
        F_climate = 10
        P = 8   # Pathogens
        
        time_index = np.arange(T, dtype=np.int32)
        
        # time_index_master
        ti_master_path = tmp_path / "time_index_master.npy"
        np.save(ti_master_path, time_index)
        
        # Trade base
        trade_base_path = tmp_path / "trade_base.zarr"
        g_tb = zarr.open_group(str(trade_base_path), mode="w")
        g_tb.create_array("trade", data=np.zeros((T, N, N, 2), dtype=np.float32), chunks=(10, N, N, 2))
        g_tb.create_array("mask", data=np.zeros((T, N, N, 2), dtype=np.uint8), chunks=(10, N, N, 2))
        g_tb.create_array("is_estimated", data=np.zeros((T, N, N, 2), dtype=np.uint8), chunks=(10, N, N, 2))
        g_tb.create_array("time_index", data=time_index, chunks=(T,))
        
        # Trade risk (canonical FAOSTAT schema)
        trade_risk_path = tmp_path / "trade_risk.zarr"
        g_tr = zarr.open_group(str(trade_risk_path), mode="w")
        g_tr.create_array("trade_risk", data=np.zeros((T, N, N, K, 2), dtype=np.float32), chunks=(10, N, N, K, 2))
        g_tr.create_array("observed_mask", data=np.zeros((T, N, N, K, 2), dtype=np.uint8), chunks=(10, N, N, K, 2))
        g_tr.create_array("is_estimated", data=np.zeros((T, N, N, K, 2), dtype=np.uint8), chunks=(10, N, N, K, 2))
        g_tr.create_array("time_index", data=time_index, chunks=(T,))
        
        # Climate tensor
        climate_path = tmp_path / "climate.zarr"
        g_c = zarr.open_group(str(climate_path), mode="w")
        g_c.create_array("climate", data=np.zeros((T, N, F_climate), dtype=np.float32), chunks=(10, N, F_climate))
        g_c.create_array("mask", data=np.zeros((T, N, F_climate), dtype=np.uint8), chunks=(10, N, F_climate))
        g_c.create_array("time_index", data=time_index, chunks=(T,))
        g_c.create_array("feature_names", data=np.array([f"feat_{i}" for i in range(F_climate)], dtype="U32"))
        
        # Climate anomalies
        climate_anoms_path = tmp_path / "climate_anoms.zarr"
        g_ca = zarr.open_group(str(climate_anoms_path), mode="w")
        g_ca.create_array("anomaly", data=np.zeros((T, N, F_climate), dtype=np.float32), chunks=(10, N, F_climate))
        g_ca.create_array("zscore", data=np.zeros((T, N, F_climate), dtype=np.float32), chunks=(10, N, F_climate))
        g_ca.create_array("mask", data=np.ones((T, N, F_climate), dtype=np.uint8), chunks=(10, N, F_climate))
        g_ca.create_array("time_index", data=time_index, chunks=(T,))
        
        # Pathogen status
        pathogen_path = tmp_path / "pathogen.zarr"
        g_p = zarr.open_group(str(pathogen_path), mode="w")
        g_p.create_array("status", data=np.zeros((T, N, P), dtype=np.float32), chunks=(10, N, P))
        g_p.create_array("mask", data=np.zeros((T, N, P), dtype=np.uint8), chunks=(10, N, P))
        g_p.create_array("time_index", data=time_index, chunks=(T,))
        
        # Meta spatial (optional, but include for complete test)
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        
        distance = np.random.rand(N, N).astype(np.float32)
        np.fill_diagonal(distance, 0.0)
        np.save(meta_dir / "distance_km.npy", distance)
        
        adjacency = np.random.randint(0, 2, size=(N, N), dtype=np.uint8)
        np.save(meta_dir / "adjacency_border.npy", adjacency)
        
        # Run verifier
        script_path = Path("tools/verify_preprocessing_complete.py")
        if not script_path.exists():
            pytest.skip(f"Script not found: {script_path}")
        
        result = subprocess.run(
            [
                sys.executable, str(script_path),
                "--time-index-master", str(ti_master_path),
                "--trade-base-zarr", str(trade_base_path),
                "--trade-risk-zarr", str(trade_risk_path),
                "--climate-zarr", str(climate_path),
                "--climate-anoms-zarr", str(climate_anoms_path),
                "--pathogen-zarr", str(pathogen_path),
                "--meta-dir", str(meta_dir),
                "--mode", "fast",
            ],
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            pytest.fail(f"Verifier failed with exit code {result.returncode}")
        
        assert "PASS" in result.stdout, "Expected PASS in output"
        assert "Selected arrays: risk=trade_risk, mask=observed_mask" in result.stdout, "Expected canonical array names in output"
        
        # Check that reports were created
        report_json = Path("data/processed/preprocessing_acceptance_report.json")
        report_txt = Path("data/processed/preprocessing_acceptance_report.txt")
        
        # Note: These will be in the actual repo, not tmpdir, so we won't check them here
        # In a real test, we'd need to make the output paths configurable
        
        print("[OK] Acceptance verifier test passed")


def test_verify_preprocessing_time_index_mismatch():
    """Test that verifier correctly fails on time_index mismatch."""
    
    try:
        import zarr  # type: ignore
    except ImportError:
        pytest.skip("zarr not available")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        T = 50
        N = 194
        
        time_index_master = np.arange(T, dtype=np.int32)
        time_index_wrong = np.arange(T, dtype=np.int32) + 1  # Offset by 1
        
        ti_master_path = tmp_path / "time_index_master.npy"
        np.save(ti_master_path, time_index_master)
        
        # Create trade base with wrong time_index
        trade_base_path = tmp_path / "trade_base.zarr"
        g_tb = zarr.open_group(str(trade_base_path), mode="w")
        g_tb.create_array("trade", data=np.zeros((T, N, N, 2), dtype=np.float32))
        g_tb.create_array("mask", data=np.zeros((T, N, N, 2), dtype=np.uint8))
        g_tb.create_array("is_estimated", data=np.zeros((T, N, N, 2), dtype=np.uint8))
        g_tb.create_array("time_index", data=time_index_wrong, chunks=(T,))  # WRONG
        
        # Create minimal other artifacts (not checked in this test)
        trade_risk_path = tmp_path / "trade_risk.zarr"
        g_tr = zarr.open_group(str(trade_risk_path), mode="w")
        g_tr.create_array("risk", data=np.zeros((T, N, N, 1, 2), dtype=np.float32))
        g_tr.create_array("mask", data=np.zeros((T, N, N, 1, 2), dtype=np.uint8))
        g_tr.create_array("is_estimated", data=np.zeros((T, N, N, 1, 2), dtype=np.uint8))
        g_tr.create_array("time_index", data=time_index_master)
        
        script_path = Path("tools/verify_preprocessing_complete.py")
        if not script_path.exists():
            pytest.skip(f"Script not found: {script_path}")
        
        # This should fail
        result = subprocess.run(
            [
                sys.executable, str(script_path),
                "--time-index-master", str(ti_master_path),
                "--trade-base-zarr", str(trade_base_path),
                "--trade-risk-zarr", str(trade_risk_path),
                "--mode", "fast",
            ],
            capture_output=True,
            text=True,
        )
        
        # Should exit with nonzero
        assert result.returncode != 0, "Expected verifier to fail on time_index mismatch"
        assert "FAIL" in result.stderr or "does not match" in result.stderr, \
            f"Expected failure message in stderr, got: {result.stderr}"
        
        print("[OK] Time index mismatch test passed")


def test_verify_preprocessing_wrong_shape():
    """Test that verifier correctly fails on wrong tensor shape."""
    
    try:
        import zarr  # type: ignore
    except ImportError:
        pytest.skip("zarr not available")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        T = 50
        N = 194
        N_wrong = 100  # WRONG
        
        time_index = np.arange(T, dtype=np.int32)
        
        ti_master_path = tmp_path / "time_index_master.npy"
        np.save(ti_master_path, time_index)
        
        # Create trade base with wrong N
        trade_base_path = tmp_path / "trade_base.zarr"
        g_tb = zarr.open_group(str(trade_base_path), mode="w")
        g_tb.create_array("trade", data=np.zeros((T, N_wrong, N, 2), dtype=np.float32))  # WRONG
        g_tb.create_array("mask", data=np.zeros((T, N_wrong, N, 2), dtype=np.uint8))
        g_tb.create_array("is_estimated", data=np.zeros((T, N_wrong, N, 2), dtype=np.uint8))
        g_tb.create_array("time_index", data=time_index)
        
        script_path = Path("tools/verify_preprocessing_complete.py")
        if not script_path.exists():
            pytest.skip(f"Script not found: {script_path}")
        
        result = subprocess.run(
            [
                sys.executable, str(script_path),
                "--time-index-master", str(ti_master_path),
                "--trade-base-zarr", str(trade_base_path),
                "--mode", "fast",
            ],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode != 0, "Expected verifier to fail on wrong shape"
        assert "FAIL" in result.stderr, f"Expected failure message, got: {result.stderr}"
        
        print("[OK] Wrong shape test passed")


def test_verify_preprocessing_legacy_risk_names():
    """Test that verifier accepts legacy 'risk'/'mask' names via alias fallback."""
    
    try:
        import zarr  # type: ignore
    except ImportError:
        pytest.skip("zarr not available")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        T = 50
        N = 194
        K = 3
        F_climate = 10
        P = 8
        
        time_index = np.arange(T, dtype=np.int32)
        
        ti_master_path = tmp_path / "time_index_master.npy"
        np.save(ti_master_path, time_index)
        
        # Trade base
        trade_base_path = tmp_path / "trade_base.zarr"
        g_tb = zarr.open_group(str(trade_base_path), mode="w")
        g_tb.create_array("trade", data=np.zeros((T, N, N, 2), dtype=np.float32))
        g_tb.create_array("mask", data=np.zeros((T, N, N, 2), dtype=np.uint8))
        g_tb.create_array("is_estimated", data=np.zeros((T, N, N, 2), dtype=np.uint8))
        g_tb.create_array("time_index", data=time_index)
        
        # Trade risk with LEGACY names (risk/mask instead of trade_risk/observed_mask)
        trade_risk_path = tmp_path / "trade_risk.zarr"
        g_tr = zarr.open_group(str(trade_risk_path), mode="w")
        g_tr.create_array("risk", data=np.zeros((T, N, N, K, 2), dtype=np.float32))
        g_tr.create_array("mask", data=np.zeros((T, N, N, K, 2), dtype=np.uint8))
        g_tr.create_array("is_estimated", data=np.zeros((T, N, N, K, 2), dtype=np.uint8))
        g_tr.create_array("time_index", data=time_index)
        
        # Climate tensor
        climate_path = tmp_path / "climate.zarr"
        g_c = zarr.open_group(str(climate_path), mode="w")
        g_c.create_array("climate", data=np.zeros((T, N, F_climate), dtype=np.float32))
        g_c.create_array("mask", data=np.zeros((T, N, F_climate), dtype=np.uint8))
        g_c.create_array("time_index", data=time_index)
        g_c.create_array("feature_names", data=np.array([f"feat_{i}" for i in range(F_climate)], dtype="U32"))
        
        # Climate anomalies
        climate_anoms_path = tmp_path / "climate_anoms.zarr"
        g_ca = zarr.open_group(str(climate_anoms_path), mode="w")
        g_ca.create_array("anomaly", data=np.zeros((T, N, F_climate), dtype=np.float32))
        g_ca.create_array("zscore", data=np.zeros((T, N, F_climate), dtype=np.float32))
        g_ca.create_array("mask", data=np.ones((T, N, F_climate), dtype=np.uint8))
        g_ca.create_array("time_index", data=time_index)
        
        # Pathogen
        pathogen_path = tmp_path / "pathogen.zarr"
        g_p = zarr.open_group(str(pathogen_path), mode="w")
        g_p.create_array("status", data=np.zeros((T, N, P), dtype=np.float32))
        g_p.create_array("mask", data=np.zeros((T, N, P), dtype=np.uint8))
        g_p.create_array("time_index", data=time_index)
        
        # Meta dir (empty, but exists)
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        
        script_path = Path("tools/verify_preprocessing_complete.py")
        if not script_path.exists():
            pytest.skip(f"Script not found: {script_path}")
        
        result = subprocess.run(
            [
                sys.executable, str(script_path),
                "--time-index-master", str(ti_master_path),
                "--trade-base-zarr", str(trade_base_path),
                "--trade-risk-zarr", str(trade_risk_path),
                "--climate-zarr", str(climate_path),
                "--climate-anoms-zarr", str(climate_anoms_path),
                "--pathogen-zarr", str(pathogen_path),
                "--meta-dir", str(meta_dir),
                "--mode", "fast",
            ],
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            pytest.fail(f"Verifier failed with exit code {result.returncode}")
        
        # Should pass and show legacy names selected
        assert "PASS" in result.stdout, "Expected PASS in output"
        assert "Selected arrays: risk=risk, mask=mask" in result.stdout, "Expected legacy array names in output"
        
        print("[OK] Legacy alias fallback test passed")


def test_verify_preprocessing_require_meta_fails():
    """Test that --require-meta fails when meta matrices missing."""
    
    try:
        import zarr  # type: ignore
    except ImportError:
        pytest.skip("zarr not available")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        T = 50
        N = 194
        K = 3
        F_climate = 10
        P = 8
        
        time_index = np.arange(T, dtype=np.int32)
        
        ti_master_path = tmp_path / "time_index_master.npy"
        np.save(ti_master_path, time_index)
        
        # Trade base
        trade_base_path = tmp_path / "trade_base.zarr"
        g_tb = zarr.open_group(str(trade_base_path), mode="w")
        g_tb.create_array("trade", data=np.zeros((T, N, N, 2), dtype=np.float32))
        g_tb.create_array("mask", data=np.zeros((T, N, N, 2), dtype=np.uint8))
        g_tb.create_array("is_estimated", data=np.zeros((T, N, N, 2), dtype=np.uint8))
        g_tb.create_array("time_index", data=time_index)
        
        # Trade risk with canonical names
        trade_risk_path = tmp_path / "trade_risk.zarr"
        g_tr = zarr.open_group(str(trade_risk_path), mode="w")
        g_tr.create_array("trade_risk", data=np.zeros((T, N, N, K, 2), dtype=np.float32))
        g_tr.create_array("observed_mask", data=np.zeros((T, N, N, K, 2), dtype=np.uint8))
        g_tr.create_array("is_estimated", data=np.zeros((T, N, N, K, 2), dtype=np.uint8))
        g_tr.create_array("time_index", data=time_index)
        
        # Climate tensor
        climate_path = tmp_path / "climate.zarr"
        g_c = zarr.open_group(str(climate_path), mode="w")
        g_c.create_array("climate", data=np.zeros((T, N, F_climate), dtype=np.float32))
        g_c.create_array("mask", data=np.zeros((T, N, F_climate), dtype=np.uint8))
        g_c.create_array("time_index", data=time_index)
        g_c.create_array("feature_names", data=np.array([f"feat_{i}" for i in range(F_climate)], dtype="U32"))
        
        # Climate anomalies
        climate_anoms_path = tmp_path / "climate_anoms.zarr"
        g_ca = zarr.open_group(str(climate_anoms_path), mode="w")
        g_ca.create_array("anomaly", data=np.zeros((T, N, F_climate), dtype=np.float32))
        g_ca.create_array("zscore", data=np.zeros((T, N, F_climate), dtype=np.float32))
        g_ca.create_array("mask", data=np.ones((T, N, F_climate), dtype=np.uint8))
        g_ca.create_array("time_index", data=time_index)
        
        # Pathogen
        pathogen_path = tmp_path / "pathogen.zarr"
        g_p = zarr.open_group(str(pathogen_path), mode="w")
        g_p.create_array("status", data=np.zeros((T, N, P), dtype=np.float32))
        g_p.create_array("mask", data=np.zeros((T, N, P), dtype=np.uint8))
        g_p.create_array("time_index", data=time_index)
        
        # Meta dir exists but NO distance/adjacency matrices
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        
        script_path = Path("tools/verify_preprocessing_complete.py")
        if not script_path.exists():
            pytest.skip(f"Script not found: {script_path}")
        
        # Run with --require-meta (should fail)
        result = subprocess.run(
            [
                sys.executable, str(script_path),
                "--time-index-master", str(ti_master_path),
                "--trade-base-zarr", str(trade_base_path),
                "--trade-risk-zarr", str(trade_risk_path),
                "--climate-zarr", str(climate_path),
                "--climate-anoms-zarr", str(climate_anoms_path),
                "--pathogen-zarr", str(pathogen_path),
                "--meta-dir", str(meta_dir),
                "--mode", "fast",
                "--require-meta",  # STRICT MODE
            ],
            capture_output=True,
            text=True,
        )
        
        # Should fail with nonzero exit code
        assert result.returncode != 0, "Expected verifier to fail when --require-meta is set and meta matrices missing"
        assert "FAIL" in result.stderr or "required" in result.stderr, \
            f"Expected failure message about required meta matrices, got: {result.stderr}"
        
        print("[OK] Require-meta strict mode test passed")
