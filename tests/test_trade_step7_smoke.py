"""Trade Step 7 - Smoke Tests.

Tests that verify the Step 7 tooling works end-to-end in a quick validation mode.
Uses max_epochs=1 and limited batches to run fast while still producing checkpoints.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
CONFIG = ROOT / "config"
TEMP_RUN_DIR = ROOT / "runs" / "_test_step7_smoke"


@pytest.fixture(scope="module")
def setup_teardown():
    """Setup and teardown for smoke tests."""
    # Cleanup before
    if TEMP_RUN_DIR.exists():
        shutil.rmtree(TEMP_RUN_DIR)
    
    yield
    
    # Cleanup after (optional - leave for debugging if needed)
    # if TEMP_RUN_DIR.exists():
    #     shutil.rmtree(TEMP_RUN_DIR)


class TestStep7VerifyArtifacts:
    """Tests for trade_step7_verify_artifacts.py."""
    
    def test_verify_artifacts_runs(self, setup_teardown):
        """Verify artifacts script runs and produces output."""
        output_path = TEMP_RUN_DIR / "artifact_verification.json"
        TEMP_RUN_DIR.mkdir(parents=True, exist_ok=True)
        
        result = subprocess.run(
            [
                sys.executable,
                str(TOOLS / "trade_step7_verify_artifacts.py"),
                "--output", str(output_path),
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        
        assert result.returncode == 0, f"Verification failed: {result.stderr}"
        assert output_path.exists(), "Output file not created"
        
        with open(output_path) as f:
            data = json.load(f)
        
        assert data["success"] is True, f"Verification reported failure: {data.get('errors')}"
        assert data["time_index_aligned"] is True
        assert data["time_range"]["T"] == 908
        assert "base_zarr" in data
        assert "risk_zarr" in data

    def test_verify_artifacts_detects_shapes(self, setup_teardown):
        """Verify that shape information is extracted correctly."""
        output_path = TEMP_RUN_DIR / "artifact_verification_shapes.json"
        
        result = subprocess.run(
            [
                sys.executable,
                str(TOOLS / "trade_step7_verify_artifacts.py"),
                "--output", str(output_path),
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        
        assert result.returncode == 0
        
        with open(output_path) as f:
            data = json.load(f)
        
        # Check base tensor schema
        base_arrays = {a["name"]: a for a in data["base_zarr"]["arrays"]}
        assert "trade" in base_arrays
        assert base_arrays["trade"]["shape"] == [908, 194, 194, 2]
        
        # Check risk tensor schema
        risk_arrays = {a["name"]: a for a in data["risk_zarr"]["arrays"]}
        assert "trade_risk" in risk_arrays
        assert risk_arrays["trade_risk"]["shape"] == [908, 194, 194, 8, 2]


@pytest.mark.slow
class TestStep7SmokeRun:
    """Smoke tests for the full Step 7 pipeline.
    
    These tests are marked slow because they involve actual training.
    Run with: pytest -m slow
    """
    
    def test_smoke_run_completes(self, setup_teardown):
        """Test that smoke run completes and produces outputs."""
        run_dir = TEMP_RUN_DIR / "smoke_run"
        
        result = subprocess.run(
            [
                sys.executable,
                str(TOOLS / "trade_step7_run_baseline.py"),
                "--smoke",
                "--run-dir", str(run_dir),
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=300,  # 5 minute timeout
        )
        
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        
        assert result.returncode == 0, f"Smoke run failed: {result.stderr}"
        
        # Check required outputs exist
        assert (run_dir / "git_commit.txt").exists(), "git_commit.txt missing"
        assert (run_dir / "artifact_verification.json").exists(), "artifact_verification.json missing"
        assert (run_dir / "environment.json").exists(), "environment.json missing"
    
    def test_smoke_run_produces_checkpoint(self, setup_teardown):
        """Test that smoke run produces at least one checkpoint."""
        run_dir = TEMP_RUN_DIR / "smoke_run_ckpt"
        
        result = subprocess.run(
            [
                sys.executable,
                str(TOOLS / "trade_step7_run_baseline.py"),
                "--smoke",
                "--run-dir", str(run_dir),
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=300,
        )
        
        assert result.returncode == 0, f"Run failed: {result.stderr}"
        
        # Find any checkpoint file
        ckpts = list(run_dir.rglob("*.ckpt"))
        assert len(ckpts) > 0, f"No checkpoints found in {run_dir}"
        print(f"Found checkpoints: {ckpts}")


class TestStep7Config:
    """Tests for Step 7 configuration."""
    
    def test_config_exists(self):
        """Test that Step 7 config file exists."""
        config_path = CONFIG / "trade_step7.yaml"
        assert config_path.exists(), f"Config not found: {config_path}"
    
    def test_config_valid_yaml(self):
        """Test that Step 7 config is valid YAML."""
        import yaml
        
        config_path = CONFIG / "trade_step7.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        
        assert "seed" in cfg
        assert cfg["seed"] == 1337
        
        assert "datamodule" in cfg
        assert cfg["datamodule"]["batch_size"] > 0
        
        assert "trainer" in cfg
        assert cfg["trainer"]["max_epochs"] > 0
        
        assert cfg["run"]["fast_dev_run"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
