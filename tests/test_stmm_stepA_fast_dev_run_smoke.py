"""
Test: ST-MM-GNN Fast-Dev-Run Smoke Test

End-to-end smoke test running the training entrypoint with real data.
Marked as slow to exclude from default test suite.
"""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
def test_stmm_stepA_fast_dev_run_smoke():
    """Test that training entrypoint runs without error."""
    # Run the training script with fast-dev-run
    result = subprocess.run(
        [
            sys.executable,
            'tools/stmm_stepA_train.py',
            '--config', 'config/stmm_stepA.yaml',
            '--fast-dev-run',
        ],
        capture_output=True,
        text=True,
        timeout=300,  # 5 min timeout
    )
    
    # Print output for debugging
    print("STDOUT:")
    print(result.stdout)
    
    if result.returncode != 0:
        print("STDERR:")
        print(result.stderr)
    
    # Assertions
    assert result.returncode == 0, f"Training script failed with exit code {result.returncode}"
    
    # Check that run directory was created
    runs_dir = Path('runs/stmm_stepA')
    assert runs_dir.exists(), f"Runs directory not found: {runs_dir}"
    
    # Check that at least one run subdirectory exists
    run_subdirs = list(runs_dir.iterdir())
    assert len(run_subdirs) > 0, "No run subdirectories created"
    
    print(f"✓ Fast-dev-run smoke test passed: {len(run_subdirs)} run(s) created")


if __name__ == '__main__':
    test_stmm_stepA_fast_dev_run_smoke()
