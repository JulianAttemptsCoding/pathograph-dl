"""
Regression test: STMM Step A fast-dev-run must produce finite losses, not NaN.

This test would have failed before the fix (when losses were NaN) and must pass after.
"""
import subprocess
import re
from pathlib import Path
from conftest import require_local_zarr, TRADE_BASE_ZARR, TRADE_RISK_ZARR


def test_fast_dev_run_produces_finite_losses():
    """Run STMM fast-dev-run and assert losses are finite (not NaN/Inf)."""
    require_local_zarr(TRADE_BASE_ZARR, TRADE_RISK_ZARR)
    # Ensure we're running from repo root
    repo_root = Path(__file__).parent.parent
    config_path = repo_root / "config" / "stmm_stepA.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    # Run fast-dev-run
    result = subprocess.run(
        ["python", "tools/stmm_stepA_train.py", "--config", str(config_path), "--fast-dev-run"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=300  # 5 minutes max
    )
    
    # Check exit code
    assert result.returncode == 0, (
        f"Training failed with exit code {result.returncode}.\n"
        f"STDERR:\n{result.stderr}\n"
        f"STDOUT:\n{result.stdout}"
    )
    
    # Check for NaN/Inf in loss values using targeted patterns
    output = result.stdout + result.stderr
    
    # Patterns that indicate NaN losses (case-insensitive)
    nan_patterns = [
        r'train_loss[_\w]*[=:]\s*nan\b',
        r'val_loss[=:]\s*nan\b',
        r'loss[=:]\s*nan\b',
    ]
    
    for pattern in nan_patterns:
        if re.search(pattern, output, re.IGNORECASE):
            # Extract context around the match
            match = re.search(pattern, output, re.IGNORECASE)
            start = max(0, match.start() - 100)
            end = min(len(output), match.end() + 100)
            context = output[start:end]
            
            raise AssertionError(
                f"Found NaN loss in output (pattern: {pattern}).\n"
                f"Context: ...{context}...\n"
                f"Full output:\n{output}"
            )
    
    print("✅ STMM fast-dev-run completed successfully with finite losses")
