"""Shared test fixtures and skip helpers for PathoGraph-DL."""
import os
from pathlib import Path

import pytest

# ── data path constants ──────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent

TRADE_BASE_ZARR = (
    REPO_ROOT / "data" / "processed" / "trade" / "imf_imts_step1" / "trade_fob_tensor.zarr"
)
TRADE_RISK_ZARR = (
    REPO_ROOT / "data" / "processed" / "trade" / "faostat_step2" / "trade_risk_tensor.zarr"
)


# ── skip helpers ─────────────────────────────────────────────────────
def require_local_zarr(*paths: Path) -> None:
    """Skip the current test if any of the given zarr store paths are missing.

    Usage at the top of a test function::

        require_local_zarr(TRADE_BASE_ZARR, TRADE_RISK_ZARR)
    """
    for p in paths:
        if not p.exists():
            pytest.skip(
                f"Local zarr data not staged: {p.relative_to(REPO_ROOT)}. "
                "Run preprocessing pipeline to generate."
            )


def require_local_path(path: str | Path) -> None:
    """Skip the current test if the given path is missing."""
    p = Path(path)
    if not p.exists():
        pytest.skip(f"Local data not found: {p}")
