"""Pathogen status tensor loader.

Provides access to monthly pathogen presence/absence status across nodes and pathogens.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import zarr


@dataclass(frozen=True)
class PathogenZarrHandle:
    """Immutable handle to pathogen status Zarr group arrays.
    
    Attributes:
        status: (T, N, P) forward-filled monthly pathogen status (0/1 or categorical codes)
        status_mask: (T, N, P) mask indicating observed/valid cells
        time_index: (T,) integer time indices (aligned with trade time_index_master)
        T: Time dimension
        N: Node/country dimension
        P: Pathogen dimension
    """
    status: Any  # zarr.Array
    status_mask: Any  # zarr.Array
    time_index: Any  # zarr.Array
    T: int
    N: int
    P: int


def open_pathogen_zarr(zarr_path: str | Path) -> PathogenZarrHandle:
    """Open pathogen status tensor Zarr group and return handle.
    
    Args:
        zarr_path: Path to status_tensor.zarr directory
        
    Returns:
        PathogenZarrHandle with validated arrays
        
    Raises:
        FileNotFoundError: If zarr_path does not exist
        AssertionError: If required arrays missing or shapes mismatch
    """
    p = Path(zarr_path)
    if not p.exists():
        raise FileNotFoundError(f"Pathogen Zarr not found: {p}")
    
    g = zarr.open_group(str(p), mode='r')
    
    # Required arrays
    if 'status' not in g:
        raise KeyError(f"Missing 'status' array in {p}; keys={list(g.array_keys())}")
    if 'status_mask' not in g:
        raise KeyError(f"Missing 'status_mask' array in {p}; keys={list(g.array_keys())}")
    
    status = g['status']
    status_mask = g['status_mask']
    
    # Validate shapes match
    if status.shape != status_mask.shape:
        raise ValueError(
            f"Shape mismatch: status {status.shape} vs status_mask {status_mask.shape}"
        )
    
    T, N, P = status.shape
    
    # Time index (optional but expected)
    time_index = g.get('time_index', None)
    if time_index is not None:
        if time_index.shape[0] != T:
            raise ValueError(
                f"time_index length {time_index.shape[0]} does not match T={T}"
            )
    
    return PathogenZarrHandle(
        status=status,
        status_mask=status_mask,
        time_index=time_index,
        T=T,
        N=N,
        P=P,
    )
