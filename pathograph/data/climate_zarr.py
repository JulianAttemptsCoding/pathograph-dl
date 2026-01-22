"""Climate and anomaly tensor loaders for multimodal ST-MM-GNN.

Provides deterministic access to climate tensors and anomalies with shape validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import zarr
import numpy as np


@dataclass(frozen=True)
class ClimateZarrHandle:
    """Immutable handle to climate tensor Zarr arrays.
    
    Attributes:
        climate: (T, N, F) climate features (e.g., temperature, precipitation)
        mask: (T, N, F) or (T, N) mask indicating observed/valid cells
        time_index: (T,) integer time indices
        T: Time dimension
        N: Node/country dimension
       F: Feature dimension (climate variables)
    """
    climate: Any  # zarr.Array
    mask: Optional[Any]  # zarr.Array or None
    time_index: Any  # zarr.Array
    T: int
    N: int
    F: int


def open_climate_zarr(zarr_path: str | Path, array_key: str = 'climate') -> ClimateZarrHandle:
    """Open climate tensor Zarr group and return handle.
    
    Args:
        zarr_path: Path to climate_tensor.zarr directory
        array_key: Name of array within group (default: 'climate')
        
    Returns:
        ClimateZarrHandle with validated arrays
        
    Raises:
        FileNotFoundError: If zarr_path does not exist
        KeyError: If array_key not found in group
        AssertionError: If shapes don't match expected (T, N, F)
    """
    p = Path(zarr_path)
    if not p.exists():
        raise FileNotFoundError(f"Climate Zarr not found: {p}")
    
    g = zarr.open_group(str(p), mode='r')
    
    if array_key not in g:
        available_keys = list(g.array_keys())
        raise KeyError(
            f"Array '{array_key}' not found in {p}. "
            f"Available keys: {available_keys}"
        )
    
    climate = g[array_key]
    
    if climate.ndim != 3:
        raise ValueError(
            f"Climate array must be 3D (T,N,F), got shape {climate.shape}"
        )
    
    T, N, F = climate.shape
    
    # Optional mask
    mask = g.get('mask', None)
    if mask is not None:
        # Mask can be (T,N,F) or (T,N) - broadcast compatible
        if mask.shape not in [(T, N, F), (T, N)]:
            raise ValueError(
                f"Mask shape {mask.shape} incompatible with climate shape {climate.shape}"
            )
    
    # Time index (optional but expected)
    time_index = g.get('time_index', None)
    if time_index is not None:
        if time_index.shape[0] != T:
            raise ValueError(
                f"time_index length {time_index.shape[0]} does not match T={T}"
            )
    
    return ClimateZarrHandle(
        climate=climate,
        mask=mask,
        time_index=time_index,
        T=T,
        N=N,
        F=F,
    )


def load_meta_matrices(
    distance_path: str | Path,
    adjacency_path: str | Path,
    expected_N: int = 194
) -> tuple[np.ndarray, np.ndarray]:
    """Load distance and adjacency meta matrices.
    
    Args:
        distance_path: Path to distance_km.npy
        adjacency_path: Path to adjacency_border.npy
        expected_N: Expected spatial dimension (default 194)
        
    Returns:
        Tuple of (distance_km, adjacency_border), both (N, N) arrays
        
    Raises:
        FileNotFoundError: If paths don't exist
        ValueError: If shapes don't match (expected_N, expected_N)
    """
    dist_p = Path(distance_path)
    adj_p = Path(adjacency_path)
    
    if not dist_p.exists():
        raise FileNotFoundError(f"Distance matrix not found: {dist_p}")
    if not adj_p.exists():
        raise FileNotFoundError(f"Adjacency matrix not found: {adj_p}")
    
    distance_km = np.load(dist_p)
    adjacency_border = np.load(adj_p)
    
    if distance_km.shape != (expected_N, expected_N):
        raise ValueError(
            f"distance_km shape {distance_km.shape} != ({expected_N}, {expected_N})"
        )
    if adjacency_border.shape != (expected_N, expected_N):
        raise ValueError(
            f"adjacency_border shape {adjacency_border.shape} != ({expected_N}, {expected_N})"
        )
    
    return distance_km.astype(np.float32), adjacency_border.astype(np.float32)
