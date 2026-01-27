"""
Unit test: STMM graph normalization must produce finite outputs even with isolated nodes.
"""
import torch
import numpy as np
import pytest
from pathograph.models.stmm_gwnet import STMMGraphWaveNet


def test_graph_norm_with_isolated_nodes():
    """Test that adjacency normalization handles isolated nodes without producing NaN."""
    # Create a small adjacency with an isolated node (row 2 has all zeros)
    adjacency = torch.tensor([
        [0, 1, 0, 1],  # Node 0: connected to 1, 3
        [1, 0, 1, 0],  # Node 1: connected to 0, 2
        [0, 0, 0, 0],  # Node 2: ISOLATED (zero degree)
        [1, 0, 0, 0],  # Node 3: connected to 0
    ], dtype=torch.float32)
    
    # Create model instance
    model = STMMGraphWaveNet(
        num_nodes=4,
        num_pathogens=2,
        residual_channels=8,
        dilation_channels=8,
        skip_channels=16,
        end_channels=32,
    )
    
    # Build supports (this will add self-loops and normalize)
    supports = model._build_supports(adjacency)
    
    # Check that supports are finite
    for i, support in enumerate(supports):
        assert torch.isfinite(support).all(), f"Support {i} contains NaN/Inf"
        
        # Check that each row sums to ~1.0 (row-stochastic)
        row_sums = support.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
            f"Support {i} is not row-stochastic: row_sums={row_sums.tolist()}"
    
    print("✅ Graph normalization produces finite, row-stochastic matrices")


def test_graph_norm_with_real_adjacency():
    """Test graph normalization with the actual adjacency_border.npy."""
    import sys
    from pathlib import Path
    
    repo_root = Path(__file__).parent.parent
    adj_path = repo_root / "data/processed/meta/adjacency_border.npy"
    
    if not adj_path.exists():
        pytest.skip(f"adjacency_border.npy not found at {adj_path}")
    
    # Load real adjacency
    adjacency_np = np.load(adj_path)
    adjacency = torch.from_numpy(adjacency_np).float()
    
    # Check for zero-degree rows
    row_sums = adjacency.sum(dim=1)
    zero_degree_count = (row_sums == 0).sum().item()
    print(f"Real adjacency has {zero_degree_count} isolated nodes (zero degree)")
    
    # Create model
    model = STMMGraphWaveNet(
        num_nodes=adjacency.shape[0],
        num_pathogens=8,
    )
    
    # Build supports
    supports = model._build_supports(adjacency)
    
    # Verify finite
    for i, support in enumerate(supports):
        assert torch.isfinite(support).all(), \
            f"Support {i} contains NaN/Inf with real adjacency"
        
        # Verify row-stochastic
        row_sums = support.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
            f"Support {i} not row-stochastic: min={row_sums.min()}, max={row_sums.max()}"
    
    print(f"✅ Real adjacency ({adjacency.shape}) normalization is finite and row-stochastic")


if __name__ == "__main__":
    test_graph_norm_with_isolated_nodes()
    test_graph_norm_with_real_adjacency()
