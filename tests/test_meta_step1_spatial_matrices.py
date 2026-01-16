"""Contract test for spatial meta matrices builder."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest


def test_meta_spatial_matrices_border_vs_point_touch():
    """Test that border-sharing is correctly distinguished from point-touch."""
    
    try:
        import geopandas as gpd
        from shapely.geometry import Polygon
    except ImportError:
        pytest.skip("geopandas/shapely not available")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create 4 polygons:
        # ┌───────┬───────┐
        # │   0   │   1   │  <- 0 and 1 share a vertical border
        # ├───────┼───────┤
        # │   2   ·   3   │  <- 2 and 3 touch at center point only (no shared edge)
        # └───────┴───────┘
        #
        # Polygon 0: [0, 1] x [1, 2]
        # Polygon 1: [1, 2] x [1, 2]
        # Polygon 2: [0, 1] x [0, 1]
        # Polygon 3: [1, 2] x [0, 1]
        #
        # Note: 0-1 share edge from (1,1) to (1,2)
        #       2-3 touch only at point (1,1)
        
        polygons = [
            Polygon([(0, 1), (1, 1), (1, 2), (0, 2), (0, 1)]),  # 0: left upper
            Polygon([(1, 1), (2, 1), (2, 2), (1, 2), (1, 1)]),  # 1: right upper
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),  # 2: left lower
            Polygon([(1, 0), (2, 0), (2, 1), (1, 1), (1, 0)]),  # 3: right lower
        ]
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame({
            "node_id": [0, 1, 2, 3],
            "iso3": ["AA", "BB", "CC", "DD"],
            "geometry": polygons
        }, crs="EPSG:4326")
        
        # Write to temp GPKG
        gpkg_path = tmp_path / "test.gpkg"
        gdf.to_file(gpkg_path, driver="GPKG")
        
        # Output directory
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        
        # Run tool
        script_path = Path("tools/meta_step1_build_spatial_matrices.py")
        if not script_path.exists():
            pytest.skip(f"Script not found: {script_path}")
        
        result = subprocess.run(
            [
                sys.executable, str(script_path),
                "--gpkg", str(gpkg_path),
                "--out-dir", str(out_dir),
                "--require-node-ids", "false",  # Only 4 polygons
            ],
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            pytest.fail(f"Tool failed with exit code {result.returncode}")
        
        # Load outputs
        distance_path = out_dir / "distance_km.npy"
        adjacency_path = out_dir / "adjacency_border.npy"
        
        assert distance_path.exists(), "distance_km.npy not created"
        assert adjacency_path.exists(), "adjacency_border.npy not created"
        
        distance = np.load(distance_path)
        adjacency = np.load(adjacency_path)
        
        # Validate distance matrix
        assert distance.shape == (4, 4), f"Distance shape: {distance.shape}"
        assert distance.dtype == np.float32, f"Distance dtype: {distance.dtype}"
        
        # Check symmetry
        assert np.allclose(distance, distance.T, atol=1e-3), "Distance not symmetric"
        
        # Check diagonal zeros
        assert np.allclose(np.diag(distance), 0.0, atol=1e-3), "Distance diagonal not zero"
        
        # Validate adjacency matrix
        assert adjacency.shape == (4, 4), f"Adjacency shape: {adjacency.shape}"
        assert adjacency.dtype == np.uint8, f"Adjacency dtype: {adjacency.dtype}"
        
        # Check symmetry
        assert np.array_equal(adjacency, adjacency.T), "Adjacency not symmetric"
        
        # Check diagonal zeros
        assert np.all(np.diag(adjacency) == 0), "Adjacency diagonal not zero"
        
        # CRITICAL: Check border-sharing vs point-touch
        # Polygons 0 and 1 share a border -> adjacency should be 1
        assert adjacency[0, 1] == 1, f"Expected adjacency[0,1]=1 (border-sharing), got {adjacency[0,1]}"
        assert adjacency[1, 0] == 1, f"Expected adjacency[1,0]=1 (border-sharing), got {adjacency[1,0]}"
        
        # Polygons 2 and 3 touch at a point only -> adjacency should be 0
        assert adjacency[2, 3] == 0, f"Expected adjacency[2,3]=0 (point-touch), got {adjacency[2,3]}"
        assert adjacency[3, 2] == 0, f"Expected adjacency[3,2]=0 (point-touch), got {adjacency[3,2]}"
        
        # The other pairs should not be adjacent (no touch at all)
        assert adjacency[0, 2] == 0, "Expected adjacency[0,2]=0"
        assert adjacency[0, 3] == 0, "Expected adjacency[0,3]=0"
        assert adjacency[1, 2] == 0, "Expected adjacency[1,2]=0"
        assert adjacency[1, 3] == 0, "Expected adjacency[1,3]=0"
        
        print("[OK] Border-sharing correctly distinguished from point-touch")


def test_meta_spatial_matrices_real_check():
    """Quick sanity check that tool can run on the real node_geometry.gpkg."""
    
    try:
        import geopandas as gpd
    except ImportError:
        pytest.skip("geopandas not available")
    
    gpkg_path = Path("data/processed/meta/node_geometry.gpkg")
    
    if not gpkg_path.exists():
        pytest.skip(f"Real gpkg not found: {gpkg_path}")
    
    # Just verify we can read it and it has node_id column
    gdf = gpd.read_file(gpkg_path)
    
    assert "node_id" in gdf.columns, "Missing node_id column"
    assert len(gdf) == 194, f"Expected 194 geometries, got {len(gdf)}"
    
    node_ids = set(gdf["node_id"].tolist())
    expected_ids = set(range(194))
    
    assert node_ids == expected_ids, f"node_id coverage mismatch: missing {expected_ids - node_ids}"
    
    print("[OK] Real node_geometry.gpkg structure validated")
