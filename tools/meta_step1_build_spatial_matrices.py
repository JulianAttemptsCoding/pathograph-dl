"""Build spatial meta matrices from node_geometry.gpkg.

Generates:
- distance_km.npy: Great-circle distances between country centroids (194x194, float32)
- adjacency_border.npy: Border-sharing adjacency matrix (194x194, uint8)
- spatial_meta_qc.json: QC metrics

Uses representative_point() for stable centroids, make_valid() for geometry repair,
and spatial index for efficient adjacency computation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def haversine_vectorized(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """
    Compute haversine distance in kilometers (vectorized).
    
    Args:
        lon1, lat1: Arrays of shape (N,) in degrees
        lon2, lat2: Arrays of shape (N,) in degrees
    
    Returns:
        Distance matrix of shape (N, N) in kilometers
    """
    # Convert to radians
    lon1_rad = np.radians(lon1)
    lat1_rad = np.radians(lat1)
    lon2_rad = np.radians(lon2)
    lat2_rad = np.radians(lat2)
    
    # Broadcasting to compute all pairwise distances
    # lon1, lat1 shape: (N,) -> (N, 1)
    # lon2, lat2 shape: (N,) -> (1, N)
    dlon = lon2_rad[np.newaxis, :] - lon1_rad[:, np.newaxis]
    dlat = lat2_rad[np.newaxis, :] - lat1_rad[:, np.newaxis]
    
    # Haversine formula
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad[:, np.newaxis]) * np.cos(lat2_rad[np.newaxis, :]) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    R = 6371.0  # Earth radius in km
    return R * c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpkg", default="data/processed/meta/node_geometry.gpkg")
    ap.add_argument("--out-dir", default="data/processed/meta")
    ap.add_argument("--layer", default=None, help="GeoPackage layer name (auto-detect if None)")
    ap.add_argument("--require-node-ids", type=bool, default=True, help="Require node_id 0..193")
    ap.add_argument("--adjacency-mode", choices=["border"], default="border")
    ap.add_argument("--touch-eps-m", type=float, default=1.0, help="Min boundary intersection length (m) for adjacency")
    args = ap.parse_args()
    
    gpkg_path = Path(args.gpkg)
    out_dir = Path(args.out_dir)
    
    print(f"[Spatial Meta Builder]")
    print(f"  Input: {gpkg_path}")
    print(f"  Output dir: {out_dir}")
    print()
    
    # Import geopandas
    try:
        import geopandas as gpd
    except ImportError as e:
        print(f"[FAIL] geopandas not available: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Check input exists
    if not gpkg_path.exists():
        print(f"[FAIL] GeoPackage not found: {gpkg_path}", file=sys.stderr)
        sys.exit(1)
    
    # Read GeoPackage
    print(f"[1/5] Reading GeoPackage...")
    try:
        gdf = gpd.read_file(gpkg_path, layer=args.layer)
    except Exception as e:
        print(f"[FAIL] Error reading GeoPackage: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"  Loaded {len(gdf)} geometries")
    
    # Validate node_id coverage
    if "node_id" not in gdf.columns:
        print("[FAIL] GeoPackage missing 'node_id' column", file=sys.stderr)
        sys.exit(1)
    
    node_ids = set(gdf["node_id"].tolist())
    expected_ids = set(range(194))
    
    if args.require_node_ids:
        if len(gdf) != 194:
            print(f"[FAIL] Expected 194 geometries, got {len(gdf)}", file=sys.stderr)
            sys.exit(1)
        
        if node_ids != expected_ids:
            missing = sorted(expected_ids - node_ids)
            extra = sorted(node_ids - expected_ids)
            print(f"[FAIL] node_id coverage mismatch:", file=sys.stderr)
            if missing:
                print(f"  Missing: {missing}", file=sys.stderr)
            if extra:
                print(f"  Extra: {extra}", file=sys.stderr)
            sys.exit(1)
        
        if len(gdf["node_id"].unique()) != 194:
            print(f"[FAIL] node_id not unique", file=sys.stderr)
            sys.exit(1)
    
    # Sort by node_id for consistent ordering
    gdf = gdf.sort_values("node_id").reset_index(drop=True)
    N = len(gdf)
    
    print(f"[OK] Validated: N={N}, node_id range [{gdf['node_id'].min()}, {gdf['node_id'].max()}]")
    print()
    
    # Repair geometries
    print(f"[2/5] Repairing geometries...")
    try:
        # Try make_valid first (shapely >= 1.8)
        from shapely.validation import make_valid
        gdf["geometry"] = gdf["geometry"].apply(make_valid)
        print("  Used make_valid()")
    except (ImportError, AttributeError):
        # Fallback to buffer(0)
        gdf["geometry"] = gdf["geometry"].buffer(0)
        print("  Used buffer(0) fallback")
    
    # Check for invalid/empty geometries after repair
    invalid_count = (~gdf["geometry"].is_valid).sum()
    empty_count = gdf["geometry"].is_empty.sum()
    
    if invalid_count > 0:
        print(f"[FAIL] {invalid_count} geometries still invalid after repair", file=sys.stderr)
        sys.exit(1)
    if empty_count > 0:
        print(f"[FAIL] {empty_count} geometries are empty after repair", file=sys.stderr)
        sys.exit(1)
    
    print(f"[OK] All geometries valid")
    print()
    
    # Compute distance matrix
    print(f"[3/5] Computing distance matrix (haversine)...")
    
    # Reproject to WGS84 for lon/lat
    gdf_wgs84 = gdf.to_crs("EPSG:4326")
    
    # Use representative_point() for stability
    repr_points = gdf_wgs84["geometry"].representative_point()
    
    lons = np.array(repr_points.x, dtype=np.float64)
    lats = np.array(repr_points.y, dtype=np.float64)
    
    # Check for NaNs
    if np.any(np.isnan(lons)) or np.any(np.isnan(lats)):
        print(f"[FAIL] NaN values in representative points", file=sys.stderr)
        sys.exit(1)
    
    # Compute haversine distances
    distance_km = haversine_vectorized(lons, lats, lons, lats).astype(np.float32)
    
    # Enforce diagonal zeros
    np.fill_diagonal(distance_km, 0.0)
    
    # Validate
    if not np.all(np.isfinite(distance_km)):
        print(f"[FAIL] Non-finite values in distance matrix", file=sys.stderr)
        sys.exit(1)
    
    if not np.allclose(distance_km, distance_km.T, atol=1e-3):
        max_diff = np.max(np.abs(distance_km - distance_km.T))
        print(f"[FAIL] Distance matrix not symmetric (max diff: {max_diff} km)", file=sys.stderr)
        sys.exit(1)
    
    diag_max = np.max(np.abs(np.diag(distance_km)))
    if diag_max > 1e-3:
        print(f"[FAIL] Distance diagonal not near-zero (max: {diag_max} km)", file=sys.stderr)
        sys.exit(1)
    
    dist_min_nonzero = np.min(distance_km[distance_km > 0])
    dist_max = np.max(distance_km)
    
    print(f"[OK] Distance matrix computed")
    print(f"  Min (nonzero): {dist_min_nonzero:.2f} km")
    print(f"  Max: {dist_max:.2f} km")
    print(f"  Symmetry: OK (max diff < 1e-3 km)")
    print()
    
    # Compute adjacency matrix
    print(f"[4/5] Computing adjacency matrix (border-sharing)...")
    
    # Reproject to metric CRS (EPSG:6933 - World Cylindrical Equal Area)
    gdf_metric = gdf.to_crs("EPSG:6933")
    
    adjacency = np.zeros((N, N), dtype=np.uint8)
    
    # Build spatial index for efficiency
    sindex = gdf_metric.sindex
    
    edges_found = 0
    
    for i in range(N):
        geom_i = gdf_metric.geometry.iloc[i]
        
        # Query spatial index for potential neighbors (touches predicate)
        # Use bounding box query as spatial index may not support touches directly
        potential_neighbors = list(sindex.query(geom_i, predicate="intersects"))
        
        for j in potential_neighbors:
            if j <= i:  # Only process upper triangle
                continue
            
            geom_j = gdf_metric.geometry.iloc[j]
            
            # Check if they touch
            if not geom_i.touches(geom_j):
                continue
            
            # Compute boundary intersection length
            try:
                boundary_int = geom_i.boundary.intersection(geom_j.boundary)
                int_length = boundary_int.length  # meters in EPSG:6933
            except Exception as e:
                print(f"[WARN] Error computing boundary intersection for ({i},{j}): {e}")
                continue
            
            # Classify as border-sharing if length > tolerance
            if int_length > args.touch_eps_m:
                adjacency[i, j] = 1
                adjacency[j, i] = 1
                edges_found += 1
    
    # Validate adjacency
    if not np.array_equal(adjacency, adjacency.T):
        print(f"[FAIL] Adjacency matrix not symmetric", file=sys.stderr)
        sys.exit(1)
    
    if not np.all(np.diag(adjacency) == 0):
        print(f"[FAIL] Adjacency diagonal not zero", file=sys.stderr)
        sys.exit(1)
    
    degrees = np.sum(adjacency, axis=1)
    degree_min = int(np.min(degrees))
    degree_max = int(np.max(degrees))
    degree_mean = float(np.mean(degrees))
    
    print(f"[OK] Adjacency matrix computed")
    print(f"  Edges: {edges_found}")
    print(f"  Degree: min={degree_min}, max={degree_max}, mean={degree_mean:.1f}")
    print(f"  Symmetry: OK")
    print()
    
    # Write outputs
    print(f"[5/5] Writing outputs...")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    distance_path = out_dir / "distance_km.npy"
    adjacency_path = out_dir / "adjacency_border.npy"
    qc_path = out_dir / "spatial_meta_qc.json"
    
    np.save(distance_path, distance_km)
    np.save(adjacency_path, adjacency)
    
    # QC report
    qc = {
        "N": int(N),
        "node_id_range": [int(gdf["node_id"].min()), int(gdf["node_id"].max())],
        "node_id_missing_count": 0 if args.require_node_ids else len(expected_ids - node_ids),
        "distance_km": {
            "dtype": "float32",
            "shape": list(distance_km.shape),
            "min_nonzero": float(dist_min_nonzero),
            "max": float(dist_max),
            "symmetry_max_diff": float(np.max(np.abs(distance_km - distance_km.T))),
            "diagonal_max": float(diag_max),
        },
        "adjacency_border": {
            "dtype": "uint8",
            "shape": list(adjacency.shape),
            "num_edges": int(edges_found),
            "degree_min": degree_min,
            "degree_max": degree_max,
            "degree_mean": degree_mean,
            "connected_nodes": int(np.sum(degrees > 0)),
            "symmetry_ok": True,
            "diagonal_ok": True,
        },
        "crs_used": {
            "distance": "EPSG:4326",
            "adjacency": "EPSG:6933",
        },
        "touch_eps_m": float(args.touch_eps_m),
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    
    with qc_path.open("w", encoding="utf-8") as f:
        json.dump(qc, f, indent=2)
    
    print(f"[OK] Wrote: {distance_path}")
    print(f"[OK] Wrote: {adjacency_path}")
    print(f"[OK] Wrote: {qc_path}")
    print()
    print(f"[COMPLETE] Spatial meta matrices built successfully")


if __name__ == "__main__":
    main()
