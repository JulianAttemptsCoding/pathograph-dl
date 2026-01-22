import os


def test_prereqs_imports():
    """Confirms all required packages including pyarrow are importable.
    
    This test checks for optional climate preprocessing dependencies.
    If PATHOGRAPH_REQUIRE_CLIMATE_EXTRAS env var is set to 1, failures cause test failure.
    Otherwise, missing optional packages cause test skip (default behavior).
    """
    # Core packages always required
    core_required = [
        'numpy',
        'pandas',
        'zarr',
    ]
    
    # Optional climate preprocessing packages
    climate_optional = [
        'cdsapi',
        'xarray',
        'netCDF4',
        'geopandas',
        'pyproj',
        'shapely',
        'pyogrio',
        'exactextract',
        'pyarrow'
    ]
    
    import importlib.util
    
    # Check core packages (must always be present)
    core_missing = [p for p in core_required if importlib.util.find_spec(p) is None]
    assert not core_missing, f"Missing core packages: {core_missing}"
    
    # Check optional climate packages
    climate_missing = [p for p in climate_optional if importlib.util.find_spec(p) is None]
    
    # If env var is set, require all; otherwise skip if missing
    require_climate = os.getenv("PATHOGRAPH_REQUIRE_CLIMATE_EXTRAS", "0") == "1"
    
    if climate_missing:
        if require_climate:
            assert False, (
                f"Missing required climate packages: {climate_missing}. "
                f"Run: conda install -n pathograph-pre -c conda-forge {' '.join(climate_missing)} -y"
            )
        else:
            # pytest.skip if available, otherwise just pass
            try:
                import pytest
                pytest.skip(
                    f"Skipping climate prereqs test: optional packages missing: {climate_missing}. "
                    f"Set PATHOGRAPH_REQUIRE_CLIMATE_EXTRAS=1 to require them."
                )
            except ImportError:
                # pytest not available, just pass silently
                pass
