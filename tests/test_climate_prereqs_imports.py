def test_prereqs_imports():
    """Confirms all required packages including pyarrow are importable."""
    required = [
        'cdsapi',
        'xarray',
        'netCDF4',
        'numpy',
        'pandas',
        'zarr',
        'geopandas',
        'pyproj',
        'shapely',
        'pyogrio',
        'exactextract',
        'pyarrow'
    ]
    import importlib.util
    missing = [p for p in required if importlib.util.find_spec(p) is None]
    assert not missing, f"Missing required packages: {missing}. Run: conda install -n pathograph-pre -c conda-forge pyarrow -y"
