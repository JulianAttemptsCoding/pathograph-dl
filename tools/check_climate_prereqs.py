import sys
from pathlib import Path
import cdsapi

def main():
    print(f"PYTHON: {sys.version}")
    
    # Check imports
    required = ['cdsapi','xarray','netCDF4','numpy','pandas','zarr','geopandas','pyproj','shapely','pyogrio','exactextract','pyarrow']
    import importlib.util
    missing = [p for p in required if importlib.util.find_spec(p) is None]
    if missing:
        print(f"MISSING_PACKAGES: {missing}")
        sys.exit(1)
    print("PACKAGES_OK: All required packages importable.")

    # Check credentials
    creds = Path.home() / ".cdsapirc"
    if not creds.exists():
        print(f"MISSING_CREDS: {creds} not found.")
        sys.exit(1)
    print(f"CREDS_OK: Found {creds}")
    
    # Check initialization
    try:
        c = cdsapi.Client()
        print("CLIENT_INIT_OK")
    except Exception as e:
        print(f"CLIENT_INIT_FAIL: {e}")
        sys.exit(1)

    # Deterministic probe for licence
    # Request a tiny slice: 2024-01-01 00:00, 1 variable
    print("PROBE: Attempting deterministic licence check (2024-01-01, 1 variable)...")
    probe_dir = Path("data/raw/era5_monthly_netcdf/_probe")
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_file = probe_dir / "licence_probe.nc"
    
    if probe_file.exists():
        probe_file.unlink()

    request = {
        "product_type": "monthly_averaged_reanalysis",
        "variable": "2m_temperature",
        "year": "2024",
        "month": "01",
        "time": "00:00",
        "format": "netcdf",
    }
    
    try:
        c.retrieve("reanalysis-era5-single-levels-monthly-means", request, str(probe_file))
        print("CDS_LICENCE_OK")
        if probe_file.exists():
            probe_file.unlink()
    except Exception as e:
        err_str = str(e).lower()
        # Look for 403 and licence text
        # 'required licences not accepted' is standard CDS text
        if "required licences not accepted" in err_str:
            print("\n" + "="*60)
            print("CDS_LICENCE_NOT_ACCEPTED")
            print("Action Required: Log in to CDS and accept the required licences.")
            print("URL: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means?tab=download#manage-licences")
            print("="*60 + "\n")
            sys.exit(2) # Distinct exit code for known blocker
        else:
            print(f"PROBE_FAIL: Unexpected error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
