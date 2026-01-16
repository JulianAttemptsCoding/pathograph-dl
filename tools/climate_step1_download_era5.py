from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path
from datetime import datetime

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Missing dependency PyYAML. Install with:\n"
            "  conda run -n pathograph-pre python -m pip install pyyaml\n"
            f"Error: {e}"
        )
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _parse_years_arg(years_arg: str, start: int, end: int) -> list[int]:
    s = years_arg.strip().lower()
    if s == "all":
        return list(range(start, end + 1))
    if ":" in s:
        a, b = s.split(":", 1)
        ya = int(a)
        yb = int(b)
        if ya > yb:
            raise SystemExit(f"Invalid --years range: {years_arg}")
        return list(range(ya, yb + 1))
    return [int(s)]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--years", default="all", help="all | YYYY | YYYY:YYYY")
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    clim = cfg["climate"]
    paths = cfg["paths"]

    raw_dir = Path(paths["raw_netcdf_dir"])
    man_dir = Path(paths["manifests_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    man_dir.mkdir(parents=True, exist_ok=True)
    
    # Temp dir for downloads/extractions
    tmp_dir = raw_dir / "_tmp_download"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    y_start = int(clim["years"]["start"])
    y_end = int(clim["years"]["end"])
    years = _parse_years_arg(args.years, y_start, y_end)

    try:
        import cdsapi  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Missing dependency cdsapi. Install with:\n"
            "  conda run -n pathograph-pre python -m pip install cdsapi\n"
            f"Error: {e}"
        )

    c = cdsapi.Client()

    dataset_id = clim["dataset_id"]
    product_type = clim["product_type"]
    variables = list(clim["variables"])
    months = list(clim["request_defaults"]["months"])
    times = list(clim["request_defaults"]["time"])
    fmt = clim["format"]

    # Single manifest file for all years
    manifest_path = man_dir / "era5_download_manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {"dataset_id": dataset_id, "entries": []}

    # Index existing entries by year for idempotency
    existing_by_year = {int(e["year"]): e for e in manifest.get("entries", []) if "year" in e}

    for year in years:
        out_path = raw_dir / f"era5_sl_monthly_{year}.nc"
        request = {
            "product_type": [product_type],
            "variable": variables,
            "year": [str(year)],
            "month": months,
            "time": times,
            "format": fmt,
        }

        # Check existing and valid
        if out_path.exists():
            # If it's a zip, it's invalid (previous run failed to clean up or user download)
            if zipfile.is_zipfile(out_path):
                print(f"[WARN] {year} Output file is a ZIP, forcing re-download: {out_path}")
                out_path.unlink()
            else:
                sha = _sha256(out_path)
                prev = existing_by_year.get(year)
                if prev and prev.get("sha256") == sha:
                    print(f"[SKIP] {year} exists and hash matches manifest: {out_path}")
                    continue
                print(f"[WARN] {year} exists but hash differs or not in manifest; keeping file and updating manifest.")

        print(f"[DOWNLOAD] {year} -> tmp")
        
        # Download to temp file
        raw_tmp_path = tmp_dir / f"download_{year}.tmp"
        
        try:
            c.retrieve(dataset_id, request, str(raw_tmp_path))
        except Exception as e:
            err_str = str(e).lower()
            if "required licences not accepted" in err_str:
                print("\n" + "="*60)
                print("CDS_LICENCE_NOT_ACCEPTED")
                print("Action Required: Log in to CDS and accept the required licences for reanalysis-era5-single-levels-monthly-means.")
                print("URL: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means?tab=download#manage-licences")
                print("="*60 + "\n")
                exit(2)
            else:
                raise e

        # Handle ZIP or NetCDF
        was_zip = False
        archive_sha = None
        
        if zipfile.is_zipfile(raw_tmp_path):
            was_zip = True
            print(f"[INFO] {year} Detected ZIP archive.")
            archive_sha = _sha256(raw_tmp_path)
            
            with zipfile.ZipFile(raw_tmp_path) as z:
                names = z.namelist()
                nc_members = [n for n in names if n.lower().endswith(".nc")]
                
                if not nc_members:
                    print(f"[ERROR] ZIP contains no .nc files. Members: {names[:50]}")
                    raise RuntimeError(f"ZIP contains no .nc files. See logs.")
                
                print(f"[INFO] Found {len(nc_members)} NetCDF members in ZIP: {nc_members}")
                
                extracted_files = []
                for mem in nc_members:
                    z.extract(mem, path=tmp_dir)
                    extracted_files.append(tmp_dir / mem)
                
                if len(extracted_files) == 1:
                    print(f"[INFO] Single member, moving to {out_path}...")
                    if out_path.exists():
                        out_path.unlink()
                    shutil.move(str(extracted_files[0]), str(out_path))
                else:
                    # Merge multiple members
                    print(f"[INFO] Merging {len(extracted_files)} files into {out_path}...")
                    try:
                        import xarray as xr
                    except ImportError:
                        raise SystemExit("Missing xarray for ZIP merging. Install xarray.")
                        
                    # Load all and merge
                    dsets = []
                    try:
                        for f in extracted_files:
                            ds = xr.open_dataset(f)
                            dsets.append(ds)
                        
                        # Merge (compat='override' might be needed if coords differ slightly, but for ERA5 usually exact)
                        merged = xr.merge(dsets, join='outer', compat='no_conflicts')
                        merged.to_netcdf(out_path)
                        print(f"[INFO] Merge complete.")
                    finally:
                        for ds in dsets:
                            ds.close()

            # Cleanup temp zip
            raw_tmp_path.unlink()
            
        else:
            # Not a zip, just move it
            print(f"[INFO] {year} Downloaded direct NetCDF.")
            if out_path.exists():
                out_path.unlink()
            shutil.move(str(raw_tmp_path), str(out_path))

        # Final SHA
        sha = _sha256(out_path)
        print(f"[OK] {year} -> {out_path} (SHA: {sha[:8]}...)")
        
        entry = {
            "year": year,
            "path": str(out_path).replace("\\", "/"),
            "bytes": out_path.stat().st_size,
            "sha256": sha,
            "was_zip": was_zip,
            "request": request,
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        if archive_sha:
            entry["archive_sha256"] = archive_sha
            
        existing_by_year[year] = entry

        # Rewrite manifest deterministically sorted by year
        manifest["entries"] = [existing_by_year[y] for y in sorted(existing_by_year.keys())]
        tmp = manifest_path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=False)
        os.replace(tmp, manifest_path)

    # Cleanup temp dir
    try:
        shutil.rmtree(tmp_dir)
    except:
        pass

    print(f"[OK] Wrote manifest: {manifest_path}")

if __name__ == "__main__":
    main()
