from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import zipfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Missing dependency PyYAML. Install:\n"
            "  conda run -n pathograph-pre python -m pip install pyyaml\n"
            f"Error: {e}"
        )
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

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

def _coord_name(ds, candidates: list[str]) -> str:
    for c in candidates:
        if c in ds.coords:
            return c
        if c in ds.dims:
            return c
    raise SystemExit(f"Missing required coordinate among {candidates}. Found coords={list(ds.coords)} dims={list(ds.dims)}")

def _normalize_lon(da, lon_name: str):
    lon = np.asarray(da[lon_name].values)
    if np.nanmax(lon) <= 180.0:
        return da
    lon_norm = ((lon + 180.0) % 360.0) - 180.0
    da = da.assign_coords({lon_name: lon_norm})
    da = da.sortby(lon_name)
    return da

def _ensure_lat_desc(da, lat_name: str):
    lat = np.asarray(da[lat_name].values)
    if len(lat) < 2:
        return da
    if lat[0] < lat[-1]:
        da = da.sortby(lat_name, ascending=False)
    return da

def _select_expver(da, policy: dict):
    if "expver" not in da.dims:
        return da
    if not policy.get("enabled", True):
        return da

    vals = np.asarray(da["expver"].values)
    prefer = policy.get("prefer_value", 1)

    # Try numeric comparison deterministically
    try:
        vals_num = vals.astype(int)
        if prefer in set(vals_num.tolist()):
            return da.sel(expver=prefer)
        mx = int(np.max(vals_num))
        return da.sel(expver=mx)
    except Exception:
        # Non-numeric expver; only allow if prefer can be matched exactly
        if prefer in set(vals.tolist()):
            return da.sel(expver=prefer)
        if policy.get("stop_on_ambiguous", True):
            raise SystemExit(f"Ambiguous non-numeric expver values={vals.tolist()} and prefer={prefer} not found.")
        # If not stopping, pick first (but we do NOT allow this by contract)
        raise SystemExit("expver is non-numeric and ambiguous; stopping per contract.")

def _expect_units(units: str | None, expected_any_of: list[str], var_name: str) -> None:
    if units is None:
        raise SystemExit(f"{var_name}: missing units attribute; STOP.")
    if units not in expected_any_of:
        raise SystemExit(f"{var_name}: unexpected units='{units}' expected one of {expected_any_of}; STOP.")

def _convert_tp_to_mm_month(tp_2d: np.ndarray, units: str | None, year: int, month: int) -> np.ndarray:
    if units is None:
        raise SystemExit("tp: missing units attribute; STOP.")

    u = units.strip()
    # depth case
    if u in {"m", "meter", "metre", "meters", "metres"}:
        return tp_2d * 1000.0

    # rate case
    if ("s-1" in u) or ("sec-1" in u):
        # seconds in calendar month
        days = pd.Period(f"{year:04d}-{month:02d}").days_in_month
        seconds = float(days) * 24.0 * 3600.0
        return tp_2d * seconds * 1000.0

    raise SystemExit(f"tp: unexpected units='{units}'. Accepted: depth(m) or rate(*s-1). STOP.")

def _es_hpa(T_c: np.ndarray) -> np.ndarray:
    # 6.112 * exp((17.67*T)/(T+243.5))  where T in Celsius, output in hPa
    return 6.112 * np.exp((17.67 * T_c) / (T_c + 243.5))

def _deduplicate_by_expver(ds, time_name: str):
    """Handles cases where expver is concatenated along time dimension."""
    if "expver" not in ds.coords or time_name not in ds.coords:
        return ds
        
    # Check if time dimension is effectively duplicated
    times = ds[time_name].values
    if len(times) <= 12:
        return ds

    # If expver is a dimension, we don't need this (xarray handles it via sel)
    # But if expver is a coordinate along time (flattened), we must filter.
    if "expver" in ds.dims:
        return ds

    # Logic: Group by month, prefer expver='1' or numeric '1'
    expvers = ds["expver"].values
    
    # normalize expver to score (1=high, 5=medium, others=low)
    def _score(e):
        s = str(e).strip()
        if s == '1' or s == '0001': return 10
        if s == '5' or s == '0005': return 5
        return 0

    import pandas as pd
    df = pd.DataFrame({"t": times, "e": expvers, "i": range(len(times))})
    df["m"] = pd.to_datetime(df["t"]).dt.to_period("M")
    df["score"] = df["e"].apply(_score)
    
    # Pick best score per month
    df = df.sort_values(["m", "score"], ascending=[True, False])
    df = df.drop_duplicates("m", keep="first")
    df = df.sort_values("i")
    
    print(f"[INFO] Deduplicating time/expver: reduced {len(times)} -> {len(df)} steps.")
    return ds.isel({time_name: df["i"].tolist()})

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--years", default="all", help="all | YYYY | YYYY:YYYY")
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    clim = cfg["climate"]
    paths = cfg["paths"]
    proc = cfg["processing"]

    # Dependencies required by contract
    try:
        import xarray as xr  # type: ignore
    except Exception as e:
        raise SystemExit(f"Missing xarray. Error: {e}")

    try:
        import geopandas as gpd  # type: ignore
    except Exception as e:
        raise SystemExit(f"Missing geopandas. Error: {e}")

    try:
        import pyarrow  # noqa: F401
    except Exception as e:
        raise SystemExit(
            "Missing pyarrow (required for Parquet). Install:\n"
            "  conda run -n pathograph-pre python -m pip install pyarrow\n"
            f"Error: {e}"
        )

    try:
        from exactextract import exact_extract  # type: ignore
        from exactextract.raster import NumPyRasterSource  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Missing exactextract. Install:\n"
            "  conda run -n pathograph-pre python -m pip install exactextract\n"
            f"Error: {e}"
        )

    # Load geometry + node index
    node_index = pd.read_csv(paths["node_index"])
    if "node_id" not in node_index.columns or "iso3" not in node_index.columns:
        raise SystemExit(f"node_index missing required columns. cols={list(node_index.columns)}")

    gdf = gpd.read_file(paths["node_geometry"])
    if len(gdf) != 194:
        raise SystemExit(f"node_geometry unexpected rowcount: {len(gdf)} != 194")
    if gdf.crs is None:
        raise SystemExit("node_geometry missing CRS; STOP.")
    if str(gdf.crs).upper().endswith("4326") is False and "4326" not in str(gdf.crs):
        raise SystemExit(f"node_geometry CRS must be EPSG:4326; found {gdf.crs}")

    # Ensure we have node_id, iso3 in geometry (if not, merge)
    if "node_id" not in gdf.columns or "iso3" not in gdf.columns:
        gdf = gdf.merge(node_index[["node_id", "iso3"]], on="node_id", how="left")
    if gdf["iso3"].isna().any():
        raise SystemExit("node_geometry iso3 merge produced NaNs; STOP.")

    # Precompute centroids for fallback
    # Fix warning: Project to 3857 for centroid calc, then back to 4326
    try:
        if gdf.crs:
            gdf_curr = gdf.to_crs(epsg=3857)
            centroids_3857 = gdf_curr.geometry.centroid
            centroids = centroids_3857.to_crs(epsg=4326)
        else:
            centroids = gdf.geometry.centroid
    except Exception:
         # Fallback if reprojection fails (shouldn't happen with valid install)
        centroids = gdf.geometry.centroid

    gdf["_centroid_lon"] = centroids.x
    gdf["_centroid_lat"] = centroids.y

    raw_dir = Path(paths["raw_netcdf_dir"])
    out_root = Path(paths["processed_country_month_dir"])
    man_dir = Path(paths["manifests_dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    man_dir.mkdir(parents=True, exist_ok=True)

    y_start = int(clim["years"]["start"])
    y_end = int(clim["years"]["end"])
    years = _parse_years_arg(args.years, y_start, y_end)

    # Load time_index_master to define valid month_index range
    ti_master_path = Path(paths["time_index_master"])
    if not ti_master_path.exists():
         raise SystemExit(f"Missing time_index_master: {ti_master_path}")
    ti_master = np.load(ti_master_path)
    TMIN = int(ti_master.min())
    TMAX = int(ti_master.max())
    print(f"[INFO] time_index_master loaded. Valid month_index range: [{TMIN}, {TMAX}]")

    # Variable aliases in ERA5 NetCDF
    VAR_MAP = {
        "2m_temperature": "t2m",
        "2m_dewpoint_temperature": "d2m",
        "surface_pressure": "sp",
        "mean_sea_level_pressure": "msl",
        "10m_u_component_of_wind": "u10",
        "10m_v_component_of_wind": "v10",
        "total_precipitation": "tp",
    }

    for year in years:
        nc_path = raw_dir / f"era5_sl_monthly_{year}.nc"
        if not nc_path.exists():
            raise SystemExit(f"Missing NetCDF for year={year}: {nc_path}")

        # Guard against ZIP input
        if zipfile.is_zipfile(nc_path):
            raise RuntimeError(f"INPUT_IS_ZIP: expected NetCDF. Re-run Step 1 after ZIP-extraction fix. Path={nc_path}")

        print(f"[OPEN] {nc_path}")
        # Explicit engine selection
        ds = xr.open_dataset(nc_path, engine="netcdf4")

        lat_name = _coord_name(ds, ["latitude", "lat"])
        lon_name = _coord_name(ds, ["longitude", "lon"])
        time_name = _coord_name(ds, ["time", "valid_time"])
        
        # Deduplicate duplicated expver/time
        ds = _deduplicate_by_expver(ds, time_name)

        # Prepare per-variable DataArrays (expver selection + lon normalize + lat descending)
        da_dict = {}
        for v_long, v_short in VAR_MAP.items():
            if v_short not in ds.data_vars and v_long in ds.data_vars:
                da = ds[v_long]
            elif v_short in ds.data_vars:
                da = ds[v_short]
            elif v_long in ds.data_vars:
                da = ds[v_long]
            else:
                raise SystemExit(f"NetCDF missing variable '{v_long}'/'{v_short}'. Found: {list(ds.data_vars)}")

            da = _select_expver(da, proc["expver_policy"])
            da = _normalize_lon(da, lon_name)
            da = _ensure_lat_desc(da, lat_name)
            da_dict[v_short] = da

        lon = np.asarray(da_dict["t2m"][lon_name].values, dtype=float)
        lat = np.asarray(da_dict["t2m"][lat_name].values, dtype=float)

        if lon.ndim != 1 or lat.ndim != 1:
            raise SystemExit("Expected 1D lon/lat coordinates; STOP.")

        if len(lon) < 2 or len(lat) < 2:
            raise SystemExit("lon/lat too short; STOP.")

        # Assume regular grid; verify near-constant spacing
        dxs = np.diff(lon)
        dys = np.diff(lat)
        dx = float(np.median(dxs))
        dy = float(np.median(np.abs(dys)))

        if not np.allclose(dxs, dx, rtol=0, atol=1e-6):
            raise SystemExit("Non-uniform lon spacing; STOP.")
        if not np.allclose(np.abs(dys), dy, rtol=0, atol=1e-6):
            raise SystemExit("Non-uniform lat spacing; STOP.")
        if dx <= 0 or dy <= 0:
            raise SystemExit("Invalid dx/dy; STOP.")

        xmin = float(lon[0]) - dx / 2.0
        xmax = float(lon[-1]) + dx / 2.0
        ymax = float(lat[0]) + dy / 2.0  # lat is descending => first is max
        ymin = float(lat[-1]) - dy / 2.0

        # Time coordinate -> months
        times = pd.to_datetime(np.asarray(da_dict["t2m"][time_name].values))
        if len(times) != 12:
            # Monthly file should normally have 12 steps; STOP if not (no guessing)
            raise SystemExit(f"Expected 12 monthly timesteps in {year}; found {len(times)}; STOP.")

        # Output per year partition
        year_dir = out_root / f"year={year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        out_parquet = year_dir / f"country_month_{year}.parquet"

        rows = []
        fallback_count = 0
        units_seen = {}

        for i, ts in enumerate(times):
            mo = int(ts.month)

            # Extract 2D arrays for base vars
            t2m = np.asarray(da_dict["t2m"].isel({time_name: i}).values, dtype=float)
            d2m = np.asarray(da_dict["d2m"].isel({time_name: i}).values, dtype=float)
            sp = np.asarray(da_dict["sp"].isel({time_name: i}).values, dtype=float)
            msl = np.asarray(da_dict["msl"].isel({time_name: i}).values, dtype=float)
            u10 = np.asarray(da_dict["u10"].isel({time_name: i}).values, dtype=float)
            v10 = np.asarray(da_dict["v10"].isel({time_name: i}).values, dtype=float)
            tp = np.asarray(da_dict["tp"].isel({time_name: i}).values, dtype=float)

            # Units validation + conversions
            t2m_units = da_dict["t2m"].attrs.get("units")
            d2m_units = da_dict["d2m"].attrs.get("units")
            sp_units = da_dict["sp"].attrs.get("units")
            msl_units = da_dict["msl"].attrs.get("units")
            tp_units = da_dict["tp"].attrs.get("units")

            units_seen.update({
                "t2m.units": t2m_units,
                "d2m.units": d2m_units,
                "sp.units": sp_units,
                "msl.units": msl_units,
                "tp.units": tp_units,
            })

            _expect_units(t2m_units, proc["unit_rules"]["temperature_expected_any_of"], "t2m")
            _expect_units(d2m_units, proc["unit_rules"]["temperature_expected_any_of"], "d2m")
            _expect_units(sp_units, proc["unit_rules"]["pressure_expected_any_of"], "sp")
            _expect_units(msl_units, proc["unit_rules"]["pressure_expected_any_of"], "msl")

            t2m_c = t2m - 273.15
            d2m_c = d2m - 273.15
            sp_pa = sp
            msl_pa = msl
            tp_mm = _convert_tp_to_mm_month(tp, tp_units, year, mo)

            # Build raster sources (names chosen so output columns are deterministic)
            rasters = [
                NumPyRasterSource(np.ma.masked_invalid(t2m_c), xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, name="t2m_c", nodata=np.nan),
                NumPyRasterSource(np.ma.masked_invalid(d2m_c), xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, name="d2m_c", nodata=np.nan),
                NumPyRasterSource(np.ma.masked_invalid(sp_pa), xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, name="sp_pa", nodata=np.nan),
                NumPyRasterSource(np.ma.masked_invalid(msl_pa), xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, name="msl_pa", nodata=np.nan),
                NumPyRasterSource(np.ma.masked_invalid(u10), xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, name="u10", nodata=np.nan),
                NumPyRasterSource(np.ma.masked_invalid(v10), xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, name="v10", nodata=np.nan),
                NumPyRasterSource(np.ma.masked_invalid(tp_mm), xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, name="tp_mm_month", nodata=np.nan),
            ]

            df = exact_extract(
                rasters,
                str(Path(paths["node_geometry"])),
                ["mean"],
                include_cols=["node_id", "iso3"],
                output="pandas"
            )

            # Rename columns from exactextract naming: <name>_mean
            rename = {
                "t2m_c_mean": "t2m_mean_c",
                "d2m_c_mean": "d2m_mean_c",
                "sp_pa_mean": "sp_mean_pa",
                "msl_pa_mean": "msl_mean_pa",
                "u10_mean": "u10_mean",
                "v10_mean": "v10_mean",
                "tp_mm_month_mean": "tp_mean_mm_month",
            }
            df = df.rename(columns=rename)

            # Derived features
            df["wind10_speed_mean"] = np.sqrt(df["u10_mean"] ** 2 + df["v10_mean"] ** 2)

            # RH and VPD from means (locked formulas)
            es_t = _es_hpa(df["t2m_mean_c"].to_numpy(dtype=float))
            es_td = _es_hpa(df["d2m_mean_c"].to_numpy(dtype=float))
            rh = 100.0 * (es_td / es_t)
            df["rh_mean"] = rh

            # VPD in kPa: es in hPa => multiply by 0.1 to kPa
            df["vpd_mean_kpa"] = (es_t * (1.0 - (df["rh_mean"].to_numpy(dtype=float) / 100.0))) * 0.1

            df["year"] = year
            df["month"] = mo
            df["month_index"] = (year - 1950) * 12 + (mo - 1)
            df["qc_flags"] = ""

            # Fallback centroid sampling only when ALL base vars are NaN
            base_cols = [
                "t2m_mean_c", "d2m_mean_c", "sp_mean_pa", "msl_mean_pa",
                "u10_mean", "v10_mean", "tp_mean_mm_month"
            ]
            all_nan = df[base_cols].isna().all(axis=1)
            if all_nan.any():
                # Build lookup arrays for nearest-cell sampling
                lat_vals = lat  # descending
                lon_vals = lon  # ascending
                # Precompute full 2D fields for this month for sampling
                fields = {
                    "t2m_mean_c": t2m_c,
                    "d2m_mean_c": d2m_c,
                    "sp_mean_pa": sp_pa,
                    "msl_mean_pa": msl_pa,
                    "u10_mean": u10,
                    "v10_mean": v10,
                    "tp_mean_mm_month": tp_mm,
                }
                # Map node_id -> centroid
                cent = gdf[["node_id", "_centroid_lon", "_centroid_lat"]].copy()
                cent_map = cent.set_index("node_id")[["_centroid_lon", "_centroid_lat"]].to_dict("index")

                for idx in df.index[all_nan]:
                    node_id = int(df.at[idx, "node_id"])
                    cxy = cent_map.get(node_id)
                    if cxy is None:
                        continue
                    clon = float(cxy["_centroid_lon"])
                    clat = float(cxy["_centroid_lat"])
                    ii = int(np.argmin(np.abs(lat_vals - clat)))
                    jj = int(np.argmin(np.abs(lon_vals - clon)))
                    for col, arr2d in fields.items():
                        val = float(arr2d[ii, jj])
                        if math.isnan(val):
                            continue
                        df.at[idx, col] = val
                    df.at[idx, "qc_flags"] = "FALLBACK_CENTROID"
                    fallback_count += 1

                # Recompute derived for rows we filled (safe to recompute vectorized)
                df["wind10_speed_mean"] = np.sqrt(df["u10_mean"] ** 2 + df["v10_mean"] ** 2)
                es_t = _es_hpa(df["t2m_mean_c"].to_numpy(dtype=float))
                es_td = _es_hpa(df["d2m_mean_c"].to_numpy(dtype=float))
                df["rh_mean"] = 100.0 * (es_td / es_t)
                df["vpd_mean_kpa"] = (es_t * (1.0 - (df["rh_mean"].to_numpy(dtype=float) / 100.0))) * 0.1

            # Missing flag
            feature_cols = cfg["processing"]["feature_order_locked"]
            df["is_missing_any"] = df[feature_cols].isna().any(axis=1)

            # Final column order (locked)
            final_cols = [
                "node_id", "iso3", "year", "month", "month_index",
                "t2m_mean_c", "d2m_mean_c", "sp_mean_pa", "msl_mean_pa",
                "u10_mean", "v10_mean", "wind10_speed_mean", "rh_mean",
                "vpd_mean_kpa", "tp_mean_mm_month",
                "is_missing_any", "qc_flags"
            ]
            df = df[final_cols]
            rows.append(df)

        out_df = pd.concat(rows, ignore_index=True)

        # Clip to valid time range
        count_before = len(out_df)
        out_df = out_df[(out_df["month_index"] >= TMIN) & (out_df["month_index"] <= TMAX)].copy()
        count_after = len(out_df)
        if count_before != count_after:
             print(f"[CLIP] year={year} before={count_before} after={count_after} tmin={TMIN} tmax={TMAX}")

        # Invariant check
        if len(out_df) > 0:
             assert out_df["month_index"].between(TMIN, TMAX).all(), f"Clipping failed for year={year}"

        out_df.to_parquet(out_parquet, index=False)

        # Write per-year manifest
        man = {
            "year": year,
            "input_netcdf": str(nc_path).replace("\\", "/"),
            "input_netcdf_sha256": _sha256(nc_path),
            "node_geometry": paths["node_geometry"],
            "node_geometry_sha256": _sha256(Path(paths["node_geometry"])),
            "output_parquet": str(out_parquet).replace("\\", "/"),
            "rows": int(len(out_df)),
            "fallback_centroid_count": int(fallback_count),
            "units_seen": units_seen,
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z"
        }
        man_path = man_dir / f"era5_country_month_manifest_{year}.json"
        tmp = man_path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(man, f, indent=2, sort_keys=False)
        os.replace(tmp, man_path)

        print(f"[OK] Wrote {out_parquet} and manifest {man_path}")

if __name__ == "__main__":
    main()
