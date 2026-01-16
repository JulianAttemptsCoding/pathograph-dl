from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib.request import urlretrieve

import geopandas as gpd
import pandas as pd

# Natural Earth 10m Admin 0 layers (try multiple to maximize coverage for small states/territories)
NE_DATASETS = [
    # Best general countries layer
    (
        "admin_0_countries",
        "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip",
        "ne_10m_admin_0_countries.shp",
    ),
    # Often includes additional “map units” that can cover some missing microstates/territories
    (
        "admin_0_map_units",
        "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_map_units.zip",
        "ne_10m_admin_0_map_units.shp",
    ),
    # Sometimes differs in coverage/attributes
    (
        "admin_0_sovereignty",
        "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_sovereignty.zip",
        "ne_10m_admin_0_sovereignty.shp",
    ),
]


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    node_index_primary: Path
    node_index_fallback: Path
    out_gpkg: Path
    out_layer: str
    raw_shapes_dir: Path


def _paths() -> Paths:
    repo_root = Path(".")
    return Paths(
        repo_root=repo_root,
        node_index_primary=Path("data/processed/trade/imf_imts_step1/node_index.csv"),
        node_index_fallback=Path("data/processed/meta/node_index.csv"),
        out_gpkg=Path("data/processed/meta/node_geometry.gpkg"),
        out_layer="node_geometry",
        raw_shapes_dir=Path("data/raw/shapes/naturalearth_admin0"),
    )


def _read_node_index(p: Paths) -> pd.DataFrame:
    node_index = p.node_index_primary if p.node_index_primary.exists() else p.node_index_fallback
    if not node_index.exists():
        raise FileNotFoundError(
            "node_index.csv not found. Expected one of:\n"
            f"  - {p.node_index_primary}\n"
            f"  - {p.node_index_fallback}"
        )

    nodes = pd.read_csv(node_index)
    required = {"node_id", "iso3"}
    missing = required - set(nodes.columns)
    if missing:
        raise ValueError(f"{node_index} missing required columns: {missing}. Found: {list(nodes.columns)}")

    nodes = nodes.copy()
    nodes["iso3"] = nodes["iso3"].astype(str).str.strip()
    if nodes["iso3"].str.len().ne(3).any():
        bad = nodes.loc[nodes["iso3"].str.len().ne(3), "iso3"].tolist()[:40]
        raise ValueError(f"node_index.csv contains non-ISO3 codes (len!=3). Examples: {bad}")

    if not nodes["node_id"].is_unique or not nodes["iso3"].is_unique:
        raise ValueError("node_index.csv must have unique node_id and unique iso3.")

    if len(nodes) != 194:
        raise ValueError(f"Expected 194 nodes in node_index.csv, got {len(nodes)}.")

    return nodes


def _download_and_extract(url: str, zip_path: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        print(f"[INFO] Downloading Natural Earth: {url}")
        urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    # Be robust to nested directories inside the zip
    shp_files = list(extract_dir.rglob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No .shp found after extracting to: {extract_dir}")
    # Prefer the largest shp (usually the real layer, not tiny ancillary)
    shp_files.sort(key=lambda p: p.stat().st_size, reverse=True)
    return shp_files[0]


def _choose_iso_series(df: gpd.GeoDataFrame) -> pd.Series:
    """
    Construct a robust ISO3 series from Natural Earth attributes.

    Many NE layers contain multiple possible ISO3-like fields; some rows have '-99'.
    We try in order and accept a 3-letter code that is not '-99'.
    """
    candidates = [
        "ISO_A3_EH",  # often best
        "ADM0_A3",    # often best
        "ISO_A3",     # sometimes '-99'
        "GU_A3",
        "SU_A3",
        "SOV_A3",
    ]
    cols = [c for c in candidates if c in df.columns]

    if not cols:
        raise KeyError(f"No known ISO3 columns found. Available columns: {list(df.columns)}")

    # Start with all empty
    iso = pd.Series([""] * len(df), index=df.index, dtype="string")

    for c in cols:
        s = df[c].astype("string").fillna("").str.strip()
        # Accept only 3-letter codes that are not '-99'
        ok = (s.str.len() == 3) & (s != "-99") & (iso == "")
        iso.loc[ok] = s.loc[ok]

    return iso


def _fix_geometry(g: gpd.GeoSeries) -> gpd.GeoSeries:
    """
    Fix common topology issues. Prefer make_valid if available; fallback to buffer(0).
    """
    try:
        # Shapely 2.x exposes shapely.make_valid
        import shapely  # type: ignore

        if hasattr(shapely, "make_valid"):
            return g.apply(lambda geom: shapely.make_valid(geom) if geom is not None else geom)
    except Exception:
        pass

    # Fallback that often resolves self-intersections
    return g.buffer(0)


def _load_layer_as_iso3_geoms(
    name: str,
    url: str,
    expected_shp_name: str,
    iso3_universe: set[str],
    base_dir: Path,
) -> gpd.GeoDataFrame:
    """
    Download/extract/read one Natural Earth layer and return dissolved geometries by iso3.
    """
    zip_path = base_dir / f"{expected_shp_name}.zip".replace(".shp.zip", ".zip")  # harmless
    extract_dir = base_dir / name

    shp = _download_and_extract(url, zip_path, extract_dir)
    print(f"[INFO] Reading polygons ({name}): {shp}")
    world = gpd.read_file(shp)

    iso = _choose_iso_series(world)
    world = world.copy()
    world["iso3"] = iso.astype(str).str.strip()

    # Filter to our ISO3 universe
    sub = world[world["iso3"].isin(iso3_universe)].copy()
    if sub.empty:
        print(f"[WARN] Layer {name} produced 0 matches to node ISO3 universe.")
        return gpd.GeoDataFrame({"iso3": [], "geometry": []}, geometry="geometry", crs="EPSG:4326")

    # Keep only needed columns, dissolve to one geometry per iso3
    sub = sub[["iso3", "geometry"]]
    sub = sub.dissolve(by="iso3", as_index=False)

    # Fix topology
    sub["geometry"] = _fix_geometry(sub["geometry"])
    sub = sub.set_crs("EPSG:4326", allow_override=True)

    # Drop any empty geometries
    sub = sub[~sub["geometry"].isna()].copy()
    return sub


def _merge_layers(layers: Iterable[gpd.GeoDataFrame], iso3_universe: list[str]) -> gpd.GeoDataFrame:
    """
    Merge iso3->geometry from multiple layers. First non-null geometry wins.
    """
    out = pd.DataFrame({"iso3": iso3_universe})
    geom_map: dict[str, object] = {}

    for layer in layers:
        if layer.empty:
            continue
        for _, row in layer.iterrows():
            code = str(row["iso3"]).strip()
            if code and (code not in geom_map) and (row["geometry"] is not None):
                geom_map[code] = row["geometry"]

    out["geometry"] = out["iso3"].map(geom_map)
    gdf = gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")
    return gdf


def main() -> int:
    p = _paths()
    nodes = _read_node_index(p)
    iso3_list = nodes["iso3"].tolist()
    iso3_set = set(iso3_list)

    layers: list[gpd.GeoDataFrame] = []
    for name, url, shp_name in NE_DATASETS:
        try:
            layer = _load_layer_as_iso3_geoms(
                name=name,
                url=url,
                expected_shp_name=shp_name,
                iso3_universe=iso3_set,
                base_dir=p.raw_shapes_dir,
            )
            layers.append(layer)
        except Exception as e:
            print(f"[WARN] Failed loading layer {name}: {e}")

    merged = _merge_layers(layers, iso3_list)

    # Join to node_id axis
    out = nodes[["node_id", "iso3"]].merge(merged, on="iso3", how="left")
    gdf = gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")

    # Validate coverage
    missing = gdf[gdf["geometry"].isna()]["iso3"].tolist()
    if missing:
        raise RuntimeError(
            f"Missing geometries for {len(missing)} ISO3 codes:\n"
            + "\n".join(missing[:60])
            + ("\n...(truncated)" if len(missing) > 60 else "")
        )

    # Validate shape
    if len(gdf) != 194:
        raise RuntimeError(f"Expected 194 rows, got {len(gdf)}")

    if not gdf["node_id"].is_unique or not gdf["iso3"].is_unique:
        raise RuntimeError("node_id/iso3 not unique in output")

    # Validate geometry validity
    # Some valid checks can be expensive; we do a strict check anyway because this gates climate aggregation.
    valid_mask = gdf.geometry.is_valid
    if not bool(valid_mask.all()):
        bad = gdf.loc[~valid_mask, "iso3"].tolist()
        raise RuntimeError(f"Invalid geometries remain for: {bad[:40]}")

    # Write output
    p.out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Writing: {p.out_gpkg} (layer={p.out_layer})")
    gdf.to_file(p.out_gpkg, layer=p.out_layer, driver="GPKG")
    print("[OK] Wrote node_geometry.gpkg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
