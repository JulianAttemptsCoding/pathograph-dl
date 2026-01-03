import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml
import pycountry

ROOT = Path.cwd()
META_DIR = ROOT / "data" / "processed" / "meta"
STRUCTURES_DIR = ROOT / "data" / "raw" / "imf_dots" / "_structures"
RAW_STRUCT = STRUCTURES_DIR
OVERRIDES = ROOT / "overrides" / "imf_ref_area_overrides.csv"
CONFIG_FILE = ROOT / "config" / "trade_ingest.yaml"

NODE_INDEX = META_DIR / "node_index.csv"
OUT_ROSETTA = META_DIR / "rosetta_codes.csv"
OUT_UNMATCHED = META_DIR / "imf_unmatched.csv"
OUT_MANIFEST = META_DIR / "entity_map_manifest.json"

DEFAULT_IMF_BASE = "https://dataservices.imf.org/REST/SDMX_JSON.svc"


def ensure_dirs():
    META_DIR.mkdir(parents=True, exist_ok=True)
    RAW_STRUCT.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[\(\)\[\]\{\},\.;:'\"/\\-]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def load_cfg() -> dict:
    if not CONFIG_FILE.exists():
        return {}
    with CONFIG_FILE.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_imf_base(cfg: dict) -> str:
    return (cfg.get("imf", {}) or {}).get("sdmx_base_url") or DEFAULT_IMF_BASE


def get_dots_hint(cfg: dict) -> str:
    return (cfg.get("imf", {}) or {}).get("dots_dataflow_hint") or "DOTS"


def get_dots_flow_id_override(cfg: dict):
    return (cfg.get("imf", {}) or {}).get("dots_flow_id")


def read_node_index():
    if not NODE_INDEX.exists():
        print(f"Missing node index: {NODE_INDEX}. Run: python -m src.ingest.make_node_index_from_iso3")
        sys.exit(2)
    rows = []
    with NODE_INDEX.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "node_id": int(row["node_id"]),
                "iso3": row["iso3"].strip().upper(),
                "iso2": row.get("iso2", "").strip().upper(),
                "name": row.get("name", "").strip()
            })
    if len(rows) != 194:
        print(f"node_index.csv must have 194 rows; found {len(rows)}")
        sys.exit(2)
    return rows


def read_overrides():
    if not OVERRIDES.exists():
        return {}
    m = {}
    with OVERRIDES.open("r", encoding="utf-8") as f:
        r = csv.DictReader(line for line in f if line.strip() and not line.strip().startswith("#"))
        for row in r:
            iso3 = (row.get("iso3") or "").strip().upper()
            code = (row.get("imf_ref_area") or "").strip()
            name = (row.get("imf_ref_area_name") or "").strip()
            if iso3:
                m[iso3] = (code, name)
    return m


# New helper: fetch with retries, offline-first
def fetch_json_with_retries(base: str, endpoint: str, cache_path: Path, max_attempts: int = 3, timeout: int = 120, offline_only: bool = False):
    """
    Offline-first: if cache_path exists load and return. Otherwise attempt HTTP GET with retries/backoff.
    If offline_only is True, do not attempt network and return None if cache absent.
    Returns parsed JSON on success, or None if fetch failed and no cache is present.
    """
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Warning: failed to read cached JSON {cache_path}: {e}")
            # fall through to attempt fetch (unless offline_only)
    url = base.rstrip("/") + "/" + endpoint.lstrip("/")

    if offline_only:
        print("OFFLINE-ONLY: cache missing for required file:")
        print(f"  Expected cache path: {cache_path}")
        print("To proceed, obtain the file on an internet-enabled machine using tools/imf_structure_pack.py and place it at the path above.")
        return None

    # Print proxy debug if present
    proxies = {k: v for k, v in os.environ.items() if k.lower() in ("http_proxy", "https_proxy")}
    if proxies:
        print("Detected proxy environment variables:")
        for k, v in proxies.items():
            print(f"  {k}={v}")

    attempt = 0
    backoff = 1
    last_exc = None
    while attempt < max_attempts:
        attempt += 1
        try:
            print(f"Fetching {url} (attempt {attempt}/{max_attempts})")
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            cache_path.write_text(resp.text, encoding="utf-8")
            return resp.json()
        except requests.exceptions.RequestException as e:
            last_exc = e
            print(f"Request attempt {attempt} failed: {e}")
            if attempt < max_attempts:
                time.sleep(backoff)
                backoff *= 2
            else:
                break
    # If fetch failed but cache now exists (maybe created concurrently), try to read it
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Failed to read cache after failed fetch: {e}")
    # No data available
    print("ERROR: Unable to fetch from IMF SDMX and no cached file available:")
    print(f"  URL attempted: {url}")
    print(f"  Cache path: {cache_path}")
    print("To proceed offline, place the appropriate JSON file(s) in the following path(s):")
    print(f"  {STRUCTURES_DIR / 'Dataflow.json'}")
    print("Then run: python -m src.ingest.build_entity_map --print-offline-requirements")
    return None


# Replace previous imf_get_json wrapper with new fetch_json_with_retries

def imf_get_json(base: str, endpoint: str, cache_path: Path, offline_only: bool = False) -> dict:
    return fetch_json_with_retries(base, endpoint, cache_path, offline_only=offline_only)


def extract_dataflows(dataflow_json: dict):
    # Try common IMF SDMX-JSON shapes.
    # Shape A: structure->dataflows->dataflow
    try:
        return dataflow_json["structure"]["dataflows"]["dataflow"]
    except Exception:
        pass
    # Shape B: Structure->Dataflows->Dataflow
    try:
        return dataflow_json["Structure"]["Dataflows"]["Dataflow"]
    except Exception:
        pass
    return []


def dataflow_id_and_name(df):
    fid = df.get("id") or df.get("ID") or ""
    name_obj = df.get("name") or df.get("Name") or ""
    if isinstance(name_obj, list) and name_obj:
        name = str(name_obj[0].get("value", ""))
    elif isinstance(name_obj, dict):
        name = str(name_obj.get("value", ""))
    else:
        name = str(name_obj)
    return str(fid), name


def discover_dots_flow_id(dataflow_json: dict, hint: str):
    flows = extract_dataflows(dataflow_json)
    hint_l = (hint or "").lower()
    candidates = []
    for df in flows:
        fid, name = dataflow_id_and_name(df)
        s = (fid + " " + name).lower()
        if ("dots" in s) or ("direction of trade" in s) or (hint_l and hint_l in s):
            candidates.append((fid, name))
    if not candidates:
        # return top flows for inspection
        preview = []
        for df in flows[:50]:
            fid, name = dataflow_id_and_name(df)
            preview.append((fid, name))
        return None, preview
    candidates.sort(key=lambda x: (len(x[0]), x[0]))
    return candidates[0][0], candidates


def extract_ref_area_codelist_id(dsd_json: dict):
    # Try recursive search since paths vary (dimensions vs dimension, casing, etc)
    found = set()
    
    def walk(obj):
        if isinstance(obj, dict):
            # Check if this object is the REF_AREA dimension
            did = str(obj.get("id") or obj.get("ID") or obj.get("concept") or "").upper()
            if did == "REF_AREA":
                 # Extract Codelist ID
                 lr = obj.get("localRepresentation") or obj.get("localrepresentation")
                 if isinstance(lr, dict):
                     enum = lr.get("enumeration") or lr.get("Enumeration")
                     if isinstance(enum, dict):
                         # Standard object: enumeration -> ref -> id
                         ref = enum.get("ref") or enum.get("Ref")
                         if isinstance(ref, dict):
                             found.add(str(ref.get("id") or ref.get("ID") or ""))
                     elif isinstance(enum, str):
                         # URN string
                         # urn:sdmx:org.sdmx.infomodel.codelist.Codelist=IMF:CL_REF_AREA(1.0)
                         tokens = enum.replace("=", ":").replace("(", ":").split(":")
                         # Look for token starting with CL_ or containing REF_AREA
                         for t in tokens:
                             if "REF_AREA" in t and len(t) < 50:
                                 found.add(t)
            
            # Recursive walk
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for i in obj:
                walk(i)

    # Start search in structure root
    walk(dsd_json)
    
    # Filter valid IDs
    valid = [x for x in found if x and x.upper() != "REF_AREA"]
    if valid:
        # Prefer CL_REF_AREA if present
        for v in valid:
            if v == "CL_REF_AREA":
                 return v
        return valid[0]
        
    return None


def parse_codelist(codelist_json: dict, codelist_id: str):
    codes = []
    # SDMX-JSON: structure->codelists->codelist
    try:
        cls = codelist_json["structure"]["codelists"]["codelist"]
        target = None
        for cl in cls:
            if str(cl.get("id", "")) == str(codelist_id):
                target = cl
                break
        if target is None and len(cls) == 1:
            target = cls[0]
        if target is None:
            return []
            
        items = target.get("codes") or target.get("Codes") or target.get("code") or target.get("Code") or []
        for c in items:
            code = str(c.get("id", "")).strip()
            name_obj = c.get("name", {})
            if isinstance(name_obj, list) and name_obj:
                name = str(name_obj[0].get("value", ""))
            elif isinstance(name_obj, dict):
                name = str(name_obj.get("value", ""))
            else:
                name = str(name_obj)
            if code:
                codes.append((code, name))
        return codes
    except Exception:
        pass

    # Fallback: Structure->CodeLists->CodeList
    try:
        cls = codelist_json["Structure"]["CodeLists"]["CodeList"]
        if isinstance(cls, dict):
            cls = [cls]
        for cl in cls:
            for c in cl.get("Code", []):
                code = str(c.get("id") or c.get("ID") or "").strip()
                name_obj = c.get("Name") or ""
                if isinstance(name_obj, dict):
                    name = next(iter(name_obj.values())) if name_obj else ""
                else:
                    name = str(name_obj)
                if code:
                    codes.append((code, name))
        return codes
    except Exception:
        return []


def iso3_to_iso2(iso3: str):
    c = pycountry.countries.get(alpha_3=iso3)
    if c is not None:
        return getattr(c, "alpha_2", "")
    if iso3 == "TWN":
        return "TW"
    return ""


def print_offline_requirements():
    print("Offline requirements helper:\n")
    dfp = STRUCTURES_DIR / "Dataflow.json"
    if not dfp.exists():
        print(f"Missing: {dfp}\nPlease obtain Dataflow JSON by calling: https://dataservices.imf.org/REST/SDMX_JSON.svc/Dataflow and place the result at the path above.")
        return
    # If Dataflow.json exists, show candidate DOTS flows
    try:
        df_json = json.loads(dfp.read_text(encoding="utf-8"))
        dots_flow_id, preview = discover_dots_flow_id(df_json, get_dots_hint(load_cfg()))
        if dots_flow_id:
            print(f"Discovered DOTS flow id from local Dataflow.json: {dots_flow_id}")
            print("You will need the corresponding DataStructure_<flow_id>.json and CodeList_<codelist_id>.json after discovery.")
        else:
            print("DOTS flow id not auto-discovered from local Dataflow.json. Top candidates (id | name):")
            for fid, name in (preview or [])[:50]:
                print(f"  {fid} | {name}")
            print("If you identify the correct DOTS flow id from the preview, place DataStructure_<flow_id>.json and CodeList_<codelist_id>.json as indicated by the DataStructure content.")
    except Exception as e:
        print(f"Failed to parse local Dataflow.json: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-offline-requirements", action="store_true", help="Print which SDMX JSON files are required for offline use and exit")
    parser.add_argument("--offline-only", action="store_true", help="Run in strict offline-only mode: do not attempt any network requests; require cached SDMX JSON files in data/raw/imf_dots/_structures")
    args = parser.parse_args()

    ensure_dirs()

    cfg = load_cfg()
    base = get_imf_base(cfg)
    hint = get_dots_hint(cfg)
    flow_override = get_dots_flow_id_override(cfg)

    nodes = read_node_index()
    overrides = read_overrides()

    if args.print_offline_requirements:
        print_offline_requirements()
        sys.exit(0)

    # Attempt to load/call Dataflow (offline-first)
    dataflow = imf_get_json(base, "Dataflow", STRUCTURES_DIR / "Dataflow.json", offline_only=args.offline_only)
    if dataflow is None:
        # fetch_json_with_retries already printed explicit offline instructions
        sys.exit(2)

    if flow_override:
        dots_flow_id = flow_override
    else:
        dots_flow_id, preview = discover_dots_flow_id(dataflow, hint)
        if not dots_flow_id:
            print("Could not auto-discover DOTS dataflow id. Preview of dataflows (id | name):")
            for fid, name in (preview or [])[:30]:
                print(f"  {fid} | {name}")
            print("NEXT ACTION: Set config/trade_ingest.yaml -> imf.dots_flow_id to the correct DOTS flow id, then re-run.")
            sys.exit(2)

    dsd = imf_get_json(base, f"DataStructure/{dots_flow_id}", STRUCTURES_DIR / f"DataStructure_{dots_flow_id}.json", offline_only=args.offline_only)
    if dsd is None:
        sys.exit(2)

    ref_area_codelist_id = extract_ref_area_codelist_id(dsd)
    if not ref_area_codelist_id:
        print("Could not find REF_AREA codelist id. Inspect cached DataStructure JSON:")
        print(str(STRUCTURES_DIR / f"DataStructure_{dots_flow_id}.json"))
        sys.exit(2)

    cl = imf_get_json(base, f"CodeList/{ref_area_codelist_id}", STRUCTURES_DIR / f"CodeList_{ref_area_codelist_id}.json", offline_only=args.offline_only)
    if cl is None:
        sys.exit(2)

    code_pairs = parse_codelist(cl, ref_area_codelist_id)
    if not code_pairs:
        print("Parsed 0 REF_AREA codes. Inspect cached CodeList JSON:")
        print(str(STRUCTURES_DIR / f"CodeList_{ref_area_codelist_id}.json"))
        sys.exit(2)

    by_code = {c[0].upper(): c[1] for c in code_pairs}
    by_name = {norm(c[1]): c[0] for c in code_pairs if c[1]}

    rosetta_rows = []
    unmatched_rows = []

    for n in nodes:
        iso3 = n["iso3"].upper()
        iso2 = (n.get("iso2") or "").upper() or iso3_to_iso2(iso3)
        name = n.get("name") or ""

        imf_code = ""
        imf_name = ""
        notes = ""

        if iso3 in overrides and overrides[iso3][0]:
            imf_code = overrides[iso3][0]
            imf_name = overrides[iso3][1] or by_code.get(imf_code.upper(), "")
            notes = "override"
        elif iso3 in by_code:
            imf_code = iso3
            imf_name = by_code[iso3]
            notes = "matched_by_code_iso3"
        elif iso2 and iso2 in by_code:
            imf_code = iso2
            imf_name = by_code[iso2]
            notes = "matched_by_code_iso2"
        else:
            nn = norm(name)
            if nn in by_name:
                imf_code = by_name[nn]
                imf_name = by_code.get(imf_code.upper(), "")
                notes = "matched_by_name"

        row = {
            "iso3": iso3,
            "iso2": iso2,
            "name": name,
            "imf_ref_area": imf_code,
            "imf_ref_area_name": imf_name,
            "comtrade_reporter_code": "",
            "twn_partner_code": "",
            "twn_partner_name": "",
            "notes": notes or ("UNMATCHED" if not imf_code else "")
        }
        rosetta_rows.append(row)
        if not imf_code:
            unmatched_rows.append({"iso3": iso3, "iso2": iso2, "name": name, "reason": "no_exact_match"})

    # Write rosetta
    rosetta_fields = [
        "iso3", "iso2", "name",
        "imf_ref_area", "imf_ref_area_name",
        "comtrade_reporter_code",
        "twn_partner_code", "twn_partner_name",
        "notes"
    ]
    with OUT_ROSETTA.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rosetta_fields)
        w.writeheader()
        w.writerows(rosetta_rows)

    # Write unmatched
    with OUT_UNMATCHED.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["iso3", "iso2", "name", "reason"])
        w.writeheader()
        w.writerows(unmatched_rows)

    # Write manifest
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "imf_sdmx_base": base,
        "dots_flow_id": dots_flow_id,
        "ref_area_codelist_id": ref_area_codelist_id,
        "files": {
            str(NODE_INDEX): sha256_file(NODE_INDEX),
            str(OUT_ROSETTA): sha256_file(OUT_ROSETTA),
            str(OUT_UNMATCHED): sha256_file(OUT_UNMATCHED),
            str(OVERRIDES): sha256_file(OVERRIDES) if OVERRIDES.exists() else ""
        },
        "cached_structures": {
            "Dataflow.json": sha256_file(RAW_STRUCT / "Dataflow.json") if (RAW_STRUCT / "Dataflow.json").exists() else "",
            f"DataStructure_{dots_flow_id}.json": sha256_file(RAW_STRUCT / f"DataStructure_{dots_flow_id}.json") if (RAW_STRUCT / f"DataStructure_{dots_flow_id}.json").exists() else "",
            f"CodeList_{ref_area_codelist_id}.json": sha256_file(RAW_STRUCT / f"CodeList_{ref_area_codelist_id}.json") if (RAW_STRUCT / f"CodeList_{ref_area_codelist_id}.json").exists() else ""
        }
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=== EntityMap Build Complete ===")
    print(f"rosetta rows: {len(rosetta_rows)}")
    print(f"unmatched rows: {len(unmatched_rows)}")
    print(f"wrote: {OUT_ROSETTA}")
    print(f"wrote: {OUT_UNMATCHED}")
    print(f"wrote: {OUT_MANIFEST}")
    if unmatched_rows:
        print("NEXT: open data/processed/meta/imf_unmatched.csv and add any needed overrides to overrides/imf_ref_area_overrides.csv, then re-run build + validate.")


if __name__ == "__main__":
    main()
