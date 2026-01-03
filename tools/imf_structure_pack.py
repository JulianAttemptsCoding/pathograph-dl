"""
Download IMF SDMX structure JSONs and produce a portable structure pack.

This version uses IMF SDMX Central (sdmxcentral.imf.org), which you can reach even when
dataservices.imf.org is blocked.

Outputs (into --out-dir):
  - Dataflow.json
  - DataStructure_<DOTS_FLOW_ID>.json
  - CodeList_<REF_AREA_CODELIST_ID>.json
  - imf_structure_pack_manifest.json
Optionally:
  - data/raw/manifests/imf_dots_structures_<DOTS_FLOW_ID>_<REF_AREA_CODELIST_ID>.zip

Usage (PowerShell):
  python -m tools.imf_structure_pack
  python -m tools.imf_structure_pack --out-dir data/raw/imf_dots/_structures --no-zip
  python -m tools.imf_structure_pack --dots-flow-id <FLOW_ID>
  python -m tools.imf_structure_pack --dots-flow-id <FLOW_ID> --dsd-id <DSD_ID>

Notes:
- Deterministic: it will only auto-select when the DOTS candidate is unambiguous.
- Offline workflow: run this on any machine that can reach sdmxcentral.imf.org, then copy the
  out-dir contents to the blocked host at: data/raw/imf_dots/_structures/
"""

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import requests

DEFAULT_BASE = "https://sdmxcentral.imf.org/sdmx/v2"


def sha256_file(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _headers(accept_language: str | None = "en") -> dict:
    # SDMX Central supports JSON via Accept header and/or format=sdmx-json param.
    # Note: explicit Accept: application/vnd.sdmx.json causes 406 on SDMX Central.
    # Rely on ?format=sdmx-json instead.
    h = {
        "User-Agent": "PathoGraph-DL imf_structure_pack/1.0",
    }
    if accept_language:
        h["Accept-Language"] = accept_language
    return h


def fetch(url: str, timeout: int = 120, max_attempts: int = 3, accept_language: str | None = "en") -> str:
    attempt = 0
    backoff = 1
    last = None

    # requests will respect HTTP_PROXY / HTTPS_PROXY automatically; printing helps debugging
    if attempt == 0:
        hp = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
        sp = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        if hp or sp:
            print(f"Detected proxies: HTTP_PROXY={bool(hp)} HTTPS_PROXY={bool(sp)}")

    while attempt < max_attempts:
        attempt += 1
        try:
            print(f"GET {url} (attempt {attempt}/{max_attempts})")
            r = requests.get(url, headers=_headers(accept_language), timeout=timeout)
            r.raise_for_status()
            return r.text
        except requests.RequestException as e:
            last = e
            print(f"Attempt {attempt} failed: {e}")
            if attempt < max_attempts:
                time.sleep(backoff)
                backoff *= 2
    raise RuntimeError(f"Failed to GET {url}: {last}")


def localised_to_str(x) -> str:
    """
    SDMX-JSON localised text can appear as:
      - string
      - dict like {"en": "...", "fr": "..."} (best-match often returned already)
      - list of dicts (rare)
    """
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        # prefer common language keys
        for k in ("en", "en-US", "en-GB"):
            if k in x and isinstance(x[k], str):
                return x[k]
        # otherwise return first string value
        for v in x.values():
            if isinstance(v, str):
                return v
        return ""
    if isinstance(x, list):
        for it in x:
            s = localised_to_str(it)
            if s:
                return s
        return ""
    return str(x)


def extract_dataflows_sdmxjson(structure_message: dict) -> list[dict]:
    data = structure_message.get("data") or structure_message.get("Data") or {}
    flows = data.get("dataflows") or data.get("dataFlows") or data.get("Dataflows") or data.get("DataFlows") or []
    if isinstance(flows, dict):
        # unexpected, but normalize
        flows = list(flows.values())
    if not isinstance(flows, list):
        return []
    return [f for f in flows if isinstance(f, dict)]


def dataflow_id_and_name(df: dict) -> tuple[str, str]:
    fid = str(df.get("id") or df.get("ID") or df.get("structureID") or df.get("StructureID") or "")
    name = localised_to_str(df.get("name") or df.get("Name") or df.get("names") or df.get("Names"))
    return fid, name


def discover_dots_candidates(flow_items: list[dict]) -> list[tuple[str, str]]:
    candidates = []
    for df in flow_items:
        fid, name = dataflow_id_and_name(df)
        s = (fid + " " + name).lower()
        if "dots" in s or "direction of trade" in s:
            candidates.append((fid, name))
    # de-dupe while preserving order
    seen = set()
    out = []
    for fid, name in candidates:
        key = (fid, name)
        if key not in seen:
            seen.add(key)
            out.append((fid, name))
    return out


def legacy_wrap_dataflows(flow_items: list[dict]) -> dict:
    # Match the older “structure/dataflows/dataflow” shape used by earlier codepaths.
    return {"structure": {"dataflows": {"dataflow": flow_items}}}


def legacy_wrap_datastructures(dsd_items: list[dict]) -> dict:
    return {"structure": {"datastructures": {"datastructure": dsd_items}}}


def legacy_wrap_codelists(codelist_items: list[dict]) -> dict:
    return {"structure": {"codelists": {"codelist": codelist_items}}}


def extract_referenced_dsds(structure_message: dict) -> list[dict]:
    data = structure_message.get("data") or structure_message.get("Data") or {}
    dsd = data.get("dataStructures") or data.get("datastructures") or data.get("DataStructures") or data.get("Datastructures") or []
    if isinstance(dsd, dict):
        dsd = list(dsd.values())
    if not isinstance(dsd, list):
        return []
    return [x for x in dsd if isinstance(x, dict)]


def extract_codelists(structure_message: dict) -> list[dict]:
    data = structure_message.get("data") or structure_message.get("Data") or {}
    cls = data.get("codelists") or data.get("codeLists") or data.get("Codelists") or data.get("CodeLists") or []
    if isinstance(cls, dict):
        cls = list(cls.values())
    if not isinstance(cls, list):
        return []
    return [x for x in cls if isinstance(x, dict)]


def deep_find_ref_area_codelist_ids(obj) -> set[str]:
    """
    Tolerant recursive search for REF_AREA's codelist id inside a DSD object.
    Returns a set (usually size 1).
    """
    found: set[str] = set()

    def try_extract_from_dimension(d: dict):
        # Identify REF_AREA
        did = str(d.get("id") or d.get("ID") or d.get("concept") or d.get("Concept") or d.get("conceptIdentity") or "")
        if did.upper() != "REF_AREA":
            return

        # Common SDMX patterns: localRepresentation/enumeration/ref/id
        for lr_key in ("localRepresentation", "localrepresentation", "LocalRepresentation"):
            lr = d.get(lr_key)
            if isinstance(lr, dict):
                enum = lr.get("enumeration") or lr.get("Enumeration")
                if isinstance(enum, dict):
                    ref = enum.get("ref") or enum.get("Ref")
                    if isinstance(ref, dict):
                        cid = ref.get("id") or ref.get("ID")
                        if cid:
                            found.add(str(cid))
                    elif isinstance(ref, str):
                        found.add(ref)
                elif isinstance(enum, str):
                    # Handle URN: urn:sdmx:org.sdmx.infomodel.codelist.Codelist=IMF:CL_REF_AREA(1.0)
                    if "=" in enum:
                        # IMF:CL_REF_AREA(1.0)
                        after_eq = enum.split("=", 1)[1]
                        # Handling "Agency:ID(Version)"
                        if ":" in after_eq:
                             # CL_REF_AREA(1.0)
                             after_colon = after_eq.split(":", 1)[1]
                             if "(" in after_colon:
                                 found.add(after_colon.split("(", 1)[0])
                             else:
                                 found.add(after_colon)
                        elif "(" in after_eq:
                             found.add(after_eq.split("(", 1)[0])
                        else:
                             found.add(after_eq)
                # sometimes enumeration directly has id
                cid2 = enum.get("id") if isinstance(enum, dict) else None
                if cid2:
                    found.add(str(cid2))

        # Some models use codelist/codeList directly
        for k in ("codelist", "codeList", "CodeList", "Codelist"):
            cid = d.get(k)
            if cid:
                found.add(str(cid))

    def walk(x):
        if isinstance(x, dict):
            # If this dict looks like a dimension definition, attempt extraction
            if ("id" in x or "ID" in x) and any(k in x for k in ("localRepresentation", "localrepresentation", "codelist", "codeList")):
                try_extract_from_dimension(x)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for it in x:
                walk(it)

    walk(obj)
    return found


def sdmxcentral_url(base: str, resource: str, agency: str, sid: str, version: str, *, fmt: str = "sdmx-json", references: str | None = None) -> str:
    # Per IMF SDMX Central guide: https://sdmxcentral.imf.org/sdmx/v2/ is the entry point,
    # and /structure/{resource}/{agencyID}/{structureID}/{version}/?... is used. Version can be
    # latest/all or a specific version string.
    # We always include trailing "/".
    # NOTE: "latest" often fails on Central; use "+" (encoded as %2B) for "latest stable".
    base = base.rstrip("/")
    url = f"{base}/structure/{resource}/{agency}/{sid}/{version}/?format={fmt}"
    if references:
        url += f"&references={references}"
    return url


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE, help="IMF SDMX Central base, default https://sdmxcentral.imf.org/sdmx/v2")
    parser.add_argument("--out-dir", default=str(Path("data") / "raw" / "imf_dots" / "_structures"))
    parser.add_argument("--dots-flow-id", default=None, help="If provided, skip DOTS flow discovery and use this flow id.")
    parser.add_argument("--dsd-id", default=None, help="Optional: force the DataStructure (DSD) id to fetch. Use if references discovery fails.")
    parser.add_argument("--dsd-agency", default="all", help="Agency for --dsd-id (default all).")
    parser.add_argument("--dsd-version", default="latest", help="Version for --dsd-id (default latest).")
    parser.add_argument("--ref-area-codelist-id", default=None, help="If provided, skip REF_AREA codelist extraction and use this codelist id.")
    parser.add_argument("--accept-language", default="en", help="Accept-Language header (default en). Use '' to omit.")
    parser.add_argument("--no-zip", dest="zip", action="store_false")
    parser.add_argument("--zip", dest="zip", action="store_true")
    parser.set_defaults(zip=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    STRUCTURES_DIR = Path("data") / "raw" / "imf_dots" / "_structures"
    out_dir.mkdir(parents=True, exist_ok=True)

    base = args.base_url.rstrip("/")
    accept_language = args.accept_language if args.accept_language != "" else None

    # Use "%2B" (plus) for versioning to satisfy SDMX Central
    sdmx_version_plus = "%2B"

    # Step 1: download Dataflow list
    # If dots_flow_id is known, fetch ONLY that flow to avoid 406 on "all/all"
    target_flow_id = args.dots_flow_id if args.dots_flow_id else "all"
    
    print(f"Fetching Dataflow with id: {target_flow_id}")
    df_url = sdmxcentral_url(base, "dataflow", "all", target_flow_id, sdmx_version_plus, fmt="sdmx-json")
    df_text = fetch(df_url, accept_language=accept_language)
    df_json = json.loads(df_text)

    flows = extract_dataflows_sdmxjson(df_json)
    if not flows:
        raise RuntimeError("No dataflows found in SDMX Central response. Cannot proceed.")

    # Write Dataflow.json in a legacy wrapper shape for downstream compatibility
    df_path = out_dir / "Dataflow.json"
    df_path.write_text(json.dumps(legacy_wrap_dataflows(flows), indent=2), encoding="utf-8")
    print("Wrote", df_path)

    # Step 2: discover DOTS flow id unless provided
    dots_flow_id = args.dots_flow_id
    if not dots_flow_id:
        candidates = discover_dots_candidates(flows)
        if not candidates:
            print("Could not auto-discover DOTS flow id. Print top 50 dataflows (id | name):")
            for i, df in enumerate(flows[:50], start=1):
                fid, name = dataflow_id_and_name(df)
                print(f"{i}. {fid} | {name}")
            print("\nRe-run with: --dots-flow-id <FLOW_ID>")
            sys.exit(2)
        if len(candidates) > 1:
            print("Multiple DOTS-like dataflow candidates found. Choose deterministically and re-run with --dots-flow-id:")
            for i, (fid, name) in enumerate(candidates, start=1):
                print(f"{i}. {fid} | {name}")
            sys.exit(2)
        dots_flow_id = candidates[0][0]
        print("Auto-selected DOTS flow id (unique match):", dots_flow_id)

    # Step 3: obtain the DataStructure (DSD)
    # Preferred: query the dataflow with references=datastructure to retrieve the referenced DSD.
    dsd_items: list[dict] = []
    dsd_id_used = None

    if args.dsd_id:
        # Forced DSD fetch
        dsd_url = sdmxcentral_url(base, "datastructure", args.dsd_agency, args.dsd_id, args.dsd_version, fmt="sdmx-json")
        dsd_text = fetch(dsd_url, accept_language=accept_language)
        dsd_json = json.loads(dsd_text)
        dsd_items = extract_referenced_dsds(dsd_json) or (dsd_json.get("data", {}).get("dataStructures") or [])
        dsd_id_used = args.dsd_id
        if not dsd_items:
            print("Could not retrieve dataStructures from forced DSD request.")
            sys.exit(2)
    else:
        # Reference discovery via dataflow query
        # Use %2B here as well
        df_detail_url = sdmxcentral_url(base, "dataflow", "all", dots_flow_id, sdmx_version_plus, fmt="sdmx-json", references="datastructure")
        df_detail_text = fetch(df_detail_url, accept_language=accept_language)
        df_detail_json = json.loads(df_detail_text)
        dsd_items = extract_referenced_dsds(df_detail_json)

        # Fallback: references=all (some servers only return DSD under all)
        if not dsd_items:
            df_detail_url2 = sdmxcentral_url(base, "dataflow", "all", dots_flow_id, sdmx_version_plus, fmt="sdmx-json", references="all")
            df_detail_text2 = fetch(df_detail_url2, accept_language=accept_language)
            df_detail_json2 = json.loads(df_detail_text2)
            dsd_items = extract_referenced_dsds(df_detail_json2)

        if not dsd_items:
            print("Could not discover referenced DataStructure (DSD) via dataflow references.")
            print("Action:")
            print("  1) Use SDMX Central UI (or query) to find the DSD id for this dataflow.")
            print("  2) Re-run with: --dots-flow-id <FLOW_ID> --dsd-id <DSD_ID> [--dsd-agency <AGENCY>] [--dsd-version <VERSION>]")
            sys.exit(2)

        # Deterministic selection: require exactly one DSD
        unique_ids = []
        seen = set()
        for d in dsd_items:
            did = d.get("id") or d.get("ID")
            if did and did not in seen:
                seen.add(did)
                unique_ids.append(did)
        if len(unique_ids) != 1:
            print("Non-unique DSD set returned from references; cannot choose deterministically.")
            print("Returned DSD ids:", unique_ids)
            print("Re-run with: --dsd-id <DSD_ID> to force selection.")
            sys.exit(2)
        dsd_id_used = str(unique_ids[0])

    # Write DataStructure_<FLOW>.json in a legacy wrapper shape
    ds_path = out_dir / f"DataStructure_{dots_flow_id}.json"
    ds_path.write_text(json.dumps(legacy_wrap_datastructures(dsd_items), indent=2), encoding="utf-8")
    print("Wrote", ds_path)

    # Step 4: extract REF_AREA codelist id unless provided
    ref_area_id = args.ref_area_codelist_id
    if not ref_area_id:
        ids = set()
        for dsd in dsd_items:
            ids |= deep_find_ref_area_codelist_ids(dsd)
        ids = {x for x in ids if x and x.lower() not in ("none", "null")}
        if len(ids) != 1:
            print("Could not deterministically extract a single REF_AREA codelist id from the DSD.")
            print("Found candidates:", sorted(ids))
            print("Re-run with: --ref-area-codelist-id <CODELIST_ID>")
            sys.exit(2)
        ref_area_id = sorted(ids)[0]
        print("Extracted REF_AREA codelist id:", ref_area_id)

    # Step 5: download CodeList for REF_AREA
    # Use %2B here as well
    cl_url = sdmxcentral_url(base, "codelist", "all", ref_area_id, sdmx_version_plus, fmt="sdmx-json")
    cl_text = fetch(cl_url, accept_language=accept_language)
    cl_json = json.loads(cl_text)
    codelists = extract_codelists(cl_json)
    if not codelists:
        print("No codelists returned for ref_area_id:", ref_area_id)
        sys.exit(2)

    # Prefer exact match by id
    chosen = None
    for cl in codelists:
        if str(cl.get("id") or cl.get("ID") or "") == str(ref_area_id):
            chosen = cl
            break
    if not chosen:
        if len(codelists) == 1:
            chosen = codelists[0]
        else:
            print("Multiple codelists returned and none matched id exactly; cannot choose deterministically.")
            print("Returned ids:", [cl.get("id") or cl.get("ID") for cl in codelists])
            sys.exit(2)

    cl_path = out_dir / f"CodeList_{ref_area_id}.json"
    cl_path.write_text(json.dumps(legacy_wrap_codelists([chosen]), indent=2), encoding="utf-8")
    print("Wrote", cl_path)

    # Step 6: write pack manifest
    manifest = {
        "created_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "base_url": base,
        "dots_flow_id": dots_flow_id,
        "dsd_id_used": dsd_id_used,
        "ref_area_codelist_id": ref_area_id,
        "files": {
            "Dataflow.json": sha256_file(df_path),
            f"DataStructure_{dots_flow_id}.json": sha256_file(ds_path),
            f"CodeList_{ref_area_id}.json": sha256_file(cl_path),
        },
    }
    manifest_path = out_dir / "imf_structure_pack_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Wrote", manifest_path)

    # Step 7: zip
    if args.zip:
        man_dir = Path("data/raw/manifests")
        man_dir.mkdir(parents=True, exist_ok=True)
        zip_name = man_dir / f"imf_dots_structures_{dots_flow_id}_{ref_area_id}.zip"
        with ZipFile(zip_name, "w", ZIP_DEFLATED) as z:
            z.write(df_path, df_path.name)
            z.write(ds_path, ds_path.name)
            z.write(cl_path, cl_path.name)
            z.write(manifest_path, manifest_path.name)
        print("Created zip", zip_name)

    print(f"\nDone. Place the out-dir content into the blocked host at {STRUCTURES_DIR} and run build_entity_map with --offline-only")


if __name__ == "__main__":
    main()
