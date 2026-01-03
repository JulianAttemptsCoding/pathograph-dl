# Step 3: IMF SDMX Offline Workflow

This document explains how to produce the IMF SDMX structure pack on an internet-enabled machine and run Step 3 offline on the blocked host.

## Problem
The main dev host cannot reach `dataservices.imf.org` over HTTPS (TCP 443). The `build_entity_map` script needs the SDMX structure JSONs (Dataflow, DataStructure, CodeList) to map project ISO3 codes to IMF REF_AREA codes.

## Solution Overview
1. On an internet-enabled machine, run the downloader to fetch SDMX structure JSONs and produce a zip structure pack.
2. Copy or unzip the structure pack into the blocked host under `data/raw/imf_dots/_structures`.
3. On the blocked host, run `build_entity_map` in offline-only mode to produce the rosetta outputs.

## Downloader (run on internet-enabled machine)
Install dependencies (if not already):

```powershell
python -m pip install -U requests pyyaml
```

Run the downloader (default out-dir `data/raw/imf_dots/_structures` and default base URL):

```powershell
python -m tools.imf_structure_pack
```

If automatic DOTS discovery fails, re-run with explicit flags:

```powershell
python -m tools.imf_structure_pack --dots-flow-id <FLOW_ID>
python -m tools.imf_structure_pack --dots-flow-id <FLOW_ID> --ref-area-codelist-id <CODELIST_ID>
```

Output: the downloader writes `Dataflow.json`, `DataStructure_<flow_id>.json`, `CodeList_<codelist_id>.json`, and `imf_structure_pack_manifest.json` into the chosen out-dir and creates a zip under `data/raw/manifests` by default.

## Copy structure pack to blocked host
Copy the contents of the downloader out-dir (or the zip) into the blocked host path:

```
<data-root>/data/raw/imf_dots/_structures/
```

## Run build in offline-only mode (blocked host)
On the blocked host, run:

```powershell
python -u -m src.ingest.build_entity_map --offline-only
python -u -m src.ingest.validate_entity_map
```

If any required file is missing, `build_entity_map --offline-only` will exit and print the exact file path(s) it expects.

## Troubleshooting
- If DOTS flow id is not auto-discovered from `Dataflow.json`, run the downloader with `--dots-flow-id` using the id you identify from the preview list.
- If mappings are missing (entries in `data/processed/meta/imf_unmatched.csv`), add exact override rows to `overrides/imf_ref_area_overrides.csv` and re-run build + validate.

## Notes
- The scripts use deterministic matching only: overrides, exact IMF code match (iso3, iso2), exact normalized name match.
- Do not edit the canonical 194 ISO3 list except to correct formatting (whitespace). Do not change country membership.


