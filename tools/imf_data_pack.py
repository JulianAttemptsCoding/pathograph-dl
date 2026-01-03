import os
import json
import time
import hashlib
import datetime
import requests
import argparse
from pathlib import Path

def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def get_with_retry(url: str, max_retries: int = 3, backoff_factor: float = 2.0):
    # No Accept header to avoid 406 on SDMX Central
    headers = {
        "User-Agent": "PathoGraph-DL imf_data_pack/1.0",
    }
    for i in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=60)
            if resp.status_code == 200:
                return resp
            if resp.status_code in [429, 503]:
                wait = backoff_factor ** i
                print(f"Server busy ({resp.status_code}), waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
        except Exception as e:
            if i == max_retries - 1:
                raise
            wait = backoff_factor ** i
            print(f"Error: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    return None

def main():
    parser = argparse.ArgumentParser(description="IMF DOTS Data Downloader")
    parser.add_argument("--base-url", default="https://sdmxcentral.imf.org/sdmx/v2", help="SDMX base URL")
    parser.add_argument("--flow-id", default="DOTS", help="Default dots flow id")
    parser.add_argument("--start-period", default="2024-01", help="YYYY-MM")
    parser.add_argument("--end-period", default="2024-03", help="YYYY-MM")
    parser.add_argument("--indicators", default="TXG_FOB_USD,TMG_CIF_USD", help="Comma separated indicators")
    parser.add_argument("--data-domain", default="DOTS", help="DATA_DOMAIN dimension value")
    parser.add_argument("--freq", default="M", help="Frequency (M/Q/A)")
    parser.add_argument("--out-dir", default="data/raw/imf_dots/downloads", help="Output directory")
    parser.add_argument("--rosetta-path", default="data/processed/meta/rosetta_codes.csv")
    parser.add_argument("--dsd-order-path", default="data/raw/imf_dots/_structures/dsd_dimension_order.json")
    parser.add_argument("--mock", action="store_true", help="Generate synthetic data instead of downloading")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load DSD order
    if not os.path.exists(args.dsd_order_path):
        raise FileNotFoundError(f"Missing {args.dsd_order_path}. Run DSD introspection first.")
    with open(args.dsd_order_path, "r") as f:
        dsd_info = json.load(f)
    ordered_ids = dsd_info["ordered_dimension_ids"]

    # 2. Load Rosetta to get reporters
    if not os.path.exists(args.rosetta_path):
        raise FileNotFoundError(f"Missing {args.rosetta_path}. Run build_entity_map first.")
    
    reporters = []
    import csv
    with open(args.rosetta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            reporters.append(row["imf_ref_area"])
    reporters = sorted(list(set(reporters))) # unique IMF codes

    indic_str = args.indicators.replace(",", "+")
    
    # 3. Download loop
    manifest_entries = []
    
    # We batch by REPORTER to stay under query limits
    # Key template depends on order: DATA_DOMAIN.REF_AREA.INDICATOR.COUNTERPART_AREA.FREQ
    
    base = args.base_url.rstrip("/")
    # Version slot + (encoded %2B)
    vplus = "%2B"
    
    # Build key parts map
    const_map = {
        "DATA_DOMAIN": args.data_domain,
        "INDICATOR": indic_str,
        "FREQ": args.freq,
        "COUNTERPART_AREA": "" # Wildcard all partners
    }

    print(f"Starting {'MOCK ' if args.mock else ''}download for {len(reporters)} reporters...")
    
    import random
    
    for rep in reporters:
        # Build key
        key_parts = []
        for d in ordered_ids:
            if d == "REF_AREA":
                key_parts.append(rep)
            else:
                key_parts.append(const_map.get(d, ""))
        
        sdmx_key = ".".join(key_parts)
        
        # Endpoint: /data/{flow}/{key}/{provider}?startPeriod=...&endPeriod=...&format=sdmx-json
        url = f"{base}/data/{args.flow_id}/{sdmx_key}/all?startPeriod={args.start_period}&endPeriod={args.end_period}&format=sdmx-json"
        
        filename = f"dots_{rep}_{args.start_period}_{args.end_period}_{sha256_str(sdmx_key)[:8]}.json"
        fpath = out_dir / filename
        
        if fpath.exists() and not args.mock: # overwrite mocks if requested? Let's just skip if exists
            print(f"Skipping {rep} (already exists)")
            manifest_entries.append({
                "reporter": rep,
                "file": filename,
                "url_queried": url,
                "status": "cached"
            })
            continue

        if args.mock:
            print(f"Generating mock for {rep}")
            # Minimal mock SDMX-JSON
            # We assume a few partners (e.g. US, CN, DE) + World
            mock_partners = ["US", "CN", "DE", "W0"]
            mock_series = {}
            
            # Dimensions: DATA_DOMAIN, REF_AREA, INDICATOR, COUNTERPART_AREA, FREQ
            # keys string: "DOTS:US:TXG_FOB_USD:W0:M"
            
            # Helper to build series key string
            # SDMX-JSON structure.dimensions.series defines order.
            # We'll just dump flat observations in a structure acceptable for a simple parser.
            # Actually, let's make the parser robust to "Structure" block or assume it.
            
            # Simplified mock payload
            mock_data = {
                "header": {"id": "MOCK", "test": True, "prepared": datetime.datetime.utcnow().isoformat()},
                "dataSets": [{
                    "action": "Information",
                    "series": {}
                }],
                "structure": {
                    "dimensions": {
                        "series": [
                            {"id": "DATA_DOMAIN", "values": [{"id": args.data_domain}]},
                            {"id": "REF_AREA", "values": [{"id": rep}]},
                            {"id": "INDICATOR", "values": [{"id": i} for i in args.indicators.split(",")]},
                            {"id": "COUNTERPART_AREA", "values": [{"id": p} for p in mock_partners]},
                            {"id": "FREQ", "values": [{"id": args.freq}]}
                        ],
                        "observation": [
                            {"id": "TIME_PERIOD", "values": [{"id": "2024-01"}, {"id": "2024-02"}, {"id": "2024-03"}]}
                        ]
                    }
                }
            }
            
            # Populate series data
            # Key format in "series": "0:0:0:0:0" (indices into structure.dimensions.series.values)
            # data_domain=0, ref_area=0, freq=0
            # indicator varies (0..N), partner varies (0..M)
            
            s_dict = mock_data["dataSets"][0]["series"]
            for i_idx, indic in enumerate(args.indicators.split(",")):
                for p_idx, partner in enumerate(mock_partners):
                    if partner == rep: continue # no self trade usually
                    # Key: DATA_DOMAIN(0) : REF_AREA(0) : INDICATOR(i_idx) : COUNTERPART(p_idx) : FREQ(0)
                    key_str = f"0:0:{i_idx}:{p_idx}:0"
                    
                    # Observations: { "0": [val], "1": [val], "2": [val] }
                    obs = {}
                    for t_idx in range(3):
                        val = random.uniform(1e6, 1e9) # Random value 1M to 1B
                        obs[str(t_idx)] = [val]
                    
                    s_dict[key_str] = {"observations": obs}

            fpath.write_text(json.dumps(mock_data), encoding="utf-8")
            manifest_entries.append({
                "reporter": rep,
                "file": filename,
                "url_queried": url,
                "status": "mocked"
            })
            continue

        print(f"Fetching {rep}: {url}")
        try:
            resp = get_with_retry(url)
            if resp:
                fpath.write_text(resp.text, encoding="utf-8")
                manifest_entries.append({
                    "reporter": rep,
                    "file": filename,
                    "url_queried": url,
                    "status": "success"
                })
            else:
                print(f"Failed to fetch {rep}")
        except Exception as e:
            print(f"Failed {rep}: {e}")

    # 4. Write manifest
    manifest = {
        "created_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "params": vars(args),
        "dimension_order": ordered_ids,
        "downloads": manifest_entries
    }
    with open(out_dir / "imf_data_pack_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print("Download complete.")

if __name__ == "__main__":
    main()
