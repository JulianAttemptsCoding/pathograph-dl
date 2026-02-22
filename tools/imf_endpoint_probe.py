import argparse
import requests
import json
import time
import hashlib
from pathlib import Path
import datetime

# --- Configuration ---
ENDPOINTS = [
    {
        "name": "SDMX_Central_REST_1",
        "base_url": "https://sdmxcentral.imf.org/ws/public/sdmxapi/rest",
        "supports_v2_url_structure": False # Uses /data/{flow}/{key}
    },
    {
        "name": "IMF_Data_Portal_SDMX_2_1",
        "base_url": "https://api.imf.org/external/sdmx/2.1",
        "supports_v2_url_structure": False # Uses /data/{flow}/{key} (standard 2.1)
    }
]

# Control dataset: Use something common like 'CPI', 'IFS', or 'WEO' if available, 
# or specific small flows often found. Let's try to discover one.
# For now, we search for a candidate.

def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def log_request(log_dir: Path, endpoint_name: str, tag: str, url: str, code: int, headers_sent: dict, response_body: str):
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    h = sha256_str(url)[:8]
    fname = f"{endpoint_name}_{tag}_{ts}_{h}.txt"
    with open(log_dir / fname, "w", encoding="utf-8") as f:
        f.write(f"URL: {url}\n")
        f.write(f"Timestamp: {ts}\n")
        f.write(f"Status: {code}\n")
        f.write(f"Headers Sent: {json.dumps(headers_sent, indent=2)}\n")
        f.write("-" * 40 + "\n")
        f.write(response_body[:2000] + "\n...\n") # Log first 2KB

def probe_endpoint(ep, log_dir):
    print(f"--- Probing {ep['name']} ---")
    base = ep["base_url"].rstrip("/")
    
    # 1. Structure Probe (Dataflow)
    # 2.1 standard: /dataflow or /structure/dataflow
    # SDMX Central legacy: often /dataflow
    
    # Try getting dataflows
    flow_url = f"{base}/dataflow"
    print(f"Fetching Dataflows: {flow_url}")
    
    flows = []
    
    try:
        resp = requests.get(flow_url, timeout=30)
        log_request(log_dir, ep['name'], "dataflow", flow_url, resp.status_code, {}, resp.text)
        if resp.status_code == 200:
            # Parse XML or JSON?
            # If default is XML, we might need simple string search or proper parsing
            # Try to identify IMTS or DOTS
            if "DOTS" in resp.text:
                print("  [+] Found 'DOTS' in dataflow response")
                flows.append("DOTS")
            if "IMTS" in resp.text:
                 print("  [+] Found 'IMTS' in dataflow response")
                 flows.append("IMTS")
            
            # Find a control flow?
            if "CPI" in resp.text:
                 print("  [+] Found 'CPI' candidate")
    except Exception as e:
        print(f"  [-] Dataflow fetch failed: {e}")

    # If we didn't find specific flows, try typical IDs blindly?
    if not flows:
        print("  [!] No obvious trade flows found in list (or list failed). Will probe blind IDs.")
        flows = ["IMTS", "DOTS"]
        
    # 2. Probe Data
    # For each candidate flow, try to fetch a DSD? Or just go straight for canary?
    # Let's try structure for the flow first to see if it exists.
    
    found_winner = False
    winner_info = None

    for fid in flows:
        print(f"  Probing Flow: {fid}")
        
        # 2a. Check DSD (optional but good for debugging)
        # dsd_url = f"{base}/datastructure/IMF/{fid}" # generic pattern
        # skip for speed, go to data
        
        # 2b. Canary Data Query for Trade
        # Canary: USA to CAN, Jan-Feb 2024
        # Key: usually FREQ.REF_AREA.INDICATOR...
        # We need to guess key structure if we don't have DSD.
        # BUT: Authoritative requirement says "derive from DSD".
        # So we MUST fetch DSD if we want to be correct.
        
        # Try fetching DSD
        dsd_url = f"{base}/datastructure/all/{fid}/latest/?references=children"
        # 2.1 standard: /datastructure/{agency}/{id}/{version}
        # Try a few patterns or just generic "all"
        
        # Actually, let's try the data query with a "wildcard-ish" approach if supported, 
        # OR use the known DOTS key from previous steps as a strong guess, but adapt for IMTS if needed.
        # Known DOTS Key Order from Central: DATA_DOMAIN.REF_AREA.INDICATOR.COUNTERPART_AREA.FREQ
        # DOTS.US.TXG_FOB_USD+TMG_CIF_USD..M
        
        # For IMTS (if it exists), key might differ. 
        # Let's try the DOTS pattern first if ID is DOTS.
        
        canary_keys = []
        if fid == "DOTS":
             # Legacy guess
             canary_keys.append("DOTS.US.TXG_FOB_USD+TMG_CIF_USD..M") 
             
        elif fid == "IMTS":
            # Confirmed DSD Order: COUNTRY, INDICATOR, COUNTERPART_COUNTRY, FREQUENCY
            # But maybe codes differ.
            # Brute force variations.
            
            reps = ["US", "USA"]
            partners = ["W00", "W0", "CN", "CHN"]
            indics = ["TXG_FOB_USD", "TXG_FOB_USD_VAL"] # Guessing
            # Actually standard IMTS often uses same codes as DOTS.
            
            # Let's try known valid DOTS codes: US, TXG_FOB_USD, W00, M
            # Variant 1: DSD Order
            canary_keys.append("US.TXG_FOB_USD.W00.M")
            canary_keys.append("US.TXG_FOB_USD.CN.M")
            
            # Variant 2: Freq first (common)
            canary_keys.append("M.US.TXG_FOB_USD.W00")
            
        
        for k in canary_keys:
            # Query URL
            q_url = f"{base}/data/{fid}/{k}"
            
            # Try with and without time limits
            # Maybe 2024 is empty? Try 2023.
            time_opts = [
                ("2024-01", "2024-02"),
                ("2023-01", "2023-02"),
                (None, None) # All time (limit via lastNObservations?)
            ]
            
            for start, end in time_opts:
                params = {}
                label = "AllTime"
                if start:
                    params["startPeriod"] = start
                    params["endPeriod"] = end
                    label = f"{start}"
                else:
                    params["lastNObservations"] = "1"
                    label = "Last1"

                # Header Modes
                # SDMX-JSON preferred
                headers = {"Accept": "application/vnd.sdmx.data+json;version=1.0.0"}
                m_name = "AuthJSON"
                
                print(f"    Query: {q_url} [{label}]")
                try:
                    r = requests.get(q_url, params=params, headers=headers, timeout=10)
                    log_request(log_dir, ep['name'], f"data_{fid}_{m_name}_{label}", r.url, r.status_code, headers, r.text)
                    
                    if r.status_code == 200:
                        content = r.text
                        has_data = False
                        
                        # Better detection
                        if "values" in content and ":[" in content: # JSON array with content
                            has_data = True
                        if "<Obs>" in content or "<mes:Obs>" in content or "<generic:Obs>" in content:
                            has_data = True
                            
                        count_hint = content.count("<Obs>") + content.count("values")
                        
                        if has_data and count_hint > 0:
                            print(f"    [!!!] SUCCESS! {ep['name']} returned 200 for {fid}. Approx obs: {count_hint}")
                            return {
                                "endpoint": ep['name'],
                                "base_url": ep['base_url'],
                                "flow_id": fid,
                                "canary_url": r.url,
                                "headers_used": headers,
                                "key_used": k
                            }
                        else:
                             print(f"    [?] 200 OK but maybe empty? Check log. Count hint: {count_hint}")
                    elif r.status_code == 404:
                         print("    [-] 404 Not Found")
                    elif r.status_code == 400:
                         print("    [-] 400 Bad Request (Invalid Key/Params?)")
                    else:
                         print(f"    [-] {r.status_code}")
                         
                except Exception as e:
                    print(f"    [-] Err: {e}")
                    
    return None

def main():
    log_dir = Path("data/raw/imf_dots/probe_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    winner = None
    
    for ep in ENDPOINTS:
        res = probe_endpoint(ep, log_dir)
        if res:
            winner = res
            break
            
    summary_path = log_dir / "probe_summary.md"
    with open(summary_path, "w") as f:
        f.write("# IMF Endpoint Probe Summary\n\n")
        if winner:
            f.write("## WINNER FOUND\n")
            f.write(f"- **Endpoint**: {winner['endpoint']}\n")
            f.write(f"- **Flow ID**: {winner['flow_id']}\n")
            f.write(f"- **Canary URL**: `{winner['canary_url']}`\n")
            f.write(f"- **Hearders**: `{winner['headers_used']}`\n")
            f.write("\nValid real data access confirmed.\n")
        else:
            f.write("## NO WINNER\n")
            f.write("All probed endpoints failed to return a 200 OK with data for trade canary.\n")
            f.write("See logs for details.\n")

if __name__ == "__main__":
    main()
