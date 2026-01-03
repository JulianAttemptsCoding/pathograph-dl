import json
import csv
import glob
import pandas as pd
from pathlib import Path
import argparse

def load_rosetta(path):
    # Map imf_ref_area -> iso3
    imf_to_iso = {}
    df = pd.read_csv(path)
    # Handle TWN special case if needed, but rosetta should have it
    for _, row in df.iterrows():
        imf_to_iso[row['imf_ref_area']] = row['iso3']
    return imf_to_iso

def load_node_index(path):
    # Map iso3 -> node_id
    iso_to_id = {}
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        iso_to_id[row['iso3']] = row['node_id']
    return iso_to_id

def parse_sdmx_json(fpath, rosetta_map):
    with open(fpath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Locate dimensions
    dims = data['structure']['dimensions']['series']
    obs_dims = data['structure']['dimensions']['observation']
    
    # Find indices
    try:
        idx_ref = next(i for i, d in enumerate(dims) if d['id'] == 'REF_AREA')
        idx_partner = next(i for i, d in enumerate(dims) if d['id'] == 'COUNTERPART_AREA')
        idx_indic = next(i for i, d in enumerate(dims) if d['id'] == 'INDICATOR')
        idx_time = next(i for i, d in enumerate(obs_dims) if d['id'] == 'TIME_PERIOD')
    except StopIteration:
        # Fallback if names differ (e.g. mock vs real if real differs)
        # But we assume DSD strictness
        return []

    # Get values lists
    ref_values = [v['id'] for v in dims[idx_ref]['values']]
    partner_values = [v['id'] for v in dims[idx_partner]['values']]
    indic_values = [v['id'] for v in dims[idx_indic]['values']]
    time_values = [v['id'] for v in obs_dims[idx_time]['values']]
    
    records = []
    
    # Series
    dsets = data.get('dataSets', [])
    if not dsets: return []
    
    series_map = dsets[0].get('series', {})
    
    for key, content in series_map.items():
        indices = [int(x) for x in key.split(':')]
        
        ref_code = ref_values[indices[idx_ref]]
        partner_code = partner_values[indices[idx_partner]]
        indic_code = indic_values[indices[idx_indic]]
        
        # Map to ISO3
        rep_iso = rosetta_map.get(ref_code)
        par_iso = rosetta_map.get(partner_code)
        
        # If partner is W0 (World) or others not in rosetta, we might skip or keep as raw
        # For tensor assignment, we need canonical partners
        # For now, keep everything, filter later? Or filter now to verify coverage?
        # Let's keep valid ISO3s.
        
        # Channel map
        if 'TXG' in indic_code:
            channel = 'exports'
        elif 'TMG' in indic_code:
            channel = 'imports'
        else:
            channel = 'other'
            
        obs = content.get('observations', {})
        for t_key, t_val in obs.items():
            t_idx = int(t_key)
            time_period = time_values[t_idx]
            val = t_val[0] if isinstance(t_val, list) else t_val
            
            records.append({
                'month': time_period,
                'reporter_iso3': rep_iso, # Can be None
                'partner_iso3': par_iso,  # Can be None
                'reporter_imf': ref_code,
                'partner_imf': partner_code,
                'channel': channel,
                'value': float(val) if val is not None else 0.0
            })
            
    return records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/raw/imf_dots/downloads")
    parser.add_argument("--rosetta-path", default="data/processed/meta/rosetta_codes.csv")
    parser.add_argument("--out-file", default="data/processed/trade/dots_long.parquet")
    args = parser.parse_args()
    
    rosetta_map = load_rosetta(args.rosetta_path)
    
    all_files = glob.glob(str(Path(args.input_dir) / "*.json"))
    print(f"Parsing {len(all_files)} files...")
    
    all_rows = []
    for f in all_files:
        if "manifest" in f: continue
        try:
            rows = parse_sdmx_json(f, rosetta_map)
            all_rows.extend(rows)
        except Exception as e:
            print(f"Error parsing {f}: {e}")
            
    df = pd.DataFrame(all_rows)
    print(f"Parsed {len(df)} rows.")
    
    # Filter for valid mappings
    valid = df.dropna(subset=['reporter_iso3', 'partner_iso3'])
    print(f"Valid rows (mapped reporter/partner): {len(valid)}")
    
    # Ensure output dir
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    
    valid.to_parquet(args.out_file, index=False)
    print(f"Wrote {args.out_file}")

if __name__ == "__main__":
    main()
