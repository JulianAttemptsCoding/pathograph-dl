import json
from pathlib import Path

def audit_scaler():
    p = Path('data/processed/trade/trade_step3_scaler.json')
    if not p.exists():
        print("Error: Scaler file not found.")
        return
        
    with open(p, 'r', encoding='utf-8') as f:
        d = json.load(f)
        
    summary = {
        "train": d.get("train"),
        "K": d.get("K"),
        "channels": d.get("channels"),
        "groups": d.get("groups"),
        "base_mean_len": len(d.get('base', {}).get('mean', [])),
        "base_std_len": len(d.get('base', {}).get('std', [])),
        "risk_mean_len": len(d.get('risk', {}).get('mean', [])),
        "risk_std_len": len(d.get('risk', {}).get('std', []))
    }
    
    out = Path('docs/reports/trade_spec_v1_2_scaler_summary.txt')
    out.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print("Scaler audit complete.")

if __name__ == "__main__":
    audit_scaler()
