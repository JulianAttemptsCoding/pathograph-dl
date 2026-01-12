import json
import os
from pathlib import Path

def audit_manifests():
    report = {"step1": {}, "step2": {}}
    
    # Step 1 Manifest
    s1_path = Path("data/processed/trade/imf_imts_step1/manifest.json")
    if s1_path.exists():
        with open(s1_path, 'r', encoding='utf-8') as f:
            d = json.load(f)
            report["step1"] = {
                "version": d.get("version"),
                "outputs": d.get("outputs"),
                "time_axis": d.get("time_axis"),
                "parameters": d.get("parameters"),
                "qc_summary": d.get("qc_summary")
            }
    
    # Step 2 Manifest
    s2_path = Path("data/processed/trade/faostat_step2/preprocessing_manifest.json")
    if s2_path.exists():
        with open(s2_path, 'r', encoding='utf-8') as f:
            d = json.load(f)
            report["step2"] = {
                "version": d.get("version"),
                "outputs": d.get("outputs"),
                "weights": d.get("weights"),
                "parameters": d.get("parameters"),
                "qc_summary": d.get("qc_summary")
            }
            
    os.makedirs("docs/reports", exist_ok=True)
    with open("docs/reports/trade_spec_v1_2_artifact_audit.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print("Manifest audit complete.")

if __name__ == "__main__":
    audit_manifests()
