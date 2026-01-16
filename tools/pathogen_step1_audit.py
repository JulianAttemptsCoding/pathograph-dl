
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

def audit_inputs():
    report = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "checks": []
    }
    
    # Check 1: Python version (implicitly checked by running this, but logging it)
    report['checks'].append({"check": "Python Version", "status": "INFO", "value": sys.version})
    
    # Check 2: node_index
    node_index_candidates = [
        "data/processed/trade/imf_imts_step1/node_index.csv",
        "data/processed/meta/node_index.csv"
    ]
    node_index_path = None
    for p in node_index_candidates:
        if Path(p).exists():
            node_index_path = Path(p)
            break
            
    if not node_index_path:
        report['checks'].append({"check": "node_index existence", "status": "FAIL", "msg": "No node_index found"})
    else:
        try:
            df = pd.read_csv(node_index_path)
            rows = len(df)
            u_node = df['node_id'].is_unique
            u_iso = df['iso3'].is_unique
            cols = list(df.columns)
            
            status = "PASS" if (rows == 194 and u_node and u_iso) else "FAIL"
            report['checks'].append({
                "check": "node_index validation",
                "status": status,
                "path": str(node_index_path),
                "rows": rows,
                "unique_node_id": bool(u_node),
                "unique_iso3": bool(u_iso),
                "columns": cols
            })
        except Exception as e:
            report['checks'].append({"check": "node_index read", "status": "FAIL", "error": str(e)})

    # Check 3: time_index_master
    time_path = Path("data/processed/meta/time_index_master.npy")
    if not time_path.exists():
        report['checks'].append({"check": "time_index_master existence", "status": "FAIL"})
    else:
        try:
            arr = np.load(time_path)
            status = "PASS" if (arr.ndim == 1 and arr.shape[0] == 908) else "FAIL"
            report['checks'].append({
                "check": "time_index_master validation",
                "status": status,
                "shape": list(arr.shape),
                "min": int(arr.min()) if arr.size > 0 else None,
                "max": int(arr.max()) if arr.size > 0 else None
            })
        except Exception as e:
            report['checks'].append({"check": "time_index_master read", "status": "FAIL", "error": str(e)})

    # Check 4: Pathogen Files
    input_dir = Path("data/raw/pathogen_curated")
    files = list(input_dir.glob("*_curated_long.csv"))
    
    report['checks'].append({
        "check": "Pathogen file count",
        "status": "PASS" if len(files) == 8 else "WARN",
        "count": len(files),
        "files": [f.name for f in files]
    })
    
    required_cols = {'iso3', 'date', 'pathogen', 'value'}
    for f in files:
        try:
            df = pd.read_csv(f)
            missing = required_cols - set(df.columns)
            
            # Date parsing check
            df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
            bad_dates = df['date_parsed'].isna().sum()
            
            # Iso3 check (against node_index if available)
            # (simplified here just to check they are strings)
            
            status = "PASS" if not missing and bad_dates == 0 else "FAIL"
            report['checks'].append({
                "check": f"File validation: {f.name}",
                "status": status,
                "missing_cols": list(missing),
                "unparseable_dates": int(bad_dates),
                "rows": len(df)
            })
        except Exception as e:
            report['checks'].append({"check": f"File read: {f.name}", "status": "FAIL", "error": str(e)})

    # Write outputs
    out_json = Path("docs/_logs/pathogen_step1_input_audit.json")
    out_md = Path("docs/_logs/pathogen_step1_input_audit.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    
    out_json.write_text(json.dumps(report, indent=2))
    
    # MD summary
    lines = ["# Pathogen Step 1 Input Audit", f"Timestamp: {report['timestamp']}", ""]
    for c in report['checks']:
        lines.append(f"- **{c['check']}**: {c.get('status', 'UNKNOWN')}")
        for k, v in c.items():
            if k not in ['check', 'status']:
                lines.append(f"  - {k}: {v}")
    
    out_md.write_text("\n".join(lines))
    print(f"Audit complete. Wrote {out_json} and {out_md}")

    # Return non-zero if any FAIL
    if any(c.get('status') == 'FAIL' for c in report['checks']):
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(audit_inputs())
