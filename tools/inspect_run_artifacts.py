import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime

def inspect_run_artifacts(runs_root, out_json_path, out_md_path):
    root = Path(runs_root)
    runs = []
    
    if root.exists():
        # Find all run directories
        for p in root.iterdir():
            if p.is_dir():
                runs.append(p)
    
    # Sort by mtime descending
    runs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    inventory = {
        "scan_time": datetime.now().isoformat(),
        "runs_root": str(root),
        "total_runs_found": len(runs),
        "runs": []
    }
    
    valid_runs = []
    
    for run_dir in runs:
        run_info = {
            "name": run_dir.name,
            "path": str(run_dir),
            "mtime": datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(),
            "has_hparams": (run_dir / "hparams.yaml").exists(),
            "has_metrics": (run_dir / "metrics.csv").exists(),
            "checkpoints": [str(c.name) for c in run_dir.glob("**/*.ckpt")],
            "files": [str(f.name) for f in run_dir.iterdir() if f.is_file()],
        }
        
        # Try to read metrics
        metrics_csv = list(run_dir.glob("**/metrics.csv"))
        if metrics_csv:
            run_info["metrics_file"] = str(metrics_csv[0])
            try:
                with open(metrics_csv[0], 'r', encoding='utf-8') as f:
                    header = f.readline().strip()
                    run_info["metric_keys"] = header.split(',')
            except Exception as e:
                run_info["metric_read_error"] = str(e)
        
        inventory["runs"].append(run_info)
        valid_runs.append(run_info)

    # Write JSON
    Path(out_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(inventory, f, indent=2)
        
    # Write MD
    Path(out_md_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_md_path, 'w', encoding='utf-8') as f:
        f.write(f"# STMM Run Artifact Inventory\n\n")
        f.write(f"**Scan Time**: {inventory['scan_time']}\n")
        f.write(f"**Root**: `{inventory['runs_root']}`\n")
        f.write(f"**Total Runs**: {inventory['total_runs_found']}\n\n")
        
        if not runs:
            f.write("No runs found.\n")
        else:
            for r in valid_runs:
                f.write(f"## Run: {r['name']}\n")
                f.write(f"- Path: `{r['path']}`\n")
                f.write(f"- Mtime: {r['mtime']}\n")
                f.write(f"- Checkpoints: {len(r['checkpoints'])} found\n")
                if r.get("metric_keys"):
                    f.write(f"- Metrics logged: {', '.join(r['metric_keys'][:10])}...\n")
                else:
                    f.write("- Metrics: None found\n")
                f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()
    
    inspect_run_artifacts(args.runs_root, args.out_json, args.out_md)
