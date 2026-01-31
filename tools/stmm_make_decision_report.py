"""
Generate Final Decision Report for STMM Eval Gate.
"""

import argparse
import json
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_md", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()
    
    # Paths (hardcoded per task spec or discovered)
    stmm_run_dir = Path("runs/stmm_stepA/eval_gate_seed42")
    baseline_run_dir = Path("runs/persistence_baseline/eval_gate_seed42")
    data_gate_json = Path("docs/_logs/stmm_data_gate.json")
    
    # Load Data Gate
    with open(data_gate_json, 'r') as f:
        data_gate = json.load(f)
        
    # Load Baseline Metrics
    with open(baseline_run_dir / "metrics.json", 'r') as f:
        baseline_metrics = json.load(f)
        
    # Load STMM Metrics
    # Can be in metrics.csv or we might have an eval result json?
    # T7 produces "calibrated_metrics.json" from stmm_eval_calibrated.py
    # and maybe we can get raw metrics from eval_logs usually in metrics.csv
    
    stmm_calib_json = stmm_run_dir / "calibrated_metrics.json"
    if stmm_calib_json.exists():
        with open(stmm_calib_json, 'r') as f:
            stmm_metrics = json.load(f)
    else:
        stmm_metrics = {}
        
    # Get raw test metrics from CSV if available (for uncalibrated comparison)
    stmm_raw_csv = list(stmm_run_dir.glob("eval_logs/**/metrics.csv"))
    if stmm_raw_csv:
        df = pd.read_csv(stmm_raw_csv[0])
        # Last row
        last_row = df.iloc[-1].to_dict()
        stmm_metrics.update(last_row)
        
    # Decision Logic
    # Primary Metric: test_auprc_macro
    metric_key = "test_auprc_macro"
    eps = 0.001
    
    baseline_val = baseline_metrics.get(metric_key, 0.0)
    stmm_raw = stmm_metrics.get(metric_key, 0.0)
    stmm_cal = stmm_metrics.get("test_cal_auprc_macro", 0.0) # From calibrated json
    if pd.isna(stmm_cal): stmm_cal = 0.0
    
    # Use best of raw or calibrated for GO condition?
    # "STMM test_auprc_macro (raw OR calibrated; report both) must be > baseline ... by at least eps"
    stmm_best = max(stmm_raw, stmm_cal)
    
    diff = stmm_best - baseline_val
    is_better = diff >= eps
    
    # Checks
    min_pos = 10
    min_valid_p = 2
    
    test_pos_bs = baseline_metrics.get('test_pos_total', 0)
    test_pos_stmm = stmm_metrics.get('test_pos_total', 0) # Raw
    valid_p_stmm = stmm_metrics.get('macro_valid_pathogens', 0)
    
    # We use data gate result technically, but runtime metrics confirm it
    gate_status = data_gate.get('gate_status', 'FAIL')
    
    decision = "INCONCLUSIVE"
    reason = []
    
    if gate_status != 'PASS':
        decision = "NO-GO"
        reason.append(f"Data Gate FAILED: {data_gate.get('fail_reasons')}")
    elif test_pos_stmm < min_pos:
        decision = "INCONCLUSIVE"
        reason.append(f"Not enough positives in test (found {test_pos_stmm}, min {min_pos})")
    elif valid_p_stmm < min_valid_p:
        decision = "INCONCLUSIVE"
        reason.append(f"Not enough valid pathogens (found {valid_p_stmm}, min {min_valid_p})")
    elif is_better:
        decision = "GO"
        reason.append(f"STMM ({stmm_best:.4f}) beats Baseline ({baseline_val:.4f}) by {diff:.4f} >= {eps}")
    else:
        decision = "NO-GO"
        reason.append(f"STMM ({stmm_best:.4f}) failed to beat Baseline ({baseline_val:.4f}) by {eps}")
        
    # Write JSON
    result = {
        "decision": decision,
        "reason": reason,
        "metrics": {
            "baseline": baseline_val,
            "stmm_raw": stmm_raw,
            "stmm_calibrated": stmm_cal,
            "diff": diff
        },
        "details": {
            "baseline_full": baseline_metrics,
            "stmm_full": stmm_metrics
        }
    }
    
    with open(args.out_json, 'w') as f:
        json.dump(result, f, indent=2)
        
    # Write MD
    with open(args.out_md, 'w') as f:
        f.write(f"# ST-MM-GNN Eval Gate Decision: {decision}\n\n")
        f.write(f"**Reason**: {'; '.join(reason)}\n\n")
        
        f.write("## Metrics Comparison\n")
        f.write("| Metric | Baseline | STMM (Raw) | STMM (Calib) | Delta (Best-Base) |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(f"| {metric_key} | {baseline_val:.4f} | {stmm_raw:.4f} | {stmm_cal:.4f} | {diff:.4f} |\n\n")
        
        f.write("## Status Counters\n")
        f.write(f"- Test Positives: {test_pos_stmm}\n")
        f.write(f"- Valid Pathogens: {valid_p_stmm}\n")
        f.write(f"- Data Gate: {gate_status}\n")

if __name__ == "__main__":
    main()
