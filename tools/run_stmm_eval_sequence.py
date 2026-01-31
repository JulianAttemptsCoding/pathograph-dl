import subprocess
import sys
from pathlib import Path

def run_command(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

def main():
    run_dir = Path("runs/stmm_stepA/eval_gate_seed42")
    config = "config/stmm_stepA.yaml"
    calib_dir = "runs/calibration/eval_gate_seed42"
    
    # 1. Find Checkpoint
    ckpts = list(run_dir.glob("**/*.ckpt"))
    # Filter for explicitly named one (not last.ckpt) if possible, or use last.ckpt
    named_ckpts = [c for c in ckpts if "val_auprc" in c.name]
    if named_ckpts:
        ckpt = named_ckpts[0] # Take first (should be best/only)
    elif ckpts:
        ckpt = ckpts[0]
    else:
        print("No checkpoint found!")
        sys.exit(1)
        
    print(f"Using checkpoint: {ckpt}")
    
    # 2. Eval (Raw)
    # writes metrics to CSV/logs in run_dir
    run_command([sys.executable, "tools/stmm_stepA_eval.py", "--config", config, "--ckpt", str(ckpt), "--run_dir", str(run_dir)])
    
    # 3. Temperature Scale (Fit on Val)
    run_command([sys.executable, "tools/stmm_temperature_scale.py", "--config", config, "--run_dir", str(run_dir), "--out_dir", calib_dir, "--ckpt", str(ckpt)])
    
    # 4. Calibrated Eval (Test with T)
    run_command([sys.executable, "tools/stmm_eval_calibrated.py", "--config", config, "--run_dir", str(run_dir), "--calib_dir", calib_dir])
    
    # 5. Decision Report
    inputs_json = "docs/_logs/stmm_eval_gate_decision.json"
    inputs_md = "docs/_logs/stmm_eval_gate_decision.md"
    run_command([sys.executable, "tools/stmm_make_decision_report.py", "--out_json", inputs_json, "--out_md", inputs_md])
    
    print("Sequence Complete.")

if __name__ == "__main__":
    main()
