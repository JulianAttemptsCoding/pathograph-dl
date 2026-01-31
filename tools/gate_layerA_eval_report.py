"""
STMM Layer-A Evaluation & Reporting Gate Runner.

Orchestrates the full gate sequence:
1. Label sanity check (strict mode, bounded)
2. Pytest (full suite)
3. Eval report (baseline-only, bounded)
4. Eval report with model (if STMM_CKPT env var is set)

Exits nonzero on first failure.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, desc):
    """Run command and return (success, output)."""
    print(f"\n{'='*60}")
    print(f"[GATE] {desc}")
    print(f"{'='*60}")
    print(f"$ {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    success = result.returncode == 0
    status = "[OK]" if success else "[FAIL]"
    print(f"\n{status} {desc}")
    
    return success, result.stdout


def main():
    parser = argparse.ArgumentParser(description='STMM Layer-A Evaluation Gate')
    parser.add_argument('--config', type=str, default='config/stmm_stepA.yaml')
    parser.add_argument('--max_batches', type=int, default=10,
                        help='Max batches for bounded sanity/eval runs')
    args = parser.parse_args()
    
    print("="*60)
    print("STMM LAYER-A EVALUATION & REPORTING GATE")
    print("="*60)
    
    repo_root = Path(__file__).parent.parent
    os.chdir(repo_root)
    
    #1. Label sanity check
    success, _ = run_cmd(
        [
            sys.executable, 
            'tools/stmm_stepA_label_sanity.py',
            '--config', args.config,
            '--split', 'all',
            '--max_batches', str(args.max_batches),
            '--strict',
        ],
        "Label sanity check (strict)"
    )
    
    if not success:
        print("\n[GATE FAILED] Label sanity check failed")
        sys.exit(1)
    
    # 2. Pytest
    success, _ = run_cmd(
        [sys.executable, '-m', 'pytest', '-q'],
        "Pytest (full suite)"
    )
    
    if not success:
        print("\n[GATE FAILED] Pytest failed")
        sys.exit(1)
    
    # 3. Eval report baseline-only
    success, _ = run_cmd(
        [
            sys.executable,
            'tools/stmm_stepA_eval_report.py',
            '--config', args.config,
            '--split', 'all',
            '--max_batches', str(args.max_batches),
            '--device', 'cpu',
            '--out_dir', 'runs/gate_eval_baseline',
        ],
        "Eval report (baseline-only, bounded)"
    )
    
    if not success:
        print("\n[GATE FAILED] Baseline eval failed")
        sys.exit(1)
    
    # 4. Eval report with model (if STMM_CKPT set)
    ckpt_path = os.environ.get('STMM_CKPT')
    if ckpt_path:
        print(f"\n[INFO] STMM_CKPT detected: {ckpt_path}")
        
        success, _ = run_cmd(
            [
                sys.executable,
                'tools/stmm_stepA_eval_report.py',
                '--config', args.config,
                '--ckpt', ckpt_path,
                '--split', 'all',
                '--max_batches', str(args.max_batches),
                '--device', 'cpu',
                '--out_dir', 'runs/gate_eval_with_model',
            ],
            "Eval report (with model, bounded)"
        )
        
        if not success:
            print("\n[GATE FAILED] Model eval failed")
            sys.exit(1)
    else:
        print("\n[INFO] STMM_CKPT not set, skipping model eval")
    
    # All passed
    print("\n" + "="*60)
    print("[GATE PASSED] All checks successful")
    print("="*60)
    print("\nArtifacts:")
    print("  - runs/gate_eval_baseline/report.md")
    if ckpt_path:
        print("  - runs/gate_eval_with_model/report.md")
    print()


if __name__ == "__main__":
    main()
