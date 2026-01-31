"""
STMM Layer-A Evaluation & Reporting Gate Runner.

Orchestrates the full gate sequence IN-PROCESS to avoid environment mismatch:
1. Label sanity check (strict mode, bounded)
2. Pytest (full suite)
3. Eval report (baseline-only, bounded)
4. Eval report with model (if STMM_CKPT env var is set)

Exits nonzero on first failure.
"""

import argparse
import os
import runpy
import sys
from datetime import datetime
from pathlib import Path


def run_script(script_path, argv_list, desc):
    """
    Run a Python script in-process with custom argv.
    
    Args:
        script_path: Path to script
        argv_list: List of arguments (script path will be prepended)
        desc: Description for output
    
    Returns:
        (success, exit_code)
    """
    print(f"\n{'='*60}")
    print(f"[GATE] {desc}")
    print(f"{'='*60}")
    print(f"$ python {script_path} {' '.join(argv_list)}")
    print()
    
    # Save original argv
    orig_argv = sys.argv.copy()
    
    try:
        # Set argv for the script
        sys.argv = [str(script_path)] + argv_list
        
        # Run in-process
        runpy.run_path(str(script_path), run_name='__main__')
        
        # If we get here, script succeeded
        print(f"\n[OK] {desc}")
        return True, 0
        
    except SystemExit as e:
        # Script called sys.exit()
        exit_code = e.code if e.code is not None else 0
        success = (exit_code == 0)
        status = "[OK]" if success else "[FAIL]"
        print(f"\n{status} {desc}")
        return success, exit_code
        
    except Exception as e:
        print(f"\n[FAIL] {desc}: {str(e)}")
        return False, 1
        
    finally:
        # Restore argv
        sys.argv = orig_argv


def main():
    parser = argparse.ArgumentParser(description='STMM Layer-A Evaluation Gate')
    parser.add_argument('--config', type=str, default='config/stmm_stepA.yaml')
    parser.add_argument('--max_batches', type=int, default=10,
                        help='Max batches for bounded sanity/eval runs')
    args = parser.parse_args()
    
    print("="*60)
    print("STMM LAYER-A EVALUATION & REPORTING GATE")
    print("="*60)
    
    # Find repo root and change to it
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)
    
    # Preflight: check torch import
    try:
        import torch
        print(f"\n[OK] Environment check: torch {torch.__version__}")
    except ImportError:
        print("\n[FAIL] Environment check: torch not importable")
        print("Please run via: conda run -n pathograph-train python tools/gate_layerA_eval_report.py ...")
        sys.exit(1)
    
    # Create gate output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    gate_out_dir = repo_root / 'runs' / 'gate_layerA_eval_report' / timestamp
    gate_out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGate output directory: {gate_out_dir}")
    
    # 1. Label sanity check
    success, _ = run_script(
        repo_root / 'tools' / 'stmm_stepA_label_sanity.py',
        [
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
    print(f"\n{'='*60}")
    print("[GATE] Pytest (full suite)")
    print(f"{'='*60}")
    print("$ python -m pytest -q")
    print()
    
    try:
        import pytest
        exit_code = pytest.main(['-q'])
        success = (exit_code == 0)
        status = "[OK]" if success else "[FAIL]"
        print(f"\n{status} Pytest (full suite)")
        
        if not success:
            print("\n[GATE FAILED] Pytest failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] Pytest: {str(e)}")
        print("\n[GATE FAILED] Pytest failed")
        sys.exit(1)
    
    # 3. Eval report baseline-only
    baseline_out = gate_out_dir / 'baseline_eval'
    success, _ = run_script(
        repo_root / 'tools' / 'stmm_stepA_eval_report.py',
        [
            '--config', args.config,
            '--split', 'all',
            '--max_batches', str(args.max_batches),
            '--device', 'cpu',
            '--out_dir', str(baseline_out),
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
        
        model_out = gate_out_dir / 'model_eval'
        success, _ = run_script(
            repo_root / 'tools' / 'stmm_stepA_eval_report.py',
            [
                '--config', args.config,
                '--ckpt', ckpt_path,
                '--split', 'all',
                '--max_batches', str(args.max_batches),
                '--device', 'cpu',
                '--out_dir', str(model_out),
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
    print(f"  - {baseline_out / 'report.md'}")
    if ckpt_path:
        print(f"  - {model_out / 'report.md'}")
    print()


if __name__ == "__main__":
    main()
