import difflib
import pathlib
import subprocess
import os

def generate_diff():
    p = pathlib.Path('docs/spec_trade_ingestion_v1.2.md')
    snap = pathlib.Path('docs/reports/trade_spec_v1_2_spec_snapshot.md')
    
    if not p.exists() or not snap.exists():
        print("Missing spec or snapshot for diff.")
        return
        
    before = snap.read_text(encoding='utf-8').splitlines(keepends=True)
    after = p.read_text(encoding='utf-8').splitlines(keepends=True)
    
    diff = difflib.unified_diff(before, after, fromfile='snapshot', tofile='v1.2')
    out = pathlib.Path('docs/reports/trade_spec_v1_2_patch_diff.md')
    out.write_text("".join(diff), encoding='utf-8')
    print("Patch diff generated.")

def run_postpatch_checks():
    print("\nRunning post-patch checks...")
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Smoketest
    print("Running smoketest...")
    res4 = subprocess.run(["python", "tools/trade_step4_smoketest.py"], env=env, capture_output=True, text=True)
    pathlib.Path("docs/reports/trade_spec_v1_2_step4_smoketest_output_postpatch.txt").write_text(res4.stdout + res4.stderr, encoding='utf-8')
    
    # Integration
    print("Running integration test...")
    res5 = subprocess.run(["python", "tools/trade_step5_integration_test.py"], env=env, capture_output=True, text=True)
    pathlib.Path("docs/reports/trade_spec_v1_2_step5_integration_output_postpatch.txt").write_text(res5.stdout + res5.stderr, encoding='utf-8')
    
    print("Post-patch checks complete.")

if __name__ == "__main__":
    generate_diff()
    run_postpatch_checks()
