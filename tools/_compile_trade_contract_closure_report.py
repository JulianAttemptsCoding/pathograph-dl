import json
from pathlib import Path

def read_text(p):
    pp = Path(p)
    if not pp.exists():
        return ''
    # tolerate BOM/UTF-16-ish by fallback
    for enc in ('utf-8', 'utf-8-sig', 'utf-16', 'latin-1'):
        try:
            return pp.read_text(encoding=enc)
        except Exception:
            pass
    return pp.read_bytes().decode('latin-1', errors='replace')

reports = {
  'repo_state': read_text('docs/reports/_repo_state.txt'),
  'env_state': read_text('docs/reports/_env_state.txt'),
  'step4_smoketest': read_text('docs/reports/_step4_smoketest_output.txt'),
  'step5_risk_debug': read_text('docs/reports/_step5_risk_loss_debug.txt'),
  'spec_hits': read_text('docs/reports/_spec_semantics_hits.txt'),
}

zarr_inv = json.loads(Path('docs/reports/_zarr_inventory.json').read_text(encoding='utf-8'))
time_audit = json.loads(Path('docs/reports/_time_index_audit.json').read_text(encoding='utf-8'))
mask_audit = json.loads(Path('docs/reports/_mask_semantics_audit.json').read_text(encoding='utf-8'))
dataset_shapes = json.loads(Path('docs/reports/_dataset_getitem_shapes.json').read_text(encoding='utf-8'))

problems = []
recommended_fixes = []

# Detect key problems automatically
if not time_audit.get('equal_arrays', False):
    problems.append('Time index mismatch between Step1 and Step2.')
    recommended_fixes.append('Rebuild/realign time_index arrays; verify both Zarrs share the same axis.')

# Risk loss zero check - Read actual debug output
debug_out = read_text('docs/reports/_step5_risk_loss_debug.txt')
if debug_out:
    # Parse mask sums
    risk_mask_sum = None
    for line in debug_out.splitlines():
        if 'risk_mask sum' in line:
            parts = line.split()
            if len(parts) >= 3:
                risk_mask_sum = int(parts[-1])
    
    if risk_mask_sum is not None and risk_mask_sum == 0:
        problems.append('Risk mask sum is 0: ALL risk trade observations are masked out (no observed entries).')
        recommended_fixes.append('Investigate why risk_mask has no observed entries (all 0s). Check Step 2 preprocessing logic and observed_mask semantics.')
else:
    problems.append('Step5 risk debug output missing or empty.')

# Spec contradiction check
if '0=Valid' in reports['spec_hits'] and 'Code 1' in reports['spec_hits']:
    problems.append('Spec likely contains contradictory mask semantics language.')
    recommended_fixes.append('Remove/replace the outdated "0=Valid" statement; keep a single truth: observed is code!=0 (typically 1).')

out_json = {
  'repo_root': 'C:/Users/bubga.JULIAN-LAPTOPE2/PycharmProjects/PathoGraph-DL',
  'git_head': 'bbda1d2749208ee2a7bf362e05cf2d6b1f068c37',
  'python_version': '3.12.4',
  'package_versions': {
    'torch': '2.5.1',
    'zarr': '3.1.5',
    'numpy': 'from pip show',
    'pytorch_lightning': 'NOT INSTALLED'
  },
  'step1_inventory': zarr_inv.get('step1', {}),
  'step2_inventory': zarr_inv.get('step2', {}),
  'time_index_summary': time_audit,
  'mask_semantics_summary': mask_audit,
  'dataset_item_shapes': dataset_shapes,
  'batch_shapes_analysis': 'See smoketest output - confirmed 5D/6D masks',
  'risk_loss_debug_summary': debug_out,
  'problems_detected': problems,
  'recommended_fixes': recommended_fixes
}

Path('docs/reports/trade_contract_closure_audit.json').write_text(json.dumps(out_json, indent=2), encoding='utf-8')

md = []
md.append('# Trade Contract Closure Audit (No-Assumptions)')
md.append('')
md.append('**Date:** 2026-01-11')
md.append('**Repo:** PathoGraph-DL')
md.append('**HEAD:** bbda1d2749208ee2a7bf362e05cf2d6b1f068c37')
md.append('')
md.append('## Repo State')
md.append('```')
md.append('C:/Users/bubga.JULIAN-LAPTOPE2/PycharmProjects/PathoGraph-DL')
md.append('bbda1d2749208ee2a7bf362e05cf2d6b1f068c37 (master)')
md.append('Modified: trade_dataset.py, trade_collate.py, trade_datamodule.py, etc.')
md.append('```')
md.append('## Environment')
md.append('- Python: 3.12.4')
md.append('- torch: 2.5.1')
md.append('- zarr: 3.1.5')
md.append('- numpy: (from pip)')
md.append('- pytorch-lightning: NOT INSTALLED')
md.append('')
md.append('## Zarr Inventory')
md.append('```json')
md.append(json.dumps(zarr_inv, indent=2))
md.append('```')
md.append('## Time Index Audit')
md.append('```json')
md.append(json.dumps(time_audit, indent=2))
md.append('```')
md.append('## Mask Semantics (Count + Mass by Code)')
md.append('```json')
md.append(json.dumps(mask_audit, indent=2)[:20000])
md.append('```')
md.append('## Dataset __getitem__ Shapes')
md.append('```json')
md.append(json.dumps(dataset_shapes, indent=2))
md.append('```')
md.append('## Step4 Batch Shapes')
md.append('```')
md.append('Confirmed 5D base_mask: [B, L, N, N, 2]')
md.append('Confirmed 6D risk_mask: [B, L, N, N, K, 2]')
md.append('```')
md.append('## Step5 Risk Loss Debug Output')
md.append('```')
md.append(debug_out.strip() if debug_out else 'NO OUTPUT')
md.append('```')
md.append('## Spec Semantics Consistency Hits')
md.append('```')
md.append(reports['spec_hits'].strip())
md.append('```')
md.append('## Problems Detected')
for p in problems:
    md.append(f'- {p}')
md.append('')
md.append('## Recommended Fixes')
for r in recommended_fixes:
    md.append(f'- {r}')
md.append('')
md.append('## Agent Constraints & Workarounds')
md.append('- **Encoding Issues:** PowerShell $env:PYTHONPATH syntax causes parse errors. Solution: Write scripts to tools/ and run directly.')
md.append('- **Blocked Paths:** docs/reports/ is gitignored. Solution: Write there anyway (allowed by agent constraints).')
md.append('- **pytorch-lightning:** Not installed. Solution: Code gracefully handles absence.')

Path('docs/reports/trade_contract_closure_audit.md').write_text('\n'.join(md), encoding='utf-8')
print('WROTE docs/reports/trade_contract_closure_audit.md')
print('WROTE docs/reports/trade_contract_closure_audit.json')
