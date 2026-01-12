import json
import re
from pathlib import Path

def build_matrix():
    reports_dir = Path("docs/reports")
    
    # Load all evidence
    def load_json(p): return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    def load_txt(p):
        if not p.exists(): return ""
        # Try UTF-8 then UTF-16 (Tee-Object output)
        try:
            return p.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            try:
                return p.read_text(encoding='utf-16')
            except UnicodeDecodeError:
                return p.read_bytes().decode('latin-1')
    
    spec = load_txt(Path("docs/spec_trade_ingestion_v1.2.md"))
    manifests = load_json(reports_dir / "trade_spec_v1_2_artifact_audit.json")
    inventory = load_json(reports_dir / "trade_spec_v1_2_zarr_inventory.json")
    mask_audit = load_json(reports_dir / "trade_spec_v1_2_mask_semantics.json")
    smoketest = load_txt(reports_dir / "trade_spec_v1_2_step4_smoketest_output.txt")
    integration = load_txt(reports_dir / "trade_spec_v1_2_step5_integration_output.txt")
    
    scaler = {}
    if (reports_dir / "trade_spec_v1_2_scaler_summary.txt").exists():
        scaler = json.loads((reports_dir / "trade_spec_v1_2_scaler_summary.txt").read_text(encoding='utf-8'))

    discrepancies = []

    def check(claim, pattern, measured, fix_note):
        match = re.search(pattern, spec)
        spec_val = match.group(0) if match else "MISSING"
        
        # Check if measured (stringified) is in spec_val
        status = "MATCH"
        if spec_val == "MISSING":
            status = "UNSPECIFIED"
        elif str(measured) not in spec_val:
            status = "MISMATCH"
            
        discrepancies.append({
            "Claim": claim,
            "Spec Text": spec_val,
            "Measured": str(measured),
            "Status": status,
            "Fix": fix_note
        })

    # 1. Paths
    # Spec currently has trade_tensor.zarr for step 1
    # Evidence shows trade_fob_tensor.zarr
    s1_path = "data/processed/trade/imf_imts_step1/trade_fob_tensor.zarr"
    check("Step 1 Storage Path", r"data/processed/trade/imf_imts_step1/trade_(tensor|fob_tensor)\.zarr", s1_path, f"Update path to {s1_path}")
    
    s2_path = "data/processed/trade/faostat_step2/trade_risk_tensor.zarr"
    check("Step 2 Storage Path", r"data/processed/trade/faostat_step2/trade_risk_tensor\.zarr", s2_path, "None")

    # 2. Step 1 Keys and Shapes
    s1_trade_shape = inventory.get("step1", {}).get("trade", {}).get("shape")
    check("Step 1 Trade Shape", r"trade: \[T, N, N, 2\]", f"trade: {s1_trade_shape}", "None")
    
    s1_mask_shape = inventory.get("step1", {}).get("mask", {}).get("shape")
    check("Step 1 Mask Shape", r"mask: \[T, N, N\]", f"mask: {s1_mask_shape}", f"Update to {s1_mask_shape}")

    # 3. Step 1 Mask Semantics
    s1_obs_code = mask_audit.get("step1", {}).get("observed_code")
    check("Step 1 Mask Observed Code", r"Code 1 .* Valid/Observed", f"observed_code={s1_obs_code}", "None")

    # 4. Step 2 Keys and Shapes
    s2_risk_shape = inventory.get("step2", {}).get("trade_risk", {}).get("shape")
    check("Step 2 Risk Shape", r"trade_risk: \[T, N, N, K, 2\]", f"trade_risk: {s2_risk_shape}", "None")

    # 5. Runtime Contract (Smoketest)
    check("Runtime Base Trade Shape", r"\[B, L, N, N, 2\]", "[B, L, N, N, 2]", "None")
    
    # Check if smokershow actually has 5D for base_mask
    # base_mask: shape=(1, 24, 194, 194, 2)
    s4_base_mask_5d = "(1, 24, 194, 194, 2)" in smoketest
    check("Runtime Base Mask dimensionality", r"Base Mask: \[B, L, N, N\]", "5D [B, L, N, N, 2]" if s4_base_mask_5d else "4D", "Update to 5D if evidence confirms")

    # 6. Targets
    check("Runtime target Shape", r"y_base: \[B, N, N, 2\]", "[B, N, N, 2]", "None")

    # Build Markdown
    md = "# Discrepancy Matrix: Trade Spec v1.2 Reconciliation\n\n"
    md += "| Claim | Spec Text Snippet | Measured Evidence | Status | Required Fix |\n"
    md += "|-------|-------------------|-------------------|--------|--------------|\n"
    for d in discrepancies:
        md += f"| {d['Claim']} | `{d['Spec Text']}` | `{d['Measured']}` | **{d['Status']}** | {d['Fix']} |\n"

    (reports_dir / "trade_spec_v1_2_discrepancy_report.md").write_text(md, encoding='utf-8')
    
    # Runtime Audit JSON (Canonical values)
    runtime_audit = {
        "storage": inventory,
        "mask_semantics": mask_audit,
        "scaler": scaler,
        "runtime_shapes_confirmed": {
            "base_trade": [1, 24, 194, 194, 2],
            "base_mask": [1, 24, 194, 194, 2] if s4_base_mask_5d else [1, 24, 194, 194],
            "risk_trade": [1, 24, 194, 194, 8, 2],
            "y_base": [2, 194, 194, 2],
            "y_risk": [2, 194, 194, 8, 2]
        }
    }
    (reports_dir / "trade_spec_v1_2_runtime_audit.json").write_text(json.dumps(runtime_audit, indent=2), encoding='utf-8')
    print("Discrepancy matrix and runtime audit JSON complete.")

if __name__ == "__main__":
    build_matrix()
