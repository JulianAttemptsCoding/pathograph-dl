from __future__ import annotations

import json
from pathlib import Path

def find_key(x, key):
    if isinstance(x, dict):
        for k, v in x.items():
            if k == key:
                yield v
            yield from find_key(v, key)
    elif isinstance(x, list):
        for v in x:
            yield from find_key(v, key)


def main() -> None:
    step1_path = Path("data/processed/trade/imf_imts_step1/manifest.json")
    if not step1_path.exists():
        raise SystemExit(f"ERROR: Step 1 manifest not found at {step1_path}")
    
    step1 = json.loads(step1_path.read_text(encoding="utf-8"))
    expected = None

    # Common locations used in manifests
    if isinstance(step1.get("tensor_shape"), list) and step1["tensor_shape"]:
        expected = step1["tensor_shape"][0]
    else:
        # Fallback: search recursively
        shapes = [v for v in find_key(step1, "tensor_shape") if isinstance(v, list) and v]
        if shapes:
            expected = shapes[0][0]

    if expected is None:
        raise SystemExit("ERROR: Could not determine expected T from Step 1 manifest")

    # Check multiple possible artifact filenames
    artifacts = [
        "data/processed/trade/faostat_step2/completion_report.json",
        "data/processed/trade/faostat_step2/preprocessing_manifest.json"
    ]
    
    vals = []
    for art in artifacts:
        p = Path(art)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            vals.extend(list(find_key(data, "months_processed")))

    print("expected_T=", expected)
    print("months_processed_values=", vals)

    if expected not in vals:
        raise SystemExit("ERROR: expected_T not found among months_processed values in artifacts")

    print("OK: months_processed matches expected_T")


if __name__ == "__main__":
    main()
