"""Trade Step 7 - Artifact Verification Tool.

Verifies Step 1 and Step 2 artifacts exist, have correct shapes, and are aligned.
Exports artifact_verification.json with schemas, hashes, and QC summary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ArrayInfo:
    name: str
    shape: List[int]
    dtype: str
    chunks: Optional[List[int]] = None


@dataclass
class ZarrGroupInfo:
    path: str
    arrays: List[ArrayInfo]
    size_bytes: int


@dataclass
class VerificationResult:
    success: bool
    timestamp: str
    base_zarr: ZarrGroupInfo
    risk_zarr: ZarrGroupInfo
    time_index_aligned: bool
    time_range: Dict[str, Any]
    split_boundaries: Dict[str, List[int]]
    manifest_hashes: Dict[str, str]
    qc_summary: Dict[str, Any]
    errors: List[str]


def compute_sha256(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_dir_size(path: Path) -> int:
    """Get total size of a directory."""
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def inspect_zarr_group(zarr_path: Path) -> ZarrGroupInfo:
    """Inspect a Zarr group and return metadata."""
    import zarr

    g = zarr.open_group(str(zarr_path), mode="r")
    arrays = []
    for name in g.array_keys():
        arr = g[name]
        arrays.append(ArrayInfo(
            name=name,
            shape=list(arr.shape),
            dtype=str(arr.dtype),
            chunks=list(arr.chunks) if arr.chunks else None,
        ))
    
    return ZarrGroupInfo(
        path=str(zarr_path),
        arrays=arrays,
        size_bytes=get_dir_size(zarr_path),
    )


def epoch_to_month(epoch: int) -> str:
    """Convert epoch (months since Jan 1950) to YYYY-MM string."""
    year = 1950 + (epoch // 12)
    month = (epoch % 12) + 1
    return f"{year:04d}-{month:02d}"


def verify_artifacts(
    base_zarr_path: Path,
    risk_zarr_path: Path,
    base_manifest_path: Path,
    risk_manifest_path: Path,
    split_train: tuple = (0, 815),
    split_val: tuple = (816, 851),
    split_test: tuple = (852, 907),
) -> VerificationResult:
    """Verify Step 1 and Step 2 artifacts."""
    import zarr

    errors: List[str] = []
    timestamp = datetime.utcnow().isoformat() + "Z"

    # Inspect Zarr groups
    try:
        base_info = inspect_zarr_group(base_zarr_path)
    except Exception as e:
        errors.append(f"Failed to open base Zarr: {e}")
        base_info = ZarrGroupInfo(path=str(base_zarr_path), arrays=[], size_bytes=0)

    try:
        risk_info = inspect_zarr_group(risk_zarr_path)
    except Exception as e:
        errors.append(f"Failed to open risk Zarr: {e}")
        risk_info = ZarrGroupInfo(path=str(risk_zarr_path), arrays=[], size_bytes=0)

    # Check time index alignment
    time_aligned = False
    time_range: Dict[str, Any] = {}
    try:
        base_g = zarr.open_group(str(base_zarr_path), mode="r")
        risk_g = zarr.open_group(str(risk_zarr_path), mode="r")
        
        base_ti = base_g["time_index"][:]
        risk_ti = risk_g["time_index"][:]
        
        time_aligned = np.array_equal(base_ti, risk_ti)
        if not time_aligned:
            errors.append("Time index arrays do not match between base and risk tensors.")
        
        time_range = {
            "T": int(len(base_ti)),
            "epoch_min": int(base_ti.min()),
            "epoch_max": int(base_ti.max()),
            "calendar_min": epoch_to_month(int(base_ti.min())),
            "calendar_max": epoch_to_month(int(base_ti.max())),
        }
    except Exception as e:
        errors.append(f"Failed to check time index alignment: {e}")

    # Expected shapes
    expected_base_shape = (908, 194, 194, 2)
    expected_risk_shape = (908, 194, 194, 8, 2)
    
    # Validate base shape
    base_trade = next((a for a in base_info.arrays if a.name == "trade"), None)
    if base_trade and tuple(base_trade.shape) != expected_base_shape:
        errors.append(f"Base trade shape mismatch: expected {expected_base_shape}, got {tuple(base_trade.shape)}")
    
    # Validate risk shape
    risk_trade = next((a for a in risk_info.arrays if a.name == "trade_risk"), None)
    if risk_trade and tuple(risk_trade.shape) != expected_risk_shape:
        errors.append(f"Risk trade shape mismatch: expected {expected_risk_shape}, got {tuple(risk_trade.shape)}")

    # Compute manifest hashes
    manifest_hashes: Dict[str, str] = {}
    for name, path in [("base_manifest", base_manifest_path), ("risk_manifest", risk_manifest_path)]:
        if path.exists():
            manifest_hashes[name] = compute_sha256(path)
        else:
            errors.append(f"Manifest not found: {path}")
            manifest_hashes[name] = "MISSING"

    # Load QC summaries
    qc_summary: Dict[str, Any] = {}
    for name, path in [("base", base_manifest_path), ("risk", risk_manifest_path)]:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                qc_summary[name] = manifest.get("qc_summary", {})
            except Exception as e:
                errors.append(f"Failed to load QC summary from {path}: {e}")

    # Split boundaries with calendar labels
    split_boundaries = {
        "train": [split_train[0], split_train[1]],
        "val": [split_val[0], split_val[1]],
        "test": [split_test[0], split_test[1]],
    }

    return VerificationResult(
        success=len(errors) == 0,
        timestamp=timestamp,
        base_zarr=base_info,
        risk_zarr=risk_info,
        time_index_aligned=time_aligned,
        time_range=time_range,
        split_boundaries=split_boundaries,
        manifest_hashes=manifest_hashes,
        qc_summary=qc_summary,
        errors=errors,
    )


def main():
    parser = argparse.ArgumentParser(description="Trade Step 7 - Artifact Verification")
    parser.add_argument("--base-zarr", default="data/processed/trade/imf_imts_step1/trade_fob_tensor.zarr")
    parser.add_argument("--risk-zarr", default="data/processed/trade/faostat_step2/trade_risk_tensor.zarr")
    parser.add_argument("--base-manifest", default="data/processed/trade/imf_imts_step1/manifest.json")
    parser.add_argument("--risk-manifest", default="data/processed/trade/faostat_step2/preprocessing_manifest.json")
    parser.add_argument("--output", "-o", default="artifact_verification.json", help="Output JSON path")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress stdout output")
    args = parser.parse_args()

    # Resolve paths relative to script location (project root)
    root = Path(__file__).resolve().parent.parent
    
    base_zarr = root / args.base_zarr
    risk_zarr = root / args.risk_zarr
    base_manifest = root / args.base_manifest
    risk_manifest = root / args.risk_manifest
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path

    result = verify_artifacts(
        base_zarr_path=base_zarr,
        risk_zarr_path=risk_zarr,
        base_manifest_path=base_manifest,
        risk_manifest_path=risk_manifest,
    )

    # Serialize dataclasses
    def serialize(obj):
        if hasattr(obj, "__dict__"):
            return {k: serialize(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [serialize(v) for v in obj]
        elif isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        return obj

    output_dict = serialize(result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=2)

    if not args.quiet:
        print(f"Artifact verification {'PASSED' if result.success else 'FAILED'}")
        print(f"  Base Zarr: {base_zarr}")
        print(f"  Risk Zarr: {risk_zarr}")
        print(f"  Time aligned: {result.time_index_aligned}")
        print(f"  Time range: {result.time_range.get('calendar_min')} to {result.time_range.get('calendar_max')}")
        if result.errors:
            print("  Errors:")
            for e in result.errors:
                print(f"    - {e}")
        print(f"  Output: {output_path}")

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
