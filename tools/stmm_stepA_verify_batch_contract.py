"""Verify multimodal batch contract for ST-MM-GNN DataModule.

This script validates that the DataModule emits all required modalities with correct shapes.
"""

from pathlib import Path
import yaml
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from pathograph.data.trade_datamodule import TradeDataModule, TradeDataModuleConfig
from pathograph.data.trade_dataset import TradeSplit


def convert_splits(cfg_dict):
    """Convert list-based split definitions to TradeSplit objects in place."""
    for key in ["split_train", "split_val", "split_test"]:
        if key in cfg_dict and isinstance(cfg_dict[key], (list, tuple)):
            vals = cfg_dict[key]
            if len(vals) != 2:
                raise ValueError(f"Split {key} must have exactly 2 elements (min, max). Got: {vals}")
            cfg_dict[key] = TradeSplit(int(vals[0]), int(vals[1]))


def main():
    cfg_path = Path("config/stmm_stepA.yaml")
    assert cfg_path.exists(), f"Missing {cfg_path}"
    
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    
    # Get datamodule config
    dm_raw = cfg.get("datamodule", {})
    ROOT = Path.cwd()
    
    # Resolve relative paths
    for p_key in [
        "base_zarr_path",
        "risk_zarr_path",
        "scaler_json_path",
        "pathogen_zarr_path",
        "climate_zarr_path",
        "climate_anoms_zarr_path",
        "meta_distance_path",
        "meta_adjacency_path",
    ]:
        if p_key in dm_raw and isinstance(dm_raw[p_key], str):
            p = Path(dm_raw[p_key])
            if not p.is_absolute():
                dm_raw[p_key] = str(ROOT / p)
    
    convert_splits(dm_raw)
    dm_cfg = TradeDataModuleConfig(**dm_raw)
    
    # Instantiate and setup datamodule
    dm = TradeDataModule(dm_cfg)
    dm.setup()
    
    # Get first batch
    batch = next(iter(dm.train_dataloader()))
    
    print("\n[MULTIMODAL BATCH CONTRACT VERIFICATION]")
    print(f"\nBatch keys: {sorted(batch.keys())}")
    print(f"\nBatch shapes:")
    for k in sorted(batch.keys()):
        v = batch[k]
        if hasattr(v, "shape"):
            print(f"  {k:25s} {str(tuple(v.shape)):30s} dtype={v.dtype}")
        else:
            print(f"  {k:25s} {str(type(v)):30s}")
    
    # Required keys for multimodal ST-MM-GNN
    required = [
        "base_trade",
        "risk_trade",
        "climate",
        "climate_anoms",
        "distance_km",
        "adjacency_border",
        "y_next",
        "y_mask",
        "t",
        "t_y",
    ]
    
    print(f"\n[REQUIRED KEYS CHECK]")
    missing = []
    for k in required:
        present = k in batch
        print(f"  {k:25s} {'✓ PRESENT' if present else '✗ MISSING'}")
        if not present:
            missing.append(k)
    
    if missing:
        print(f"\nERROR: Missing required keys: {missing}")
        sys.exit(1)
    
    # Expected shapes
    expected_shapes = {
        "base_trade": (1, 24, 194, 194, 2),
        "risk_trade": (1, 24, 194, 194, 8, 2),
        "climate": (1, 24, 194, 10),
        "climate_anoms": (1, 24, 194, 10),
        "distance_km": (194, 194),
        "adjacency_border": (194, 194),
        "y_next": (1, 194, 8),
        "y_mask": (1, 194, 8),
    }
    
    print(f"\n[SHAPE VERIFICATION]")
    shape_errors = []
    for k, expected_shape in expected_shapes.items():
        actual_shape = tuple(batch[k].shape)
        match = actual_shape == expected_shape
        print(f"  {k:25s} expected={str(expected_shape):30s} actual={str(actual_shape):30s} {'✓' if match else '✗'}")
        if not match:
            shape_errors.append((k, expected_shape, actual_shape))
    
    if shape_errors:
        print(f"\nERROR: Shape mismatches detected:")
        for k, exp, act in shape_errors:
            print(f"  {k}: expected {exp}, got {act}")
        sys.exit(1)
    
    # Check y_mask has positives
    y_mask_nonzero = int((batch["y_mask"] > 0).sum().item())
    print(f"\n[TARGET MASK CHECK]")
    print(f"  y_mask nonzero: {y_mask_nonzero} (must be > 0)")
    
    if y_mask_nonzero == 0:
        print("ERROR: y_mask has no positive values")
        sys.exit(1)
    
    print(f"\n[✓ ALL CHECKS PASSED]")
    print("Multimodal batch contract verified successfully.")
    print(f"All required keys present with correct shapes.")


if __name__ == "__main__":
    main()
