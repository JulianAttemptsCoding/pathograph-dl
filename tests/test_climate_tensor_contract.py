from pathlib import Path
import numpy as np

def test_climate_tensor_contract_if_exists():
    zpath = Path("data/processed/climate/climate_tensor.zarr")
    if not zpath.exists():
        return  # contract test runs only when outputs exist

    import zarr  # type: ignore

    g = zarr.open_group(str(zpath), mode="r")

    for k in ["climate", "mask", "time_index", "feature_names"]:
        assert k in g, f"Missing array '{k}' in {zpath}"

    climate = g["climate"]
    mask = g["mask"]
    time_index = np.asarray(g["time_index"][:]).astype(np.int32)
    feature_names = list(np.asarray(g["feature_names"][:]).astype(str))

    assert climate.shape == (908, 194, 10)
    assert str(climate.dtype) == "float32"
    assert mask.shape == (908, 194, 10)
    assert str(mask.dtype) in ("uint8", "ubyte")

    master = np.load("data/processed/meta/time_index_master.npy").astype(np.int32)
    assert np.array_equal(time_index, master)

    # Feature order must match locked config
    import yaml  # type: ignore
    cfg = yaml.safe_load(Path("config/climate_step1.yaml").read_text(encoding="utf-8"))
    locked = cfg["processing"]["feature_order_locked"]
    assert feature_names == locked
