from pathlib import Path

def test_climate_config_schema():
    import yaml  # type: ignore

    p = Path("config/climate_step1.yaml")
    assert p.exists(), "Missing config/climate_step1.yaml"

    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert "climate" in cfg
    assert "paths" in cfg
    assert "processing" in cfg

    clim = cfg["climate"]
    proc = cfg["processing"]

    assert clim["dataset_id"] == "reanalysis-era5-single-levels-monthly-means"
    assert clim["product_type"] == "monthly_averaged_reanalysis"
    assert clim["format"] == "netcdf"

    vars_ = clim["variables"]
    assert isinstance(vars_, list)
    assert len(vars_) == 7

    order = proc["feature_order_locked"]
    assert isinstance(order, list)
    assert len(order) == 10
