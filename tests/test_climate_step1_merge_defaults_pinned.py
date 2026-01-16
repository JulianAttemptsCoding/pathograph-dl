from pathlib import Path

def test_step1_xarray_merge_defaults_are_pinned():
    p = Path('tools/climate_step1_download_era5.py')
    assert p.exists(), f"Missing file: {p}"
    txt = p.read_text(encoding='utf-8')
    # Must explicitly pin current effective defaults to avoid FutureWarning behavior drift
    assert 'xr.merge' in txt, 'Expected xr.merge call not found'
    # Strong but simple: ensure parameters appear in the merge call area
    assert "join='outer'" in txt or 'join="outer"' in txt, 'join=outer not pinned'
    assert "compat='no_conflicts'" in txt or 'compat="no_conflicts"' in txt, 'compat=no_conflicts not pinned'
