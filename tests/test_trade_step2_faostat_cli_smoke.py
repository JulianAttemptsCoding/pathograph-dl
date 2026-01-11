"""
Tests for FAOSTAT Step 2: CLI Smoke Tests

Tests:
1. Template generator mode
2. Full pipeline mode
3. Manifest validation
4. Deterministic outputs
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import zarr


def create_synthetic_step1_manifest(temp_dir):
    """Create a minimal synthetic Step 1 manifest and artifacts."""
    step1_dir = Path(temp_dir) / 'step1'
    step1_dir.mkdir(parents=True, exist_ok=True)
    
    # Create node index
    node_index_path = step1_dir / 'node_index.csv'
    node_index_df = pd.DataFrame({
        'node_id': [0, 1, 2],
        'iso3': ['USA', 'CAN', 'MEX']
    })
    node_index_df.to_csv(node_index_path, index=False)
    
    # Create minimal base tensor
    zarr_path = step1_dir / 'trade_fob_tensor.zarr'
    
    try:
        store = zarr.DirectoryStore(str(zarr_path))
    except AttributeError:
        from zarr.storage import LocalStore
        store = LocalStore(str(zarr_path))
    
    root = zarr.open_group(store=store, mode='w')
    
    # Create small tensor: 24 months (2 years), 3 nodes, 2 channels
    # Months cover 2020-2021 (t_min=840)
    T = 24
    N = 3
    
    trade = np.random.rand(T, N, N, 2).astype(np.float32) * 1000.0
    mask = np.ones((T, N, N, 2), dtype=np.uint8)
    is_estimated = np.zeros((T, N, N, 2), dtype=np.uint8)
    time_index = np.arange(840, 840 + T, dtype=np.int32)
    
    # Zero diagonal
    for t in range(T):
        for i in range(N):
            trade[t, i, i, :] = 0.0
            mask[t, i, i, :] = 0
    
    def _create(name, data, chunks):
    	data = np.asarray(data)
    	arr = root.create_array(name, shape=data.shape, dtype=data.dtype, chunks=chunks)
    	arr[:] = data

    
    _create('trade', trade, (12, 3, 3, 2))
    _create('mask', mask, (12, 3, 3, 2))
    _create('is_estimated', is_estimated, (12, 3, 3, 2))
    _create('time_index', time_index, (24,))
    
    # Create manifest
    manifest = {
        'version': '1.0',
        'outputs': {
            'zarr_path': str(zarr_path),
            'tensor_shape': [T, N, N, 2],
            'tensor_dtype': 'float32',
            'mask_dtype': 'uint8',
            'is_estimated_dtype': 'uint8'
        },
        'time_axis': {
            't_min': 840,
            't_max': 840 + T - 1,
            'T_full': T
        },
        'inputs': {
            'node_index': {
                'path': str(node_index_path),
                'sha256': 'dummy_hash'
            }
        }
    }
    
    manifest_path = step1_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    return str(manifest_path)


def create_synthetic_faostat_csv(temp_dir):
    """Create a minimal synthetic FAOSTAT CSV (Wide Format for Template Scanner compatibility)."""
    faostat_path = Path(temp_dir) / 'faostat.csv'
    
    # M49 Mapping
    iso3_to_m49 = {'USA': '840', 'CAN': '124', 'MEX': '484'}
    
    # Create synthetic data for 2019-2020, 3 countries, 3 items
    # Pivot logic: keys are (Reporter, Partner, Item Code)
    data = {}
    
    for reporter in ['USA', 'CAN', 'MEX']:
        for partner in ['USA', 'CAN', 'MEX']:
            if reporter == partner:
                continue
            for item_code in [15, 27, 44]:
                key = (reporter, partner, item_code)
                # Random values for 2019, 2020
                data[key] = {
                    'Y2019': np.random.rand() * 1000.0,
                    'Y2020': np.random.rand() * 1000.0
                }
    
    rows = []
    for (rep, part, item), vals in data.items():
        row = {
            'Reporter Country Code': iso3_to_m49[rep],  # Dummy FAO code
            'Reporter Country Code (M49)': iso3_to_m49[rep],
            'Partner Country Code': iso3_to_m49[part],  # Dummy FAO code
            'Partner Country Code (M49)': iso3_to_m49[part],
            'Item Code': item,
            'Item': f"Item {item}",
            'Element': 'Export Value',
            'Unit': 'USD',
            'Y2019': vals['Y2019'],
            'Y2020': vals['Y2020'],
            'Flag': ''
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(faostat_path, index=False)
    
    return str(faostat_path)


def create_groups_csv(temp_dir):
    """Create a minimal groups mapping CSV."""
    groups_path = Path(temp_dir) / 'faostat_groups.csv'
    
    df = pd.DataFrame({
        'item_code': [15, 27, 44],
        'group_id': ['CEREALS', 'LIVESTOCK', 'VEGETABLES'],
        'group_name': ['Cereals', 'Livestock', 'Vegetables']
    })
    df.to_csv(groups_path, index=False)
    
    return str(groups_path)


def test_cli_help():
    """Test that CLI --help works."""
    result = subprocess.run(
        [sys.executable, '-m', 'tools.trade_step2_faostat', '--help'],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    assert 'FAOSTAT Step 2' in result.stdout
    assert '--emit-groups-template' in result.stdout
    assert '--step1-manifest' in result.stdout


def test_template_generator_mode():
    """Test template generator mode."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create synthetic FAOSTAT file
        faostat_path = create_synthetic_faostat_csv(temp_dir)
        
        # Create node index for filtering
        node_index_path = Path(temp_dir) / 'node_index.csv'
        node_index_df = pd.DataFrame({
            'node_id': [0, 1, 2],
            'iso3': ['USA', 'CAN', 'MEX']
        })
        node_index_df.to_csv(node_index_path, index=False)
        
        output_dir = Path(temp_dir) / 'output'
        
        # Run template generator with noncanonical mode
        result = subprocess.run(
            [
                sys.executable, '-m', 'tools.trade_step2_faostat',
                '--faostat-path', faostat_path,
                '--node-index', str(node_index_path),
                '--allow-noncanonical-n',
                '--emit-groups-template', str(output_dir)
            ],
            capture_output=True,
            text=True
        )
        
        # Check exit code
        assert result.returncode == 0, f"Template generator failed:\n{result.stderr}"
        
        # Check that template files were created
        items_path = output_dir / 'faostat_items_present.csv'
        template_path = output_dir / 'faostat_groups.template.csv'
        
        assert items_path.exists(), "faostat_items_present.csv not created"
        assert template_path.exists(), "faostat_groups.template.csv not created"
        
        # Validate items file
        items_df = pd.read_csv(items_path)
        assert 'item_code' in items_df.columns
        assert 'total_value' in items_df.columns
        assert 'row_count' in items_df.columns
        assert len(items_df) == 3  # 3 items
        
        # Validate template file
        template_df = pd.read_csv(template_path)
        assert 'item_code' in template_df.columns
        assert 'group_id' in template_df.columns
        assert len(template_df) == 3


def test_full_pipeline_mode():
    """Test full pipeline mode."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create synthetic inputs
        step1_manifest_path = create_synthetic_step1_manifest(temp_dir)
        faostat_path = create_synthetic_faostat_csv(temp_dir)
        groups_csv_path = create_groups_csv(temp_dir)
        
        output_dir = Path(temp_dir) / 'step2_output'
        
        # Run full pipeline with noncanonical mode
        result = subprocess.run(
            [
                sys.executable, '-m', 'tools.trade_step2_faostat',
                '--step1-manifest', step1_manifest_path,
                '--faostat-path', faostat_path,
                '--groups-csv', groups_csv_path,
                '--out-dir', str(output_dir),
                '--allow-noncanonical-n',
                '--max-trace', '10'
            ],
            capture_output=True,
            text=True
        )
        
        # Check exit code
        assert result.returncode == 0, f"Full pipeline failed:\n{result.stderr}"
        
        # Check that all expected outputs exist
        assert (output_dir / 'weights_corridor_year.zarr').exists()
        assert (output_dir / 'trade_risk_tensor.zarr').exists()
        assert (output_dir / 'qc_report.json').exists()
        assert (output_dir / 'trace_samples.jsonl').exists()
        assert (output_dir / 'preprocessing_manifest.json').exists()
        assert (output_dir / 'completion_report.json').exists()
        
        # Validate manifest
        with open(output_dir / 'preprocessing_manifest.json', 'r') as f:
            manifest = json.load(f)
        
        assert manifest['version'] == '1.0'
        assert manifest['step'] == 'faostat_step2'
        assert 'inputs' in manifest
        assert 'outputs' in manifest
        assert 'weights' in manifest
        assert manifest['weights']['K'] == 3  # 3 groups
        
        # Validate QC report
        with open(output_dir / 'qc_report.json', 'r') as f:
            qc = json.load(f)
        
        assert 'time_coverage' in qc
        assert 'base_to_risk_coverage' in qc
        assert 'group_distributions' in qc
        
        # Validate trace samples
        trace_samples = []
        with open(output_dir / 'trace_samples.jsonl', 'r') as f:
            for line in f:
                trace_samples.append(json.loads(line))
        
        assert len(trace_samples) > 0
        assert len(trace_samples) <= 10  # max_trace=10
        
        # Each sample should have required fields
        for sample in trace_samples:
            assert 'exporter_iso3' in sample
            assert 'importer_iso3' in sample
            assert 'base_flow_usd' in sample
            assert 'weights' in sample
            assert 'risk_flows' in sample
            assert 'selected_weight_year' in sample
        
        # Validate risk tensor Zarr
        try:
            store = zarr.DirectoryStore(str(output_dir / 'trade_risk_tensor.zarr'))
        except AttributeError:
            from zarr.storage import LocalStore
            store = LocalStore(str(output_dir / 'trade_risk_tensor.zarr'))
        
        root = zarr.open_group(store=store, mode='r')
        
        assert 'trade_risk' in root
        assert 'observed_mask' in root
        assert 'is_estimated' in root
        
        trade_risk = root['trade_risk']
        assert trade_risk.shape == (24, 3, 3, 3, 2)  # (T=24, N=3, N=3, K=3, C=2)


def test_deterministic_outputs():
    """Test that running twice with identical inputs produces identical outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create synthetic inputs
        step1_manifest_path = create_synthetic_step1_manifest(temp_dir)
        faostat_path = create_synthetic_faostat_csv(temp_dir)
        groups_csv_path = create_groups_csv(temp_dir)
        
        output_dir1 = Path(temp_dir) / 'step2_output1'
        output_dir2 = Path(temp_dir) / 'step2_output2'
        
        # Run pipeline twice with noncanonical mode
        for output_dir in [output_dir1, output_dir2]:
            result = subprocess.run(
                [
                    sys.executable, '-m', 'tools.trade_step2_faostat',
                    '--step1-manifest', step1_manifest_path,
                    '--faostat-path', faostat_path,
                    '--groups-csv', groups_csv_path,
                    '--out-dir', str(output_dir),
                    '--allow-noncanonical-n',
                    '--max-trace', '10'
                ],
                capture_output=True,
                text=True
            )
            
            assert result.returncode == 0
        
        # Compare manifests (excluding timestamps if any)
        with open(output_dir1 / 'preprocessing_manifest.json', 'r') as f:
            manifest1 = json.load(f)
        
        with open(output_dir2 / 'preprocessing_manifest.json', 'r') as f:
            manifest2 = json.load(f)
        
        # Compare key fields (excluding timestamps)
        assert manifest1['weights'] == manifest2['weights']
        assert manifest1['parameters'] == manifest2['parameters']
        assert manifest1['mapping_stats'] == manifest2['mapping_stats']
        
        # Compare risk tensors
        try:
            store1 = zarr.DirectoryStore(str(output_dir1 / 'trade_risk_tensor.zarr'))
            store2 = zarr.DirectoryStore(str(output_dir2 / 'trade_risk_tensor.zarr'))
        except AttributeError:
            from zarr.storage import LocalStore
            store1 = LocalStore(str(output_dir1 / 'trade_risk_tensor.zarr'))
            store2 = LocalStore(str(output_dir2 / 'trade_risk_tensor.zarr'))
        
        root1 = zarr.open_group(store=store1, mode='r')
        root2 = zarr.open_group(store=store2, mode='r')
        
        # Note: trace samples may differ due to random sampling, so we skip that check
        # But the main tensors should be identical
        np.testing.assert_array_equal(np.array(root1['trade_risk']), np.array(root2['trade_risk']))
        np.testing.assert_array_equal(np.array(root1['observed_mask']), np.array(root2['observed_mask']))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
