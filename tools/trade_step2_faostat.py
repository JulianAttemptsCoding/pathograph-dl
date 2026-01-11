"""
CLI Tool: FAOSTAT Step 2 Pipeline

Usage (Template Generator):
    python -m tools.trade_step2_faostat \
        --faostat-path data/raw/faostat/Trade_DetailedTradeMatrix_E_All_Data.zip \
        --emit-groups-template config/

Usage (Full Pipeline):
    python -m tools.trade_step2_faostat \
        --step1-manifest data/processed/trade/imts_step1/manifest.json \
        --faostat-path data/raw/faostat/Trade_DetailedTradeMatrix_E_All_Data.zip \
        --groups-csv config/faostat_groups.csv \
        --out-dir data/processed/trade/faostat_step2

This script orchestrates the complete Step 2 pipeline:
1. Template Generator Mode: Generate groups mapping template from FAOSTAT data
2. Full Pipeline Mode: Compute weights, apply lag, generate risk tensor
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pathograph.trade.faostat_step2 import (
    load_step1_manifest,
    validate_step1_artifacts,
    load_node_index,
    detect_faostat_file,
    validate_faostat_schema,
    ingest_faostat_chunked,
    generate_groups_template,
    load_group_mapping,
    apply_group_mapping,
    compute_corridor_year_weights,
    apply_backoff_policy,
    build_month_to_year_mapping,
    apply_lag_and_generate_pseudoflows,
    write_weights_zarr,
    write_risk_tensor_zarr,
    compute_qc_metrics,
    generate_trace_samples,
    compute_file_sha256
)


def setup_logging(log_path: Path):
    """Configure logging to file and console."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )


def mode_emit_groups_template(args):
    """
    Template Generator Mode: Generate groups mapping template from FAOSTAT data.
    
    This mode:
    1. Ingests FAOSTAT file (chunked)
    2. Computes distinct item codes with totals
    3. Writes faostat_items_present.csv
    4. Writes faostat_groups.template.csv
    5. Exits successfully
    """
    import os
    import traceback
    import pandas as pd
    from pathograph.trade.faostat_step2 import scan_faostat_items_present_streaming

    os.makedirs(args.emit_groups_template, exist_ok=True)
    error_log_path = os.path.join('reports', 'faostat_step2_emit_groups_template_error.log')
    os.makedirs('reports', exist_ok=True)

    try:

        # Load Step1 manifest to get node_index_path
        if args.step1_manifest:
            with open(args.step1_manifest, 'r', encoding='utf-8') as f:
                step1 = json.load(f)
            if 'inputs' in step1 and 'node_index' in step1['inputs']:
                node_index_path = step1['inputs']['node_index'].get('path')
            else:
                node_index_path = step1.get('node_index_path')
        elif args.node_index:
            node_index_path = args.node_index
        else:
            raise ValueError("--step1-manifest OR --node-index is required")
            
        if not node_index_path:
            raise ValueError('Could not determine node_index_path from inputs')

        node_df = pd.read_csv(node_index_path)
        iso3_set = set(node_df['iso3'].astype(str).str.upper().str.strip().tolist())

        # Crosswalk path (fixed contract)
        m49_csv = os.path.join('config', 'm49_to_iso3.csv')
        if not os.path.exists(m49_csv):
            raise FileNotFoundError(f"Missing crosswalk: {m49_csv}")

        items_df = scan_faostat_items_present_streaming(
            faostat_path=args.faostat_path,
            iso3_set=iso3_set,
            m49_to_iso3_csv=m49_csv,
            chunksize=args.chunksize,
            zip_member=getattr(args, 'zip_member', None),
        )

        # Write items_present
        items_present_path = os.path.join(args.emit_groups_template, 'faostat_items_present.csv')
        items_df[['item_code', 'item_name', 'total_value', 'row_count']].to_csv(items_present_path, index=False)

        # Write groups template
        tmpl = items_df[['item_code', 'item_name']].copy()
        tmpl['group_id'] = ''
        tmpl['group_name'] = ''
        groups_tmpl_path = os.path.join(args.emit_groups_template, 'faostat_groups.template.csv')
        tmpl.to_csv(groups_tmpl_path, index=False)

        print(f"Wrote: {items_present_path}")
        print(f"Wrote: {groups_tmpl_path}")
        return 0

    except Exception:
        tb = traceback.format_exc()
        print(tb)
        with open(error_log_path, 'w', encoding='utf-8') as f:
            f.write(tb)
        raise


def mode_full_pipeline(args):
    """
    Full Pipeline Mode: Compute weights, apply lag, generate risk tensor.
    """
    out_dir = Path(args.out_dir)
    log_path = out_dir / 'logs' / 'step2.log'
    
    # Setup logging
    setup_logging(log_path)
    
    logging.info("=" * 80)
    logging.info("FAOSTAT Step 2: Full Pipeline Mode")
    logging.info("=" * 80)
    logging.info(f"Step 1 manifest: {args.step1_manifest}")
    logging.info(f"FAOSTAT file: {args.faostat_path}")
    logging.info(f"Groups CSV: {args.groups_csv}")
    logging.info(f"Output dir: {out_dir}")
    logging.info(f"Use backoff: {args.use_backoff}")
    logging.info(f"Lag: {args.lag} year(s)")
    
    start_time = datetime.now(timezone.utc)
    
    try:
        # ====================================================================
        # Step 1: Preflight - Load and validate Step 1 artifacts
        # ====================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Step 1: Preflight - Validating Step 1 artifacts")
        logging.info("=" * 80)
        
        manifest_step1 = load_step1_manifest(args.step1_manifest)
        validate_step1_artifacts(manifest_step1, allow_noncanonical=args.allow_noncanonical_n)
        
        # Load node index
        node_index_path = manifest_step1['inputs']['node_index']['path']
        iso3_to_id, id_to_iso3, iso3_set, N = load_node_index(node_index_path, allow_noncanonical=args.allow_noncanonical_n)
        
        # Load Step 1 base tensor
        zarr_path_step1 = manifest_step1['outputs']['zarr_path']
        
        import zarr
        try:
            store = zarr.DirectoryStore(zarr_path_step1)
        except AttributeError:
            from zarr.storage import LocalStore
            store = LocalStore(zarr_path_step1)
        
        root_step1 = zarr.open_group(store=store, mode='r')
        base_tensor = np.array(root_step1['trade'])
        base_mask = np.array(root_step1['mask'])
        base_is_estimated = np.array(root_step1['is_estimated'])
        time_index = np.array(root_step1['time_index'])
        
        t_min = manifest_step1['time_axis']['t_min']
        t_max = manifest_step1['time_axis']['t_max']
        T = len(time_index)
        
        logging.info(f"Loaded Step 1 base tensor: shape={base_tensor.shape}")
        
        # ====================================================================
        # Step 2: Ingest FAOSTAT
        # ====================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Step 2: Ingesting FAOSTAT data")
        logging.info("=" * 80)
        
        file_path, zip_member = detect_faostat_file(args.faostat_path)
        
        # Read header
        if zip_member:
            import zipfile
            import io
            with zipfile.ZipFile(file_path, 'r') as zf:
                with zf.open(zip_member) as csv_file:
                    df_header = pd.read_csv(io.TextIOWrapper(csv_file, encoding='utf-8'), nrows=0)
        else:
            df_header = pd.read_csv(file_path, nrows=0)
        
        schema_mapping = validate_faostat_schema(df_header)
        
        # Ingest chunked
        df_faostat = ingest_faostat_chunked(file_path, zip_member, schema_mapping, iso3_set, args.chunksize)
        
        # ====================================================================
        # Step 3: Load groups mapping and apply
        # ====================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Step 3: Loading groups mapping")
        logging.info("=" * 80)
        
        item_to_group, group_names, K, group_order = load_group_mapping(args.groups_csv)
        df_faostat, mapping_stats = apply_group_mapping(df_faostat, item_to_group, group_order)
        
        # ====================================================================
        # Step 4: Compute corridor-year weights
        # ====================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Step 4: Computing corridor-year weights")
        logging.info("=" * 80)
        
        W, weight_mask, weight_stats = compute_corridor_year_weights(
            df_faostat, iso3_to_id, group_order, N, K
        )
        
        weight_year_min = weight_stats['year_min']
        weight_year_max = weight_stats['year_max']
        
        # ====================================================================
        # Step 5: Apply backoff policy (if enabled)
        # ====================================================================
        backoff_code = None
        backoff_stats = None
        
        if args.use_backoff:
            logging.info("\n" + "=" * 80)
            logging.info("Step 5: Applying backoff policy")
            logging.info("=" * 80)
            
            # Reconstruct V from df_faostat for backoff
            Y = weight_stats['Y']
            V = np.zeros((Y, N, N, K), dtype=np.float64)
            
            group_to_k = {gid: k for k, gid in enumerate(group_order)}
            
            for _, row in df_faostat.iterrows():
                year = int(row['year'])
                y = year - weight_year_min
                i = iso3_to_id[row['reporter_iso3']]
                j = iso3_to_id[row['partner_iso3']]
                k = group_to_k[row['group_id']]
                V[y, i, j, k] += float(row['value'])
            
            W, backoff_code, backoff_stats = apply_backoff_policy(W, weight_mask, V, N, K)
        else:
            logging.info("\n" + "=" * 80)
            logging.info("Step 5: Backoff policy disabled (default: missing-only)")
            logging.info("=" * 80)
        
        # ====================================================================
        # Step 6: Apply lag and generate pseudo-flows
        # ====================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Step 6: Applying lag and generating pseudo-flows")
        logging.info("=" * 80)
        
        E, observed_risk, is_estimated_risk, backoff_risk, pseudoflow_stats = apply_lag_and_generate_pseudoflows(
            base_tensor, base_mask, base_is_estimated,
            W, weight_mask, backoff_code,
            time_index, t_min, weight_year_min, args.lag, N, K
        )
        
        # ====================================================================
        # Step 7: Write Zarr outputs
        # ====================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Step 7: Writing Zarr outputs")
        logging.info("=" * 80)
        
        weights_zarr_path = out_dir / 'weights_corridor_year.zarr'
        risk_tensor_zarr_path = out_dir / 'trade_risk_tensor.zarr'
        
        write_weights_zarr(W, weight_mask, backoff_code, weight_year_min, group_order, str(weights_zarr_path))
        write_risk_tensor_zarr(E, observed_risk, is_estimated_risk, backoff_risk, time_index, group_order, str(risk_tensor_zarr_path))
        
        # ====================================================================
        # Step 8: Compute QC metrics
        # ====================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Step 8: Computing QC metrics")
        logging.info("=" * 80)
        
        qc_metrics = compute_qc_metrics(
            E, observed_risk, base_tensor, base_mask,
            W, weight_mask, backoff_code, group_order, id_to_iso3
        )
        
        qc_report_path = out_dir / 'qc_report.json'
        with open(qc_report_path, 'w') as f:
            json.dump(qc_metrics, f, indent=2)
        
        logging.info(f"Wrote QC report: {qc_report_path}")
        
        # ====================================================================
        # Step 9: Generate trace samples
        # ====================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Step 9: Generating trace samples")
        logging.info("=" * 80)
        
        month_years = build_month_to_year_mapping(time_index, t_min)
        
        trace_samples = generate_trace_samples(
            E, observed_risk, base_tensor, base_mask,
            W, weight_mask, backoff_code,
            time_index, month_years, weight_year_min, args.lag,
            group_order, id_to_iso3, args.max_trace
        )
        
        trace_samples_path = out_dir / 'trace_samples.jsonl'
        with open(trace_samples_path, 'w') as f:
            for sample in trace_samples:
                f.write(json.dumps(sample) + '\n')
        
        logging.info(f"Wrote trace samples: {trace_samples_path}")
        
        # ====================================================================
        # Step 10: Generate manifest
        # ====================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Step 10: Generating manifest")
        logging.info("=" * 80)
        
        # Compute hashes
        step1_manifest_hash = compute_file_sha256(args.step1_manifest)
        faostat_hash = compute_file_sha256(args.faostat_path)
        groups_csv_hash = compute_file_sha256(args.groups_csv)
        
        end_time = datetime.now(timezone.utc)
        duration_sec = (end_time - start_time).total_seconds()
        
        manifest = {
            'version': '1.0',
            'step': 'faostat_step2',
            'inputs': {
                'step1_manifest': {
                    'path': str(args.step1_manifest),
                    'sha256': step1_manifest_hash
                },
                'faostat_file': {
                    'path': str(args.faostat_path),
                    'sha256': faostat_hash,
                    'zip_member': zip_member
                },
                'groups_csv': {
                    'path': str(args.groups_csv),
                    'sha256': groups_csv_hash
                }
            },
            'parameters': {
                'lag': args.lag,
                'use_backoff': args.use_backoff,
                'max_trace': args.max_trace,
                'chunksize': args.chunksize
            },
            'weights': {
                'year_min': int(weight_year_min),
                'year_max': int(weight_year_max),
                'Y': int(weight_stats['Y']),
                'K': int(K),
                'groups': group_order,
                'coverage': weight_stats['coverage']
            },
            'outputs': {
                'weights_zarr': str(weights_zarr_path),
                'risk_tensor_zarr': str(risk_tensor_zarr_path),
                'qc_report': str(qc_report_path),
                'trace_samples': str(trace_samples_path),
                'risk_tensor_shape': list(E.shape),
                'risk_tensor_dtype': str(E.dtype)
            },
            'mapping_stats': mapping_stats,
            'weight_stats': weight_stats,
            'backoff_stats': backoff_stats,
            'pseudoflow_stats': pseudoflow_stats,
            'qc_summary': {
                'time_coverage': qc_metrics['time_coverage'],
                'base_to_risk_coverage': qc_metrics['base_to_risk_coverage']
            }
        }
        
        manifest_path = out_dir / 'preprocessing_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logging.info(f"Wrote manifest: {manifest_path}")
        
        # ====================================================================
        # Step 11: Generate completion report
        # ====================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Step 11: Generating completion report")
        logging.info("=" * 80)
        
        completion_report = {
            'step': 'faostat_step2',
            'status': 'complete',
            'created_at': start_time.isoformat(),
            'completed_at': end_time.isoformat(),
            'duration_seconds': duration_sec,
            'artifacts': {
                'weights_zarr': str(weights_zarr_path.relative_to(out_dir)),
                'risk_tensor_zarr': str(risk_tensor_zarr_path.relative_to(out_dir)),
                'qc_report': str(qc_report_path.relative_to(out_dir)),
                'trace_samples': str(trace_samples_path.relative_to(out_dir)),
                'manifest': str(manifest_path.relative_to(out_dir)),
                'log': str(log_path.relative_to(out_dir))
            },
            'summary': {
                'K_groups': K,
                'weight_year_range': f"{weight_year_min}..{weight_year_max}",
                'time_coverage': f"{qc_metrics['time_coverage']:.2%}",
                'base_to_risk_coverage': f"{qc_metrics['base_to_risk_coverage']:.2%}",
                'lag_years': args.lag,
                'backoff_enabled': args.use_backoff
            }
        }
        
        completion_report_path = out_dir / 'completion_report.json'
        with open(completion_report_path, 'w') as f:
            json.dump(completion_report, f, indent=2)
        
        logging.info(f"Wrote completion report: {completion_report_path}")
        
        # ====================================================================
        # Done
        # ====================================================================
        logging.info("\n" + "=" * 80)
        logging.info("FAOSTAT Step 2 Pipeline Completed Successfully")
        logging.info("=" * 80)
        logging.info(f"Duration: {duration_sec:.1f} seconds")
        logging.info(f"Output directory: {out_dir}")
        logging.info(f"Completion report: {completion_report_path}")
        
        return 0
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        return 3


def main():
    parser = argparse.ArgumentParser(
        description='FAOSTAT Step 2: Annual Bilateral Trade to Risk-Weighted Monthly Pseudo-Flows',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate groups template
  python -m tools.trade_step2_faostat \\
      --faostat-path data/raw/faostat/Trade.zip \\
      --emit-groups-template config/

  # Run full pipeline
  python -m tools.trade_step2_faostat \\
      --step1-manifest data/processed/trade/imts_step1/manifest.json \\
      --faostat-path data/raw/faostat/Trade.zip \\
      --groups-csv config/faostat_groups.csv \\
      --out-dir data/processed/trade/faostat_step2
        """
    )
    
    # Mode selection
    parser.add_argument('--emit-groups-template', metavar='DIR',
                       help='Generate groups template and exit (template generator mode)')
    
    # Required for full pipeline
    parser.add_argument('--step1-manifest',
                       help='Path to Step 1 preprocessing_manifest.json (required for full pipeline)')
    parser.add_argument('--faostat-path', required=True,
                       help='Path to FAOSTAT CSV or ZIP file')
    parser.add_argument('--groups-csv',
                       help='Path to faostat_groups.csv (required for full pipeline)')
    parser.add_argument('--out-dir',
                       help='Output directory (required for full pipeline)')
    
    # Optional parameters
    parser.add_argument('--node-index',
                       help='Path to node_index.csv (for template mode or to override Step 1 manifest)')
    parser.add_argument('--allow-noncanonical-n', action='store_true',
                       help='Allow non-canonical node universe (N != 194) for TEST/DEV ONLY (default: strict N=194)')
    parser.add_argument('--use-backoff', action='store_true',
                       help='Enable backoff policy for missing weights (default: OFF)')
    parser.add_argument('--lag', type=int, default=1,
                       help='Lag in years (default: 1)')
    parser.add_argument('--max-trace', type=int, default=50,
                       help='Maximum trace samples (default: 50)')
    parser.add_argument('--chunksize', type=int, default=100000,
                       help='FAOSTAT CSV chunk size (default: 100000)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Determine mode
    if args.emit_groups_template:
        # Template generator mode
        return mode_emit_groups_template(args)
    else:
        # Full pipeline mode - validate required arguments
        if not args.step1_manifest:
            parser.error("--step1-manifest is required for full pipeline mode")
        if not args.groups_csv:
            parser.error("--groups-csv is required for full pipeline mode")
        if not args.out_dir:
            parser.error("--out-dir is required for full pipeline mode")
        
        return mode_full_pipeline(args)


if __name__ == '__main__':
    sys.exit(main())
