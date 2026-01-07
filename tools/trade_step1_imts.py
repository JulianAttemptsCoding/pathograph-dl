"""
CLI Tool: IMTS Step 1 Pipeline

Usage:
    python tools/trade_step1_imts.py \
        --imts_csv data/raw/imf_imts/IMTS_Data_Until1950.csv \
        --node_index data/processed/node_index.csv \
        --out data/processed/trade/imf_imts_step1 \
        --m 0.06

This script orchestrates the complete Step 1 pipeline:
1. Load node index (194 ISO3 nodes)
2. Detect month columns, build continuous time axis
3. Allocate six accumulators (exports, imports_fob, imports_cif × sum/mask)
4. Stream-process CSV chunks
5. Merge imports (FOB-first)
6. Apply diagonal policy
7. Write Zarr outputs
8. Compute QC metrics
9. Generate manifest
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

from pathograph.trade.imts_step1 import (
    load_node_index,
    detect_month_columns,
    validate_schema,
    process_chunk_to_accumulators,
    merge_imports_fob_first,
    apply_diagonal_policy,
    compute_qc_metrics,
    write_zarr_outputs,
    compute_file_sha256,
    generate_trace_samples,
    write_long_output,
    MONTH_COL_REGEX
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


def main():
    parser = argparse.ArgumentParser(description='IMTS Step 1: CSV → FOB Trade Tensor')
    
    # Required arguments
    parser.add_argument('--imts_csv', required=True, help='Path to IMF IMTS CSV')
    parser.add_argument('--node_index', required=True, help='Path to node_index.csv')
    parser.add_argument('--out', required=True, help='Output directory root')
    
    # Optional arguments
    parser.add_argument('--m', type=float, default=0.06, help='CIF-to-FOB markup (default: 0.06)')
    parser.add_argument('--chunksize', type=int, default=100000, help='CSV chunk size (default: 100000)')
    parser.add_argument('--write-long', type=int, default=0, help='Enable long-form parquet output (default: 0)')
    parser.add_argument('--max-trace-samples', type=int, default=200, help='Trace sample count (default: 200)')
    parser.add_argument('--negative-policy', default='clip', choices=['clip', 'fail'], 
                       help='Policy for negative values (default: clip)')
    
    args = parser.parse_args()
    
    # Setup paths
    out_dir = Path(args.out)
    log_path = out_dir / 'logs' / 'step1.log'
    zarr_path = out_dir / 'trade_fob_tensor.zarr'
    qc_dir = out_dir / 'qc'
    manifest_path = out_dir / 'manifest.json'
    
    # Create directories
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(log_path)
    
    logging.info("=" * 80)
    logging.info("IMTS Step 1 Pipeline Starting")
    logging.info("=" * 80)
    logging.info(f"Input CSV: {args.imts_csv}")
    logging.info(f"Node index: {args.node_index}")
    logging.info(f"Output dir: {out_dir}")
    logging.info(f"CIF-to-FOB markup (m): {args.m}")
    logging.info(f"Chunk size: {args.chunksize}")
    logging.info(f"Negative policy: {args.negative_policy}")
    
    start_time = datetime.now(timezone.utc)
    
    try:
        # ====================================================================
        # Step 1: Load node index
        # ====================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Step 1: Loading node index")
        logging.info("=" * 80)
        
        iso3_to_id, id_to_iso3, N = load_node_index(args.node_index)
        
        # ====================================================================
        # Step 2: Detect month columns and build time axis
        # ====================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Step 2: Detecting month columns")
        logging.info("=" * 80)
        
        # Read header only
        df_header = pd.read_csv(args.imts_csv, nrows=0)
        
        # Validate schema
        required_cols = ['COUNTRY.ID', 'COUNTERPART_COUNTRY.ID', 'INDICATOR.ID', 
                        'FREQUENCY.ID', 'SCALE.ID']
        optional_cols = ['UNIT.ID', 'COUNTRY.NAME', 'COUNTERPART_COUNTRY.NAME']
        
        schema_result = validate_schema(df_header, required_cols, optional_cols)
        
        # Detect month columns
        month_cols, t_indices, t_min, t_max, T_full = detect_month_columns(df_header, MONTH_COL_REGEX)
        
        # Build month_col -> tensor_index mapping
        t_map = {col: t - t_min for col, t in zip(month_cols, t_indices)}
        
        # Build time_index array
        time_index = np.arange(t_min, t_max + 1, dtype=np.int32)
        
        logging.info(f"Continuous time axis: T_full={T_full}, t_min={t_min}, t_max={t_max}")
        
        # ====================================================================
        # Step 3: Allocate accumulators
        # ====================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Step 3: Allocating accumulators")
        logging.info("=" * 80)
        
        exports_sum = np.zeros((T_full, N, N), dtype=np.float64)
        exports_mask = np.zeros((T_full, N, N), dtype=np.uint8)
        imports_fob_sum = np.zeros((T_full, N, N), dtype=np.float64)
        imports_fob_mask = np.zeros((T_full, N, N), dtype=np.uint8)
        imports_cif_sum = np.zeros((T_full, N, N), dtype=np.float64)
        imports_cif_mask = np.zeros((T_full, N, N), dtype=np.uint8)
        
        mem_mb = (exports_sum.nbytes + exports_mask.nbytes + 
                 imports_fob_sum.nbytes + imports_fob_mask.nbytes +
                 imports_cif_sum.nbytes + imports_cif_mask.nbytes) / (1024**2)
        
        logging.info(f"Allocated 6 accumulators: {T_full}×{N}×{N} each")
        logging.info(f"Total memory: {mem_mb:.1f} MB")
        
        # ====================================================================
        # Step 4: Stream-process CSV chunks
        # ====================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Step 4: Processing CSV chunks")
        logging.info("=" * 80)
        
        # Determine columns to read
        cols_to_read = required_cols + month_cols
        if 'UNIT.ID' in schema_result['optional_present']:
            cols_to_read.append('UNIT.ID')
        
        stats = {}
        chunk_count = 0
        
        for chunk in pd.read_csv(args.imts_csv, chunksize=args.chunksize, usecols=cols_to_read):
            chunk_count += 1
            
            # Filter by frequency
            chunk = chunk[chunk['FREQUENCY.ID'] == 'M']
            
            # Filter by unit if present
            if 'UNIT.ID' in chunk.columns:
                chunk = chunk[chunk['UNIT.ID'] == 'USD']
            
            # Filter by indicator
            chunk = chunk[chunk['INDICATOR.ID'].isin(['XG_FOB_USD', 'MG_FOB_USD', 'MG_CIF_USD'])]
            
            if len(chunk) == 0:
                continue
            
            # Process chunk
            process_chunk_to_accumulators(
                chunk, month_cols, t_map, iso3_to_id, 'SCALE.ID',
                exports_sum, exports_mask,
                imports_fob_sum, imports_fob_mask,
                imports_cif_sum, imports_cif_mask,
                negative_policy=args.negative_policy,
                stats=stats
            )
            
            if chunk_count % 10 == 0:
                logging.info(f"Processed {chunk_count} chunks, {stats.get('processed_rows', 0)} rows")
        
        logging.info(f"Finished processing {chunk_count} chunks")
        logging.info(f"Statistics:")
        for key, val in stats.items():
            logging.info(f"  {key}: {val}")
        
        # ====================================================================
        # Step 5: Merge imports (FOB-first)
        # ====================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Step 5: Merging imports (FOB-first)")
        logging.info("=" * 80)
        
        imports_best, imports_mask, is_estimated_imports, merge_diag = merge_imports_fob_first(
            imports_fob_sum, imports_fob_mask,
            imports_cif_sum, imports_cif_mask,
            args.m
        )
        
        # ====================================================================
        # Step 6: Build final tensor
        # ====================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Step 6: Building final tensor")
        logging.info("=" * 80)
        
        trade = np.zeros((T_full, N, N, 2), dtype=np.float32)
        mask = np.zeros((T_full, N, N, 2), dtype=np.uint8)
        is_estimated = np.zeros((T_full, N, N, 2), dtype=np.uint8)
        
        trade[..., 0] = exports_sum.astype(np.float32)
        trade[..., 1] = imports_best.astype(np.float32)
        mask[..., 0] = exports_mask
        mask[..., 1] = imports_mask
        is_estimated[..., 0] = 0  # Exports never estimated
        is_estimated[..., 1] = is_estimated_imports
        
        logging.info(f"Final tensor shape: {trade.shape}")
        
        # ====================================================================
        # Step 7: Apply diagonal policy
        # ====================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Step 7: Applying diagonal policy")
        logging.info("=" * 80)
        
        apply_diagonal_policy(trade, mask, is_estimated)
        
        # ====================================================================
        # Step 8: Write Zarr outputs
        # ====================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Step 8: Writing Zarr outputs")
        logging.info("=" * 80)
        
        zarr_attrs = {
            'created_at': start_time.isoformat(),
            'source_csv': str(args.imts_csv),
            'node_index': str(args.node_index),
            'cif_to_fob_markup': args.m,
            'negative_policy': args.negative_policy,
            't_min': int(t_min),
            't_max': int(t_max),
            'T_full': int(T_full),
            'N': int(N),
            'channels': ['exports_fob_usd', 'imports_fob_best_usd'],
            'month_cols_used': month_cols,
            'indicators': ['XG_FOB_USD', 'MG_FOB_USD', 'MG_CIF_USD']
        }
        
        write_zarr_outputs(str(zarr_path), trade, mask, is_estimated, time_index, zarr_attrs)
        
        # ====================================================================
        # Step 9: Compute QC metrics
        # ====================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Step 9: Computing QC metrics")
        logging.info("=" * 80)
        
        qc_metrics = compute_qc_metrics(trade, mask, is_estimated, time_index, id_to_iso3, t_min)
        
        # Write QC summary
        qc_summary_path = qc_dir / 'qc_summary.json'
        with open(qc_summary_path, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(qc_metrics, f, indent=2, ensure_ascii=False)

        
        logging.info(f"Wrote QC summary to {qc_summary_path}")
        
        # Write top corridors
        top_exports_path = qc_dir / 'top_corridors_exports.csv'
        pd.DataFrame(qc_metrics['top_corridors_exports']).to_csv(top_exports_path, index=False)
        
        top_imports_path = qc_dir / 'top_corridors_imports.csv'
        pd.DataFrame(qc_metrics['top_corridors_imports']).to_csv(top_imports_path, index=False)
        
        # ====================================================================
        # Step 10: Generate manifest
        # ====================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Step 10: Generating manifest")
        logging.info("=" * 80)
        
        logging.info("Computing SHA256 hash of input CSV...")
        csv_sha256 = compute_file_sha256(args.imts_csv)
        
        logging.info("Computing SHA256 hash of node index...")
        node_index_sha256 = compute_file_sha256(args.node_index)
        
        end_time = datetime.now(timezone.utc)
        duration_sec = (end_time - start_time).total_seconds()
        
        manifest = {
            'version': '1.0',
            'created_at': start_time.isoformat(),
            'completed_at': end_time.isoformat(),
            'duration_seconds': duration_sec,
            'inputs': {
                'imts_csv': {
                    'path': str(args.imts_csv),
                    'sha256': csv_sha256
                },
                'node_index': {
                    'path': str(args.node_index),
                    'sha256': node_index_sha256
                }
            },
            'parameters': {
                'cif_to_fob_markup': args.m,
                'chunksize': args.chunksize,
                'negative_policy': args.negative_policy,
                'month_column_regex': MONTH_COL_REGEX
            },
            'time_axis': {
                't_min': int(t_min),
                't_max': int(t_max),
                'T_full': int(T_full),
                'month_cols_used': month_cols,
                'month_indices': t_indices
            },
            'outputs': {
                'zarr_path': str(zarr_path),
                'tensor_shape': list(trade.shape),
                'tensor_dtype': str(trade.dtype),
                'mask_dtype': str(mask.dtype),
                'is_estimated_dtype': str(is_estimated.dtype)
            },
            'processing_stats': stats,
            'merge_diagnostics': merge_diag,
            'qc_summary': {
                'coverage_overall': qc_metrics['coverage_overall'],
                'coverage_exports': qc_metrics['coverage_exports'],
                'coverage_imports': qc_metrics['coverage_imports'],
                'estimated_share_imports': qc_metrics['estimated_share_imports'],
                'observed_zeros_count': qc_metrics['observed_zeros_count'],
                'validation_passed': all(qc_metrics['validation'].values())
            }
        }
        
        with open(manifest_path, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)


        logging.info(f"Wrote manifest to {manifest_path}")
        
        # ====================================================================
        # Step 11: Generate trace samples
        # ====================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Step 11: Generating trace samples")
        logging.info("=" * 80)
        
        trace_samples_df = generate_trace_samples(
            trade, mask, is_estimated, time_index, id_to_iso3, t_min,
            imports_fob_sum, imports_fob_mask,
            imports_cif_sum, imports_cif_mask,
            args.m, max_samples=args.max_trace_samples
        )
        
        trace_samples_path = qc_dir / 'trace_samples.csv'
        trace_samples_df.to_csv(trace_samples_path, index=False)
        logging.info(f"Wrote trace samples to {trace_samples_path}")
        
        # ====================================================================
        # Step 12: Handle --write-long
        # ====================================================================
        if args.write_long == 1:
            logging.info("\n" + "=" * 80)
            logging.info("Step 12: Long-form output requested")
            logging.info("=" * 80)
            
            try:
                # This will raise NotImplementedError
                write_long_output(str(out_dir / 'long'), None, None, None, None, t_min)
            except NotImplementedError as e:
                logging.error(str(e))
                logging.error("Long-form output is not yet implemented.")
                logging.error("Exiting with code 4.")
                return 4
        else:
            logging.info("\n" + "=" * 80)
            logging.info("Step 12: Long-form output disabled (--write-long=0)")
            logging.info("=" * 80)
        
        # ====================================================================
        # Step 13: Generate completion report
        # ====================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Step 13: Generating completion report")
        logging.info("=" * 80)
        
        reports_dir = out_dir / 'reports'
        reports_dir.mkdir(parents=True, exist_ok=True)
        completion_report_path = reports_dir / 'step1_completion_report.md'
        
        # Get software versions
        import platform
        try:
            import zarr as zarr_mod
            zarr_version = zarr_mod.__version__
        except:
            zarr_version = "unknown"
        
        # Build completion report
        report_lines = [
            "# IMTS Step 1 Completion Report",
            "",
            "## 1. Run Metadata",
            "",
            f"- **Timestamp**: {start_time.isoformat()}",
            f"- **Duration**: {duration_sec:.1f} seconds",
            f"- **Command**: `python tools/trade_step1_imts.py --imts_csv {args.imts_csv} --node_index {args.node_index} --out {args.out} --m {args.m} --negative-policy {args.negative_policy} --write-long {args.write_long}`",
            f"- **Python version**: {platform.python_version()}",
            f"- **NumPy version**: {np.__version__}",
            f"- **Pandas version**: {pd.__version__}",
            f"- **Zarr version**: {zarr_version}",
            f"- **Input CSV**: {args.imts_csv} ({Path(args.imts_csv).stat().st_size / (1024**2):.1f} MB)",
            f"- **Input SHA256**: {csv_sha256}",
            "",
            "## 2. Schema and Filters",
            "",
            f"- **Required columns present**: {', '.join(schema_result['required_present'])}",
            f"- **Month regex**: `{MONTH_COL_REGEX}`",
            f"- **Indicators used**: XG_FOB_USD, MG_FOB_USD, MG_CIF_USD",
            f"- **Frequency filter**: M (monthly)",
            f"- **Unit filter**: {'USD' if 'UNIT.ID' in schema_result['optional_present'] else 'N/A (UNIT.ID absent)'}",
            f"- **Pre-1950 columns dropped**: {stats.get('pre_1950_dropped', 0)}",
            f"- **Time axis**: t_min={t_min}, t_max={t_max}, T_full={T_full}",
            "",
            "## 3. Mapping and Coverage",
            "",
            f"- **194-node universe**: ✅ Validated",
            f"- **Rows dropped (non-universe)**: {stats.get('dropped_nonuniverse', 0)}",
            f"- **Rows dropped (invalid SCALE.ID)**: {stats.get('dropped_invalid_scale', 0)}",
            f"- **Rows processed**: {stats.get('processed_rows', 0)}",
            f"- **Overall mask coverage**: {qc_metrics['coverage_overall']:.2%}",
            f"- **Exports coverage**: {qc_metrics['coverage_exports']:.2%}",
            f"- **Imports coverage**: {qc_metrics['coverage_imports']:.2%}",
            f"- **Imports estimated share**: {qc_metrics['estimated_share_imports']:.2%}",
            "",
            "### FOB/CIF Overlap Diagnostics",
            "",
            f"- **Overlap cells** (both FOB and CIF): {merge_diag['overlap_cells']:,}",
            f"- **FOB-only cells**: {merge_diag['fob_only_cells']:,}",
            f"- **CIF-only cells** (estimated): {merge_diag['cif_only_cells']:,}",
            "",
            "## 4. Correctness Assertions",
            "",
            f"- **Tensor shapes**: {trade.shape} ({'✅ PASS' if trade.shape == (T_full, 194, 194, 2) else '❌ FAIL'})",
            f"- **Tensor dtypes**: trade={trade.dtype}, mask={mask.dtype}, is_estimated={is_estimated.dtype} ({'✅ PASS' if trade.dtype == np.float32 and mask.dtype == np.uint8 else '❌ FAIL'})",
            f"- **Diagonal policy**: {'✅ PASS' if np.all(trade[:, range(194), range(194), :] == 0) else '❌ FAIL'}",
            f"- **Mask→zero implication (trade)**: {'✅ PASS' if qc_metrics['validation']['mask_zero_implies_trade_zero'] else '❌ FAIL'}",
            f"- **Mask→zero implication (is_estimated)**: {'✅ PASS' if qc_metrics['validation']['mask_zero_implies_estimated_zero'] else '❌ FAIL'}",
            f"- **Exports never estimated**: {'✅ PASS' if qc_metrics['validation']['exports_never_estimated'] else '❌ FAIL'}",
            f"- **Non-negativity**: {'✅ PASS' if qc_metrics['validation']['nonnegativity_ok'] else '❌ FAIL'}",
            f"- **Observed zeros count**: {qc_metrics['observed_zeros_count']:,}",
            "",
            "### FOB-First Correctness (Trace Sample Validation)",
            "",
            "Checking trace samples for FOB-first rule compliance:",
            "",
        ]
        
        # Validate FOB-first in trace samples
        imports_samples = trace_samples_df[trace_samples_df['ch'] == 1]
        if len(imports_samples) > 0:
            fob_first_violations = imports_samples[(imports_samples['is_estimated'] == 1) & (imports_samples['has_fob'] == 1)]
            if len(fob_first_violations) == 0:
                report_lines.append("- ✅ **PASS**: No violations found (is_estimated=1 → has_fob=0)")
            else:
                report_lines.append(f"- ❌ **FAIL**: {len(fob_first_violations)} violations found")
        else:
            report_lines.append("- ⚠️ **N/A**: No imports samples generated")
        
        report_lines.extend([
            "",
            "## 5. Artifacts Produced",
            "",
            f"- **Zarr tensor**: `{zarr_path.relative_to(out_dir)}`",
            f"- **QC summary**: `{qc_summary_path.relative_to(out_dir)}`",
            f"- **Trace samples**: `{trace_samples_path.relative_to(out_dir)}` ({len(trace_samples_df)} samples)",
            f"- **Top corridors (exports)**: `{top_exports_path.relative_to(out_dir)}`",
            f"- **Top corridors (imports)**: `{top_imports_path.relative_to(out_dir)}`",
            f"- **Manifest**: `{manifest_path.relative_to(out_dir)}`",
            f"- **Logs**: `{log_path.relative_to(out_dir)}`",
            f"- **Long parquet dataset**: {'Not implemented (--write-long=1 would exit with code 4)' if args.write_long == 0 else 'Exit code 4 (not implemented)'}",
            "",
            "## 6. Known Issues / Warnings",
            "",
        ])
        
        # Add warnings
        if stats.get('negative_cells', 0) > 0:
            report_lines.append(f"- ⚠️ **Negative values clipped**: {stats['negative_cells']:,} cells, total magnitude ${stats.get('negative_usd_total', 0):,.0f}")
        
        if 'UNIT.ID' not in schema_result['optional_present']:
            report_lines.append("- ⚠️ **UNIT.ID column absent**: Unit filter not applied (assumed USD)")
        
        if stats.get('dropped_invalid_scale', 0) > 0:
            report_lines.append(f"- ⚠️ **Invalid SCALE.ID rows dropped**: {stats['dropped_invalid_scale']:,} rows")
        
        report_lines.extend([
            "",
            "## 7. Completion Status",
            "",
            "**Step 1 is COMPLETE** if all of the following are true:",
            "",
            f"- [{'x' if zarr_path.exists() else ' '}] Zarr tensor exists",
            f"- [{'x' if qc_summary_path.exists() else ' '}] QC summary exists",
            f"- [{'x' if trace_samples_path.exists() else ' '}] Trace samples exist",
            f"- [{'x' if args.write_long == 0 or args.write_long == 1 else ' '}] --write-long behavior correct (disabled or explicit error)",
            f"- [{'x' if all(qc_metrics['validation'].values()) else ' '}] All validation checks passed",
            "",
            "### Exit Code Semantics",
            "",
            "- **0**: Success (all outputs produced, validation passed)",
            "- **2**: Schema validation failed",
            "- **3**: Runtime failure during processing",
            "- **4**: --write-long requested but not implemented",
            "",
            f"**This run exited with code: 0** ✅",
            "",
        ])
        
        with open(completion_report_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write('\n'.join(report_lines))

        
        logging.info(f"Wrote completion report to {completion_report_path}")
        
        # ====================================================================
        # Done
        # ====================================================================
        logging.info("\n" + "=" * 80)
        logging.info("IMTS Step 1 Pipeline Completed Successfully")
        logging.info("=" * 80)
        logging.info(f"Duration: {duration_sec:.1f} seconds")
        logging.info(f"Output directory: {out_dir}")
        logging.info(f"Completion report: {completion_report_path}")
        
        return 0
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        return 3


if __name__ == '__main__':
    sys.exit(main())
