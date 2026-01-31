# Pathogen Step 1 QC Report
Generated: 2026-01-28T00:21:48

## Inputs
- node_index: ['data\\processed\\trade\\imf_imts_step1\\node_index.csv', 'data\\processed\\meta\\node_index.csv']
- time_index_master: data\processed\meta\time_index_master.npy (T=908)
- pathogen inputs: data\raw\pathogen_curated / *_curated_long.csv (files=8)

## Outputs
- zarr: data\processed\pathogen\status_tensor.zarr
- trace: docs\_logs\pathogen_step1_trace_samples.csv

## Tensor Shapes
- status: (908, 194, 8) dtype=uint8
- status_mask: (908, 194, 8) dtype=uint8
- evidence: (908, 194, 8) dtype=uint8
- evidence_mask: (908, 194, 8) dtype=uint8

## Events per Pathogen (sum of evidence)
- BBTD: 19 (first_month_index=96)
- Cassava: 10 (first_month_index=336)
- CitrusGreening: 5 (first_month_index=12)
- Clubroot: 3 (first_month_index=636)
- PPV: 9 (first_month_index=72)
- TR4: 19 (first_month_index=204)
- WheatBlast: 10 (first_month_index=420)
- XylellaFastidiosa: 12 (first_month_index=444)

## Per-file stats
- {'file': 'data\\raw\\pathogen_curated\\BBTD_curated_long.csv', 'rows_in_file': 19, 'rows_kept': 19, 'date_min': '1958-01-01', 'date_max': '2020-01-01', 'unique_iso3': 19, 'unique_pathogen': 1}
- {'file': 'data\\raw\\pathogen_curated\\Cassava_curated_long.csv', 'rows_in_file': 10, 'rows_kept': 10, 'date_min': '1978-01-01', 'date_max': '2022-01-01', 'unique_iso3': 9, 'unique_pathogen': 1}
- {'file': 'data\\raw\\pathogen_curated\\CitrusGreening_curated_long.csv', 'rows_in_file': 5, 'rows_kept': 5, 'date_min': '1951-01-01', 'date_max': '2009-01-01', 'unique_iso3': 5, 'unique_pathogen': 1}
- {'file': 'data\\raw\\pathogen_curated\\Clubroot_curated_long.csv', 'rows_in_file': 3, 'rows_kept': 3, 'date_min': '2003-01-01', 'date_max': '2017-01-01', 'unique_iso3': 3, 'unique_pathogen': 1}
- {'file': 'data\\raw\\pathogen_curated\\PPV_curated_long.csv', 'rows_in_file': 9, 'rows_kept': 9, 'date_min': '1956-01-01', 'date_max': '2017-01-01', 'unique_iso3': 9, 'unique_pathogen': 1}
- {'file': 'data\\raw\\pathogen_curated\\TR4_curated_long.csv', 'rows_in_file': 19, 'rows_kept': 19, 'date_min': '1967-01-01', 'date_max': '2023-01-01', 'unique_iso3': 19, 'unique_pathogen': 1}
- {'file': 'data\\raw\\pathogen_curated\\WheatBlast_curated_long.csv', 'rows_in_file': 10, 'rows_kept': 10, 'date_min': '1985-01-01', 'date_max': '2018-01-01', 'unique_iso3': 10, 'unique_pathogen': 1}
- {'file': 'data\\raw\\pathogen_curated\\XylellaFastidiosa_curated_long.csv', 'rows_in_file': 12, 'rows_kept': 12, 'date_min': '1987-01-01', 'date_max': '2018-01-01', 'unique_iso3': 12, 'unique_pathogen': 1}
