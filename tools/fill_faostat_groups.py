import argparse
import sys
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='in_path', required=True)
    ap.add_argument('--out', dest='out_path', required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.in_path)
    required = ['item_code', 'item_name', 'group_id', 'group_name']
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"Missing required columns: {missing_cols}")

    # Normalize types
    df['item_code'] = pd.to_numeric(df['item_code'], errors='coerce').astype('Int64')
    df['item_name'] = df['item_name'].astype(str)

    # Deterministic K-small group catalog
    GROUPS = {
        'BANANA': 'Banana & plantain (BBTD/TR4 proxy)',
        'CITRUS': 'Citrus (HLB proxy)',
        'OLIVE_GRAPE': 'Olive + grape sector (Xylella proxy)',
        'PRUNUS': 'Prunus/stone fruit sector (PPV + Xylella proxy)',
        'WHEAT': 'Wheat sector (wheat blast proxy)',
        'CASSAVA': 'Cassava sector (CMD/CBSD proxy)',
        'BRASSICA': 'Brassica sector (clubroot proxy)',
        'OTHER': 'Other commodities'
    }

    # Explicit item_code sets (fast, deterministic, zero ambiguity)
    BANANA = {486, 489}
    CITRUS = {490, 495, 497, 507, 512, 513, 514, 491, 492, 498, 499, 509, 510, 496}
    OLIVE_GRAPE = {260, 262, 261, 274, 273, 560, 561, 562, 563, 564, 565, 566}
    PRUNUS = {
        221, 231, 217, 230, 232, 222, 229, 233, 234,
        531, 530, 534, 536, 537, 526, 527, 541
    }
    WHEAT = {15, 16, 17, 19}
    CASSAVA = {125, 126, 127, 128, 129}
    BRASSICA = {358, 393, 270, 271, 272, 292, 294, 646, 649}

    def assign_group(code: int) -> str:
        if code in BANANA:
            return 'BANANA'
        if code in CITRUS:
            return 'CITRUS'
        if code in CASSAVA:
            return 'CASSAVA'
        if code in BRASSICA:
            return 'BRASSICA'
        if code in OLIVE_GRAPE:
            return 'OLIVE_GRAPE'
        if code in PRUNUS:
            return 'PRUNUS'
        if code in WHEAT:
            return 'WHEAT'
        return 'OTHER'

    # Apply mapping
    df['group_id'] = df['item_code'].apply(lambda x: assign_group(int(x)) if pd.notna(x) else 'OTHER')
    df['group_name'] = df['group_id'].map(GROUPS)

    # Hard fail if anything is unmapped (should never happen)
    if df['group_id'].isna().any() or df['group_name'].isna().any():
        bad = df[df['group_id'].isna() | df['group_name'].isna()][['item_code', 'item_name', 'group_id', 'group_name']]
        raise SystemExit(f"Unmapped rows detected:\n{bad.head(50).to_string(index=False)}")

    # QC summary
    counts = df['group_id'].value_counts(dropna=False).sort_index()
    qc = pd.DataFrame({'group_id': counts.index, 'n_items': counts.values})
    qc_path = 'config/faostat_groups_qc_summary.csv'
    qc.to_csv(qc_path, index=False)

    df.to_csv(args.out_path, index=False)
    print(f"Wrote updated groups: {args.out_path}")
    print(f"Wrote QC summary: {qc_path}")
    print(qc.to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
