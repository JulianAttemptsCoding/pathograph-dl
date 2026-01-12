import pathlib
import re

def audit_code():
    files = {
        "dataset": "pathograph/data/trade_dataset.py",
        "datamodule": "pathograph/data/trade_datamodule.py",
        "collate": "pathograph/data/trade_collate.py"
    }
    
    patterns = {
        "dataset": r'base_mask|y_base_m|return_targets|target_kind|split_train|split_val|split_test|lookback|horizon',
        "datamodule": r'TradeDataModule|split|setup|DataLoader|return_targets|target_kind|lookback|horizon',
        "collate": r'mask|y_base|y_risk|multiply|collate'
    }
    
    out_dir = pathlib.Path("docs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for key, path_str in files.items():
        p = pathlib.Path(path_str)
        if not p.exists():
            print(f"Warning: {path_str} not found.")
            continue
            
        lines = p.read_text(encoding='utf-8').splitlines()
        hits = []
        for i, l in enumerate(lines):
            if re.search(patterns[key], l):
                hits.append(f"{i+1}: {l}")
        
        out_file = out_dir / f"trade_spec_v1_2_{key}_code_hits.txt"
        out_file.write_text("\n".join(hits), encoding='utf-8')
        print(f"Code audit for {key} complete.")

if __name__ == "__main__":
    audit_code()
