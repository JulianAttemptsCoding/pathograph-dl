import zarr
from pathlib import Path

def inspect():
    p = 'data/processed/trade/imf_imts_step1/trade_fob_tensor.zarr'
    root = zarr.open(p, mode='r')
    print(f"Type: {type(root)}")
    print(f"Attributes: {[m for m in dir(root) if not m.startswith('_')]}")
    
    if hasattr(root, 'members'):
        print(f"Members: {list(root.members())}")
    
    # Try iterating if it works as a sequence
    try:
        print(f"Keys (iter): {list(root.keys())}")
    except:
        print("Keys (iter) failed")

if __name__ == "__main__":
    inspect()
