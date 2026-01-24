import zarr
import glob
import os
import numpy as np

def check_zarr():
    root = "data/processed"
    lines = []
    lines.append(f"Scanning {root}...")
    zarr_groups = sorted(glob.glob(os.path.join(root, "**", "*.zarr"), recursive=True))
    lines.append(f"Found {len(zarr_groups)} zarr groups")

    for zpath in zarr_groups:
        lines.append(f"\nChecking: {zpath}")
        try:
            # Mode 'r' is default
            grp = zarr.open_group(zpath, mode='r')
            
            # Manual walk for Zarr v3 compat
            results = []
            def walk(g, pfx=""):
                # Handle arrays in current group
                try:
                    for k, v in g.arrays():
                        results.append((pfx + k, v.shape, str(v.dtype)))
                except AttributeError:
                    pass # Maybe checking wrong object

                # Recurse into subgroups
                try:
                    for k, v in g.groups():
                        walk(v, pfx + k + "/")
                except AttributeError:
                    pass

            walk(grp)
            
            if not results:
                lines.append("  (No arrays found or traversal failed)")
            
            for name, shape, dtype in sorted(results):
                lines.append(f"  Array: {name}, Shape: {shape}, Dtype: {dtype}")

        except Exception as e:
            lines.append(f"  FAILED to open: {e}")

    # Check .npy for meta
    lines.append("\nScanning .npy files in data/processed/meta...")
    meta_path = os.path.join(root, "meta")
    if os.path.exists(meta_path):
        for npy in glob.glob(os.path.join(meta_path, "*.npy")):
            try:
                arr = np.load(npy)
                lines.append(f"  File: {os.path.basename(npy)}, Shape: {arr.shape}, Dtype: {arr.dtype}")
            except Exception as e:
                lines.append(f"  FAILED to load {npy}: {e}")
    else:
        lines.append(f"  meta dir not found at {meta_path}")
    
    with open("prereq_shapes_report.txt", "w") as f:
        f.write("\n".join(lines))
    print("DONE: written to prereq_shapes_report.txt")

if __name__ == "__main__":
    check_zarr()
