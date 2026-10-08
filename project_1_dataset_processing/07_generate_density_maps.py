#!/usr/bin/env python3
import argparse
import json
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter

def read_jsonl(p: Path) -> list[dict]:
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: rows.append(json.loads(line))
    return rows

def write_jsonl(rows: list[dict], p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def generate_density_adaptive(shape, centers, boxes, beta=0.2):
    """
    Most optimized density generation:
    Sigma is 20% of the average box dimension (beta=0.2).
    """
    density = np.zeros(shape, dtype=np.float32)
    for i, (cx, cy) in enumerate(centers):
        # Calculate individual sigma based on box size
        x1, y1, x2, y2 = boxes[i]
        w, h = (x2 - x1), (y2 - y1)
        sigma = ((w + h) / 2) * beta
        
        # Create a tiny localized kernel to save computation
        temp_bin = np.zeros(shape, dtype=np.float32)
        temp_bin[int(clip(cy, 0, shape[0]-1)), int(clip(cx, 0, shape[1]-1))] = 1.0
        density += gaussian_filter(temp_bin, sigma=sigma)
        
    return density

def clip(val, lo, hi):
    return max(lo, min(hi, val))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default=".")
    ap.add_argument("--S", type=int, default=512)
    ap.add_argument("--beta", type=float, default=0.2, help="Sigma scale factor")
    args = ap.parse_args()

    root = Path(args.project_root)
    base_processing = root / "project_1_dataset_processing"
    out_density_root = base_processing / "density_maps"
    out_density_root.mkdir(parents=True, exist_ok=True)

    print(f"--- [07] Generating Adaptive Density Maps ({args.S}x{args.S}) ---")

    splits = ["train_files", "validation_files", "test_files"]
    for split_dir in splits:
        folder = base_processing / "modified_splits" / split_dir
        for jsonl_file in folder.glob("*.jsonl"):
            recs = read_jsonl(jsonl_file)
            updated = []
            
            for r in recs:
                centers = r.get("centers_512", [])
                boxes = r.get("boxes_512", [])
                
                # Generate the map
                density_map = generate_density_adaptive((args.S, args.S), centers, boxes, args.beta)
                
                # Save as .npy
                stem = Path(r["filename"]).stem
                den_path = out_density_root / f"{stem}_density.npy"
                np.save(den_path, density_map)

                rr = dict(r)
                rr["density_path"] = str(den_path)
                rr["density_sum"] = float(np.sum(density_map))
                updated.append(rr)

            write_jsonl(updated, jsonl_file)
            print(f"[OK] Created density maps for {jsonl_file.name}")

if __name__ == "__main__":
    main()