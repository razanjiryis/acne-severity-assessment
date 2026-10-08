#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

def read_jsonl(p: Path) -> list[dict]:
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: rows.append(json.loads(line))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default=".")
    args = ap.parse_args()

    root = Path(args.project_root)
    base_processing = root / "project_1_dataset_processing"
    
    print("--- [08] Final Bundle Integrity Check ---")

    splits = ["train_files", "validation_files", "test_files"]
    for split_dir in splits:
        folder = base_processing / "modified_splits" / split_dir
        for jsonl_file in folder.glob("*.jsonl"):
            recs = read_jsonl(jsonl_file)
            valid = []
            
            for r in recs:
                # Critical check: Does every piece of the puzzle exist?
                img_ok = Path(r.get("image_std_path", "")).exists()
                mask_ok = Path(r.get("attn_mask_path", "")).exists()
                dens_ok = Path(r.get("density_path", "")).exists()

                if img_ok and mask_ok and dens_ok:
                    valid.append(r)
                else:
                    print(f"[WARN] Missing files for {r['filename']}, skipping.")

            print(f"[OK] {jsonl_file.name}: {len(valid)}/{len(recs)} samples verified.")

if __name__ == "__main__":
    main()