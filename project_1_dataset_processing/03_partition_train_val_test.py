#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

def read_split_txt(path: Path):
    """Reads the refined text files and converts them to list of dictionaries."""
    rows = []
    with path.open("r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3: continue
            # Format: filename severity objective_count
            rows.append({
                "filename": parts[0], 
                "sev_gt": int(parts[1]), 
                "count_xml": int(parts[2])
            })
    return rows

def write_jsonl(rows: list[dict], p: Path) -> None:
    """Writes list of dictionaries to a JSONL file."""
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default=".")
    ap.add_argument("--modified_dir", type=str, default="project_1_dataset_processing/modified_splits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    args = ap.parse_args()

    root = Path(args.project_root)
    base_path = root / args.modified_dir
    
    # Output Directories
    out_dirs = {
        "train": base_path / "train_files",
        "val": base_path / "validation_files",
        "test": base_path / "test_files"
    }

    # Initialize random generator for consistent splitting
    rnd = random.Random(args.seed)

    print("--- [03] Partitioning Data into JSONL Splits (90/10 Train-Val) ---")

    for k in range(5):
        # 1. Load the data created in Step 01
        trainval_txt = base_path / f"fold_{k}_trainval_xml.txt"
        test_txt = base_path / f"fold_{k}_test_xml.txt"

        if not trainval_txt.exists() or not test_txt.exists():
            print(f"[SKIP] Fold {k}: Files not found.")
            continue

        trainval_rows = read_split_txt(trainval_txt)
        test_rows = read_split_txt(test_txt)

        # 2. Shuffle and Split Train/Val
        rnd.shuffle(trainval_rows)
        val_size = int(len(trainval_rows) * args.val_ratio)
        
        val_rows = trainval_rows[:val_size]
        train_rows = trainval_rows[val_size:]

        # 3. Save as JSONL in the specific subdirectories
        write_jsonl(train_rows, out_dirs["train"] / f"fold_{k}_train.jsonl")
        write_jsonl(val_rows, out_dirs["val"] / f"fold_{k}_val.jsonl")
        write_jsonl(test_rows, out_dirs["test"] / f"fold_{k}_test.jsonl")

        print(f"[OK] Fold {k}: Train={len(train_rows)}, Val={len(val_rows)}, Test={len(test_rows)}")

    print(f"\n[DONE] Partitioning complete. JSONL files saved in {base_path}")

if __name__ == "__main__":
    main()