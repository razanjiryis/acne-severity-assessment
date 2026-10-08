#!/usr/bin/env python3
import argparse
import json
import numpy as np
from pathlib import Path

def read_jsonl(p: Path) -> list[dict]:
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_jsonl(rows: list[dict], p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default=".")
    ap.add_argument("--S", type=int, default=512)
    args = ap.parse_args()

    root = Path(args.project_root)
    base_processing = root / "project_1_dataset_processing"
    
    # Where the .npy mask files will be stored
    out_mask_root = base_processing / "attention_masks"
    out_mask_root.mkdir(parents=True, exist_ok=True)

    print(f"--- [06] Generating Binary Attention Masks ({args.S}x{args.S}) ---")

    splits = ["train_files", "validation_files", "test_files"]
    for split_dir in splits:
        folder = base_processing / "modified_splits" / split_dir
        
        for jsonl_file in folder.glob("*.jsonl"):
            recs = read_jsonl(jsonl_file)
            updated = []
            
            for r in recs:
                # Create a blank black mask
                mask = np.zeros((args.S, args.S), dtype=np.float32)
                
                # Get the boxes we mapped in Step 05
                boxes = r.get("boxes_512", [])
                
                for b in boxes:
                    x1, y1, x2, y2 = b
                    
                    # Convert to integer pixel indices
                    # Use floor/ceil to ensure we don't 'lose' small lesions
                    xs = int(np.floor(x1))
                    ys = int(np.floor(y1))
                    xe = int(np.ceil(x2))
                    ye = int(np.ceil(y2))

                    # Clamp to image boundaries
                    xs = max(0, min(args.S, xs))
                    ys = max(0, min(args.S, ys))
                    xe = max(0, min(args.S, xe))
                    ye = max(0, min(args.S, ye))

                    # Fill the rectangle with 1.0 (Acne)
                    if xe > xs and ye > ys:
                        mask[ys:ye, xs:xe] = 1.0

                # Save the mask as a .npy file
                # We use the original filename stem to keep them organized
                stem = Path(r["filename"]).stem
                mask_path = out_mask_root / f"{stem}_mask.npy"
                np.save(mask_path, mask)

                # Update the record with the mask path
                rr = dict(r)
                rr["attn_mask_path"] = str(mask_path)
                updated.append(rr)

            # Update the JSONL manifest
            write_jsonl(updated, jsonl_file)
            print(f"[OK] Generated masks for {len(updated)} images in {jsonl_file.name}")

if __name__ == "__main__":
    main()