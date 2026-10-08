#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default=".")
    args = ap.parse_args()

    root = Path(args.project_root)
    base_processing = root / "project_1_dataset_processing"
    out_dir = base_processing / "final_manifests"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("--- [09] Finalizing Lightweight Training Manifests ---")

    splits = ["train_files", "validation_files", "test_files"]
    for split_dir in splits:
        src_folder = base_processing / "modified_splits" / split_dir
        for jsonl_file in src_folder.glob("*.jsonl"):
            with jsonl_file.open("r") as f_in, (out_dir / jsonl_file.name).open("w") as f_out:
                for line in f_in:
                    r = json.loads(line)
                    # KEEP only what the model needs to train
                    clean_r = {
                        "filename": r["filename"],
                        "sev_gt": r["sev_gt"],
                        "count_xml": r["count_xml"],
                        "image_path": r["image_std_path"],
                        "mask_path": r["attn_mask_path"],
                        "density_path": r["density_path"]
                    }
                    f_out.write(json.dumps(clean_r) + "\n")
            print(f"[OK] Finalized: {jsonl_file.name}")

if __name__ == "__main__":
    main()