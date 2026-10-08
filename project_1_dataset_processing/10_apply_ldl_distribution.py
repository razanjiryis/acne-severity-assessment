#!/usr/bin/env python3
import argparse
import json
import numpy as np
from pathlib import Path

def generate_ldl_distribution(target_label, num_classes=4, sigma=0.5):
    """Converts a hard label into a Gaussian soft label distribution."""
    x = np.arange(num_classes)
    dist = np.exp(-(x - target_label)**2 / (2 * sigma**2))
    return (dist / dist.sum()).tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default=".")
    ap.add_argument("--sigma", type=float, default=0.5, help="Spread of the LDL distribution")
    args = ap.parse_args()

    root = Path(args.project_root)
    manifest_dir = root / "project_1_dataset_processing/final_manifests"

    print(f"--- [10] Applying Label Distribution Learning (Sigma={args.sigma}) ---")

    for jsonl_file in manifest_dir.glob("*.jsonl"):
        updated = []
        with jsonl_file.open("r") as f:
            for line in f:
                r = json.loads(line)
                # Logic: map 1-4 scale to 0-3 if necessary, then create distribution
                # We assume 0-3 based on Step 02 audit
                r["ldl_label"] = generate_ldl_distribution(r["sev_gt"], sigma=args.sigma)
                updated.append(r)
        
        with jsonl_file.open("w") as f:
            for r in updated:
                f.write(json.dumps(r) + "\n")
        
        print(f"[SUCCESS] LDL Applied to {jsonl_file.name}")

    print("\n[ALL STEPS COMPLETE] Project 1 Dataset Processing is finished.")

if __name__ == "__main__":
    main()