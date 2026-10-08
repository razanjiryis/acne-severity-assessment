#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

def parse_split_line(line: str) -> tuple[str, int]:
    """Extracts filename and clinical severity from the original split file."""
    parts = line.strip().split()
    if len(parts) < 2:
        raise ValueError(f"Bad line format: {line!r}")
    return parts[0], int(parts[1])

def xml_count_objects(xml_path: Path) -> int:
    """Calculates the objective count by counting individual <object> tags in the XML."""
    tree = ET.parse(xml_path)
    return len(tree.getroot().findall("object"))

def process_one_file(in_path: Path, out_path: Path, xml_dir: Path):
    """Reads clinical labels, syncs with XML count, and writes refined ground truth."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_out = 0
    with in_path.open("r") as f_in, out_path.open("w") as f_out:
        for line in f_in:
            if not line.strip(): continue
            try:
                fn, sev_gt = parse_split_line(line)
                xml_path = xml_dir / (Path(fn).stem + ".xml")
                
                if xml_path.exists():
                    # The central logic: Count boxes instead of trusting the text file
                    cnt = xml_count_objects(xml_path)
                    # Format: filename severity_grade objective_count
                    f_out.write(f"{fn} {sev_gt} {cnt}\n")
                    n_out += 1
            except Exception as e:
                print(f"Error processing line {line.strip()}: {e}")
                
    return n_out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default=".")
    # Path to Raw Project 0 Data
    ap.add_argument("--xml_dir", type=str, default="project_0_datasets/bounding_boxes")
    # Path to Refined Project 1 Labels
    ap.add_argument("--out_dir", type=str, default="project_1_dataset_processing/modified_splits")
    args = ap.parse_args()

    root = Path(args.project_root)
    out_dir = root / args.out_dir
    xml_dir = root / args.xml_dir
    
    # Mapping to the folders in Project 0
    split_root = root / "project_0_datasets/original_splits"
    paths = {
        "trainval": split_root / "train_files",
        "test": split_root / "test_files"
    }

    print("--- [STEP 01] Synchronizing Ground Truth from XML Counts ---")

    for k in range(5):
        for split, folder in paths.items():
            in_file = folder / f"NNEW_{split}_{k}.txt"
            if not in_file.exists():
                continue
            
            # This creates the 'Refined' files that all future steps will use
            out_file = out_dir / f"fold_{k}_{split}_xml.txt"
            count = process_one_file(in_file, out_file, xml_dir)
            print(f"[SUCCESS] Fold {k} {split}: {count} samples refined -> {out_file.name}")

if __name__ == "__main__":
    main()