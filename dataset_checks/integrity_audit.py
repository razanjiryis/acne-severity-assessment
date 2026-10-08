#!/usr/bin/env python3
from __future__ import annotations
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

def get_xml_count(xml_path: Path) -> int:
    """Helper to extract instance count from XML."""
    try:
        tree = ET.parse(xml_path)
        return len(tree.getroot().findall("object"))
    except Exception:
        return -1 

def main():
    # Root of Project 0
    ds_root = Path("project_0_datasets")
    img_dir = ds_root / "images"
    xml_dir = ds_root / "bounding_boxes"
    split_root = ds_root / "original_splits"
    
    # Audit Output folder
    audit_out = ds_root / "dataset_check/reports"
    audit_out.mkdir(parents=True, exist_ok=True)

    # Dictionary to track every image across every fold it appears in
    image_tracker = defaultdict(list)
    
    print("--- [AUDIT] Starting Dataset Integrity Check ---")

    # Loop through all split files
    for split_path in split_root.rglob("*.txt"):
        with split_path.open("r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3: continue
                
                fn, sev, doc_cnt = parts[0], int(parts[1]), int(parts[2])
                xml_path = xml_dir / (Path(fn).stem + ".xml")
                
                xml_actual = get_xml_count(xml_path) if xml_path.exists() else -2
                
                image_tracker[fn].append({
                    "split_file": split_path.name,
                    "clinical_severity": sev,
                    "doctor_count": doc_cnt,
                    "xml_count": xml_actual,
                    "img_found": (img_dir / fn).exists()
                })

    # Identify Mismatches
    mismatch_list = []
    for fn, history in image_tracker.items():
        # Check if Doctor Count != XML Count in any fold
        has_mismatch = any(h["doctor_count"] != h["xml_count"] for h in history)
        if has_mismatch:
            mismatch_list.append({"filename": fn, "data": history})

    # Save Results
    summary = {
        "total_unique_images": len(image_tracker),
        "total_mismatches_found": len(mismatch_list),
        "mismatch_rate_percent": (len(mismatch_list)/len(image_tracker)*100) if image_tracker else 0
    }
    
    (audit_out / "audit_summary.json").write_text(json.dumps(summary, indent=4))
    (audit_out / "mismatch_details.json").write_text(json.dumps(mismatch_list, indent=4))
    print(f"--- [DONE] Summary: {summary['total_mismatches_found']} mismatches detected. ---")

if __name__ == "__main__":
    main()