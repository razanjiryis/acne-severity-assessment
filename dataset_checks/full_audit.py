#!/usr/bin/env python3
from __future__ import annotations
import json
import csv
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image
from collections import defaultdict, Counter

def get_hayashi_sev(count: int) -> int:
    if count <= 5: return 0
    if 6 <= count <= 20: return 1
    if 21 <= count <= 50: return 2
    return 3

def main():
    # --- 0. Setup Paths ---
    root = Path("project_0_datasets")
    img_dir = root / "images"
    xml_dir = root / "bounding_boxes"
    split_root = root / "original_splits"
    out_dir = root / "dataset_checks"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Data Collections ---
    images_in_folder = {f.name for f in img_dir.glob("*.jpg")}
    xmls_in_folder = {f.name for f in xml_dir.glob("*.xml")}
    images_in_splits = set()
    split_data = defaultdict(list) 
    
    # --- 1. Structural Integrity & Split Coverage ---
    for txt_file in split_root.rglob("*.txt"):
        with txt_file.open("r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2: continue
                fn, sev = parts[0], int(parts[1])
                images_in_splits.add(fn)
                split_data[fn].append({"sev": sev, "fold": txt_file.name})

    img_no_xml = [img for img in images_in_folder if (Path(img).stem + ".xml") not in xmls_in_folder]
    xml_no_img = [xml for xml in xmls_in_folder if (Path(xml).stem + ".jpg") not in images_in_folder]
    split_no_img = [img for img in images_in_splits if img not in images_in_folder]
    orphans = [img for img in images_in_folder if img not in images_in_splits]

    with open(out_dir / "missing_pairs.log", "w") as log:
        log.write(f"--- Structural Integrity Log ---\n")
        log.write(f"Images without XML: {len(img_no_xml)}\n{img_no_xml}\n\n")
        log.write(f"XMLs without Image: {len(xml_no_img)}\n{xml_no_img}\n\n")
        log.write(f"Split entries without Image file: {len(split_no_img)}\n{split_no_img}\n\n")
        log.write(f"Orphan files: {len(orphans)}\n{orphans}\n")

    # --- 2. Content Quality & 3. Clinical Reliability ---
    image_stats = []
    hayashi_conflicts = []
    
    for fn in images_in_splits:
        img_path = img_dir / fn
        xml_path = xml_dir / (Path(fn).stem + ".xml")
        
        if not img_path.exists() or not xml_path.exists(): continue

        with Image.open(img_path) as img:
            w, h = img.size
            image_stats.append({"fn": fn, "w": w, "h": h})

        tree = ET.parse(xml_path)
        objects = tree.getroot().findall("object")
        xml_count = len(objects)
        
        sevs = [d['sev'] for d in split_data[fn]]
        sev_consistent = len(set(sevs)) == 1
        clinical_sev = sevs[0]

        hayashi_sev = get_hayashi_sev(xml_count)
        if clinical_sev != hayashi_sev or not sev_consistent:
            hayashi_conflicts.append({
                "filename": fn,
                "clinical_sev": clinical_sev,
                "xml_count": xml_count,
                "hayashi_sev": hayashi_sev,
                "consistent_across_folds": sev_consistent,
                "all_fold_sevs": str(sevs)
            })

    # --- 4. Save Final Reports ---
    avg_w = sum(s['w'] for s in image_stats) / len(image_stats)
    avg_h = sum(s['h'] for s in image_stats) / len(image_stats)
    
    # FIX: Convert tuple keys (w, h) to strings "WxH" for JSON compatibility
    res_counts = Counter([(s['w'], s['h']) for s in image_stats])
    res_formatted = {f"{w}x{h}": count for (w, h), count in res_counts.items()}

    with open(out_dir / "image_statistics.json", "w") as f:
        json.dump({
            "total_images": len(image_stats),
            "avg_width": avg_w,
            "avg_height": avg_h,
            "resolutions": res_formatted
        }, f, indent=4)

    with open(out_dir / "hayashi_conflicts.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "clinical_sev", "xml_count", "hayashi_sev", "consistent_across_folds", "all_fold_sevs"])
        writer.writeheader()
        writer.writerows(hayashi_conflicts)

    print(f"Audit Complete. Reports generated in {out_dir}")

if __name__ == "__main__":
    main()