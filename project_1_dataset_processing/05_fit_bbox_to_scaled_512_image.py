#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

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

def parse_voc_boxes(xml_path: Path) -> list[tuple[float, float, float, float]]:
    """Extracts (xmin, ymin, xmax, ymax) from VOC XML in original coordinates."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        if bnd is None: continue
        xmin = float(bnd.findtext("xmin"))
        ymin = float(bnd.findtext("ymin"))
        xmax = float(bnd.findtext("xmax"))
        ymax = float(bnd.findtext("ymax"))
        boxes.append((xmin, ymin, xmax, ymax))
    return boxes

def map_box_to_512(box: tuple[float, float, float, float], 
                   scale: float, pad_x: int, pad_y: int, S: int):
    """Applies scaling and padding to a single bounding box."""
    x1, y1, x2, y2 = box
    # The Math: Multiply by scale factor, then add the black bar offset
    nx1 = x1 * scale + pad_x
    ny1 = y1 * scale + pad_y
    nx2 = x2 * scale + pad_x
    ny2 = y2 * scale + pad_y
    
    # Ensure they stay inside 0-512 boundaries
    return [
        max(0.0, min(S, nx1)),
        max(0.0, min(S, ny1)),
        max(0.0, min(S, nx2)),
        max(0.0, min(S, ny2))
    ]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default=".")
    ap.add_argument("--S", type=int, default=512)
    args = ap.parse_args()

    root = Path(args.project_root)
    xml_dir = root / "project_0_datasets/bounding_boxes"
    base_processing = root / "project_1_dataset_processing"
    
    print(f"--- [05] Fitting Bounding Boxes to {args.S}px Letterboxed Space ---")

    splits = ["train_files", "validation_files", "test_files"]
    for split_dir in splits:
        folder = base_processing / "modified_splits" / split_dir
        for jsonl_file in folder.glob("*.jsonl"):
            recs = read_jsonl(jsonl_file)
            updated = []
            
            for r in recs:
                # Get the original XML
                xml_path = xml_dir / (Path(r["filename"]).stem + ".xml")
                if not xml_path.exists():
                    continue

                orig_boxes = parse_voc_boxes(xml_path)
                
                # Retrieve the geometric metadata from Step 04
                scale = r["scale"]
                px, py = r["pad_x"], r["pad_y"]
                
                boxes_512 = []
                centers_512 = []
                
                for b in orig_boxes:
                    nb = map_box_to_512(b, scale, px, py, args.S)
                    # Skip boxes that became tiny or invalid after scaling
                    if (nb[2] - nb[0]) < 1.0 or (nb[3] - nb[1]) < 1.0:
                        continue
                    
                    boxes_512.append(nb)
                    # Store center points (crucial for Gaussian Density Maps later)
                    centers_512.append([(nb[0] + nb[2]) / 2.0, (nb[1] + nb[3]) / 2.0])

                rr = dict(r)
                rr.update({
                    "boxes_512": boxes_512,
                    "centers_512": centers_512
                })
                updated.append(rr)

            # Update the JSONL with the new spatial coordinates
            write_jsonl(updated, jsonl_file)
            print(f"[OK] Mapped boxes for {len(updated)} images in {jsonl_file.name}")

if __name__ == "__main__":
    main()