#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
from PIL import Image

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

@dataclass
class LetterboxInfo:
    scale: float
    pad_x: int
    pad_y: int
    new_w: int
    new_h: int
    orig_w: int
    orig_h: int
    S: int

def letterbox(img: Image.Image, S: int) -> Tuple[Image.Image, LetterboxInfo]:
    orig_w, orig_h = img.size
    scale = min(S / orig_w, S / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    img_rs = img.resize((new_w, new_h), resample=Image.BILINEAR)
    
    # Create black canvas
    new_img = Image.new("RGB", (S, S), (0, 0, 0))
    pad_x = (S - new_w) // 2
    pad_y = (S - new_h) // 2
    new_img.paste(img_rs, (pad_x, pad_y))

    info = LetterboxInfo(
        scale=scale, pad_x=pad_x, pad_y=pad_y,
        new_w=new_w, new_h=new_h,
        orig_w=orig_w, orig_h=orig_h, S=S
    )
    return new_img, info

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default=".")
    ap.add_argument("--S", type=int, default=512)
    args = ap.parse_args()

    root = Path(args.project_root)
    src_images = root / "project_0_datasets/images"
    base_processing = root / "project_1_dataset_processing"
    
    # Where the images will go
    out_img_root = base_processing / f"images_{args.S}"
    out_img_root.mkdir(parents=True, exist_ok=True)

    print(f"--- [04] Standardizing Images to {args.S}px (Letterbox) ---")

    splits = ["train_files", "validation_files", "test_files"]
    for split_dir in splits:
        folder = base_processing / "modified_splits" / split_dir
        for jsonl_file in folder.glob("*.jsonl"):
            recs = read_jsonl(jsonl_file)
            updated = []
            
            for r in recs:
                img_path = src_images / r["filename"]
                if not img_path.exists():
                    continue

                img = Image.open(img_path).convert("RGB")
                img_std, info = letterbox(img, S=args.S)

                # Save physical file
                out_path = out_img_root / f"{img_path.stem}.png"
                img_std.save(out_path)

                # Update record with geometric metadata
                rr = dict(r)
                rr.update({
                    "image_std_path": str(out_path),
                    "scale": float(info.scale),
                    "pad_x": int(info.pad_x),
                    "pad_y": int(info.pad_y),
                    "orig_w": int(info.orig_w),
                    "orig_h": int(info.orig_h)
                })
                updated.append(rr)

            # Overwrite the JSONL with the new metadata
            write_jsonl(updated, jsonl_file)
            print(f"[OK] Processed {len(updated)} images for {jsonl_file.name}")

if __name__ == "__main__":
    main()