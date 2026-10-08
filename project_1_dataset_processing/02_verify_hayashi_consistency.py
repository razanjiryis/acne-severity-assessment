#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from collections import Counter, defaultdict
import csv

def parse_line(line: str) -> tuple[str, int, int]:
    # Expected format from Step 01: filename sev_gt count_xml
    parts = line.strip().split()
    if len(parts) < 3:
        raise ValueError(f"Bad line: {line!r}")
    fn = parts[0]
    sev = int(parts[1])
    cnt = int(parts[2])
    return fn, sev, cnt

def hayashi_sev_from_count(count_xml: int) -> int:
    """
    Medical standard for acne severity:
      0 (Mild): 1–5 lesions
      1 (Moderate): 6–20 lesions
      2 (Severe): 21–50 lesions
      3 (Very Severe): >50 lesions
    """
    if count_xml <= 5:
        return 0
    if 6 <= count_xml <= 20:
        return 1
    if 21 <= count_xml <= 50:
        return 2
    return 3

def load_split_file(path: Path) -> list[tuple[str, int, int]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rows.append(parse_line(line))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default=".")
    # INPUT: Points to the output of Step 01
    ap.add_argument("--splits_xml_dir", type=str, default="project_1_dataset_processing/modified_splits")
    # OUTPUT: A new folder for medical audit reports
    ap.add_argument("--out_dir", type=str, default="project_1_dataset_processing/audit_reports/hayashi")
    ap.add_argument("--folds", type=str, default="0,1,2,3,4")
    args = ap.parse_args()

    project_root = Path(args.project_root)
    splits_dir = project_root / args.splits_xml_dir
    out_dir = project_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    folds = [int(x.strip()) for x in args.folds.split(",") if x.strip()]

    print("--- [STEP 02] Verifying Clinical Logic (Hayashi Consistency) ---")

    for k in folds:
        for split_name in ["trainval", "test"]:
            p = splits_dir / f"fold_{k}_{split_name}_xml.txt"
            if not p.exists():
                print(f"[WARN] Missing: {p}")
                continue

            rows = load_split_file(p)
            
            # Check for 1-4 vs 0-3 labeling shift
            sevs = [sev for _, sev, _ in rows]
            sev_shift = 1 if (len(sevs) > 0 and min(sevs) >= 1 and max(sevs) <= 4) else 0

            mismatches = []
            conf = Counter()
            delta_by_sev = defaultdict(list)

            for fn, sev_gt_raw, cnt_xml in rows:
                sev_gt = sev_gt_raw - sev_shift
                sev_h = hayashi_sev_from_count(cnt_xml)

                conf[(sev_gt, sev_h)] += 1
                delta_by_sev[sev_gt].append(cnt_xml)

                if sev_gt != sev_h:
                    mismatches.append((fn, sev_gt_raw, sev_gt, cnt_xml, sev_h))

            total = len(rows)
            bad = len(mismatches)
            rate = bad / max(total, 1)

            # 1. Write Detailed CSV of conflicts
            csv_path = out_dir / f"fold_{k}_{split_name}_hayashi_mismatches.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["filename", "sev_gt_raw", "sev_gt_0to3", "count_xml", "sev_from_count_0to3"])
                for r in mismatches:
                    w.writerow(list(r))

            # 2. Write Summary Report
            rep_path = out_dir / f"fold_{k}_{split_name}_hayashi_report.txt"
            with rep_path.open("w", encoding="utf-8") as f:
                f.write(f"Fold {k} | Split: {split_name}\n")
                f.write(f"Total samples: {total}\n")
                f.write(f"Hayashi mismatches: {bad} ({rate*100:.2f}%)\n\n")

                f.write("Confusion Matrix (Clinical Grade vs Medical Standard):\n")
                for gt in range(4):
                    row = [str(conf.get((gt, pred), 0)) for pred in range(4)]
                    f.write(f"  Grade {gt}: " + "  ".join(row) + "\n")

            print(f"[OK] Fold {k} {split_name}: {bad} clinical inconsistencies found.")

    print("\n[DONE] Hayashi Audit Complete. Reports saved to project_1_dataset_processing/audit_reports/hayashi")

if __name__ == "__main__":
    main()