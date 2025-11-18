"""
Simple analyzer for `data/trackings/evaluation.csv`.
Saves `worst_iou.csv` and `worst_center_distance.csv` into the same folder.
Usage: python tools/analyze_evaluation.py [--n 20] [--csv path/to/evaluation.csv]
"""
import csv
import sys
import os
import argparse
from statistics import mean, median


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/trackings/evaluation.csv")
    p.add_argument("--n", type=int, default=30)
    return p.parse_args()


def safe_float(s):
    try:
        return float(s)
    except Exception:
        return None


def main():
    args = parse_args()
    path = args.csv
    if not os.path.exists(path):
        print(f"Evaluation CSV not found: {path}")
        sys.exit(1)

    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not r or r.get('frame') in (None, ''):
                continue
            iou = safe_float(r.get('iou'))
            cd = safe_float(r.get('center_distance'))
            rows.append({
                'frame': r.get('frame'),
                'track_id': r.get('track_id'),
                'iou': iou,
                'center_distance': cd,
                'gt_box': r.get('gt_box'),
                'pred_box': r.get('pred_box')
            })

    ious = [r['iou'] for r in rows if r['iou'] is not None]
    cds = [r['center_distance'] for r in rows if r['center_distance'] is not None]

    print("Rows parsed:", len(rows))
    if ious:
        print(f"Average IoU: {mean(ious):.4f}  Median IoU: {median(ious):.4f}")
    if cds:
        print(f"Average center distance: {mean(cds):.2f}  Median: {median(cds):.2f}")

    # find worst IoU (lowest) and worst center distances (highest)
    worst_iou = sorted([r for r in rows if r['iou'] is not None], key=lambda x: x['iou'])[: args.n]
    worst_cd = sorted([r for r in rows if r['center_distance'] is not None], key=lambda x: -x['center_distance'])[: args.n]

    out_dir = os.path.dirname(path)
    out_iou = os.path.join(out_dir, 'worst_iou.csv')
    out_cd = os.path.join(out_dir, 'worst_center_distance.csv')

    def write_csv(outpath, rows):
        with open(outpath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['frame','track_id','iou','center_distance','gt_box','pred_box'])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    write_csv(out_iou, worst_iou)
    write_csv(out_cd, worst_cd)

    print(f"Wrote worst IoU -> {out_iou}")
    print(f"Wrote worst center distance -> {out_cd}")

    # quick diagnostics: common patterns in worst IoU
    print("\nTop patterns in worst IoU examples (first 10):")
    for r in worst_iou[:10]:
        print(f"frame={r['frame']} track={r['track_id']} iou={r['iou']} cd={r['center_distance']}")


if __name__ == '__main__':
    main()
