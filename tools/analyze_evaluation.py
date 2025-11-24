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


def safe_int(s):
    try:
        return int(s)
    except Exception:
        return None


def parse_box(s):
    """Parse a box string like "[x1, y1, x2, y2]" into a tuple (x1,y1,x2,y2).
    Returns None on failure."""
    if not s:
        return None
    try:
        # remove brackets and whitespace
        s2 = s.strip()
        if s2.startswith('[') and s2.endswith(']'):
            s2 = s2[1:-1]
        parts = [p.strip() for p in s2.split(',')]
        if len(parts) < 4:
            return None
        vals = [float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])]
        return tuple(vals)
    except Exception:
        return None


def box_center(box):
    """Return center (cx, cy) of box (x1,y1,x2,y2) or None if invalid."""
    if not box:
        return None
    try:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return (cx, cy)
    except Exception:
        return None


def main():
    args = parse_args()
    path = args.csv
    out_dir = os.path.dirname(path)
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
            # parse boxes and recompute center distance to avoid relying on CSV precomputed value
            gt_box = parse_box(r.get('gt_box'))
            pred_box = parse_box(r.get('pred_box'))
            gt_c = box_center(gt_box)
            pred_c = box_center(pred_box)
            cd_computed = None
            if gt_c is not None and pred_c is not None:
                dx = gt_c[0] - pred_c[0]
                dy = gt_c[1] - pred_c[1]
                cd_computed = (dx * dx + dy * dy) ** 0.5

            rows.append({
                'frame': r.get('frame'),
                'frame_int': safe_int(r.get('frame')),
                'track_id': r.get('track_id'),
                'iou': iou,
                # center_distance now computed from boxes when possible, fallback to CSV col
                'center_distance': cd_computed if cd_computed is not None else safe_float(r.get('center_distance')),
                'gt_box': r.get('gt_box'),
                'pred_box': r.get('pred_box'),
                '_gt_center': gt_c,
                '_pred_center': pred_c
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

    # Compute per-track ADE (average displacement error) and FDE (final displacement error)
    tracks = {}
    for r in rows:
        tid = r.get('track_id')
        if tid is None or tid == '':
            continue
        tracks.setdefault(tid, []).append(r)

    track_metrics = []
    for tid, entries in tracks.items():
        # sort entries by frame
        entries_sorted = sorted([e for e in entries if e.get('frame_int') is not None], key=lambda x: x['frame_int'])
        # distances
        dists = [e['center_distance'] for e in entries_sorted if e['center_distance'] is not None]
        if not dists:
            continue
        ade = mean(dists)
        fde = dists[-1]

        # compute RMSE for this track if centers are available
        sqerrs = []
        for e in entries_sorted:
            gc = e.get('_gt_center')
            pc = e.get('_pred_center')
            if gc is None or pc is None:
                continue
            dx = gc[0] - pc[0]
            dy = gc[1] - pc[1]
            sqerrs.append(dx * dx + dy * dy)
        rmse = (mean(sqerrs) ** 0.5) if sqerrs else None

        track_metrics.append({'track_id': tid, 'frames': len(dists), 'ade': ade, 'fde': fde, 'rmse': rmse})

    # overall ADE/FDE (averaged across tracks)
    if track_metrics:
        overall_ade = mean([t['ade'] for t in track_metrics])
        overall_fde = mean([t['fde'] for t in track_metrics])
        print(f"Overall ADE (mean per-track ADE): {overall_ade:.2f}")
        print(f"Overall FDE (mean per-track FDE): {overall_fde:.2f}")
        # overall RMSE across tracks (only include tracks with rmse)
        rmses = [t['rmse'] for t in track_metrics if t.get('rmse') is not None]
        if rmses:
            print(f"Overall RMSE (mean per-track RMSE): {mean(rmses):.2f}")

        # save per-track metrics
        out_tracks = os.path.join(out_dir, 'tracks_metrics.csv')
        with open(out_tracks, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['track_id', 'frames', 'ade', 'fde', 'rmse'])
            writer.writeheader()
            for t in track_metrics:
                writer.writerow(t)
        print(f"Wrote per-track metrics -> {out_tracks}")

        # compute per-frame ADE and RMSE
        frames_stats = {}
        for r in rows:
            fi = r.get('frame_int')
            if fi is None:
                continue
            frames_stats.setdefault(fi, []).append(r.get('center_distance'))

        frames_out = os.path.join(out_dir, 'frames_metrics.csv')
        with open(frames_out, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['frame', 'count', 'ade', 'rmse'])
            writer.writeheader()
            for fi in sorted(frames_stats.keys()):
                vals = [v for v in frames_stats[fi] if v is not None]
                if not vals:
                    continue
                ade_f = mean(vals)
                rmse_f = (mean([v * v for v in vals]) ** 0.5) if vals else None
                writer.writerow({'frame': fi, 'count': len(vals), 'ade': f"{ade_f:.3f}", 'rmse': f"{rmse_f:.3f}" if rmse_f is not None else ''})
        print(f"Wrote per-frame metrics -> {frames_out}")

    out_dir = os.path.dirname(path)
    out_iou = os.path.join(out_dir, 'worst_iou.csv')
    out_cd = os.path.join(out_dir, 'worst_center_distance.csv')

    def write_csv(outpath, rows):
        with open(outpath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['frame','track_id','iou','center_distance','gt_box','pred_box'])
            writer.writeheader()
            for r in rows:
                # filter out any internal keys (like 'frame_int') before writing
                filtered = {k: v for k, v in r.items() if k in ('frame','track_id','iou','center_distance','gt_box','pred_box')}
                writer.writerow(filtered)

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
