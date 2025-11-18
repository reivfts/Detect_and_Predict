"""
Visualize worst examples from a CSV created by `tools/analyze_evaluation.py`.
It looks for image files under `--images-root` whose filename contains the frame id string.

Usage:
  python tools/visualize_worst.py --csv data/trackings/worst_iou.csv --images-root <path to frames> --out-dir data/trackings/visuals --n 20

If it cannot find an image for a row, it prints a message and skips it.
"""
import argparse
import os
import csv
import glob
import cv2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/trackings/worst_iou.csv")
    p.add_argument("--images-root", default='.')
    p.add_argument("--out-dir", default='data/trackings/visuals')
    p.add_argument("--n", type=int, default=30)
    return p.parse_args()


def find_image_for_frame(images_root, frame_id):
    # frame_id may be an int or string; try several patterns
    s = str(frame_id)
    # search for files containing the frame substring
    matches = []
    for ext in ('*.jpg','*.jpeg','*.png','*.bmp'):
        for path in glob.glob(os.path.join(images_root, '**', ext), recursive=True):
            fname = os.path.basename(path)
            if s in fname:
                matches.append(path)
    # if multiple, choose shortest path (likely exact file)
    if matches:
        matches.sort(key=lambda p: (len(p), p))
        return matches[0]
    return None


def draw_box(img, box, color, label=None, thickness=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label is not None:
        cv2.putText(img, label, (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def main():
    args = parse_args()
    if not os.path.exists(args.csv):
        print('CSV not found:', args.csv)
        return
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    with open(args.csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not r or r.get('frame') in (None, ''):
                continue
            rows.append(r)

    count = 0
    for r in rows:
        if count >= args.n:
            break
        frame = r.get('frame')
        img_path = find_image_for_frame(args.images_root, frame)
        if img_path is None:
            print(f"Image for frame {frame} not found under {args.images_root}")
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue
        # parse boxes
        try:
            gt = eval(r.get('gt_box'))
        except Exception:
            gt = None
        try:
            pred = eval(r.get('pred_box'))
        except Exception:
            pred = None

        vis = img.copy()
        if gt:
            draw_box(vis, gt, (0, 255, 0), label='GT')
        if pred:
            draw_box(vis, pred, (0, 0, 255), label='PRED')

        out_name = f"frame_{frame}_track_{r.get('track_id')}_iou_{r.get('iou')}.jpg"
        out_path = os.path.join(args.out_dir, out_name)
        cv2.imwrite(out_path, vis)
        print(f"Wrote {out_path} (source: {img_path})")
        count += 1

    if count == 0:
        print("No visualizations created. Provide --images-root pointing to frame images (jpg/png).")

if __name__ == '__main__':
    main()
