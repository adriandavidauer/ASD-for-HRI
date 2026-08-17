"""Score one model's predictions with the official AVA active-speaker script.

Every ground-truth box a detection matched is scored against that detection's
speaking probability; unmatched ground truth is left out, so the AP reflects the
classifier rather than the detector.
"""

import argparse
import bisect
import csv
import glob
import os
import re
import subprocess
import sys
from collections import defaultdict

import numpy as np
from paz.backend.boxes import compute_iou

HERE = os.path.dirname(os.path.abspath(__file__))
OFFICIAL = os.path.join(HERE, 'eval', 'get_ava_active_speaker_performance.py')

GT_COLUMNS = 8
POSITIVE = 'SPEAKING_AUDIBLE'

FIELDS = ['dataset', 'model', 'video_id', 'ap', 'n_gt', 'n_scored', 'n_pos',
          'entities', 'entities_hit', 'frames', 'elapsed', 'fps']
GLOBAL_FIELDS = ['dataset', 'model', 'ap', 'videos', 'n_gt', 'n_scored', 'n_pos']
DETAIL_FIELDS = ['frame_timestamp', 'entity_id', 'gt_label',
                 'x1', 'y1', 'x2', 'y2', 'iou', 'score', 'matched']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--gt', required=True, help='Ground-truth CSV covering every video')
    p.add_argument('--predictions_dir', required=True)
    p.add_argument('--out_dir', required=True, help='Directory for the converted CSV pairs')
    p.add_argument('--results_dir', required=True)
    p.add_argument('--gt_has_header', action='store_true')
    p.add_argument('--iou_threshold', type=float, default=0.5)
    p.add_argument('--timestamp_tolerance', type=float, default=0.02,
                   help='Max |pred.ts - gt.ts| in seconds to align a prediction to a GT frame')
    p.add_argument('--detail', action='store_true',
                   help='Also write <video_id>.detail.csv, one row per GT box')
    return p.parse_args()


def load_preds(path):
    """Read a prediction CSV -> (sorted timestamps, {timestamp: [(box, score), ...]})."""
    by_ts = defaultdict(list)
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            box = (float(row['x1']), float(row['y1']), float(row['x2']), float(row['y2']))
            by_ts[float(row['timestamp'])].append((box, float(row.get('score') or 0.0)))
    return sorted(by_ts), by_ts


def nearest(ts_sorted, by_ts, ts, tolerance):
    """Boxes from the prediction frame closest to ts, or [] if none within tolerance."""
    i = bisect.bisect_left(ts_sorted, ts)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(ts_sorted) and abs(ts_sorted[j] - ts) <= tolerance:
            if best is None or abs(ts_sorted[j] - ts) < abs(ts_sorted[best] - ts):
                best = j
    return by_ts[ts_sorted[best]] if best is not None else []


def match(gt_box, candidates, iou_threshold):
    """Highest-IoU prediction for one GT box -> (score, iou)."""
    if not candidates:
        return 0.0, 0.0
    ious = compute_iou(np.asarray(gt_box), np.array([box for box, _ in candidates]))
    best = int(np.argmax(ious))
    best_iou = float(ious[best])
    return (candidates[best][1], best_iou) if best_iou >= iou_threshold else (0.0, best_iou)


def convert_video(video_id, gt_rows, preds, out_dir, args):
    """Write the (gt, pred) pair over matched boxes, return diagnostics."""
    ts_sorted, by_ts = preds
    entities, entities_hit = set(), set()
    n_scored = n_pos = 0
    detail = []

    gt_path = os.path.join(out_dir, f'{video_id}.gt.csv')
    pred_path = os.path.join(out_dir, f'{video_id}.pred.csv')

    with open(gt_path, 'w', newline='') as fg, open(pred_path, 'w', newline='') as fp:
        for parts in gt_rows:
            label, entity_id = parts[6], parts[7]
            gt_box = tuple(map(float, parts[2:6]))
            entities.add(entity_id)

            candidates = nearest(ts_sorted, by_ts, float(parts[1]), args.timestamp_tolerance)
            score, overlap = match(gt_box, candidates, args.iou_threshold)
            is_matched = overlap >= args.iou_threshold

            if is_matched:
                n_scored += 1
                n_pos += label == POSITIVE
                entities_hit.add(entity_id)
                fg.write(','.join(parts[:GT_COLUMNS]) + '\n')
                fp.write(','.join(parts[:6] + [POSITIVE, entity_id, f'{score:.6f}']) + '\n')

            if args.detail:
                detail.append([parts[1], entity_id, label, *parts[2:6],
                               f'{overlap:.4f}', f'{score:.6f}', is_matched])

    if args.detail:
        with open(os.path.join(out_dir, f'{video_id}.detail.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(DETAIL_FIELDS)
            w.writerows(detail)

    return {'video_id': video_id, 'n_gt': len(gt_rows), 'n_scored': n_scored,
            'n_pos': n_pos, 'entities': len(entities), 'entities_hit': len(entities_hit)}


def average_precision(gt_path, pred_path):
    """Official AP for one (gt, pred) pair, or '' when the script rejects the input."""
    r = subprocess.run([sys.executable, '-O', OFFICIAL, '-g', gt_path, '-p', pred_path],
                       capture_output=True, text=True)
    m = re.search(r'average precision:\s*(\S+)', r.stdout)
    if not m:
        last = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else 'no output'
        print(f'  no AP for {os.path.basename(gt_path)}: {last}')
        return ''
    return m.group(1)


def load_times(predictions_dir):
    """video_id -> (frames, elapsed, fps) from aggregate_time.csv, when present."""
    path = os.path.join(predictions_dir, 'aggregate_time.csv')
    if not os.path.isfile(path):
        return {}
    times = {}
    for row in csv.DictReader(open(path, newline='')):
        try:
            frames, elapsed = int(row['frames_processed']), float(row['elapsed_seconds'])
        except (KeyError, ValueError):
            continue
        if elapsed > 0:
            times[row['video_id']] = (frames, elapsed, frames / elapsed)
    return times


def merge(path, fieldnames, key, rows):
    """Replace this run's rows in a results CSV, keeping every other run's rows."""
    new_keys = {tuple(n[k] for k in key) for n in rows}
    existing = []
    if os.path.isfile(path):
        existing = [r for r in csv.DictReader(open(path, newline=''))
                    if tuple(r[k] for k in key) not in new_keys]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(existing + rows)


def combine(out_dir, pattern, dest_name):
    """Concatenate the per-video parts into one file for the global score."""
    dest = os.path.join(out_dir, dest_name)
    with open(dest, 'w') as out:
        for part in sorted(glob.glob(os.path.join(out_dir, pattern))):
            if not os.path.basename(part).startswith('_all'):
                out.write(open(part).read())
    return dest


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    available = {os.path.splitext(f)[0] for f in os.listdir(args.predictions_dir)
                 if f.endswith('.csv')} - {'aggregate_time'}

    by_video = defaultdict(list)
    skipped = set()
    with open(args.gt) as f:
        if args.gt_has_header:
            next(f)
        for line in f:
            parts = line.rstrip('\n').split(',')
            if parts[0] in available:
                by_video[parts[0]].append(parts)
            else:
                skipped.add(parts[0])

    times = load_times(args.predictions_dir)
    rows = []
    for video_id, gt_rows in sorted(by_video.items()):
        preds = load_preds(os.path.join(args.predictions_dir, f'{video_id}.csv'))
        d = convert_video(video_id, gt_rows, preds, args.out_dir, args)
        frames, elapsed, fps = times.get(video_id, ('', '', ''))
        # No SPEAKING_AUDIBLE rows makes recall divide by zero upstream.
        ap = average_precision(os.path.join(args.out_dir, f'{video_id}.gt.csv'),
                               os.path.join(args.out_dir, f'{video_id}.pred.csv')) \
            if d['n_pos'] else ''
        rows.append({'dataset': args.dataset, 'model': args.model, 'ap': ap,
                     'frames': frames,
                     'elapsed': f'{elapsed:.3f}' if elapsed != '' else '',
                     'fps': f'{fps:.2f}' if fps != '' else '', **d})

    global_row = {'dataset': args.dataset, 'model': args.model, 'videos': len(rows),
                  'ap': average_precision(combine(args.out_dir, '*.gt.csv', '_all.gt.csv'),
                                          combine(args.out_dir, '*.pred.csv', '_all.pred.csv')),
                  **{k: sum(int(r[k]) for r in rows) for k in ('n_gt', 'n_scored', 'n_pos')}}

    merge(os.path.join(args.results_dir, 'results.csv'), FIELDS,
          ('dataset', 'model', 'video_id'), rows)
    merge(os.path.join(args.results_dir, 'results_global.csv'), GLOBAL_FIELDS,
          ('dataset', 'model'), [global_row])

    if skipped:
        print(f'skipped (no predictions): {" ".join(sorted(skipped))}')
    scored, n_gt = global_row['n_scored'], global_row['n_gt']
    print(f"\n{args.dataset}/{args.model} over {global_row['videos']} videos")
    print(f"  AP = {global_row['ap']}  [{scored} of {n_gt} GT boxes matched "
          f"({100 * scored / max(1, n_gt):.1f}%), positive rate "
          f"{global_row['n_pos'] / max(1, scored):.4f}]")


if __name__ == '__main__':
    main()
