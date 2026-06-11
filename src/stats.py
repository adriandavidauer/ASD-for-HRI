"""Score VVAD predictions against UniTalk ground truth — purely from CSVs.
Having no dependencies from other files is intentional - Wanted to run in local setup without any use of docker.
"""

import os
import csv
import bisect
import argparse
import logging
from collections import defaultdict, namedtuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import pandas as pd

CONTAINMENT_THRESHOLD = 0.5          # accept match when smaller box is ≥50% covered
TIMESTAMP_TOLERANCE_MS = 200.0        # |pred.ts − gt.ts| must be within this to align frames
LOGGER = logging.getLogger('UniTalk_VVAD')

_LABEL_MAP = {
    'SPEAKING_AUDIBLE': 'speaking',
    'NOT_SPEAKING':     'not-speaking',
}

_PredBox = namedtuple('_PredBox', ['coordinates', 'class_name'])
_GtBox   = namedtuple('_GtBox',   ['timestamp', 'index', 'entity_id', 'vvad_label', 'bbox'])

_DETAIL_FIELDS = ['frame_timestamp', 'x1', 'y1', 'x2', 'y2',
                  'iou', 'containment', 'entity_id', 'gt_label', 'pred_label', 'matched']

_SUMMARY_FIELDS = ['video_id', 'tp', 'tn', 'fp', 'fn', 'precision', 'recall', 'f1',
                   'missed_detections', 'total_gt_boxes', 'missed_pct',
                   'entities_detected', 'entities_correctly_identified', 'total_entities']

def parse_args():
    p = argparse.ArgumentParser(description='Score VVAD prediction CSVs against ground truth')
    p.add_argument('--predictions_dir', default='data/predictions',
                   help='Directory of per-video prediction CSVs')
    p.add_argument('--groundtruth_csv', default='data/csv/val_orig.csv',
                   help='Master ground-truth CSV covering every video')
    p.add_argument('--result_dir',      default='data/results',
                   help='Directory for the aggregate CSV and per-video detail CSVs')
    p.add_argument('--video',           default=None,
                   help='Score a single video_id; omit to score every CSV in predictions_dir')
    p.add_argument('--iou_threshold',   type=float, default=0.5)
    p.add_argument('--timestamp_tolerance_ms', type=float, default=TIMESTAMP_TOLERANCE_MS,
                   help='Max |pred.ts- gt.ts| (ms) to align a prediction to a GT frame')
    p.add_argument('--workers',         type=int, default=None,
                   help='Parallel worker processes for scoring (default: all CPUs)')
    p.add_argument('--verbose', '-v',   action='store_true')
    return p.parse_args()

def setup_logging(log_name='unitalk_stats', verbose=False):
    """Configure file + console logging; return the log file path."""
    os.makedirs('logs', exist_ok=True)
    path = f'logs/{log_name}_{datetime.now():%Y%m%d_%H%M%S}.log'
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.handlers.clear()
    LOGGER.propagate = False
    fh = logging.FileHandler(path, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    LOGGER.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO if verbose else logging.WARNING)
    ch.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    LOGGER.addHandler(ch)
    return path

# ── data loading ─────────────────────────────────────────────────────────────

def load_ground_truth(csv_path):
    """Read the master ground-truth CSV → dict[video_id] -> list[gt row dict].
    """
    df = pd.read_csv(csv_path)
    df['vvad_label'] = df['label'].map(_LABEL_MAP).fillna('not-speaking')

    by_video = {}
    for vid, group in df.groupby('video_id', sort=False):
        by_video[vid] = [
            {
                'frame_timestamp': r.frame_timestamp,
                'bbox':            (r.entity_box_x1, r.entity_box_y1,
                                    r.entity_box_x2, r.entity_box_y2),
                'vvad_label':      r.vvad_label,
                'entity_id':       r.entity_id,
            }
            for r in group.itertuples(index=False)
        ]
    LOGGER.info('ground_truth videos=%d rows=%d path=%s',
                len(by_video), len(df), csv_path)
    return by_video


def load_predictions_csv(path):
    """Read a prediction CSV → sorted list of (timestamp, [_PredBox, ...]).
    """
    by_ts = defaultdict(list)
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            if not row.get('label'):
                continue
            by_ts[float(row['timestamp'])].append(_PredBox(
                (float(row['x1']), float(row['y1']),
                 float(row['x2']), float(row['y2'])),
                row['label'],
            ))
    return by_ts


class GroundTruthIndex:
    """Sorted-timestamp index  with binary search over one video's ground truth annotations.
    """

    def __init__(self, gt_rows):
        by_ts = defaultdict(list)
        for r in gt_rows:
            by_ts[r['frame_timestamp']].append(r)

        self.timestamps = sorted(by_ts.keys())
        self._buckets   = [None] * len(self.timestamps)  # parallel to timestamps: list of tuple(_GtBox, ...)
        self.entity_gt_rows = defaultdict(int)     # entity_id -> number of GT boxes
        for i, ts in enumerate(self.timestamps):
            boxes = tuple(
                _GtBox(ts, gi, r['entity_id'], r['vvad_label'], r['bbox'])
                for gi, r in enumerate(by_ts[ts])
            )
            self._buckets[i] = boxes
            for box in boxes:
                self.entity_gt_rows[box.entity_id] += 1
        self.entities = set(self.entity_gt_rows)
        self.total_gt_boxes = sum(self.entity_gt_rows.values())

    def nearest(self, timestamp, tolerance_s):
        """Return the closest GT frame's boxes within tolerance, or () if none."""
        ts = self.timestamps
        if not ts:
            return ()
        i = bisect.bisect_left(ts, timestamp)
        best_i, best_diff = None, tolerance_s
        for j in (i - 1, i): #checks the slot before and at the insertion point - timestamps cluster closely so the nearest is adjacent
            if 0 <= j < len(ts):
                diff = abs(ts[j] - timestamp)
                if diff <= best_diff:
                    best_i, best_diff = j, diff
        return self._buckets[best_i] if best_i is not None else ()

    def all_boxes(self):
        """Iterate every GT box in timestamp order."""
        for boxes in self._buckets:
            yield from boxes


# ── IoU / box matching ─────────────────────


def _overlap(a, b):
    """Return (IoU, containment) for two boxes, computing the intersection once.

    Containment is the fraction of the smaller box covered by the intersection.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    iou  = inter / max(1e-9, area_a + area_b - inter)
    cont = inter / max(1e-9, min(area_a, area_b))
    return iou, cont


def match_frame(pred_boxes, gt_boxes, iou_threshold):
    """Greedy highest-IoU matching of one frame's predictions to its GT boxes.
    """
    overlaps    = {}
    best_per_gt = [(0.0, 0.0, '')] * len(gt_boxes)
    for pi, pred in enumerate(pred_boxes):
        for gi, gt in enumerate(gt_boxes):
            iou, cont = _overlap(pred.coordinates, gt.bbox)
            overlaps[(pi, gi)] = (iou, cont)
            if iou > best_per_gt[gi][0]:
                best_per_gt[gi] = (iou, cont, pred.class_name)

    matches, used = [], set()
    for pi in range(len(pred_boxes)):
        best_iou = best_cont = 0.0
        best_gi = None
        for gi in range(len(gt_boxes)):
            if gi in used:
                continue
            iou, cont = overlaps[(pi, gi)]
            if iou > best_iou or (iou == best_iou and cont > best_cont):
                best_iou, best_cont, best_gi = iou, cont, gi
        if best_gi is not None and (
            best_iou >= iou_threshold or best_cont >= CONTAINMENT_THRESHOLD
        ):
            matches.append((pi, best_gi))
            used.add(best_gi)
    return matches, best_per_gt


# ── stats accumulator ────────────────────────────────────────────────────────

class Stats:
    """Accumulates precision / recall / F1, missed detections and entity identification."""

    def __init__(self, gt_index):
        self.total_gt_boxes = gt_index.total_gt_boxes
        self.gt_entities    = set(gt_index.entities)
        self.entity_gt_rows = dict(gt_index.entity_gt_rows)

        # binary classification on matched boxes (positive class = speaking)
        self.tp = self.tn = self.fp = self.fn = 0

        self.matched_boxes     = 0          # GT boxes matched by a prediction
        self.detected_entities = set()
        self.entity_matched    = defaultdict(int)
        self.entity_correct    = defaultdict(int)

    def update(self, pred_boxes, gt_boxes, matches):
        gt_for_pred = {pi: gi for pi, gi in matches}

        for pi, pred in enumerate(pred_boxes):
            label = pred.class_name
            gi    = gt_for_pred.get(pi)

            if gi is None:                       # spurious detection (no GT box)
                if label == 'speaking':
                    self.fp += 1
                elif label == 'not-speaking':
                    self.tn += 1
                continue

            gt = gt_boxes[gi]
            self.matched_boxes += 1
            self.detected_entities.add(gt.entity_id)
            self.entity_matched[gt.entity_id] += 1

            if label == gt.vvad_label:
                self.entity_correct[gt.entity_id] += 1
                if label == 'speaking':
                    self.tp += 1
                else:
                    self.tn += 1
            elif label == 'speaking':
                self.fp += 1
            else:
                self.fn += 1

    @property
    def precision(self):
        return self.tp / max(1, self.tp + self.fp)

    @property
    def recall(self):
        return self.tp / max(1, self.tp + self.fn)

    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2 * p * r / max(1e-9, p + r)

    @property
    def missed_detections(self):
        """GT boxes that no prediction matched (detection misses)."""
        return self.total_gt_boxes - self.matched_boxes

    @property
    def correctly_identified_entities(self):
        """Entities detected spatially and labelled atleast once."""
        return {eid for eid in self.detected_entities
                if self.entity_matched[eid]}


def compute_stats_from_predictions(predictions_csv, gt_rows, iou_threshold,
                                   timestamp_tolerance_s):
    """Score one video's prediction CSV and return (Stats, per-GT-box match info).
    """
    gt_index = GroundTruthIndex(gt_rows)
    stats = Stats(gt_index)
    # (timestamp, gi) -> [iou, containment, pred_label, matched] for the best
    # overlapping prediction seen (several pred frames may align to one GT frame)
    match_info = {}

    for ts, pred_boxes in load_predictions_csv(predictions_csv).items():
        gt_boxes = gt_index.nearest(ts, timestamp_tolerance_s)
        if not gt_boxes:
            continue
        matches, best_per_gt = match_frame(pred_boxes, gt_boxes, iou_threshold)
        stats.update(pred_boxes, gt_boxes, matches)

        accepted = {gi for _pi, gi in matches}
        for gi, gt in enumerate(gt_boxes):
            iou, cont, label = best_per_gt[gi]
            key  = (gt.timestamp, gi)
            prev = match_info.get(key)
            matched = gi in accepted or (prev is not None and prev[3])
            # keep the record with the strongest overlap; OR-in the matched flag
            if prev is None or iou > prev[0]:
                match_info[key] = [iou, cont, label, matched]
            elif matched:
                match_info[key][3] = True

    return stats, _detail_rows(gt_index, match_info)


def _detail_rows(gt_index, match_info):
    """Build per-GT-box detail rows (one per GT box, missed boxes included)."""
    rows = []
    for gt in gt_index.all_boxes():
        iou, cont, pred_label, matched = match_info.get((gt.timestamp, gt.index),
                                                         (0.0, 0.0, '', False))
        x1, y1, x2, y2 = gt.bbox
        rows.append({
            'frame_timestamp': f'{gt.timestamp:.3f}',
            'x1': f'{x1:.6f}', 'y1': f'{y1:.6f}', 'x2': f'{x2:.6f}', 'y2': f'{y2:.6f}',
            'iou':         f'{iou:.4f}',
            'containment': f'{cont:.4f}',
            'entity_id':   gt.entity_id,
            'gt_label':    gt.vvad_label,
            'pred_label':  pred_label,
            'matched':     matched,
        })
    return rows


# ── reporting ─────────────────────────────────────────────────────────────────


def write_detail_csv(result_dir, video_id, detail_rows):
    """Write the per-video detail CSV: one row per GT box with its match info."""
    os.makedirs(result_dir, exist_ok=True)
    path = os.path.join(result_dir, f'{video_id}_detail.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_DETAIL_FIELDS)
        writer.writeheader()
        writer.writerows(detail_rows)
    LOGGER.debug('detail_csv path=%s rows=%d', path, len(detail_rows))



def write_aggregate_csv(result_dir, per_video):
    """Write one CSV holding a summary row per video.

    ``per_video`` is a list of (video_id, Stats).
    """
    if not per_video:
        return
    os.makedirs(result_dir, exist_ok=True)
    path = os.path.join(result_dir, 'aggregate_results.csv')
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
        writer.writeheader()
        for video_id, stats in per_video:
            writer.writerow({
                'video_id':          video_id,
                'tp': stats.tp, 'tn': stats.tn, 'fp': stats.fp, 'fn': stats.fn,
                'precision':         f'{stats.precision:.4f}',
                'recall':            f'{stats.recall:.4f}',
                'f1':                f'{stats.f1:.4f}',
                'missed_detections': stats.missed_detections,
                'total_gt_boxes':    stats.total_gt_boxes,
                'missed_pct':        f'{100.0 * stats.missed_detections / max(1, stats.total_gt_boxes):.2f}',
                'entities_detected':             len(stats.detected_entities),
                'entities_correctly_identified': len(stats.correctly_identified_entities),
                'total_entities':                len(stats.gt_entities),
            })
    LOGGER.debug('aggregate_results path=%s videos=%d', path, len(per_video))




def _score_video(vid, predictions_csv, gt_rows, iou_threshold, tol_s, result_dir):
    """Worker: score one video, write its detail CSV, return (vid, Stats).
    """
    stats, detail_rows = compute_stats_from_predictions(
        predictions_csv, gt_rows, iou_threshold, tol_s)
    write_detail_csv(result_dir, vid, detail_rows)
    return vid, stats


def main():
    args = parse_args()
    log_path = setup_logging('unitalk_stats', args.verbose)
    tol_s = args.timestamp_tolerance_ms / 1000.0
    LOGGER.info('stats run start predictions_dir=%s gt=%s tol_ms=%.1f log=%s',
                args.predictions_dir, args.groundtruth_csv,
                args.timestamp_tolerance_ms, log_path)

    gt_by_video = load_ground_truth(args.groundtruth_csv)
    os.makedirs(args.result_dir, exist_ok=True)

    if args.video:
        video_ids = [args.video]
    else:
        video_ids = [
            os.path.splitext(f)[0]
            for f in os.listdir(args.predictions_dir)
            if f.endswith('.csv') and os.path.splitext(f)[0] in gt_by_video
        ]

    tasks = []
    for vid in video_ids:
        predictions_csv = os.path.join(args.predictions_dir, f'{vid}.csv')
        if not os.path.isfile(predictions_csv):
            LOGGER.warning('Skipping video=%s reason=predictions_not_found path=%s',
                           vid, predictions_csv)
            continue
        gt_rows = gt_by_video.get(vid)
        if not gt_rows:
            LOGGER.warning('Skipping video=%s reason=no_ground_truth', vid)
            continue
        tasks.append((vid, predictions_csv, gt_rows, args.iou_threshold, tol_s,
                      args.result_dir))

    # Score videos in parallel
    per_video = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_score_video, *task): task[0] for task in tasks}
        for future in as_completed(futures):
            vid = futures[future]
            try:
                vid, stats = future.result()
                per_video.append((vid, stats))
            except Exception:
                LOGGER.exception('Failed scoring video=%s', vid)

    write_aggregate_csv(args.result_dir, per_video)
    LOGGER.info('stats run complete videos_scored=%d', len(per_video))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
