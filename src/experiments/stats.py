"""Score VVAD predictions against UniTalk ground truth — purely from CSVs.
Having no dependencies from other files is intentional - Wanted to run in local setup without any use of docker.
"""

import os
import csv
import bisect
import argparse
import logging
import importlib.util
from collections import defaultdict, namedtuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

CONTAINMENT_THRESHOLD = 0.5          # accept match when smaller box is ≥50% covered
TIMESTAMP_TOLERANCE_MS = 20.0        # |pred.ts − gt.ts| must be within this to align frames
LOGGER = logging.getLogger('UniTalk_VVAD')

_AVA_POSITIVE = 'SPEAKING_AUDIBLE'
_AVA_EVAL_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'eval', 'get_ava_active_speaker_performance.py')
_AVA_COLUMNS = ['video_id', 'frame_timestamp', 'entity_box_x1', 'entity_box_y1',
                'entity_box_x2', 'entity_box_y2', 'label', 'entity_id']

_LABEL_MAP = {
    'SPEAKING_AUDIBLE': 'speaking',
    'NOT_SPEAKING':     'not-speaking',
}

_PredBox = namedtuple('_PredBox', ['coordinates', 'class_name', 'score'])
_GtBox   = namedtuple('_GtBox',   ['timestamp', 'index', 'entity_id', 'vvad_label', 'bbox',
                                   'ava_label'])

_DETAIL_FIELDS = ['frame_timestamp', 'x1', 'y1', 'x2', 'y2',
                  'iou', 'containment', 'entity_id', 'gt_label', 'pred_label', 'score', 'matched']

_SUMMARY_FIELDS = ['video_id', 'tp', 'tn', 'fp', 'fn', 'accuracy', 'precision', 'recall', 'f1',
                   'ap', 'ap_scored_boxes', 'ap_positives',
                   'missed_detections', 'total_gt_boxes', 'missed_pct',
                   'entities_detected', 'entities_correctly_identified', 'total_entities',
                   'frames_processed', 'elapsed_seconds', 'fps']

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
    os.makedirs('logs_stats', exist_ok=True)
    path = f'logs_stats/{log_name}_{datetime.now():%Y%m%d_%H%M%S}.log'
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
                'ava_label':       r.label,
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
                float(row['score']),
            ))
    return by_ts


def load_processing_times(predictions_dir):
    """Read ``aggregate_time.csv`` -> dict[video_id] -> {frames, elapsed, fps}.
    """
    path = os.path.join(predictions_dir, 'aggregate_time.csv')
    if not os.path.isfile(path):
        LOGGER.warning('aggregate_time.csv not found path=%s', path)
        return {}
    by_video = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            try:
                frames  = int(row['frames_processed'])
                elapsed = float(row['elapsed_seconds'])
            except (KeyError, ValueError, TypeError):
                continue
            if elapsed <= 0:
                continue
            by_video[row['video_id']] = {
                'frames_processed': frames,
                'elapsed_seconds':  elapsed,
                'fps':              frames / elapsed,
            }
    LOGGER.info('processing_times videos=%d path=%s', len(by_video), path)
    return by_video


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
                _GtBox(ts, gi, r['entity_id'], r['vvad_label'], r['bbox'], r['ava_label'])
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
    best_per_gt = [(0.0, 0.0, '', 0.0)] * len(gt_boxes)
    for pi, pred in enumerate(pred_boxes):
        for gi, gt in enumerate(gt_boxes):
            iou, cont = _overlap(pred.coordinates, gt.bbox)
            overlaps[(pi, gi)] = (iou, cont)
            if iou > best_per_gt[gi][0]:
                best_per_gt[gi] = (iou, cont, pred.class_name, pred.score)

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

        # average precision over the matched boxes, plus the pooled inputs it came from
        self.ap           = float('nan')
        self.ap_scores    = np.empty(0)
        self.ap_positives = np.empty(0, dtype=bool)

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
    def accuracy(self):
        """Label accuracy on matched boxes: entity_correct / entity_matched.

        Counts only spatially-matched GT boxes;
        """
        correct = sum(self.entity_correct.values())
        matched = sum(self.entity_matched.values())
        return correct / max(1, matched)

    @property
    def missed_detections(self):
        """GT boxes that no prediction matched (detection misses)."""
        return self.total_gt_boxes - self.matched_boxes

    @property
    def correctly_identified_entities(self):
        """Entities detected spatially and labelled atleast once."""
        return {eid for eid in self.detected_entities
                if self.entity_matched[eid]}


def compute_stats_from_predictions(video_id, predictions_csv, gt_rows, iou_threshold,
                                   timestamp_tolerance_s):
    """Score one video's prediction CSV and return (Stats, per-GT-box match info).
    """
    gt_index = GroundTruthIndex(gt_rows)
    stats = Stats(gt_index)
    # (timestamp, gi) -> [iou, containment, pred_label, matched, score] for the best
    # overlapping prediction seen (several pred frames may align to one GT frame)
    match_info = {}

    for ts, pred_boxes in load_predictions_csv(predictions_csv).items():
        gt_boxes = gt_index.nearest(ts, timestamp_tolerance_s)
        if not gt_boxes:
            continue
        matches, best_per_gt = match_frame(pred_boxes, gt_boxes, iou_threshold)
        stats.update(pred_boxes, gt_boxes, matches)

        accepted = {gi: pi for pi, gi in matches}
        for gi, gt in enumerate(gt_boxes):
            iou, cont, label, score = best_per_gt[gi]
            if gi in accepted:
                # the assigned pair can differ from the highest-IoU one on a containment match
                pred = pred_boxes[accepted[gi]]
                label, score = pred.class_name, pred.score
            key  = (gt.timestamp, gi)
            prev = match_info.get(key)
            matched = gi in accepted or (prev is not None and prev[3])
            # keep the record with the strongest overlap; OR-in the matched flag
            if prev is None or iou > prev[0]:
                match_info[key] = [iou, cont, label, matched, score]
            elif matched:
                if not prev[3]:            # first accepted match supplies the scored label
                    prev[2], prev[4] = label, score
                prev[3] = True

    ap_rows = _ap_rows(video_id, gt_index, match_info)
    stats.ap_scores = np.array([r[8] for r in ap_rows], dtype=float)
    stats.ap_positives = np.array([r[6] == _AVA_POSITIVE for r in ap_rows], dtype=bool)
    stats.ap = average_precision(ap_rows)
    return stats, _detail_rows(gt_index, match_info)


def _detail_rows(gt_index, match_info):
    """Build per-GT-box detail rows (one per GT box, missed boxes included)."""
    rows = []
    for gt in gt_index.all_boxes():
        iou, cont, pred_label, matched, score = match_info.get(
            (gt.timestamp, gt.index), (0.0, 0.0, '', False, 0.0))
        x1, y1, x2, y2 = gt.bbox
        rows.append({
            'frame_timestamp': f'{gt.timestamp:.3f}',
            'x1': f'{x1:.6f}', 'y1': f'{y1:.6f}', 'x2': f'{x2:.6f}', 'y2': f'{y2:.6f}',
            'iou':         f'{iou:.4f}',
            'containment': f'{cont:.4f}',
            'entity_id':   gt.entity_id,
            'gt_label':    gt.vvad_label,
            'pred_label':  pred_label,
            'score':       f'{score:.6f}',
            'matched':     matched,
        })
    return rows


# ── average precision (official AVA scorer) ───────────────────────────────────


_AVA_EVAL_MODULE = None


def _ava_eval():
    """Import the official AVA active-speaker scorer by path, once per process."""
    global _AVA_EVAL_MODULE
    if _AVA_EVAL_MODULE is None:
        spec = importlib.util.spec_from_file_location('ava_active_speaker_eval', _AVA_EVAL_SCRIPT)
        _AVA_EVAL_MODULE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_AVA_EVAL_MODULE)
    return _AVA_EVAL_MODULE


def _ap_rows(video_id, gt_index, match_info):
    """AVA-format rows for every scored GT box: the 8 GT columns plus the predicted score.

    Only matched boxes carry a score, so unmatched ground truth is left out and the AP
    reflects the classifier rather than the detector.
    """
    rows = []
    for gt in gt_index.all_boxes():
        record = match_info.get((gt.timestamp, gt.index))
        if record is None or not record[3] or record[4] < 0.0:
            continue                        # unmatched, or the classifier had no prediction yet
        rows.append((video_id, gt.timestamp, *gt.bbox, gt.ava_label, gt.entity_id, record[4]))
    return rows


def average_precision(ap_rows):
    """AP for one video, run through the official scorer's own merge and PR code."""
    positives = sum(r[6] == _AVA_POSITIVE for r in ap_rows)
    if not positives:
        return float('nan')                 # recall would divide by zero
    ava = _ava_eval()
    df = pd.DataFrame(ap_rows, columns=_AVA_COLUMNS + ['score'])
    df['uid'] = df['frame_timestamp'].map(str) + ':' + df['entity_id']
    df_groundtruth = df.drop(columns='score')
    df_predictions = df.assign(label=_AVA_POSITIVE)
    df_merged = ava.merge_groundtruth_and_predictions(df_groundtruth, df_predictions)
    precision, recall = ava.calculate_precision_recall(df_merged)
    return float(ava.compute_average_precision(precision, recall))


def pooled_average_precision(scores, positives):
    """AP over every video's scored boxes pooled into one ranking.

    Builds the merged frame directly instead of re-running the scorer's CSV alignment
    checks, which the per-video pass has already applied to these same boxes.
    """
    if not positives.any():
        return float('nan')
    ava = _ava_eval()
    df_merged = pd.DataFrame({
        'uid':               np.arange(len(scores)),   # only ever counted, never joined on
        'score':             scores,
        'label_groundtruth': np.where(positives, _AVA_POSITIVE, 'NOT_SPEAKING'),
        'label_prediction':  _AVA_POSITIVE,
    }).sort_values(by=['score'], ascending=False).reset_index(drop=True)
    precision, recall = ava.calculate_precision_recall(df_merged)
    return float(ava.compute_average_precision(precision, recall))


def _format_ap(ap):
    """Blank out an AP that has no positives to score."""
    return '' if np.isnan(ap) else f'{ap:.4f}'


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



def write_aggregate_csv(result_dir, per_video, times_by_video=None):
    """Write one CSV holding a summary row per video.
    """
    if not per_video:
        return
    times_by_video = times_by_video or {}
    os.makedirs(result_dir, exist_ok=True)
    path = os.path.join(result_dir, 'aggregate_results.csv')
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
        writer.writeheader()

        tot_tp = tot_tn = tot_fp = tot_fn = 0
        tot_correct = tot_matched = 0          # for matched-box label accuracy
        tot_missed = tot_gt_boxes = 0
        tot_detected = tot_identified = tot_entities = 0
        tot_frames = 0
        tot_elapsed = 0.0
        fps_values = []                        # per-video fps, for the average row

        for video_id, stats in per_video:
            timing = times_by_video.get(video_id)
            row = {
                'video_id':          video_id,
                'tp': stats.tp, 'tn': stats.tn, 'fp': stats.fp, 'fn': stats.fn,
                'accuracy':          f'{stats.accuracy:.4f}',
                'precision':         f'{stats.precision:.4f}',
                'recall':            f'{stats.recall:.4f}',
                'f1':                f'{stats.f1:.4f}',
                'ap':                _format_ap(stats.ap),
                'ap_scored_boxes':   len(stats.ap_scores),
                'ap_positives':      int(stats.ap_positives.sum()),
                'missed_detections': stats.missed_detections,
                'total_gt_boxes':    stats.total_gt_boxes,
                'missed_pct':        f'{100.0 * stats.missed_detections / max(1, stats.total_gt_boxes):.2f}',
                'entities_detected':             len(stats.detected_entities),
                'entities_correctly_identified': len(stats.correctly_identified_entities),
                'total_entities':                len(stats.gt_entities),
                'frames_processed': timing['frames_processed'],
                 'elapsed_seconds': f"{timing['elapsed_seconds']:.3f}", 
                 'fps': f"{timing['fps']:.2f}"
            }
                
            writer.writerow(row)

            tot_tp += stats.tp
            tot_tn += stats.tn
            tot_fp += stats.fp
            tot_fn += stats.fn
            tot_correct += sum(stats.entity_correct.values())
            tot_matched += sum(stats.entity_matched.values())
            tot_missed += stats.missed_detections
            tot_gt_boxes += stats.total_gt_boxes
            tot_detected += len(stats.detected_entities)
            tot_identified += len(stats.correctly_identified_entities)
            tot_entities += len(stats.gt_entities)
            tot_frames  += timing['frames_processed']
            tot_elapsed += timing['elapsed_seconds']

        # micro-averaged summary row recomputed from the pooled counts
        micro_precision = tot_tp / max(1, tot_tp + tot_fp)
        micro_recall    = tot_tp / max(1, tot_tp + tot_fn)
        micro_f1        = 2 * micro_precision * micro_recall / max(1e-9, micro_precision + micro_recall)
        micro_accuracy  = tot_correct / max(1, tot_matched)
        avg_fps = tot_frames / tot_elapsed
        pooled_scores    = np.concatenate([s.ap_scores for _vid, s in per_video])
        pooled_positives = np.concatenate([s.ap_positives for _vid, s in per_video])
        writer.writerow({
            'video_id':          'micro_average',
            'tp': tot_tp, 'tn': tot_tn, 'fp': tot_fp, 'fn': tot_fn,
            'accuracy':          f'{micro_accuracy:.4f}',
            'precision':         f'{micro_precision:.4f}',
            'recall':            f'{micro_recall:.4f}',
            'f1':                f'{micro_f1:.4f}',
            'ap':                _format_ap(pooled_average_precision(pooled_scores,
                                                                     pooled_positives)),
            'ap_scored_boxes':   len(pooled_scores),
            'ap_positives':      int(pooled_positives.sum()),
            'missed_detections': tot_missed,
            'total_gt_boxes':    tot_gt_boxes,
            'missed_pct':        f'{100.0 * tot_missed / max(1, tot_gt_boxes):.2f}',
            'entities_detected':             tot_detected,
            'entities_correctly_identified': tot_identified,
            'total_entities':                tot_entities,
            'frames_processed':  tot_frames ,
            'elapsed_seconds':   f'{tot_elapsed:.3f}',
            'fps':               f'{avg_fps:.2f}',
        })
    LOGGER.debug('aggregate_results path=%s videos=%d', path, len(per_video))




def _score_video(vid, predictions_csv, gt_rows, iou_threshold, tol_s, result_dir):
    """Worker: score one video, write its detail CSV, return (vid, Stats).
    """
    stats, detail_rows = compute_stats_from_predictions(
        vid, predictions_csv, gt_rows, iou_threshold, tol_s)
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
    times_by_video = load_processing_times(args.predictions_dir)
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

    write_aggregate_csv(args.result_dir, per_video, times_by_video)
    LOGGER.info('stats run complete videos_scored=%d', len(per_video))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
