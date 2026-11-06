"""Score ASD predictions against AVA-format ground truth, purely from CSVs.

Having no dependencies on other files of the repo is intentional: it runs in a local setup
without docker (pandas + numpy only).
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
LOGGER = logging.getLogger('UniTalk_ASD')

_AVA_POSITIVE = 'SPEAKING_AUDIBLE'
_AVA_EVAL_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'eval', 'get_ava_active_speaker_performance.py')
_AVA_COLUMNS = ['video_id', 'frame_timestamp', 'entity_box_x1', 'entity_box_y1',
                'entity_box_x2', 'entity_box_y2', 'label', 'entity_id']

_LABEL_MAP = {
    'SPEAKING_AUDIBLE': 'speaking',
    'NOT_SPEAKING':     'not-speaking',
}

_PredBox = namedtuple('_PredBox', ['coordinates', 'class_name', 'score', 'entity_id'],
                      defaults=(None,))
_GtBox   = namedtuple('_GtBox',   ['timestamp', 'index', 'entity_id', 'asd_label', 'bbox',
                                   'ava_label'])

_DETAIL_FIELDS = ['frame_timestamp', 'x1', 'y1', 'x2', 'y2',
                  'iou', 'containment', 'entity_id', 'gt_label', 'pred_label', 'score', 'matched']

_SUMMARY_FIELDS = ['video_id', 'tp', 'tn', 'fp', 'fn', 'accuracy', 'precision', 'recall', 'f1',
                   'ap', 'ap_scored_boxes', 'ap_positives',
                   'missed_detections', 'total_gt_boxes', 'missed_pct',
                   'entities_detected', 'entities_correctly_identified', 'total_entities',
                   'frames_processed', 'elapsed_seconds', 'fps']

def parse_args():
    """Parse the command-line flags.

    Returns:
        argparse.Namespace: The parsed flags.
    """
    p = argparse.ArgumentParser(description='Score ASD prediction CSVs against ground truth')
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

def setup_logging(log_name='unitalk_stats', verbose=False, log_dir='logs_stats'):
    """Configure file and console logging.

    Args:
        log_name: Prefix of the log file name.
        verbose: If True, INFO messages also go to the console.
        log_dir: Directory for the log file.

    Returns:
        str: Path of the log file.
    """
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f'{log_name}_{datetime.now():%Y%m%d_%H%M%S}.log')
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
    """Read the master ground-truth CSV.

    Args:
        csv_path: AVA-format CSV with a header row.

    Returns:
        dict: video_id -> list of GT row dicts (timestamp, bbox, labels, entity id).
    """
    df = pd.read_csv(csv_path)
    df['asd_label'] = df['label'].map(_LABEL_MAP).fillna('not-speaking')

    by_video = {}
    for vid, group in df.groupby('video_id', sort=False):
        by_video[vid] = [
            {
                'frame_timestamp': r.frame_timestamp,
                'bbox':            (r.entity_box_x1, r.entity_box_y1,
                                    r.entity_box_x2, r.entity_box_y2),
                'asd_label':       r.asd_label,
                'ava_label':       r.label,
                'entity_id':       r.entity_id,
            }
            for r in group.itertuples(index=False)
        ]
    LOGGER.info('ground_truth videos=%d rows=%d path=%s',
                len(by_video), len(df), csv_path)
    return by_video


_LABEL_SCORES = {'speaking': 1.0, 'not-speaking': 0.0}


def _row_score(row):
    """Score of a prediction row.

    Runs older than the score column get 1/0 from the label, and -1 for 'No Prediction yet'.

    Args:
        row: One row of a prediction CSV as a dict.

    Returns:
        float: The score.
    """
    if row.get('score') not in (None, ''):
        return float(row['score'])
    return _LABEL_SCORES.get(row['label'], -1.0)


def load_predictions_csv(path):
    """Read a prediction CSV.

    Args:
        path: Prediction CSV written by ``run_asd_on_video``.

    Returns:
        dict: timestamp -> list of ``_PredBox``.
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
                _row_score(row),
                row.get('entity_id') or None,   # set only by --boxes gt runs
            ))
    return by_ts


def load_processing_times(predictions_dir):
    """Read ``aggregate_time.csv`` of a predictions folder.

    Args:
        predictions_dir: Folder with the prediction CSVs.

    Returns:
        dict: video_id -> {frames_processed, elapsed_seconds, fps, width, height}; empty if the
        file is missing.
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
                # frame size in pixels; only in runs written after #75, needed for face-size bins
                'width':            _int_or_none(row.get('width')),
                'height':           _int_or_none(row.get('height')),
            }
    LOGGER.info('processing_times videos=%d path=%s', len(by_video), path)
    return by_video


def _int_or_none(value):
    """Parse an int from a CSV cell.

    Args:
        value: Cell content, possibly None or blank.

    Returns:
        int or None: The number, or None for blanks and missing columns.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class GroundTruthIndex:
    """Sorted-timestamp index  with binary search over one video's ground truth annotations."""

    def __init__(self, gt_rows):
        """Index the GT rows of one video by timestamp."""
        by_ts = defaultdict(list)
        for r in gt_rows:
            by_ts[r['frame_timestamp']].append(r)

        self.timestamps = sorted(by_ts.keys())
        self._buckets   = [None] * len(self.timestamps)  # parallel to timestamps: list of tuple(_GtBox, ...)
        self.entity_gt_rows = defaultdict(int)     # entity_id -> number of GT boxes
        for i, ts in enumerate(self.timestamps):
            boxes = tuple(
                _GtBox(ts, gi, r['entity_id'], r['asd_label'], r['bbox'], r['ava_label'])
                for gi, r in enumerate(by_ts[ts])
            )
            self._buckets[i] = boxes
            for box in boxes:
                self.entity_gt_rows[box.entity_id] += 1
        self.entities = set(self.entity_gt_rows)
        self.total_gt_boxes = sum(self.entity_gt_rows.values())

    def nearest(self, timestamp, tolerance_s):
        """Return the closest GT frame's boxes within the tolerance.

        Args:
            timestamp: Prediction timestamp in seconds.
            tolerance_s: Max distance in seconds.

        Returns:
            tuple: The GT boxes of the closest frame, or () if none is close enough.
        """
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
        """Iterate every GT box in timestamp order.

        Yields:
            _GtBox: The next GT box.
        """
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

    A pair matches at IoU >= ``iou_threshold`` or containment >= ``CONTAINMENT_THRESHOLD``.

    Args:
        pred_boxes: Predictions of the frame.
        gt_boxes: GT boxes of the frame.
        iou_threshold: Minimum IoU.

    Returns:
        tuple: ``(matches, best_per_gt)``: list of (pred index, gt index), and per GT box the
        best (iou, containment, label, score) of any prediction.
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
        """Start empty counts for the GT of one video (a ``GroundTruthIndex``)."""
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
        """Add one frame's predictions and their matches to the counts.

        Args:
            pred_boxes: Predictions of the frame.
            gt_boxes: GT boxes of the frame.
            matches: (prediction index, GT index) pairs from ``match_frame``.
        """
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

            if label == gt.asd_label:
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
        """Precision of the matched-box classification.

        Returns:
            float: tp / (tp + fp).
        """
        return self.tp / max(1, self.tp + self.fp)

    @property
    def recall(self):
        """Recall of the matched-box classification.

        Returns:
            float: tp / (tp + fn).
        """
        return self.tp / max(1, self.tp + self.fn)

    @property
    def f1(self):
        """F1 of the matched-box classification.

        Returns:
            float: Harmonic mean of precision and recall.
        """
        p, r = self.precision, self.recall
        return 2 * p * r / max(1e-9, p + r)

    @property
    def accuracy(self):
        """Label accuracy on matched boxes (``entity_correct`` / ``entity_matched``).

        Returns:
            float: The accuracy over spatially matched GT boxes.
        """
        correct = sum(self.entity_correct.values())
        matched = sum(self.entity_matched.values())
        return correct / max(1, matched)

    @property
    def missed_detections(self):
        """GT boxes that no prediction matched (detection misses).

        Returns:
            int: The number of missed GT boxes.
        """
        return self.total_gt_boxes - self.matched_boxes

    @property
    def correctly_identified_entities(self):
        """Entities detected spatially and labelled at least once.

        Returns:
            set: Their entity ids.
        """
        return {eid for eid in self.detected_entities
                if self.entity_matched[eid]}


def compute_stats_from_predictions(video_id, predictions_csv, gt_rows, iou_threshold,
                                   timestamp_tolerance_s):
    """Score one video's prediction CSV with the matched-box metrics.

    Args:
        video_id: Video id.
        predictions_csv: Prediction CSV of the video.
        gt_rows: GT rows of the video (``load_ground_truth``).
        iou_threshold: Minimum IoU for a match.
        timestamp_tolerance_s: Max distance between prediction and GT timestamps.

    Returns:
        tuple: ``(Stats, detail_rows)``.
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
    """Build the per-GT-box detail rows, missed boxes included.

    Args:
        gt_index: ``GroundTruthIndex`` of the video.
        match_info: (timestamp, gt index) -> [iou, containment, label, matched, score].

    Returns:
        list[dict]: One row per GT box.
    """
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
            'gt_label':    gt.asd_label,
            'pred_label':  pred_label,
            'score':       f'{score:.6f}',
            'matched':     matched,
        })
    return rows


# ── average precision (official AVA scorer) ───────────────────────────────────


_AVA_EVAL_MODULE = None


def _ava_eval():
    """Import the official AVA active-speaker scorer by path, once per process.

    Returns:
        module: The scorer module.
    """
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

    Args:
        video_id: Video id.
        gt_index: ``GroundTruthIndex`` of the video.
        match_info: (timestamp, gt index) -> [iou, containment, label, matched, score].

    Returns:
        list[tuple]: One row per scored GT box.
    """
    rows = []
    for gt in gt_index.all_boxes():
        record = match_info.get((gt.timestamp, gt.index))
        if record is None or not record[3] or record[4] < 0.0:
            continue                        # unmatched, or the classifier had no prediction yet
        rows.append((video_id, gt.timestamp, *gt.bbox, gt.ava_label, gt.entity_id, record[4]))
    return rows


def average_precision(ap_rows):
    """AP for one video, run through the official scorer's own merge and PR code.

    Args:
        ap_rows: AVA rows (8 GT columns + score) of the matched GT boxes.

    Returns:
        float: The AP, NaN without positives.
    """
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
    
    Args:
        scores: Scores of all scored boxes.
        positives: True for SPEAKING_AUDIBLE boxes.

    Returns:
        float: The pooled AP, NaN without positives.
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
    """Format an AP for the CSV.

    Args:
        ap: The AP.

    Returns:
        str: 4 decimals, or blank if the AP is NaN.
    """
    return '' if np.isnan(ap) else f'{ap:.4f}'



# ── comparable protocols (#75) ────────────────────────────────────────────────
#
# Both protocols give every GT box exactly one score and rank them with the official AVA scorer:
#
#   gt        The classifier ran on the ground-truth boxes (``--boxes gt``). Predictions carry the
#             GT ``entity_id`` and are joined on (timestamp, entity_id). This is the protocol of
#             every AVA / UniTalk / WASD paper.
#   detector  End-to-end mAP@0.5 (Ego4D protocol). Detections are matched one-to-one to GT boxes
#             at IoU >= 0.5 (greedy, highest IoU first). Detections without a GT box in an
#             annotated frame are added to the ranking as negatives (false positives).
#
# A GT box without a score (missed by the detector, or no prediction yet during buffer warm-up)
# gets score 0, the lowest possible score, exactly as the official script requires a score for
# every GT row.

E2E_IOU_THRESHOLD = 0.5
DECISION_THRESHOLD = 0.5
FACE_SIZE_BINS = (('small', 0, 64), ('medium', 64, 128), ('large', 128, float('inf')))
_MISSING_SCORE = 0.0


def face_size_bin(width_px):
    """Return the AVA face-size bin of a face.

    Args:
        width_px: Face width in pixels, or None/NaN if unknown.

    Returns:
        str: 'small', 'medium', 'large', or '' if unknown.
    """
    if width_px is None or np.isnan(width_px):
        return ''
    for name, lo, hi in FACE_SIZE_BINS:
        if lo <= width_px < hi:
            return name
    return ''


def _match_iou(pred_boxes, gt_boxes, iou_threshold):
    """One-to-one greedy matching by IoU only, highest IoU first.

    Args:
        pred_boxes: Predictions of the frame.
        gt_boxes: GT boxes of the frame.
        iou_threshold: Minimum IoU.

    Returns:
        dict: GT index -> prediction index.
    """
    pairs = []
    for pi, pred in enumerate(pred_boxes):
        for gi, gt in enumerate(gt_boxes):
            iou, _ = _overlap(pred.coordinates, gt.bbox)
            if iou >= iou_threshold:
                pairs.append((iou, pi, gi))
    pairs.sort(key=lambda p: -p[0])
    used_p, used_g, out = set(), set(), {}
    for _iou, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        out[gi] = pi
    return out


def protocol_rows(video_id, gt_index, preds_by_ts, timestamp_tolerance_s,
                  iou_threshold=E2E_IOU_THRESHOLD, frame_width=None):
    """Assign one score to every GT box and collect unmatched detections.

    The protocol is chosen from the predictions: rows with an ``entity_id`` come from a
    ``--boxes gt`` run and are joined on it, otherwise detections are matched by IoU.

    Args:
        video_id: Video id, written into every row.
        gt_index: ``GroundTruthIndex`` of the video.
        preds_by_ts: dict timestamp -> list of ``_PredBox`` (``load_predictions_csv``).
        timestamp_tolerance_s: Max distance between a GT and a prediction timestamp.
        iou_threshold: IoU needed for a detection to match a GT box.
        frame_width: Frame width in pixels, for the face-size bins; None if unknown.

    Returns:
        tuple: ``(protocol, gt_df, fp_scores, n_detections)``. ``protocol`` is 'gt' or
        'detector'. ``gt_df`` has one row per GT box with the AVA columns plus ``score``,
        ``has_score``, ``detected`` and ``size_bin``. ``fp_scores`` are the scores of detections
        in annotated frames that match no GT box. ``n_detections`` counts all detections in
        annotated frames.
    """
    pred_ts = np.array(sorted(preds_by_ts))
    protocol = 'gt' if any(p.entity_id for boxes in preds_by_ts.values() for p in boxes) \
        else 'detector'

    rows, fp_scores, n_det = [], [], 0
    for i, ts in enumerate(gt_index.timestamps):
        gt_boxes = gt_index._buckets[i]
        preds = ()
        if len(pred_ts):
            j = int(np.clip(np.searchsorted(pred_ts, ts), 1, len(pred_ts) - 1)) \
                if len(pred_ts) > 1 else 0
            cand = [k for k in (j - 1, j) if 0 <= k < len(pred_ts)]
            k = min(cand, key=lambda k: abs(pred_ts[k] - ts))
            if abs(pred_ts[k] - ts) <= timestamp_tolerance_s:
                preds = preds_by_ts[pred_ts[k]]

        if protocol == 'gt':
            by_entity = {p.entity_id: p for p in preds}
            assigned = {gi: by_entity.get(gt.entity_id) for gi, gt in enumerate(gt_boxes)}
        else:
            match = _match_iou(preds, gt_boxes, iou_threshold)
            assigned = {gi: preds[pi] for gi, pi in match.items()}
            n_det += len(preds)
            matched_p = {id(p) for p in assigned.values()}
            fp_scores.extend(p.score for p in preds
                             if id(p) not in matched_p and p.score >= 0.0)

        for gi, gt in enumerate(gt_boxes):
            pred = assigned.get(gi)
            has_score = pred is not None and pred.score >= 0.0   # -1 = no prediction yet
            width_px = (gt.bbox[2] - gt.bbox[0]) * frame_width if frame_width else float('nan')
            rows.append((video_id, gt.timestamp, *gt.bbox, gt.ava_label, gt.entity_id,
                         pred.score if has_score else _MISSING_SCORE, has_score,
                         pred is not None, face_size_bin(width_px)))

    gt_df = pd.DataFrame(rows, columns=_AVA_COLUMNS + ['score', 'has_score', 'detected',
                                                       'size_bin'])
    return protocol, gt_df, np.asarray(fp_scores, dtype=float), n_det


def official_map(gt_df, fp_scores=()):
    """Compute the mAP with the official AVA scorer's own merge, precision/recall and AP code.

    Every GT row must carry a score. Unmatched detections (``fp_scores``) are appended as
    NOT_SPEAKING rows, which the official precision/recall code counts as false positives.

    Args:
        gt_df: GT rows with the AVA columns and ``score``.
        fp_scores: Scores of detections without a GT box.

    Returns:
        float: The mAP, NaN if there is no positive GT box.
    """
    if gt_df.empty or not (gt_df['label'] == _AVA_POSITIVE).any():
        return float('nan')
    ava = _ava_eval()
    df = gt_df[_AVA_COLUMNS + ['score']].copy()
    df['uid'] = df['frame_timestamp'].map(str) + ':' + df['entity_id'].map(str) + ':' \
        + df['video_id'].map(str)
    df_merged = ava.merge_groundtruth_and_predictions(
        df.drop(columns='score'), df.assign(label=_AVA_POSITIVE))
    if len(fp_scores):
        fp = pd.DataFrame({'uid': [f'fp:{i}' for i in range(len(fp_scores))],
                           'score': fp_scores, 'label_groundtruth': 'NOT_SPEAKING',
                           'label_prediction': _AVA_POSITIVE})
        df_merged = pd.concat([df_merged[['uid', 'score', 'label_groundtruth',
                                          'label_prediction']], fp])
        df_merged = df_merged.sort_values(by=['score'], ascending=False).reset_index(drop=True)
    precision, recall = ava.calculate_precision_recall(df_merged)
    return float(ava.compute_average_precision(precision, recall))


def roc_auc(scores, positives):
    """Area under the ROC curve (Mann-Whitney U with tie correction).

    Args:
        scores: Scores.
        positives: True for positive samples.

    Returns:
        float: The AUC, NaN if one class is empty.
    """
    scores = np.asarray(scores, dtype=float)
    positives = np.asarray(positives, dtype=bool)
    n_pos, n_neg = positives.sum(), (~positives).sum()
    if not n_pos or not n_neg:
        return float('nan')
    ranks = pd.Series(scores).rank(method='average').to_numpy()
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def protocol_metrics(gt_df, fp_scores, n_detections, protocol,
                     decision_threshold=DECISION_THRESHOLD):
    """Headline metrics of one protocol over the given GT rows and unmatched detections.

    Threshold metrics treat a GT box without a score as predicted not-speaking and every
    unmatched detection above the threshold as a false positive.

    Returns:
        dict: ``map``, ``auc``, ``precision``, ``recall``, ``f1``, ``balanced_accuracy``,
        ``coverage`` (share of GT boxes with a score), detector ``det_recall`` /
        ``det_precision`` (detector protocol only) and per face-size bin ``map_<bin>``,
        ``det_recall_<bin>`` and ``n_<bin>``.
    """
    fp_scores = np.asarray(fp_scores, dtype=float)
    pos = (gt_df['label'] == _AVA_POSITIVE).to_numpy()
    scores = gt_df['score'].to_numpy(dtype=float)
    pred_pos = scores > decision_threshold
    fp_pos = fp_scores > decision_threshold
    tp = int((pred_pos & pos).sum())
    fp = int((pred_pos & ~pos).sum() + fp_pos.sum())
    fn = int((~pred_pos & pos).sum())
    tn = int((~pred_pos & ~pos).sum() + (~fp_pos).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    out = {
        'protocol':          protocol,
        'map':               official_map(gt_df, fp_scores),
        'auc':               roc_auc(np.concatenate([scores, fp_scores]),
                                     np.concatenate([pos, np.zeros(len(fp_scores), bool)])),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision':         precision,
        'recall':            recall,
        'f1':                2 * precision * recall / max(1e-9, precision + recall),
        'balanced_accuracy': 0.5 * (recall + tn / max(1, tn + fp)),
        'coverage':          float(gt_df['has_score'].mean()) if len(gt_df) else float('nan'),
        'gt_boxes':          len(gt_df),
        'gt_positives':      int(pos.sum()),
    }
    if protocol == 'detector':
        detected = gt_df['detected'].to_numpy(dtype=bool)
        out['det_recall'] = float(detected.mean()) if len(gt_df) else float('nan')
        out['det_precision'] = (float(detected.sum() / n_detections) if n_detections
                                else float('nan'))
        out['dropped_faces'] = int((~detected).sum())
    for name, _lo, _hi in FACE_SIZE_BINS:
        sub = gt_df[gt_df['size_bin'] == name]
        out[f'n_{name}'] = len(sub)
        # per-bin mAP ranks the bin's GT boxes only, as in the AVA / LoCoNet face-size tables
        out[f'map_{name}'] = official_map(sub) if len(sub) else float('nan')
        if protocol == 'detector':
            out[f'det_recall_{name}'] = (float(sub['detected'].mean()) if len(sub)
                                         else float('nan'))
    return out


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
    """Write one CSV holding a summary row per video."""
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
            # runs without aggregate_time.csv still get scored, with blank timing columns
            timing = times_by_video.get(video_id) or {
                'frames_processed': 0, 'elapsed_seconds': 0.0, 'fps': 0.0}
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
        avg_fps = tot_frames / tot_elapsed if tot_elapsed else 0.0
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




def _score_video(vid, predictions_csv, gt_rows, iou_threshold, tol_s, result_dir,
                 frame_width=None):
    """Worker: score one video, write its detail CSV.

    Returns:
        tuple: ``(vid, Stats, protocol, gt_df, fp_scores, n_detections)``; the last four are the
        comparable-protocol inputs from ``protocol_rows``.
    """
    stats, detail_rows = compute_stats_from_predictions(
        vid, predictions_csv, gt_rows, iou_threshold, tol_s)
    write_detail_csv(result_dir, vid, detail_rows)
    protocol, gt_df, fp_scores, n_det = protocol_rows(
        vid, GroundTruthIndex(gt_rows), load_predictions_csv(predictions_csv), tol_s,
        E2E_IOU_THRESHOLD, frame_width)
    return vid, stats, protocol, gt_df, fp_scores, n_det


_PROTOCOL_FIELDS = ['video_id', 'protocol', 'map', 'auc', 'precision', 'recall', 'f1',
                    'balanced_accuracy', 'tp', 'fp', 'fn', 'tn', 'coverage', 'gt_boxes',
                    'gt_positives', 'det_recall', 'det_precision', 'dropped_faces'] + \
    [f'{m}_{b}' for b, _lo, _hi in FACE_SIZE_BINS for m in ('n', 'map', 'det_recall')]


def _fmt(value):
    """Format a metric for the CSV.

    Args:
        value: The metric.

    Returns:
        The value with 4 decimals for floats, blank for NaN/None, unchanged otherwise.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ''
    if isinstance(value, float):
        return f'{value:.4f}'
    return value


def write_protocol_results(result_dir, per_video_protocol, times_by_video=None, run_info=None):
    """Write ``protocol_results.csv`` (per video + pooled), the AVA CSVs and ``summary.json``.

    The pooled row ranks the GT boxes of all videos together, which is how papers report mAP
    on a validation set. ``ava_groundtruth.csv`` / ``ava_predictions.csv`` let anyone re-run the
    official script on a ``gt`` run:
    ``python src/eval/get_ava_active_speaker_performance.py -g ava_groundtruth.csv -p
    ava_predictions.csv``.

    Args:
        result_dir: Output directory.
        per_video_protocol: List of ``(video_id, protocol, gt_df, fp_scores, n_detections)``.
        times_by_video: Output of ``load_processing_times``, for the speed numbers.
        run_info: Extra key/values for ``summary.json`` (dataset, model, hardware, ...).

    Returns:
        dict: The pooled metrics, as written to ``summary.json``.
    """
    import json
    if not per_video_protocol:
        return {}
    os.makedirs(result_dir, exist_ok=True)
    protocols = {p for _v, p, *_ in per_video_protocol}
    if len(protocols) > 1:
        LOGGER.warning('mixed protocols in one predictions_dir: %s', protocols)
    protocol = sorted(protocols)[0]

    rows = []
    for vid, prot, gt_df, fp_scores, n_det in sorted(per_video_protocol, key=lambda r: r[0]):
        rows.append({'video_id': vid, **protocol_metrics(gt_df, fp_scores, n_det, prot)})
    all_gt = pd.concat([r[2] for r in per_video_protocol], ignore_index=True)
    all_fp = np.concatenate([r[3] for r in per_video_protocol])
    n_det = sum(r[4] for r in per_video_protocol)
    pooled = {'video_id': 'pooled', **protocol_metrics(all_gt, all_fp, n_det, protocol)}
    rows.append(pooled)

    with open(os.path.join(result_dir, 'protocol_results.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_PROTOCOL_FIELDS, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k)) for k in _PROTOCOL_FIELDS})

    if protocol == 'gt':
        all_gt[_AVA_COLUMNS].to_csv(os.path.join(result_dir, 'ava_groundtruth.csv'),
                                    header=False, index=False)
        all_gt[_AVA_COLUMNS].assign(label=_AVA_POSITIVE, score=all_gt['score']).to_csv(
            os.path.join(result_dir, 'ava_predictions.csv'), header=False, index=False)

    times = [t for t in (times_by_video or {}).values()]
    frames = sum(t['frames_processed'] for t in times)
    elapsed = sum(t['elapsed_seconds'] for t in times)
    summary = {**(run_info or {}),
               'videos': len(per_video_protocol),
               **{k: (None if isinstance(v, float) and np.isnan(v) else v)
                  for k, v in pooled.items() if k != 'video_id'},
               'frames_processed': frames,
               'elapsed_seconds': round(elapsed, 3),
               'fps': round(frames / elapsed, 3) if elapsed else None}
    with open(os.path.join(result_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    LOGGER.info('protocol=%s pooled mAP=%s', protocol, _fmt(pooled['map']))
    return summary


def run_stats(predictions_dir, groundtruth_csv, result_dir, video=None, iou_threshold=0.5,
              timestamp_tolerance_ms=TIMESTAMP_TOLERANCE_MS, workers=None, verbose=False,
              log_dir='logs_stats', run_info=None):
    """Score every prediction CSV in predictions_dir and write detail + aggregate CSVs.

    Also writes the comparable-protocol results (``protocol_results.csv``, ``summary.json``),
    see ``write_protocol_results``.

    Args:
        predictions_dir: Folder with the prediction CSVs and ``aggregate_time.csv``.
        groundtruth_csv: AVA-format GT CSV.
        result_dir: Output folder.
        video: Score only this video id.
        iou_threshold: Minimum IoU for the matched-box metrics.
        timestamp_tolerance_ms: Max prediction/GT timestamp offset.
        workers: Parallel scoring processes (None = all CPUs).
        verbose: Log INFO to the console.
        log_dir: Folder for the log file.
        run_info: Copied into ``summary.json``.

    Returns:
        dict: The summary (pooled metrics and run info).
    """
    log_path = setup_logging('unitalk_stats', verbose, log_dir)
    tol_s = timestamp_tolerance_ms / 1000.0
    LOGGER.info('stats run start predictions_dir=%s gt=%s tol_ms=%.1f log=%s',
                predictions_dir, groundtruth_csv,
                timestamp_tolerance_ms, log_path)

    gt_by_video = load_ground_truth(groundtruth_csv)
    times_by_video = load_processing_times(predictions_dir)
    os.makedirs(result_dir, exist_ok=True)

    if video:
        video_ids = [video]
    else:
        video_ids = [
            os.path.splitext(f)[0]
            for f in os.listdir(predictions_dir)
            if f.endswith('.csv') and os.path.splitext(f)[0] in gt_by_video
        ]

    tasks = []
    for vid in video_ids:
        predictions_csv = os.path.join(predictions_dir, f'{vid}.csv')
        if not os.path.isfile(predictions_csv):
            LOGGER.warning('Skipping video=%s reason=predictions_not_found path=%s',
                           vid, predictions_csv)
            continue
        gt_rows = gt_by_video.get(vid)
        if not gt_rows:
            LOGGER.warning('Skipping video=%s reason=no_ground_truth', vid)
            continue
        tasks.append((vid, predictions_csv, gt_rows, iou_threshold, tol_s,
                      result_dir, (times_by_video.get(vid) or {}).get('width')))

    # Score videos in parallel
    per_video = []
    per_video_protocol = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_score_video, *task): task[0] for task in tasks}
        for future in as_completed(futures):
            vid = futures[future]
            try:
                vid, stats, protocol, gt_df, fp_scores, n_det = future.result()
                per_video.append((vid, stats))
                per_video_protocol.append((vid, protocol, gt_df, fp_scores, n_det))
            except Exception:
                LOGGER.exception('Failed scoring video=%s', vid)

    write_aggregate_csv(result_dir, per_video, times_by_video)
    summary = write_protocol_results(result_dir, per_video_protocol, times_by_video, run_info)
    LOGGER.info('stats run complete videos_scored=%d', len(per_video))
    return summary


def main():
    """Command-line entry point: parse flags and run the scoring."""
    args = parse_args()
    run_stats(args.predictions_dir, args.groundtruth_csv, args.result_dir,
              video=args.video, iou_threshold=args.iou_threshold,
              timestamp_tolerance_ms=args.timestamp_tolerance_ms,
              workers=args.workers, verbose=args.verbose)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
