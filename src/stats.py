"""Compute and report VVAD evaluation stats from per-video prediction CSVs.

Reads predictions CSVs written by run_vvad_on_unitalk_video.py and matches
them against UniTalk ground-truth annotations to produce precision/recall/F1,
detection and timestamp accuracy, per-video result files and an aggregate
text result file.  No pipeline / model code here.
"""

import csv
import os
import argparse
import logging
from collections import defaultdict, namedtuple
from datetime import datetime

import cv2
import numpy as np
from paz.backend.boxes import compute_iou as _paz_iou

CONTAINMENT_THRESHOLD = 0.5          # accept match when smaller box is ≥50% covered
CONSECUTIVE_FRAME_THRESHOLD = 38     # entities below this max consecutive run are flagged
LOGGER = logging.getLogger('UniTalk_VVAD')

_LABEL_MAP = {
    'SPEAKING_AUDIBLE':       'speaking',
    'NOT_SPEAKING':           'not-speaking',
}

_PredBox = namedtuple('_PredBox', ['coordinates', 'class_name'])


# ── helpers ──────────────────────────────────────────────────────────────────

def _max_consecutive_run(sorted_frame_indices):
    """Longest run of annotation frames with no gap larger than 1.5x the typical step."""
    if not sorted_frame_indices:
        return 0
    if len(sorted_frame_indices) == 1:
        return 1
    diffs = [sorted_frame_indices[i + 1] - sorted_frame_indices[i]
             for i in range(len(sorted_frame_indices) - 1)]
    positive_diffs = [d for d in diffs if d > 0]
    step = min(positive_diffs) if positive_diffs else 1
    tolerance = step * 1.5
    max_run = cur = 1
    for d in diffs:
        if d <= tolerance:
            cur += 1
        else:
            max_run = max(max_run, cur)
            cur = 1
    return max(max_run, cur)




# ── data loading ─────────────────────────────────────────────────────────────





# ── IoU / box matching ───────────────────────────────────────────────────────

def _iou(a, b):
    return float(_paz_iou(np.array(a, np.float32), np.array([b], np.float32))[0])


def _containment(a, b):
    """Fraction of the smaller box covered by the intersection."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1e-6, min(area_a, area_b))


def compute_iou_matrix(pred_boxes, gt_boxes):
    """Compute IoU and containment for every GT↔pred pair."""
    scores = {}
    for pi, pred in enumerate(pred_boxes):
        pc = pred.coordinates
        for gi, gt in enumerate(gt_boxes):
            gc = gt['bbox_pixel']
            scores[(pi, gi)] = {
                'iou':         _iou(pc, gc),
                'containment': _containment(pc, gc),
            }
    return scores


def match_predictions_to_gt(pred_boxes, gt_boxes, iou_threshold, iou_matrix=None):
    """Greedy highest-IoU matching; accept on IoU≥thresh OR containment≥CONTAINMENT_THRESHOLD."""
    if iou_matrix is None:
        iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
    matches, used = [], set()
    for pi, _pred in enumerate(pred_boxes):
        best_iou = best_cont = 0.0
        best_gi = None
        for gi in range(len(gt_boxes)):
            if gi in used:
                continue
            s    = iou_matrix[(pi, gi)]
            iou  = s['iou']
            cont = s['containment']
            if iou > best_iou or (iou == best_iou and cont > best_cont):
                best_iou, best_cont, best_gi = iou, cont, gi
        if best_gi is not None and (
            best_iou >= iou_threshold or best_cont >= CONTAINMENT_THRESHOLD
        ):
            matches.append((pi, best_gi, best_iou))
            used.add(best_gi)
    return matches


def _build_frame_map(annots, fps, width, height, video_id=None):
    """Map each annotation timestamp to its nearest frame index at the given fps.

    The runner writes predictions at TARGET_FPS, so the caller should pass
    TARGET_FPS here so GT timestamps align with CSV frame_idx values.
    """
    frame_map = defaultdict(list)
    skipped   = 0
    for ann in annots:
        if video_id is not None and ann.get('video_id') != video_id:
            skipped += 1
            continue
        fi = int(round(ann['timestamp'] * fps))
        x1, y1, x2, y2 = ann['bbox']
        frame_map[fi].append({
            **ann,
            'bbox_pixel': (x1 * width, y1 * height, x2 * width, y2 * height),
        })
    if skipped:
        LOGGER.warning('frame_map dropped=%d annotations video_id=%s', skipped, video_id)
    return frame_map


# ── stats accumulator ────────────────────────────────────────────────────────

class Stats:
    """Accumulates per-frame evaluation results.

    Two update paths:
      record_frame()  – called for every frame fed to the pipeline.
      update()        – called only for frames that also have GT annotations.
    """

    def __init__(self, all_annots, total_video_frames):
        self.total_video_frames = total_video_frames

        self.frames_fed_to_pipeline      = 0
        self.frames_with_predictions     = 0
        self.total_predictions           = 0
        self.no_pred_frames_without_gt   = 0

        self.preds_on_annot_frames    = 0
        self.correct_predictions      = 0

        # binary classification (positive class = speaking)
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

        self.gt_entities             = set(a['entity_id'] for a in all_annots)
        self.detected_entities       = set()
        self.entity_matched          = defaultdict(int)
        self.entity_correct          = defaultdict(int)
        self.entity_gt_rows          = defaultdict(int)
        self.entity_gt_frame_indices = defaultdict(list)
        for a in all_annots:
            self.entity_gt_rows[a['entity_id']] += 1

        self.total_annotated_frames      = 0
        self.annot_frames_no_predictions = 0
        self.correct_timestamps          = 0

    def record_frame(self, pred_boxes, has_gt=False):
        self.frames_fed_to_pipeline += 1
        if pred_boxes:
            self.frames_with_predictions += 1
            self.total_predictions += len(pred_boxes)
        elif not has_gt:
            self.no_pred_frames_without_gt += 1

    def update(self, pred_boxes, gt_boxes_pixel, matches, frame_idx=None):
        self.total_annotated_frames += 1
        self.preds_on_annot_frames  += len(pred_boxes)
        if not pred_boxes:
            self.annot_frames_no_predictions += 1

        matched_pi = {pi for pi, gi, _ in matches}

        for gt in gt_boxes_pixel:
            eid = gt['entity_id']
            if frame_idx is not None:
                self.entity_gt_frame_indices[eid].append(frame_idx)

        correctly_predicted_gt = set()

        if pred_boxes:
            for pi, gi, _iou in matches:
                pred_label = getattr(pred_boxes[pi], 'class_name', None)
                gt         = gt_boxes_pixel[gi]
                gt_label   = gt['vvad_label']
                eid        = gt['entity_id']

                self.detected_entities.add(eid)
                self.entity_matched[eid] += 1

                if pred_label == gt_label:
                    self.correct_predictions += 1
                    self.entity_correct[eid] += 1
                    correctly_predicted_gt.add(gi)
                    if pred_label == 'speaking':
                        self.tp += 1
                    else:
                        self.tn += 1
                else:
                    if pred_label == 'speaking':
                        self.fp += 1
                    else:
                        self.fn += 1

            for pi in range(len(pred_boxes)):
                if pi not in matched_pi:
                    pred_label = getattr(pred_boxes[pi], 'class_name', None)
                    if pred_label == 'speaking':
                        self.fp += 1
                    elif pred_label == 'not-speaking':
                        self.tn += 1

        if len(correctly_predicted_gt) == len(gt_boxes_pixel):
            self.correct_timestamps += 1

    @property
    def prediction_accuracy_pct(self):
        return 100.0 * self.correct_predictions / max(1, self.preds_on_annot_frames)

    @property
    def timestamp_accuracy_pct(self):
        return 100.0 * self.correct_timestamps / max(1, self.total_annotated_frames)

    @property
    def entity_detection_rate(self):
        return len(self.detected_entities) / max(1, len(self.gt_entities))

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

    def entities_below_consecutive_threshold(self, threshold=CONSECUTIVE_FRAME_THRESHOLD):
        result = []
        for eid in sorted(self.gt_entities):
            frames  = sorted(self.entity_gt_frame_indices.get(eid, []))
            max_run = _max_consecutive_run(frames)
            if max_run < threshold:
                result.append((eid, max_run))
        return result


# ── scoring entry point ──────────────────────────────────────────────────────

def compute_stats_from_predictions(predictions_csv, video_path, annots,
                                   iou_threshold, video_id=None):
    """Replay predictions against GT and return a populated Stats object.

    Predictions are written at TARGET_FPS; GT timestamps are mapped to frame
    indices at TARGET_FPS so CSV rows and GT annotations align.  The video
    file is read only for width/height (used to scale normalised GT bboxes
    into pixel coords) and total frame count.

    Args:
        predictions_csv: per-video predictions CSV produced by the runner.
        video_path:      source video file.
        annots:          GT annotation rows for this video.
        iou_threshold:   IoU threshold for GT↔prediction matching.
        video_id:        when supplied, filters annotations whose video_id differs.
    """
    if video_id is None and annots:
        video_id = annots[0].get('video_id')

    fps, width, height, total_video_frames = read_video_metadata(video_path)
    by_frame  = load_predictions_csv(predictions_csv)
    frame_map = _build_frame_map(annots, fps, width, height, video_id=video_id)
    stats     = Stats(annots, total_video_frames)

    for frame_idx in sorted(by_frame):
        pred_boxes = by_frame[frame_idx]
        has_gt = frame_idx in frame_map
        stats.record_frame(pred_boxes, has_gt=has_gt)
        if has_gt:
            gt_boxes   = frame_map[frame_idx]
            iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
            matches    = match_predictions_to_gt(pred_boxes, gt_boxes,
                                                 iou_threshold, iou_matrix)
            stats.update(pred_boxes, gt_boxes, matches, frame_idx)
    return stats


# ── logging helpers ──────────────────────────────────────────────────────────

def log_video_stats(video_id, video_path, output_path, stats, elapsed):
    """Emit per-video summary as flat structured log lines."""
    missed_pct = 100.0 * stats.annot_frames_no_predictions / max(1, stats.total_annotated_frames)
    below      = stats.entities_below_consecutive_threshold()

    LOGGER.info('video=%s input=%s elapsed=%.1fs', video_id, video_path, elapsed)
    if output_path is not None:
        LOGGER.info('video=%s output=%s', video_id, output_path)
    LOGGER.info('video=%s frames total=%d fed=%d with_pred=%d no_pred_with_gt=%d no_pred_excluded=%d total_predictions=%d',
                video_id, stats.total_video_frames, stats.frames_fed_to_pipeline,
                stats.frames_with_predictions, stats.annot_frames_no_predictions,
                stats.no_pred_frames_without_gt, stats.total_predictions)
    LOGGER.info('video=%s accuracy preds_on_annot=%d correct=%d pct=%.2f',
                video_id, stats.preds_on_annot_frames, stats.correct_predictions,
                stats.prediction_accuracy_pct)
    LOGGER.info('video=%s metrics tp=%d tn=%d fp=%d fn=%d precision=%.3f recall=%.3f f1=%.3f',
                video_id, stats.tp, stats.tn, stats.fp, stats.fn,
                stats.precision, stats.recall, stats.f1)
    LOGGER.info('video=%s missed_detections=%d/%d pct=%.1f',
                video_id, stats.annot_frames_no_predictions,
                stats.total_annotated_frames, missed_pct)
    LOGGER.info('video=%s entities detected=%d/%d pct=%.1f below_consec_threshold=%d threshold=%d',
                video_id, len(stats.detected_entities), len(stats.gt_entities),
                100 * stats.entity_detection_rate, len(below), CONSECUTIVE_FRAME_THRESHOLD)
    LOGGER.info('video=%s timestamps total=%d correct=%d accuracy=%.2f',
                video_id, stats.total_annotated_frames, stats.correct_timestamps,
                stats.timestamp_accuracy_pct)

    for eid in sorted(stats.gt_entities):
        gt_rows  = stats.entity_gt_rows.get(eid, 0)
        mat      = stats.entity_matched.get(eid, 0)
        cor      = stats.entity_correct.get(eid, 0)
        acc      = cor / mat if mat else 0.0
        detected = eid in stats.detected_entities
        max_run  = _max_consecutive_run(sorted(stats.entity_gt_frame_indices.get(eid, [])))
        LOGGER.debug('video=%s entity=%s detected=%s matched=%d/%d correct=%d/%d acc=%.2f max_consecutive=%d',
                     video_id, eid, detected, mat, gt_rows, cor, mat, acc, max_run)


def log_aggregate_stats(all_stats):
    """Emit aggregate summary across processed videos as flat log lines."""
    if len(all_stats) < 2:
        return
    tvf  = sum(s.total_video_frames          for s in all_stats)
    fwp  = sum(s.frames_with_predictions     for s in all_stats)
    tp   = sum(s.total_predictions           for s in all_stats)
    cp   = sum(s.correct_predictions         for s in all_stats)
    pa   = sum(s.preds_on_annot_frames       for s in all_stats)
    atp  = sum(s.tp                          for s in all_stats)
    atn  = sum(s.tn                          for s in all_stats)
    afp  = sum(s.fp                          for s in all_stats)
    afn  = sum(s.fn                          for s in all_stats)
    det  = sum(len(s.detected_entities)      for s in all_stats)
    tot  = sum(len(s.gt_entities)            for s in all_stats)
    ct   = sum(s.correct_timestamps          for s in all_stats)
    tf   = sum(s.total_annotated_frames      for s in all_stats)
    mnd  = sum(s.annot_frames_no_predictions for s in all_stats)
    npng = sum(s.no_pred_frames_without_gt   for s in all_stats)
    below_count = sum(len(s.entities_below_consecutive_threshold()) for s in all_stats)

    p_agg  = atp / max(1, atp + afp)
    r_agg  = atp / max(1, atp + afn)
    f1_agg = 2 * p_agg * r_agg / max(1e-9, p_agg + r_agg)

    LOGGER.info('aggregate videos=%d frames=%d with_pred=%d total_predictions=%d no_pred_excluded=%d',
                len(all_stats), tvf, fwp, tp, npng)
    LOGGER.info('aggregate accuracy correct=%d/%d pct=%.2f',
                cp, pa, 100 * cp / max(1, pa))
    LOGGER.info('aggregate metrics tp=%d tn=%d fp=%d fn=%d precision=%.3f recall=%.3f f1=%.3f',
                atp, atn, afp, afn, p_agg, r_agg, f1_agg)
    LOGGER.info('aggregate missed_detections=%d/%d pct=%.1f',
                mnd, tf, 100 * mnd / max(1, tf))
    LOGGER.info('aggregate entities detected=%d/%d pct=%.1f below_consec_threshold=%d threshold=%d',
                det, tot, 100 * det / max(1, tot), below_count, CONSECUTIVE_FRAME_THRESHOLD)
    LOGGER.info('aggregate timestamps correct=%d/%d pct=%.2f',
                ct, tf, 100 * ct / max(1, tf))


# ── result file writers ──────────────────────────────────────────────────────

def write_result_file(result_dir, video_id, stats, elapsed):
    """Write a per-video evaluation summary to <result_dir>/<video_id>_results.txt."""
    os.makedirs(result_dir, exist_ok=True)
    path = os.path.join(result_dir, f'{video_id}_results.txt')

    below      = stats.entities_below_consecutive_threshold()
    below_pct  = 100.0 * len(below) / max(1, len(stats.gt_entities))
    missed_pct = 100.0 * stats.annot_frames_no_predictions / max(1, stats.total_annotated_frames)
    no_pred_total   = stats.frames_fed_to_pipeline - stats.frames_with_predictions

    lines = [
        f'Video: {video_id}',
        f'Elapsed: {elapsed:.1f} s',
        '',
        '── Frame overview ────────────────────────────────────────',
        f'  Total frames in video           : {stats.total_video_frames}',
        f'  Frames fed to pipeline          : {stats.frames_fed_to_pipeline}',
        f'  Frames with predictions (>=1)   : {stats.frames_with_predictions}',
        f'  Frames with no predictions      : {no_pred_total}',
        f'    with GT (missed detections)   : {stats.annot_frames_no_predictions}',
        f'    without GT (excluded)         : {stats.no_pred_frames_without_gt}',
        f'  Total predictions returned      : {stats.total_predictions}',
        '',
        '── Prediction accuracy (annotation frames only) ──────────',
        f'  Predictions on annot frames     : {stats.preds_on_annot_frames}',
        f'  Overall correct (TP+TN)         : {stats.correct_predictions} / {stats.preds_on_annot_frames}'
        f'  ({stats.prediction_accuracy_pct:.2f}%)',
        '',
        '── Precision / Recall / F1 (positive class = speaking) ───',
        '      TP    TN    FP    FN    P       R       F1',
        f'      {stats.tp:4d}  {stats.tn:4d}  {stats.fp:4d}  {stats.fn:4d}'
        f'  {stats.precision:.3f}   {stats.recall:.3f}   {stats.f1:.3f}',
        '',
        '── Missed detections ─────────────────────────────────────',
        f'  GT timestamps with 0 detections : {stats.annot_frames_no_predictions}'
        f' / {stats.total_annotated_frames}  ({missed_pct:.1f}%)',
        '',
        '── Entity detection rate ─────────────────────────────────',
        f'  Entities detected : {len(stats.detected_entities)} / {len(stats.gt_entities)}'
        f'  ({100 * stats.entity_detection_rate:.1f}%)',
        '',
        f'── Entity consecutive-frame coverage (threshold={CONSECUTIVE_FRAME_THRESHOLD}) ────',
        f'  Entities with max consecutive run < {CONSECUTIVE_FRAME_THRESHOLD}'
        f' : {len(below)} / {len(stats.gt_entities)}  ({below_pct:.1f}%)',
    ]
    if below:
        lines.append('  Entities below threshold:')
        for eid, max_run in below:
            lines.append(f'    {eid:<32s}  max_consecutive={max_run}')
    lines += [
        '',
        '── Timestamp accuracy ────────────────────────────────────',
        f'  Total timestamps evaluated : {stats.total_annotated_frames}',
        f'  Correct timestamps         : {stats.correct_timestamps}'
        f'  ({stats.timestamp_accuracy_pct:.2f}%)',
        '',
        '── Per-entity breakdown ──────────────────────────────────',
    ]
    for eid in sorted(stats.gt_entities):
        gt_rows  = stats.entity_gt_rows.get(eid, 0)
        mat      = stats.entity_matched.get(eid, 0)
        cor      = stats.entity_correct.get(eid, 0)
        acc      = cor / mat if mat else 0.0
        detected = eid in stats.detected_entities
        max_run  = _max_consecutive_run(sorted(stats.entity_gt_frame_indices.get(eid, [])))
        lines.append(
            f'  {eid:<32s}  detected={str(detected):<5s}'
            f'  matched={mat}/{gt_rows}'
            f'  correct={cor}/{mat} ({100*acc:.0f}%)'
            f'  max_consecutive={max_run}'
        )

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    LOGGER.debug('per_video_results path=%s', path)


def write_aggregate_result_file(result_dir, all_stats, video_ids):
    """Write aggregate stats across all scored videos to <result_dir>/aggregate_results.txt."""
    if not all_stats:
        return
    os.makedirs(result_dir, exist_ok=True)
    path = os.path.join(result_dir, 'aggregate_results.txt')

    tvf  = sum(s.total_video_frames          for s in all_stats)
    fwp  = sum(s.frames_with_predictions     for s in all_stats)
    tp   = sum(s.total_predictions           for s in all_stats)
    cp   = sum(s.correct_predictions         for s in all_stats)
    pa   = sum(s.preds_on_annot_frames       for s in all_stats)
    atp  = sum(s.tp                          for s in all_stats)
    atn  = sum(s.tn                          for s in all_stats)
    afp  = sum(s.fp                          for s in all_stats)
    afn  = sum(s.fn                          for s in all_stats)
    det  = sum(len(s.detected_entities)      for s in all_stats)
    tot  = sum(len(s.gt_entities)            for s in all_stats)
    ct   = sum(s.correct_timestamps          for s in all_stats)
    tf   = sum(s.total_annotated_frames      for s in all_stats)
    mnd  = sum(s.annot_frames_no_predictions for s in all_stats)
    npng = sum(s.no_pred_frames_without_gt   for s in all_stats)
    below_count = sum(len(s.entities_below_consecutive_threshold()) for s in all_stats)

    p_agg  = atp / max(1, atp + afp)
    r_agg  = atp / max(1, atp + afn)
    f1_agg = 2 * p_agg * r_agg / max(1e-9, p_agg + r_agg)

    lines = [
        f'AGGREGATE  ({len(all_stats)} videos)',
        f'Videos: {", ".join(video_ids)}',
        '',
        '── Frame overview ────────────────────────────────────────',
        f'  Total frames in videos          : {tvf}',
        f'  Frames with predictions         : {fwp}',
        f'  Total predictions returned      : {tp}',
        f'  No-pred frames without GT (excl): {npng}',
        '',
        '── Prediction accuracy ───────────────────────────────────',
        f'  Overall accuracy                : {100 * cp / max(1, pa):.2f}%  ({cp} / {pa})',
        '',
        '── Precision / Recall / F1 (positive class = speaking) ───',
        '      TP    TN    FP    FN    P       R       F1',
        f'      {atp:4d}  {atn:4d}  {afp:4d}  {afn:4d}  {p_agg:.3f}   {r_agg:.3f}   {f1_agg:.3f}',
        '',
        '── Missed detections ─────────────────────────────────────',
        f'  GT timestamps with 0 detections : {mnd} / {tf}  ({100 * mnd / max(1, tf):.1f}%)',
        '',
        '── Entity statistics ─────────────────────────────────────',
        f'  Entity detection rate           : {det} / {tot}  ({100 * det / max(1, tot):.1f}%)',
        f'  Entities below {CONSECUTIVE_FRAME_THRESHOLD} consec frames  : {below_count} / {tot}'
        f'  ({100 * below_count / max(1, tot):.1f}%)',
        '',
        '── Timestamp accuracy ────────────────────────────────────',
        f'  Correct timestamps              : {ct} / {tf}  ({100 * ct / max(1, tf):.2f}%)',
    ]

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    LOGGER.debug('aggregate_results path=%s', path)


# ── standalone CLI ───────────────────────────────────────────────────────────

def setup_logging(log_file=None, verbose=False):
    os.makedirs('logs', exist_ok=True)
    path = log_file or f'logs/unitalk_stats_{datetime.now():%Y%m%d_%H%M%S}.log'
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.handlers.clear()
    LOGGER.propagate = False
    fh = logging.FileHandler(path, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    LOGGER.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO if verbose else logging.WARNING)
    ch.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    LOGGER.addHandler(ch)
    return path


def parse_args():
    p = argparse.ArgumentParser(description='Compute VVAD stats from per-video predictions CSVs')
    p.add_argument('--data_dir',        default='data',
                   help='Root data dir containing csv/, videos/, predictions/')
    p.add_argument('--predictions_dir', default=None,
                   help='Override predictions CSV directory (default: <data_dir>/predictions)')
    p.add_argument('--video',           default=None,
                   help='Score a single video_id; omit to score every video with a predictions CSV')
    p.add_argument('--iou_threshold',   type=float, default=0.5)
    p.add_argument('--log_file',        default=None)
    p.add_argument('--verbose', '-v',   action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    log_path = setup_logging(args.log_file, args.verbose)
    LOGGER.info('stats run start data_dir=%s log_file=%s', args.data_dir, log_path)

    csv_path        = os.path.join(args.data_dir, 'csv', 'val_orig.csv')
    video_dir       = os.path.join(args.data_dir, 'videos', 'val')
    predictions_dir = args.predictions_dir or os.path.join(args.data_dir, 'predictions')
    result_dir      = os.path.join(args.data_dir, 'results')
    os.makedirs(result_dir, exist_ok=True)

    all_annots = load_annotations(csv_path)
    video_ids  = [args.video] if args.video else sorted(all_annots)

    all_stats = []
    processed = []
    for vid in video_ids:
        predictions_csv = os.path.join(predictions_dir, f'{vid}_results.csv')
        video_path      = os.path.join(video_dir, f'{vid}.mp4')
        if not os.path.isfile(predictions_csv):
            LOGGER.warning('Skipping video=%s reason=predictions_not_found path=%s',
                           vid, predictions_csv)
            continue
        if not os.path.isfile(video_path):
            LOGGER.warning('Skipping video=%s reason=video_not_found path=%s', vid, video_path)
            continue
        annots = all_annots.get(vid, [])
        if not annots:
            LOGGER.warning('Skipping video=%s reason=no_annotations', vid)
            continue

        LOGGER.info('scoring video=%s gt_rows=%d', vid, len(annots))
        try:
            stats = compute_stats_from_predictions(
                predictions_csv, video_path, annots, args.iou_threshold, video_id=vid,
            )
            log_video_stats(vid, video_path, None, stats, 0.0)
            write_result_file(result_dir, vid, stats, 0.0)
            all_stats.append(stats)
            processed.append(vid)
        except Exception:
            LOGGER.exception('Failed scoring video=%s', vid)

    log_aggregate_stats(all_stats)
    write_aggregate_result_file(result_dir, all_stats, processed)
    LOGGER.info('stats run complete videos_scored=%d', len(processed))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
