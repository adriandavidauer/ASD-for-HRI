
import csv
import logging
import os
from collections import defaultdict, namedtuple
from datetime import datetime
import subprocess

import cv2
import numpy as np
import pandas as pd
from paz.backend.boxes import compute_iou as _paz_iou

LOGGER = logging.getLogger(__name__)

CONTAINMENT_THRESHOLD = 0.5  # accept a match when the smaller box is ≥50% covered

_LABEL_MAP = {
    'SPEAKING_AUDIBLE': 'speaking',
    'NOT_SPEAKING':     'not-speaking',
}

_PredBox = namedtuple('_PredBox', ['coordinates', 'class_name'])


def setup_logging(log_name="unitalk", verbose=False):
    os.makedirs('/app/data/logs', exist_ok=True)
    path = f'/app/data/logs/{log_name}_{datetime.now():%Y%m%d_%H%M%S}.log'

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    fh = logging.FileHandler(path, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO if verbose else logging.WARNING)
    ch.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    root.addHandler(ch)

    LOGGER.info('logging initialised file=%s', path)
    return path


class FFmpegWriter:
    def __init__(self, path, fps, width, height):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self._proc = subprocess.Popen(
            ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
             '-s', f'{width}x{height}', '-pix_fmt', 'bgr24', '-r', str(fps),
             '-i', 'pipe:', '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
             '-preset', 'fast', '-crf', '23', path],
            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        LOGGER.debug('output_video codec=libx264 path=%s', path)

    def write(self, frame_bgr):
        self._proc.stdin.write(frame_bgr.tobytes())

    def release(self):
        try:
            self._proc.stdin.close()
        except BrokenPipeError:
            pass
        self._proc.wait()


def create_writer(requested_path, fps, width, height):
    path = os.path.splitext(requested_path)[0] + '.mp4'
    return FFmpegWriter(path, fps, width, height), path


def draw_pipeline_frame(frame_bgr, frame_idx, pred_boxes):
    """Draw pipeline-detected boxes on a BGR frame in place (no GT comparison).

    Colour by predicted label: green = speaking, red = not-speaking,
    white = unlabelled.  A small footer shows frame_idx and detection count.
    """
    h, w = frame_bgr.shape[:2]
    for pred in pred_boxes:
        label = getattr(pred, 'class_name', '') or ''
        x1, y1, x2, y2 = (int(v) for v in pred.coordinates)
        color = {
            'speaking':     (0, 255, 0),
            'not-speaking': (0, 0, 255),
        }.get(label, (255, 255, 255))
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        tx = max(0, min(x1, w - 120))
        cv2.putText(frame_bgr, label or '?', (tx, max(20, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    cv2.putText(frame_bgr, f'frame={frame_idx} detections={len(pred_boxes)}',
                (10, max(20, h - 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


def draw_annotated_frame(out_bgr, frame_idx, gt_boxes, pred_boxes, matches, iou_matrix):
    """Draw GT boxes, prediction boxes and frame summary onto out_bgr in-place.

    Colour scheme (BGR):
      green                ground truth
      red    (0,0,255)     matched to GT, correct label
      yellow (0,255,255)   matched to GT, wrong label
      black  (0,0,0)       unmatched prediction
    """
    matched_gt = {gi for _, gi, _ in matches}
    n_correct   = sum(1 for pi, gi, _ in matches
                      if getattr(pred_boxes[pi], 'class_name', None) == gt_boxes[gi]['vvad_label'])

    pred_best = {
        pi: (max((iou_matrix[(pi, gi)] for gi in range(len(gt_boxes))), key=lambda s: s['iou'])
             if gt_boxes else {'iou': 0.0, 'containment': 0.0})
        for pi in range(len(pred_boxes))
    }
    pred_status = {pi: ('correct' if getattr(pred_boxes[pi], 'class_name', None) == gt_boxes[gi]['vvad_label'] else 'wrong')
                   for pi, gi, _ in matches}
    for pi in range(len(pred_boxes)):
        pred_status.setdefault(pi, 'unmatched')

    for gi, gt in enumerate(gt_boxes):
        x1, y1, x2, y2 = (int(v) for v in gt['bbox_pixel'])
        color = (0, 255, 0)
        cv2.rectangle(out_bgr, (x1, y1), (x2, y2), color, 2)
        if gi in matched_gt:
            cv2.circle(out_bgr, (x1 + 6, y1 + 6), 5, color, -1)
        else:
            cv2.circle(out_bgr, (x1 + 6, y1 + 6), 5, (0, 128, 0), 1)
        cv2.putText(out_bgr, f'GT {gt["entity_id"]} {gt["vvad_label"]}',
                    (x1, max(8, y1 - 13)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)

    for pi, pred in enumerate(pred_boxes):
        pred_label = getattr(pred, 'class_name', None)
        if pred_label is None:
            continue
        x1, y1, x2, y2 = (int(v) for v in pred.coordinates)
        color = {'correct': (0, 0, 255), 'wrong': (0, 255, 255), 'unmatched': (0, 0, 0)}.get(pred_status[pi], (0, 0, 255))
        cv2.rectangle(out_bgr, (x1, y1), (x2, y2), color, 2)
        b = pred_best[pi]
        tx = max(0, min(x1, out_bgr.shape[1] - 190))
        cv2.putText(out_bgr,
                    f'PRED {pred_label} [{pred_status[pi]}] IoU:{b["iou"]:.2f} cont:{b["containment"]:.2f}',
                    (tx, max(20, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1, cv2.LINE_AA)

    n_wrong = len(matches) - n_correct
    n_unmatched = len(pred_boxes) - len(matches)
    cv2.putText(out_bgr,
                f'Frame {frame_idx} | GT:{len(gt_boxes)}(green)  '
                f'correct:{n_correct}(red) wrong:{n_wrong}(yellow) unmatched:{n_unmatched}(black)',
                (10, max(20, out_bgr.shape[0] - 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)


# ── IoU / box matching ────────────────────────────────────────────────────────

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
            s = iou_matrix[(pi, gi)]
            iou = s['iou']
            cont = s['containment']
            if iou > best_iou or (iou == best_iou and cont > best_cont):
                best_iou, best_cont, best_gi = iou, cont, gi
        if best_gi is not None and (
            best_iou >= iou_threshold or best_cont >= CONTAINMENT_THRESHOLD
        ):
            matches.append((pi, best_gi, best_iou))
            used.add(best_gi)
    return matches


def annotate_debug_frame(frame, frame_idx, pred_boxes, gt_boxes, iou_threshold):
    """Match this frame's predictions to GT and draw the annotated overlay in place.
    """
    iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
    matches = match_predictions_to_gt(pred_boxes, gt_boxes, iou_threshold, iou_matrix)
    draw_annotated_frame(frame, frame_idx, gt_boxes, pred_boxes, matches, iou_matrix)


def load_annotations(csv_path):
    """Load a UniTalk per-video annotation CSV from *csv_path* and return a DataFrame.

    """
    df = pd.read_csv(csv_path)
    df['vvad_label'] = df['label'].map(_LABEL_MAP).fillna('not-speaking')
    return df


def build_frame_map(annots_df, fps, width, height, video_id=None):
    """Group annotation rows by frame index for overlay/eval against predictions.

    Maps each annotation timestamp to its nearest frame index at *fps* and scales
    the normalised GT bbox into pixel coords.  Returns dict[frame_idx] -> list of
    {'entity_id', 'vvad_label', 'bbox_pixel'}.

    Args:
        annots_df: DataFrame returned by load_annotations().
        fps:       the video's native frame rate (predictions are keyed by
                   native frame index, so GT must use the same fps).
        width:     video width in pixels (to scale normalised x coords).
        height:    video height in pixels (to scale normalised y coords).
        video_id:  when given, only rows for this video_id are used.
    """
    df = annots_df
    if video_id is not None:
        df = df[df['video_id'] == video_id]

    frame_map = defaultdict(list)
    for row in df.itertuples(index=False):
        fi = int(round(float(row.frame_timestamp) * fps))
        x1, y1 = float(row.entity_box_x1), float(row.entity_box_y1)
        x2, y2 = float(row.entity_box_x2), float(row.entity_box_y2)
        frame_map[fi].append({
            'entity_id':  row.entity_id,
            'vvad_label': row.vvad_label,
            'bbox_pixel': (x1 * width, y1 * height, x2 * width, y2 * height),
        })
    return frame_map


def load_predictions_csv(path):
    """Read a predictions CSV → dict[frame_idx] -> list[_PredBox].
    """
    by_frame = defaultdict(list)
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            fi = int(row['frame_idx'])
            by_frame[fi]
            if row['label']:
                by_frame[fi].append(_PredBox(
                    (float(row['x1']), float(row['y1']),
                     float(row['x2']), float(row['y2'])),
                    row['label'],
                ))
    return dict(by_frame)
