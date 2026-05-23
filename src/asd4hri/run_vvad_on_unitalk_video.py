"""Evaluate UniTalk val split with the DetectVVAD pipeline from pypaz.

Every frame is passed through DetectVVAD().  At frames that have ground-truth
annotations, predicted bounding boxes are matched to GT boxes by IoU

An annotated output video is written showing:
    Colour scheme (BGR):
      green - ground truth (GT)
      red    (0,0,255)     – matched to GT, correct label
      yellow (0,255,255)   – matched to GT, wrong label
      black  (0,0,0)       – unmatched (no GT box overlaps sufficiently)

Usage:
    # single video
    python run_vvad_on_unitalk_video.py --data_dir data/ --video qv3-HaaxGUc --enable_logging

    # all val videos
    python run_vvad_on_unitalk_video.py --data_dir data/
"""

import contextlib
import csv
import os
import argparse
import logging
import subprocess
import time
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
# import paz.pipelines.detection as dt
from .asd import DetectVVAD
from paz.backend.boxes import compute_iou as _paz_iou

TARGET_FPS = 25
CONTAINMENT_THRESHOLD = 0.5          # accept match when smaller box is ≥50% covered
CONSECUTIVE_FRAME_THRESHOLD = 38     # entities below this max consecutive run are flagged
LOGGER = logging.getLogger('UniTalk_VVAD')


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


# ── label mapping ─────────────────────────────────────────────────────────────

_LABEL_MAP = {
    'SPEAKING_AUDIBLE':       'speaking',
    'NOT_SPEAKING':           'not-speaking',
}


# ── logging ───────────────────────────────────────────────────────────────────

def setup_logging(log_file=None, to_file=False):
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.handlers.clear()
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    LOGGER.addHandler(ch)
    if to_file:
        os.makedirs('logs', exist_ok=True)
        path = log_file or f'logs/unitalk_{datetime.now():%Y%m%d_%H%M%S}.log'
        fh = logging.FileHandler(path, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        LOGGER.addHandler(fh)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Evaluate DetectVVAD on UniTalk val videos')
    p.add_argument('--data_dir',      default='data',
                   help='Root data dir containing csv/ and videos/')
    p.add_argument('--video',         default=None,
                   help='Single video_id to process; omit to process all val videos')
    p.add_argument('--output_dir',    default=None,
                   help='Directory for annotated output videos')
    p.add_argument('--iou_threshold', type=float, default=0.5,
                   help='IoU threshold for GT↔prediction matching (default 0.5)')
    p.add_argument('--log_file',      default=None)
    p.add_argument('--enable_logging', action='store_true',
                   help='Write a debug log file in addition to console output')
    return p.parse_args()


# ── data loading ──────────────────────────────────────────────────────────────

def load_annotations(csv_path):
    """Read val_orig.csv and group annotations by video_id.

    Returns:
        dict[str, list[dict]]: video_id → list of annotation dicts with keys
        timestamp, bbox (normalised x1,y1,x2,y2), label, entity_id, vvad_label.
    """
    by_video = defaultdict(list)
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            label    = row['label']
            video_id = row['video_id']
            by_video[video_id].append({
                'video_id':    video_id,
                'timestamp':   float(row['frame_timestamp']),
                'bbox':        (float(row['entity_box_x1']), float(row['entity_box_y1']),
                                float(row['entity_box_x2']), float(row['entity_box_y2'])),
                'label':       label,
                'entity_id':   row['entity_id'],
                'vvad_label':  _LABEL_MAP.get(label, 'not-speaking'),
            })
    return dict(by_video)


# ── video helpers ─────────────────────────────────────────────────────────────

def open_video(path):
    """Open a video and return (cap, fps, width, height, total_frames)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {path}')
    fps          = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame   = cap.read()
    if not ret:
        raise RuntimeError(f'Cannot read first frame: {path}')
    h, w = frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return cap, float(fps), w, h, total_frames


class FFmpegWriter:
    """Encode video frames to H264 MP4 by piping raw BGR24 into FFmpeg.

    Produces a standard H264/yuv420p MP4 that opens on Mac, Windows, VSCode,
    VLC, and any modern browser — without needing OpenCV to have a working
    hardware encoder.

    Args:
        path:   Output .mp4 file path.
        fps:    Frames per second of the output video.
        width:  Frame width in pixels.
        height: Frame height in pixels.
    """

    def __init__(self, path, fps, width, height):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self._path = path
        self._proc = subprocess.Popen(
            [
                'ffmpeg', '-y',
                '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}',
                '-pix_fmt', 'bgr24',
                '-r', str(fps),
                '-i', 'pipe:',
                '-vcodec', 'libx264',
                '-pix_fmt', 'yuv420p',  # broadest player compatibility
                '-preset', 'fast',
                '-crf', '23',
                path,
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        LOGGER.info('Output codec: libx264 (H264 MP4)  →  %s', path)

    def write(self, frame_bgr):
        """Write one BGR frame."""
        self._proc.stdin.write(frame_bgr.tobytes())

    def release(self):
        """Flush, close the pipe and wait for FFmpeg to finish."""
        try:
            self._proc.stdin.close()
        except BrokenPipeError:
            pass
        self._proc.wait()


def create_writer(requested_path, fps, width, height):
    """Return an FFmpegWriter targeting a .mp4 output path.

    Returns:
        tuple[FFmpegWriter, str]: (writer, actual_output_path)
    """
    path = os.path.splitext(requested_path)[0] + '.mp4'
    return FFmpegWriter(path, fps, width, height), path


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
    """Compute IoU and containment for every GT↔pred pair.

    Returns:
        dict[(pi, gi)] -> {'iou': float, 'containment': float}
    """
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
    """Greedy highest-IoU matching between predicted and GT boxes.

    A pair is accepted when IoU ≥ iou_threshold OR containment ≥ CONTAINMENT_THRESHOLD
    (handles cases where the pipeline detects a larger region than the tight GT crop).

    Args:
        pred_boxes:   list of Box2D objects with a .coordinates attribute [x1,y1,x2,y2].
        gt_boxes:     list of GT dicts with 'bbox_pixel' key [x1,y1,x2,y2].
        iou_threshold: float
        iou_matrix:   optional pre-computed dict from compute_iou_matrix(); computed
                      internally when not supplied.

    Returns:
        list of (pred_idx, gt_idx, iou) triplets.
    """
    if iou_matrix is None:
        iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)

    matches, used = [], set()
    for pi, pred in enumerate(pred_boxes):
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


# ── drawing ───────────────────────────────────────────────────────────────────

def _draw_gt_box(bgr, bbox_pixel, entity_id, vvad_label, matched, best_iou, best_cont):
    """Draw a GT bounding box in green with IoU/containment diagnostics.

    - Filled circle  → this GT was matched to a prediction.
    - Hollow circle  → no prediction was matched (missed detection).
    """
    x1, y1, x2, y2 = (int(v) for v in bbox_pixel)
    color = (0, 255, 0)                                            # green
    cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
    cx, cy = x1 + 6, y1 + 6
    if matched:
        cv2.circle(bgr, (cx, cy), 5, color, -1)                   # filled  → matched
    else:
        cv2.circle(bgr, (cx, cy), 5, (0, 128, 0), 1)              # hollow  → missed
    header = f'GT {entity_id} {vvad_label}'
    cv2.putText(bgr, header,  (x1, max(8,  y1 - 13)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)



def _draw_pred_box(bgr, pred_box, best_iou, best_cont, status):
    """Draw a pipeline-detected bounding box with a status-coded colour.

    Colour scheme (BGR):
      red    (0,0,255)     – matched to GT, correct label
      yellow (0,255,255)   – matched to GT, wrong label
      black  (0,0,0)       – unmatched (no GT box overlaps sufficiently)

    Args:
        status: 'correct' | 'wrong' | 'unmatched'
    """
    pred_label = getattr(pred_box, 'class_name', None)
    if pred_label is None:
        return
    x1, y1, x2, y2 = (int(v) for v in pred_box.coordinates)

    color = {
        'correct':   (0, 0, 255),     # red
        'wrong':     (0, 255, 255),   # yellow
        'unmatched': (0, 0, 0),       # black
    }.get(status, (0, 0, 255))

    cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
    w = bgr.shape[1]
    tx, ty = max(0, min(x1, w - 190)), max(20, y1 - 4)
    cv2.putText(bgr,
                f'PRED {pred_label} [{status}] IoU:{best_iou:.2f} cont:{best_cont:.2f}',
                (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1, cv2.LINE_AA)


def _draw_frame_summary(bgr, frame_idx, n_gt, n_pred, n_matched, n_correct):
    h = bgr.shape[0]
    n_wrong     = n_matched - n_correct
    n_unmatched = n_pred - n_matched
    txt = (f'Frame {frame_idx} | GT:{n_gt}(green)  '
           f'correct:{n_correct}(red) wrong:{n_wrong}(yellow) unmatched:{n_unmatched}(black)')
    cv2.putText(bgr, txt, (10, max(20, h - 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)


# ── statistics accumulator ────────────────────────────────────────────────────



class Stats:
    """Accumulates per-frame evaluation results.

    Two update paths:
      record_frame()  – called for every frame read from the video (all frames).
      update()        – called only for frames that also have GT annotations.
    """

    def __init__(self, all_annots, total_video_frames):
        # ── video-level ───────────────────────────────────────────────────────
        self.total_video_frames = total_video_frames

        # ── pipeline output across ALL frames ─────────────────────────────────
        # A frame is "evaluated" once the pipeline's 36-frame buffer is full and
        # it starts returning predictions.
        self.frames_fed_to_pipeline      = 0   # every frame passed to pipeline
        self.frames_with_predictions     = 0   # frames where pipeline returned ≥1 box
        self.total_predictions           = 0   # total pred boxes across all frames
        self.no_pred_frames_without_gt   = 0   # no prediction AND no GT → excluded from eval

        # ── prediction correctness (annotation frames only) ───────────────────
        # Correctness can only be assessed where we have ground truth.
        self.preds_on_annot_frames    = 0   # pred boxes returned on annotation frames
        self.correct_predictions      = 0   # those that matched GT with correct label

        # per-label breakdown: matched predictions and correctness
        self.speaking_predictions     = 0
        self.correct_speaking         = 0   # TP for speaking
        self.not_speaking_predictions = 0
        self.correct_not_speaking     = 0   # TP for not-speaking

        # F1/Precision/Recall: FP and FN per class
        # TP_speaking = correct_speaking, TP_not_speaking = correct_not_speaking
        self.fp_speaking      = 0   # pred=speaking, gt≠speaking OR unmatched speaking pred
        self.fn_speaking      = 0   # gt=speaking, pred≠speaking OR unmatched speaking GT
        self.fp_not_speaking  = 0
        self.fn_not_speaking  = 0

        # ── entity-level ──────────────────────────────────────────────────────
        self.gt_entities            = set(a['entity_id'] for a in all_annots)
        self.detected_entities      = set()
        self.entity_matched         = defaultdict(int)
        self.entity_correct         = defaultdict(int)
        self.entity_gt_rows         = defaultdict(int)
        self.entity_gt_frame_indices = defaultdict(list)   # eid → sorted frame indices
        for a in all_annots:
            self.entity_gt_rows[a['entity_id']] += 1

        # ── annotation-frame / timestamp level ───────────────────────────────
        # Each unique frame_timestamp maps to one annotated frame.
        # A timestamp is "correct" when every GT entity in that frame was both
        # spatially matched to a prediction AND given the right label.
        # Missed detection: pipeline returned 0 boxes for a GT-annotated timestamp.
        # If ≥1 entity was detected at the timestamp, it is NOT a missed detection.
        self.total_annotated_frames      = 0   # = total unique timestamps evaluated
        self.annot_frames_no_predictions = 0   # GT exists but pipeline returned 0 boxes
        self.correct_timestamps          = 0   # timestamps where ALL entities correct

    # ── called for every frame ────────────────────────────────────────────────

    def record_frame(self, pred_boxes, has_gt=False):
        """Track pipeline output for one frame (annotation or not).

        Args:
            pred_boxes: list of Box2D objects returned by the pipeline.
            has_gt:     True when this frame has GT annotations.
                        Frames with no predictions AND no GT are excluded from eval.
        """
        self.frames_fed_to_pipeline += 1
        if pred_boxes:
            self.frames_with_predictions += 1
            self.total_predictions += len(pred_boxes)
        elif not has_gt:
            self.no_pred_frames_without_gt += 1

    # ── called only for annotation frames ─────────────────────────────────────

    def update(self, pred_boxes, gt_boxes_pixel, matches, frame_idx=None):
        """Record GT-matching results for one annotated frame (= one timestamp).

        Missed detection: pred_boxes is empty (pipeline returned nothing).
        If ≥1 box was detected — even if unmatched to GT — it is NOT a missed detection.
        """
        self.total_annotated_frames += 1
        self.preds_on_annot_frames  += len(pred_boxes)
        if not pred_boxes:
            self.annot_frames_no_predictions += 1

        matched_pi = {pi for pi, gi, _ in matches}
        matched_gi = {gi for pi, gi, _ in matches}

        # Track GT frame indices per entity for consecutive-run analysis
        for gt in gt_boxes_pixel:
            eid = gt['entity_id']
            if frame_idx is not None:
                self.entity_gt_frame_indices[eid].append(frame_idx)

        correctly_predicted_gt = set()

        # ── F1 / Precision / Recall accumulation ─────────────────────────────
        #
        # Empty pred frame  → skip F1 entirely (excluded, not a classification
        #                     decision; warm-up gaps must not inflate FN).
        #
        # Non-empty pred frame → standard F1:
        #   matched + correct label   → TP
        #   matched + wrong label     → FP (pred class) + FN (GT class)
        #   unmatched prediction      → FP
        #   unmatched GT box          → not counted (detection failure, not classification)
        if pred_boxes:
            for pi, gi, _iou in matches:
                pred_label = getattr(pred_boxes[pi], 'class_name', None)
                gt         = gt_boxes_pixel[gi]
                gt_label   = gt['vvad_label']
                eid        = gt['entity_id']

                self.detected_entities.add(eid)
                self.entity_matched[eid] += 1

                if pred_label == 'speaking':
                    self.speaking_predictions += 1
                elif pred_label == 'not-speaking':
                    self.not_speaking_predictions += 1

                is_correct = pred_label == gt_label
                if is_correct:                          # matched + correct label → TP
                    self.correct_predictions += 1
                    self.entity_correct[eid] += 1
                    correctly_predicted_gt.add(gi)
                    if pred_label == 'speaking':
                        self.correct_speaking += 1
                    elif pred_label == 'not-speaking':
                        self.correct_not_speaking += 1
                else:                                   # matched + wrong label → FP + FN
                    if pred_label == 'speaking' and gt_label == 'not-speaking':
                        self.fp_speaking     += 1
                        self.fn_not_speaking += 1
                    elif pred_label == 'not-speaking' and gt_label == 'speaking':
                        self.fp_not_speaking += 1
                        self.fn_speaking     += 1

            for pi in range(len(pred_boxes)):           # unmatched prediction → FP
                if pi not in matched_pi:
                    pred_label = getattr(pred_boxes[pi], 'class_name', None)
                    if pred_label == 'speaking':
                        self.fp_speaking     += 1
                    elif pred_label == 'not-speaking':
                        self.fp_not_speaking += 1

        # timestamp is correct only when every GT entity was correctly predicted
        if len(correctly_predicted_gt) == len(gt_boxes_pixel):
            self.correct_timestamps += 1

    # ── derived metrics ───────────────────────────────────────────────────────

    @property
    def frames_actually_evaluated(self):
        """Frames where the pipeline returned predictions (buffer was full)."""
        return self.frames_with_predictions

    @property
    def prediction_accuracy_pct(self):
        """% of matched predictions on annotation frames that were correct."""
        return 100.0 * self.correct_predictions / max(1, self.preds_on_annot_frames)

    @property
    def timestamp_accuracy_pct(self):
        """% of timestamps where every GT entity was correctly predicted."""
        return 100.0 * self.correct_timestamps / max(1, self.total_annotated_frames)

    @property
    def entity_detection_rate(self):
        return len(self.detected_entities) / max(1, len(self.gt_entities))

    # ── per-class Precision / Recall / F1 ────────────────────────────────────
    # TP_speaking = correct_speaking; TP_not_speaking = correct_not_speaking

    @property
    def precision_speaking(self):
        tp = self.correct_speaking
        return tp / max(1, tp + self.fp_speaking)

    @property
    def recall_speaking(self):
        tp = self.correct_speaking
        return tp / max(1, tp + self.fn_speaking)

    @property
    def f1_speaking(self):
        p, r = self.precision_speaking, self.recall_speaking
        return 2 * p * r / max(1e-9, p + r)

    @property
    def precision_not_speaking(self):
        tp = self.correct_not_speaking
        return tp / max(1, tp + self.fp_not_speaking)

    @property
    def recall_not_speaking(self):
        tp = self.correct_not_speaking
        return tp / max(1, tp + self.fn_not_speaking)

    @property
    def f1_not_speaking(self):
        p, r = self.precision_not_speaking, self.recall_not_speaking
        return 2 * p * r / max(1e-9, p + r)

    def entities_below_consecutive_threshold(self, threshold=CONSECUTIVE_FRAME_THRESHOLD):
        """Return list of (entity_id, max_consecutive_run) for entities below threshold."""
        result = []
        for eid in sorted(self.gt_entities):
            frames   = sorted(self.entity_gt_frame_indices.get(eid, []))
            max_run  = _max_consecutive_run(frames)
            if max_run < threshold:
                result.append((eid, max_run))
        return result


# ── core: single-pass video evaluation ───────────────────────────────────────

def _build_frame_map(annots, native_fps, width, height, video_id=None):
    """Map each annotation to the nearest video frame index (at native FPS).

    Args:
        video_id: When provided, only annotations whose 'video_id' field matches
                  are included.  Logs a warning for any that are filtered out so
                  accidental cross-video annotation leakage is immediately visible.
    """
    frame_map = defaultdict(list)
    skipped   = 0
    for ann in annots:
        if video_id is not None and ann.get('video_id') != video_id:
            skipped += 1
            continue
        fi = int(round(ann['timestamp'] * native_fps))
        x1, y1, x2, y2 = ann['bbox']
        frame_map[fi].append({
            **ann,
            'bbox_pixel': (x1 * width, y1 * height, x2 * width, y2 * height),
        })
    if skipped:
        LOGGER.warning(
            '_build_frame_map: dropped %d annotation(s) whose video_id != %r',
            skipped, video_id,
        )
    return frame_map


def evaluate_video(video_path, annots, iou_threshold, output_path, video_id=None, architecture='CNN2Plus1D_Light'):
    """Run DetectVVAD on every frame; evaluate and annotate at GT timestamps.

    Single pass: pipeline output, GT matching, drawing, and video writing all
    happen frame-by-frame so the video is read only once.

    Args:
        video_path:    path to input .mp4
        annots:        list of annotation dicts for this video
        iou_threshold: float
        output_path:   desired output path (codec fallback may change extension)
        video_id:      expected video_id; annotations not matching this id are
                       dropped before building the frame map.  Inferred from the
                       first annotation's 'video_id' field when not supplied.
        architecture:  which DetectVVAD architecture to use (default 'CNN2Plus1D_Light')

    Returns:
        tuple[Stats, str, float]: (stats, actual_output_path, elapsed_seconds)
    """
    if video_id is None and annots:
        video_id = annots[0].get('video_id')

    cap, native_fps, width, height, total_video_frames = open_video(video_path)
    writer, actual_out = create_writer(output_path, native_fps, width, height)

    frame_map = _build_frame_map(annots, native_fps, width, height, video_id=video_id)
    max_frame  = max(frame_map) if frame_map else 0

    # This function call parameters increase the number of predictions returned since stride is reduced and min_frames for buffer is also reduced
    pipeline = DetectVVAD(stride=1, averaging_window_size=1, min_frames=25, patience=10, architecture=architecture)
    stats    = Stats(annots, total_video_frames)
    t0       = time.time()

    LOGGER.info('Total frames in video : %d  (%.1f s @ %.2f fps)',
                total_video_frames, total_video_frames / native_fps, native_fps)

    frame_idx         = 0
    empty_run_start   = None   # first frame of a consecutive no-prediction run
    first_prediction  = None   # frame index when pipeline first returned a box

    try:
        while frame_idx <= max_frame:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # ── Phase 1: run pipeline ─────────────────────────────────────────
            try:
                out        = pipeline(frame_rgb)
                out_rgb    = out.get('image', frame_rgb) if isinstance(out, dict) else frame_rgb
                pred_boxes = out.get('boxes2D', []) if isinstance(out, dict) else []
            except Exception as exc:
                LOGGER.warning('Pipeline error frame %d: %s', frame_idx, exc)
                out_rgb, pred_boxes = frame_rgb, []

            # track pipeline output for every frame (buffer awareness)
            has_gt = frame_idx in frame_map
            stats.record_frame(pred_boxes, has_gt=has_gt)

            # ── track prediction gaps for diagnosis ───────────────────────────
            has_pred = bool(pred_boxes)
            if has_pred:
                if first_prediction is None:
                    first_prediction = frame_idx
                    LOGGER.info('First prediction at frame %d  '
                                '(buffer filled after %d frames)',
                                frame_idx, frame_idx)
                if empty_run_start is not None:
                    gap_len = frame_idx - empty_run_start
                    LOGGER.info('Prediction gap  frames %d–%d  (%d frames, %.2f s)  '
                                '— no pipeline output in this range',
                                empty_run_start, frame_idx - 1,
                                gap_len, gap_len / native_fps)
                    empty_run_start = None
            else:
                if empty_run_start is None:
                    empty_run_start = frame_idx

            out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

            # ── Phase 2: match predictions to GT (annotation frames only) ─────
            if frame_idx in frame_map:
                gt_boxes   = frame_map[frame_idx]

                if not pred_boxes:
                    # pipeline found nothing despite GT annotations existing
                    gt_sizes = [
                        (abs(g['bbox_pixel'][2] - g['bbox_pixel'][0]),
                         abs(g['bbox_pixel'][3] - g['bbox_pixel'][1]))
                        for g in gt_boxes
                    ]
                    LOGGER.debug(
                        'frame %d: pipeline returned 0 predictions '
                        '(%d GT faces present, sizes px: %s)',
                        frame_idx, len(gt_boxes),
                        [(f'{w:.0f}x{h:.0f}' for w, h in gt_sizes)]
                    )

                iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
                matches    = match_predictions_to_gt(pred_boxes, gt_boxes,
                                                     iou_threshold, iou_matrix)
                stats.update(pred_boxes, gt_boxes, matches, frame_idx)

                matched_gt_indices = {gi for _, gi, _ in matches}
                n_correct = sum(
                    1 for pi, gi, _ in matches
                    if getattr(pred_boxes[pi], 'class_name', None) == gt_boxes[gi]['vvad_label']
                )

                # best IoU/containment for each GT box against any prediction
                gt_best = {
                    gi: (max((iou_matrix[(pi, gi)] for pi in range(len(pred_boxes))),
                             key=lambda s: s['iou'])
                         if pred_boxes else {'iou': 0.0, 'containment': 0.0})
                    for gi in range(len(gt_boxes))
                }
                # best IoU/containment for each pred box against any GT
                pred_best = {
                    pi: (max((iou_matrix[(pi, gi)] for gi in range(len(gt_boxes))),
                             key=lambda s: s['iou'])
                         if gt_boxes else {'iou': 0.0, 'containment': 0.0})
                    for pi in range(len(pred_boxes))
                }

                # log all pair scores so mismatches are diagnosable
                for gi, gt in enumerate(gt_boxes):
                    for pi in range(len(pred_boxes)):
                        s = iou_matrix[(pi, gi)]
                        LOGGER.debug(
                            'frame %d  GT[%d](%s) ↔ PRED[%d]  '
                            'IoU=%.3f  cont=%.3f  threshold=%.2f/%.2f  %s',
                            frame_idx, gi, gt['entity_id'], pi,
                            s['iou'], s['containment'],
                            iou_threshold, CONTAINMENT_THRESHOLD,
                            'MATCHED' if any(m[0] == pi and m[1] == gi for m in matches)
                            else 'unmatched',
                        )

                # build per-prediction status for colour coding
                pred_status = {}
                for pi, gi, _ in matches:
                    pred_label = getattr(pred_boxes[pi], 'class_name', None)
                    pred_status[pi] = (
                        'correct' if pred_label == gt_boxes[gi]['vvad_label'] else 'wrong'
                    )
                for pi in range(len(pred_boxes)):
                    pred_status.setdefault(pi, 'unmatched')

                # green: GT boxes with IoU diagnostic
                for gi, gt in enumerate(gt_boxes):
                    b = gt_best[gi]
                    _draw_gt_box(out_bgr, gt['bbox_pixel'], gt['entity_id'],
                                 gt['vvad_label'],
                                 matched=gi in matched_gt_indices,
                                 best_iou=b['iou'], best_cont=b['containment'])

                # red/yellow/black: ALL pipeline detections, colour-coded by status
                for pi, pred in enumerate(pred_boxes):
                    b = pred_best[pi]
                    _draw_pred_box(out_bgr, pred,
                                   best_iou=b['iou'], best_cont=b['containment'],
                                   status=pred_status[pi])

                _draw_frame_summary(out_bgr, frame_idx,
                                    n_gt=len(gt_boxes), n_pred=len(pred_boxes),
                                    n_matched=len(matches), n_correct=n_correct)

                LOGGER.debug('frame %d: %d GT, %d pred, %d matches, %d correct',
                             frame_idx, len(gt_boxes), len(pred_boxes),
                             len(matches), n_correct)

            writer.write(out_bgr)
            frame_idx += 1

    finally:
        cap.release()
        writer.release()

    # report any trailing gap that reached the end of the processed range
    if empty_run_start is not None:
        gap_len = frame_idx - empty_run_start
        LOGGER.info('Prediction gap  frames %d–%d  (%d frames, %.2f s)  '
                    '— no pipeline output through end of processed range',
                    empty_run_start, frame_idx - 1,
                    gap_len, gap_len / native_fps)

    if first_prediction is None:
        LOGGER.warning('Pipeline never returned any predictions for this video')

    return stats, actual_out, time.time() - t0


# ── statistics printing ───────────────────────────────────────────────────────

def print_video_stats(video_id, video_path, output_path, stats, elapsed):
    LOGGER.info('')
    LOGGER.info('=' * 62)
    LOGGER.info('Video  : %s', video_id)
    LOGGER.info('Input  : %s', video_path)
    LOGGER.info('Output : %s', output_path)
    LOGGER.info('Time   : %.1f s', elapsed)
    LOGGER.info('')

    # ── Frame-level pipeline overview ─────────────────────────────────────────
    no_pred_total = stats.frames_fed_to_pipeline - stats.frames_with_predictions
    no_pred_with_gt    = stats.annot_frames_no_predictions
    no_pred_without_gt = stats.no_pred_frames_without_gt
    LOGGER.info('  ┌─ Frame overview ─────────────────────────────────────')
    LOGGER.info('  │  Total frames in video           : %d', stats.total_video_frames)
    LOGGER.info('  │  Frames fed to pipeline          : %d', stats.frames_fed_to_pipeline)
    LOGGER.info('  │  Frames with predictions (≥1)    : %d', stats.frames_with_predictions)
    LOGGER.info('  │  Frames with no predictions      : %d', no_pred_total)
    LOGGER.info('  │    ↳ with GT (missed detections) : %d', no_pred_with_gt)
    LOGGER.info('  │    ↳ without GT (excluded)       : %d'
                '  ← not used for evaluation', no_pred_without_gt)
    LOGGER.info('  │  Total predictions returned      : %d  (across all frames)',
                stats.total_predictions)
    LOGGER.info('  │  (check log for exact gap ranges if mid-video gaps are present)')
    LOGGER.info('  │')

    # ── Prediction accuracy ───────────────────────────────────────────────────
    sp_acc  = (100.0 * stats.correct_speaking     / max(1, stats.speaking_predictions))
    nsp_acc = (100.0 * stats.correct_not_speaking / max(1, stats.not_speaking_predictions))
    LOGGER.info('  ├─ Prediction accuracy (annotation frames only) ───────')
    LOGGER.info('  │  Predictions on annot frames     : %d', stats.preds_on_annot_frames)
    LOGGER.info('  │  Overall correct                 : %d / %d  (%.2f%%)',
                stats.correct_predictions, stats.preds_on_annot_frames,
                stats.prediction_accuracy_pct)
    LOGGER.info('  │  Speaking     predictions        : %d  →  correct %d  (%.2f%%)',
                stats.speaking_predictions, stats.correct_speaking, sp_acc)
    LOGGER.info('  │  Not-speaking predictions        : %d  →  correct %d  (%.2f%%)',
                stats.not_speaking_predictions, stats.correct_not_speaking, nsp_acc)
    LOGGER.info('  │  (correct = spatially matched to a GT box + right label)')
    LOGGER.info('  │')

    # ── Per-class Precision / Recall / F1 ────────────────────────────────────
    LOGGER.info('  ├─ Per-class Precision / Recall / F1 ────────────────')
    LOGGER.info('  │                        TP    FP    FN    P       R       F1')
    LOGGER.info('  │  speaking          : %4d  %4d  %4d  %.3f   %.3f   %.3f',
                stats.correct_speaking, stats.fp_speaking, stats.fn_speaking,
                stats.precision_speaking, stats.recall_speaking, stats.f1_speaking)
    LOGGER.info('  │  not-speaking      : %4d  %4d  %4d  %.3f   %.3f   %.3f',
                stats.correct_not_speaking, stats.fp_not_speaking, stats.fn_not_speaking,
                stats.precision_not_speaking, stats.recall_not_speaking,
                stats.f1_not_speaking)
    LOGGER.info('  │  (TP = matched pair with correct label; FP = wrong-label match or'
                ' unmatched pred; FN = unmatched GT or wrong-label match)')
    LOGGER.info('  │')

    # ── Missed detections (GT exists, pipeline returned nothing) ─────────────
    missed_pct = 100.0 * stats.annot_frames_no_predictions / max(1, stats.total_annotated_frames)
    LOGGER.info('  ├─ Missed detections ─────────────────────────────────')
    LOGGER.info('  │  GT timestamps with 0 detections : %d / %d  (%.1f%%)',
                stats.annot_frames_no_predictions, stats.total_annotated_frames, missed_pct)
    LOGGER.info('  │  Definition: pipeline returned NO boxes at a GT-annotated timestamp.')
    LOGGER.info('  │  If ≥1 entity detected (even unmatched), it is NOT a missed detection.')
    LOGGER.info('  │  Possible causes: face too small, unusual angle, buffer refilling.')
    LOGGER.info('  │  Run with --enable_logging to see GT box sizes for missed frames.')
    LOGGER.info('  │')

    # ── Entity detection rate ─────────────────────────────────────────────────
    LOGGER.info('  ├─ Entity detection rate ─────────────────────────────')
    LOGGER.info('  │  Entities detected : %d / %d  (%.1f%%)',
                len(stats.detected_entities), len(stats.gt_entities),
                100 * stats.entity_detection_rate)
    LOGGER.info('  │  (an entity is "detected" when ≥1 frame has a spatial match)')
    LOGGER.info('  │')

    # ── Entities below consecutive-frame threshold ────────────────────────────
    below = stats.entities_below_consecutive_threshold()
    below_pct = 100.0 * len(below) / max(1, len(stats.gt_entities))
    LOGGER.info('  ├─ Entity consecutive-frame coverage (threshold=%d) ──',
                CONSECUTIVE_FRAME_THRESHOLD)
    LOGGER.info('  │  Entities with max consecutive run < %d : %d / %d  (%.1f%%)',
                CONSECUTIVE_FRAME_THRESHOLD, len(below), len(stats.gt_entities), below_pct)
    if below:
        LOGGER.info('  │  These entities may be under-evaluated (short appearance):')
        for eid, max_run in below:
            LOGGER.info('  │    %-30s  max_consecutive_frames=%d', eid, max_run)
    LOGGER.info('  │')

    # ── Timestamp accuracy ────────────────────────────────────────────────────
    LOGGER.info('  └─ Timestamp accuracy ────────────────────────────────')
    LOGGER.info('     Total timestamps evaluated  : %d', stats.total_annotated_frames)
    LOGGER.info('     Correct timestamps          : %d  (%.2f%%)',
                stats.correct_timestamps, stats.timestamp_accuracy_pct)
    LOGGER.info('     (a timestamp is correct when ALL GT entities in that frame')
    LOGGER.info('      were spatially matched and given the right label)')

    # per-entity breakdown (debug log only)
    LOGGER.debug('')
    LOGGER.debug('Per-entity breakdown:')
    for eid in sorted(stats.gt_entities):
        gt      = stats.entity_gt_rows.get(eid, 0)
        mat     = stats.entity_matched.get(eid, 0)
        cor     = stats.entity_correct.get(eid, 0)
        acc     = cor / mat if mat else 0.0
        detected = eid in stats.detected_entities
        frames  = sorted(stats.entity_gt_frame_indices.get(eid, []))
        max_run = _max_consecutive_run(frames)
        LOGGER.debug('  %-30s  detected=%-3s  matched=%d/%d  correct=%d/%d (%.0f%%)  '
                     'max_consecutive=%d',
                     eid, detected, mat, gt, cor, mat, 100 * acc, max_run)


def print_aggregate_stats(all_stats):
    if len(all_stats) < 2:
        return
    tvf  = sum(s.total_video_frames          for s in all_stats)
    fwp  = sum(s.frames_with_predictions     for s in all_stats)
    tp   = sum(s.total_predictions           for s in all_stats)
    cp   = sum(s.correct_predictions         for s in all_stats)
    pa   = sum(s.preds_on_annot_frames       for s in all_stats)
    sp   = sum(s.speaking_predictions        for s in all_stats)
    csp  = sum(s.correct_speaking            for s in all_stats)
    fpsp = sum(s.fp_speaking                 for s in all_stats)
    fnsp = sum(s.fn_speaking                 for s in all_stats)
    nsp  = sum(s.not_speaking_predictions    for s in all_stats)
    cnsp = sum(s.correct_not_speaking        for s in all_stats)
    fpns = sum(s.fp_not_speaking             for s in all_stats)
    fnns = sum(s.fn_not_speaking             for s in all_stats)
    det  = sum(len(s.detected_entities)      for s in all_stats)
    tot  = sum(len(s.gt_entities)            for s in all_stats)
    ct   = sum(s.correct_timestamps          for s in all_stats)
    tf   = sum(s.total_annotated_frames      for s in all_stats)
    mnd  = sum(s.annot_frames_no_predictions for s in all_stats)
    npng = sum(s.no_pred_frames_without_gt   for s in all_stats)

    # aggregate below-threshold entities (may overlap across videos if same entity_id)
    below_count = sum(len(s.entities_below_consecutive_threshold()) for s in all_stats)

    # aggregate F1
    p_sp  = csp  / max(1, csp  + fpsp)
    r_sp  = csp  / max(1, csp  + fnsp)
    f1_sp = 2 * p_sp * r_sp / max(1e-9, p_sp + r_sp)
    p_ns  = cnsp / max(1, cnsp + fpns)
    r_ns  = cnsp / max(1, cnsp + fnns)
    f1_ns = 2 * p_ns * r_ns / max(1e-9, p_ns + r_ns)

    LOGGER.info('')
    LOGGER.info('═' * 62)
    LOGGER.info('AGGREGATE  (%d videos)', len(all_stats))
    LOGGER.info('═' * 62)
    LOGGER.info('  Total frames in videos            : %d', tvf)
    LOGGER.info('  Frames with predictions           : %d', fwp)
    LOGGER.info('  Total predictions returned        : %d', tp)
    LOGGER.info('  Frames with no pred, no GT (excl) : %d', npng)
    LOGGER.info('')
    LOGGER.info('  Overall prediction accuracy       : %.2f%%  (%d / %d)',
                100 * cp / max(1, pa), cp, pa)
    LOGGER.info('')
    LOGGER.info('  Per-class Precision / Recall / F1:')
    LOGGER.info('                        TP    FP    FN    P       R       F1')
    LOGGER.info('  speaking          : %4d  %4d  %4d  %.3f   %.3f   %.3f',
                csp, fpsp, fnsp, p_sp, r_sp, f1_sp)
    LOGGER.info('  not-speaking      : %4d  %4d  %4d  %.3f   %.3f   %.3f',
                cnsp, fpns, fnns, p_ns, r_ns, f1_ns)
    LOGGER.info('')
    LOGGER.info('  Missed detections (GT, 0 detections): %d / %d  (%.1f%%)',
                mnd, tf, 100 * mnd / max(1, tf))
    LOGGER.info('  Entity detection rate               : %d / %d  (%.1f%%)',
                det, tot, 100 * det / max(1, tot))
    LOGGER.info('  Entities below %d consecutive frames: %d / %d  (%.1f%%)',
                CONSECUTIVE_FRAME_THRESHOLD, below_count, tot,
                100 * below_count / max(1, tot))
    LOGGER.info('  Timestamp accuracy                  : %d / %d  (%.2f%%)',
                ct, tf, 100 * ct / max(1, tf))
    LOGGER.info('  (timestamp correct = all GT entities in that frame matched + right label)')


# ── result file writer ────────────────────────────────────────────────────────

def write_result_file(result_dir, video_id, stats, elapsed):
    """Write per-video evaluation stats to data/results/<video_id>_results.txt."""
    os.makedirs(result_dir, exist_ok=True)
    path = os.path.join(result_dir, f'{video_id}_results.txt')

    below = stats.entities_below_consecutive_threshold()
    below_pct = 100.0 * len(below) / max(1, len(stats.gt_entities))
    missed_pct = 100.0 * stats.annot_frames_no_predictions / max(1, stats.total_annotated_frames)
    sp_acc  = 100.0 * stats.correct_speaking     / max(1, stats.speaking_predictions)
    nsp_acc = 100.0 * stats.correct_not_speaking / max(1, stats.not_speaking_predictions)
    no_pred_total    = stats.frames_fed_to_pipeline - stats.frames_with_predictions
    no_pred_with_gt  = stats.annot_frames_no_predictions
    no_pred_excl     = stats.no_pred_frames_without_gt

    lines = [
        f'Video: {video_id}',
        f'Elapsed: {elapsed:.1f} s',
        '',
        '── Frame overview ────────────────────────────────────────',
        f'  Total frames in video           : {stats.total_video_frames}',
        f'  Frames fed to pipeline          : {stats.frames_fed_to_pipeline}',
        f'  Frames with predictions (>=1)   : {stats.frames_with_predictions}',
        f'  Frames with no predictions      : {no_pred_total}',
        f'    with GT (missed detections)   : {no_pred_with_gt}',
        f'    without GT (excluded)         : {no_pred_excl}',
        f'  Total predictions returned      : {stats.total_predictions}',
        '',
        '── Prediction accuracy (annotation frames only) ──────────',
        f'  Predictions on annot frames     : {stats.preds_on_annot_frames}',
        f'  Overall correct                 : {stats.correct_predictions} / {stats.preds_on_annot_frames}'
        f'  ({stats.prediction_accuracy_pct:.2f}%)',
        f'  Speaking   predictions          : {stats.speaking_predictions}'
        f'  -> correct {stats.correct_speaking}  ({sp_acc:.2f}%)',
        f'  Not-speaking predictions        : {stats.not_speaking_predictions}'
        f'  -> correct {stats.correct_not_speaking}  ({nsp_acc:.2f}%)',
        '',
        '── Per-class Precision / Recall / F1 ─────────────────────',
        f'{"":22s}  TP    FP    FN    P       R       F1',
        f'  speaking        : {stats.correct_speaking:4d}  {stats.fp_speaking:4d}  {stats.fn_speaking:4d}'
        f'  {stats.precision_speaking:.3f}   {stats.recall_speaking:.3f}   {stats.f1_speaking:.3f}',
        f'  not-speaking    : {stats.correct_not_speaking:4d}  {stats.fp_not_speaking:4d}  {stats.fn_not_speaking:4d}'
        f'  {stats.precision_not_speaking:.3f}   {stats.recall_not_speaking:.3f}   {stats.f1_not_speaking:.3f}',
        '',
        '── Missed detections ─────────────────────────────────────',
        f'  GT timestamps with 0 detections : {stats.annot_frames_no_predictions}'
        f' / {stats.total_annotated_frames}  ({missed_pct:.1f}%)',
        '  (missed = pipeline returned NO boxes; >=1 entity detected = not missed)',
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
        frames   = sorted(stats.entity_gt_frame_indices.get(eid, []))
        max_run  = _max_consecutive_run(frames)
        lines.append(
            f'  {eid:<32s}  detected={str(detected):<5s}'
            f'  matched={mat}/{gt_rows}'
            f'  correct={cor}/{mat} ({100*acc:.0f}%)'
            f'  max_consecutive={max_run}'
        )

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    LOGGER.info('Results written to %s', path)


def write_aggregate_result_file(result_dir, all_stats, video_ids):
    """Write aggregate stats across all processed videos to data/results/aggregate_results.txt."""
    if not all_stats:
        return
    os.makedirs(result_dir, exist_ok=True)
    path = os.path.join(result_dir, 'aggregate_results.txt')

    tvf  = sum(s.total_video_frames          for s in all_stats)
    fwp  = sum(s.frames_with_predictions     for s in all_stats)
    tp   = sum(s.total_predictions           for s in all_stats)
    cp   = sum(s.correct_predictions         for s in all_stats)
    pa   = sum(s.preds_on_annot_frames       for s in all_stats)
    csp  = sum(s.correct_speaking            for s in all_stats)
    fpsp = sum(s.fp_speaking                 for s in all_stats)
    fnsp = sum(s.fn_speaking                 for s in all_stats)
    cnsp = sum(s.correct_not_speaking        for s in all_stats)
    fpns = sum(s.fp_not_speaking             for s in all_stats)
    fnns = sum(s.fn_not_speaking             for s in all_stats)
    det  = sum(len(s.detected_entities)      for s in all_stats)
    tot  = sum(len(s.gt_entities)            for s in all_stats)
    ct   = sum(s.correct_timestamps          for s in all_stats)
    tf   = sum(s.total_annotated_frames      for s in all_stats)
    mnd  = sum(s.annot_frames_no_predictions for s in all_stats)
    npng = sum(s.no_pred_frames_without_gt   for s in all_stats)
    below_count = sum(len(s.entities_below_consecutive_threshold()) for s in all_stats)

    p_sp  = csp  / max(1, csp  + fpsp)
    r_sp  = csp  / max(1, csp  + fnsp)
    f1_sp = 2 * p_sp * r_sp / max(1e-9, p_sp + r_sp)
    p_ns  = cnsp / max(1, cnsp + fpns)
    r_ns  = cnsp / max(1, cnsp + fnns)
    f1_ns = 2 * p_ns * r_ns / max(1e-9, p_ns + r_ns)

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
        '── Per-class Precision / Recall / F1 ─────────────────────',
        f'{"":22s}  TP    FP    FN    P       R       F1',
        f'  speaking        : {csp:4d}  {fpsp:4d}  {fnsp:4d}  {p_sp:.3f}   {r_sp:.3f}   {f1_sp:.3f}',
        f'  not-speaking    : {cnsp:4d}  {fpns:4d}  {fnns:4d}  {p_ns:.3f}   {r_ns:.3f}   {f1_ns:.3f}',
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
    LOGGER.info('Aggregate results written to %s', path)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    setup_logging(args.log_file, args.enable_logging)
    LOGGER.info('UniTalk VVAD evaluation | data_dir=%s', args.data_dir)

    csv_path   = os.path.join(args.data_dir, 'csv', 'val_orig.csv')
    video_dir  = os.path.join(args.data_dir, 'videos', 'val')
    output_dir = args.output_dir or os.path.join(args.data_dir, 'output_videos', 'val')
    result_dir = os.path.join(args.data_dir, 'results')
    os.makedirs(output_dir, exist_ok=True)

    all_annots = load_annotations(csv_path)
    video_ids  = [args.video] if args.video else sorted(all_annots)

    all_stats      = []
    processed_vids = []
    for vid in video_ids:
        video_path = os.path.join(video_dir, f'{vid}.mp4')
        if not os.path.isfile(video_path):
            LOGGER.warning('Video not found, skipping: %s', video_path)
            continue
        annots = all_annots.get(vid, [])
        if not annots:
            LOGGER.warning('No annotations for %s, skipping', vid)
            continue

        LOGGER.info('\nProcessing %s  (%d GT rows)', vid, len(annots))
        output_path = os.path.join(output_dir, f'{vid}_vvad.mp4')

        try:
            stats, actual_out, elapsed = evaluate_video(
                video_path, annots, args.iou_threshold, output_path
            )
            print_video_stats(vid, video_path, actual_out, stats, elapsed)
            write_result_file(result_dir, vid, stats, elapsed)
            all_stats.append(stats)
            processed_vids.append(vid)
        except Exception as exc:
            LOGGER.exception('Failed on %s: %s', vid, exc)

    print_aggregate_stats(all_stats)
    write_aggregate_result_file(result_dir, all_stats, processed_vids)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.info('Interrupted')
    except Exception as exc:
        LOGGER.exception('Fatal: %s', exc)
        raise SystemExit(1)
