"""Evaluate UniTalk val split with the DetectVVAD pipeline from pypaz.

Every frame is passed through dt.DetectVVAD().  At frames that have ground-truth
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
import paz.pipelines.detection as dt
from paz.backend.boxes import compute_iou as _paz_iou

TARGET_FPS = 25
CONTAINMENT_THRESHOLD = 0.5          # accept match when smaller box is ≥50% covered
LOGGER = logging.getLogger('UniTalk_VVAD')


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
            label = row['label']
            by_video[row['video_id']].append({
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
        self.frames_fed_to_pipeline   = 0   # every frame passed to pipeline
        self.frames_with_predictions  = 0   # frames where pipeline returned ≥1 box
        self.total_predictions        = 0   # total pred boxes across all frames

        # ── prediction correctness (annotation frames only) ───────────────────
        # Correctness can only be assessed where we have ground truth.
        self.preds_on_annot_frames    = 0   # pred boxes returned on annotation frames
        self.correct_predictions      = 0   # those that matched GT with correct label

        # per-label breakdown: how many predictions of each class were made and
        # how many were correct (i.e. the GT label also matches)
        self.speaking_predictions     = 0
        self.correct_speaking         = 0
        self.not_speaking_predictions = 0
        self.correct_not_speaking     = 0

        # ── entity-level ──────────────────────────────────────────────────────
        self.gt_entities       = set(a['entity_id'] for a in all_annots)
        self.detected_entities = set()
        self.entity_matched    = defaultdict(int)
        self.entity_correct    = defaultdict(int)
        self.entity_gt_rows    = defaultdict(int)
        for a in all_annots:
            self.entity_gt_rows[a['entity_id']] += 1

        # ── annotation-frame / timestamp level ───────────────────────────────
        # Each unique frame_timestamp maps to one annotated frame.
        # A timestamp is "correct" when every GT entity in that frame was both
        # spatially matched to a prediction AND given the right label.
        self.total_annotated_frames      = 0   # = total unique timestamps evaluated
        self.annot_frames_no_predictions = 0   # GT exists but pipeline returned nothing
        self.correct_timestamps          = 0   # timestamps where ALL entities correct

    # ── called for every frame ────────────────────────────────────────────────

    def record_frame(self, pred_boxes):
        """Track pipeline output for one frame (annotation or not)."""
        self.frames_fed_to_pipeline += 1
        if pred_boxes:
            self.frames_with_predictions += 1
            self.total_predictions += len(pred_boxes)

    # ── called only for annotation frames ─────────────────────────────────────

    def update(self, pred_boxes, gt_boxes_pixel, matches):
        """Record GT-matching results for one annotated frame (= one timestamp)."""
        self.total_annotated_frames += 1
        self.preds_on_annot_frames  += len(pred_boxes)
        if not pred_boxes:
            self.annot_frames_no_predictions += 1

        # collect which GT indices were correctly predicted in this frame
        correctly_predicted_gt = set()

        for pi, gi, _iou in matches:
            pred_label = getattr(pred_boxes[pi], 'class_name', None)
            gt  = gt_boxes_pixel[gi]
            eid = gt['entity_id']

            self.detected_entities.add(eid)
            self.entity_matched[eid] += 1

            # per-label prediction counts
            if pred_label == 'speaking':
                self.speaking_predictions += 1
            elif pred_label == 'not-speaking':
                self.not_speaking_predictions += 1

            is_correct = pred_label == gt['vvad_label']
            if is_correct:
                self.correct_predictions    += 1
                self.entity_correct[eid]    += 1
                correctly_predicted_gt.add(gi)
                if pred_label == 'speaking':
                    self.correct_speaking     += 1
                elif pred_label == 'not-speaking':
                    self.correct_not_speaking += 1

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


# ── core: single-pass video evaluation ───────────────────────────────────────

def _build_frame_map(annots, native_fps, width, height):
    """Map each annotation to the nearest video frame index (at native FPS)."""
    frame_map = defaultdict(list)
    for ann in annots:
        fi = int(round(ann['timestamp'] * native_fps))
        x1, y1, x2, y2 = ann['bbox']
        frame_map[fi].append({
            **ann,
            'bbox_pixel': (x1 * width, y1 * height, x2 * width, y2 * height),
        })
    return frame_map


def evaluate_video(video_path, annots, iou_threshold, output_path):
    """Run DetectVVAD on every frame; evaluate and annotate at GT timestamps.

    Single pass: pipeline output, GT matching, drawing, and video writing all
    happen frame-by-frame so the video is read only once.

    Args:
        video_path:    path to input .mp4
        annots:        list of annotation dicts for this video
        iou_threshold: float
        output_path:   desired output path (codec fallback may change extension)

    Returns:
        tuple[Stats, str, float]: (stats, actual_output_path, elapsed_seconds)
    """
    cap, native_fps, width, height, total_video_frames = open_video(video_path)
    writer, actual_out = create_writer(output_path, native_fps, width, height)

    frame_map = _build_frame_map(annots, native_fps, width, height)
    max_frame  = max(frame_map) if frame_map else 0

    # This function call parameters increase the number of predictions returned since stride is reduced and min_frames for buffer is also reduced
    pipeline = dt.DetectVVAD(stride=1, averaging_window_size=1, min_frames=25, patience=10)
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
            stats.record_frame(pred_boxes)

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
                stats.update(pred_boxes, gt_boxes, matches)

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
    LOGGER.info('  ┌─ Frame overview ─────────────────────────────────────')
    LOGGER.info('  │  Total frames in video       : %d', stats.total_video_frames)
    LOGGER.info('  │  Frames fed to pipeline      : %d', stats.frames_fed_to_pipeline)
    LOGGER.info('  │  Frames actually evaluated   : %d'
                '  (frames where pipeline returned ≥1 prediction)',
                stats.frames_actually_evaluated)
    LOGGER.info('  │  Frames with no predictions  : %d'
                '  (buffer filling + any mid-video gaps)',
                stats.frames_fed_to_pipeline - stats.frames_actually_evaluated)
    LOGGER.info('  │  Total predictions returned  : %d  (across all evaluated frames)',
                stats.total_predictions)
    LOGGER.info('  │  (check log for exact gap ranges if mid-video gaps are present)')
    LOGGER.info('  │')

    # ── Prediction accuracy ───────────────────────────────────────────────────
    sp_acc  = (100.0 * stats.correct_speaking     / max(1, stats.speaking_predictions))
    nsp_acc = (100.0 * stats.correct_not_speaking / max(1, stats.not_speaking_predictions))
    LOGGER.info('  ├─ Prediction accuracy (annotation frames only) ───────')
    LOGGER.info('  │  Predictions on annot frames : %d', stats.preds_on_annot_frames)
    LOGGER.info('  │  Overall correct             : %d / %d  (%.2f%%)',
                stats.correct_predictions, stats.preds_on_annot_frames,
                stats.prediction_accuracy_pct)
    LOGGER.info('  │  Speaking     predictions    : %d  →  correct %d  (%.2f%%)',
                stats.speaking_predictions, stats.correct_speaking, sp_acc)
    LOGGER.info('  │  Not-speaking predictions    : %d  →  correct %d  (%.2f%%)',
                stats.not_speaking_predictions, stats.correct_not_speaking, nsp_acc)
    LOGGER.info('  │  (correct = spatially matched to a GT box + right label)')
    LOGGER.info('  │')

    # ── Missed detections (GT exists, pipeline found nothing) ────────────────
    missed_pct = 100.0 * stats.annot_frames_no_predictions / max(1, stats.total_annotated_frames)
    LOGGER.info('  ├─ Missed detections ─────────────────────────────────')
    LOGGER.info('  │  Annot frames with 0 predictions : %d / %d  (%.1f%%)',
                stats.annot_frames_no_predictions, stats.total_annotated_frames, missed_pct)
    LOGGER.info('  │  (GT annotations exist but the pipeline returned no boxes at all)')
    LOGGER.info('  │  Possible causes: face too small, unusual angle, '
                'buffer refilling after scene cut')
    LOGGER.info('  │  Run with --enable_logging to see GT box sizes for missed frames')
    LOGGER.info('  │')

    # ── Entity detection rate ─────────────────────────────────────────────────
    LOGGER.info('  ├─ Entity detection rate ─────────────────────────────')
    LOGGER.info('  │  Entities detected : %d / %d  (%.1f%%)',
                len(stats.detected_entities), len(stats.gt_entities),
                100 * stats.entity_detection_rate)
    LOGGER.info('  │  (an entity is "detected" when ≥1 frame has a spatial match)')
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
        gt  = stats.entity_gt_rows.get(eid, 0)
        mat = stats.entity_matched.get(eid, 0)
        cor = stats.entity_correct.get(eid, 0)
        acc = cor / mat if mat else 0.0
        detected = eid in stats.detected_entities
        LOGGER.debug('  %-30s  detected=%-3s  matched=%d/%d  correct=%d/%d (%.0f%%)',
                     eid, detected, mat, gt, cor, mat, 100 * acc)


def print_aggregate_stats(all_stats):
    if len(all_stats) < 2:
        return
    tvf = sum(s.total_video_frames          for s in all_stats)
    fwp = sum(s.frames_with_predictions     for s in all_stats)
    tp  = sum(s.total_predictions           for s in all_stats)
    cp  = sum(s.correct_predictions         for s in all_stats)
    pa  = sum(s.preds_on_annot_frames       for s in all_stats)
    sp  = sum(s.speaking_predictions        for s in all_stats)
    csp = sum(s.correct_speaking            for s in all_stats)
    nsp = sum(s.not_speaking_predictions    for s in all_stats)
    cnsp= sum(s.correct_not_speaking        for s in all_stats)
    det = sum(len(s.detected_entities)      for s in all_stats)
    tot = sum(len(s.gt_entities)            for s in all_stats)
    ct  = sum(s.correct_timestamps          for s in all_stats)
    tf  = sum(s.total_annotated_frames      for s in all_stats)

    LOGGER.info('')
    LOGGER.info('═' * 62)
    LOGGER.info('AGGREGATE  (%d videos)', len(all_stats))
    LOGGER.info('═' * 62)
    LOGGER.info('  Total frames in videos        : %d', tvf)
    LOGGER.info('  Frames actually evaluated     : %d  (pipeline returned predictions)', fwp)
    LOGGER.info('  Total predictions returned    : %d', tp)
    LOGGER.info('  Overall prediction accuracy   : %.2f%%  (%d / %d)',
                100 * cp / max(1, pa), cp, pa)
    LOGGER.info('  Speaking     accuracy         : %.2f%%  (%d / %d)',
                100 * csp / max(1, sp), csp, sp)
    LOGGER.info('  Not-speaking accuracy         : %.2f%%  (%d / %d)',
                100 * cnsp / max(1, nsp), cnsp, nsp)
    LOGGER.info('  Entity detection rate         : %d / %d  (%.1f%%)',
                det, tot, 100 * det / max(1, tot))
    LOGGER.info('  Timestamp accuracy            : %d / %d  (%.2f%%)',
                ct, tf, 100 * ct / max(1, tf))
    LOGGER.info('  (timestamp correct = all GT entities in that frame matched + right label)')


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    setup_logging(args.log_file, args.enable_logging)
    LOGGER.info('UniTalk VVAD evaluation | data_dir=%s', args.data_dir)

    csv_path   = os.path.join(args.data_dir, 'csv', 'val_orig.csv')
    video_dir  = os.path.join(args.data_dir, 'videos', 'val')
    output_dir = args.output_dir or os.path.join(args.data_dir, 'output_videos', 'val')
    os.makedirs(output_dir, exist_ok=True)

    all_annots = load_annotations(csv_path)
    video_ids  = [args.video] if args.video else sorted(all_annots)

    all_stats = []
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
            all_stats.append(stats)
        except Exception as exc:
            LOGGER.exception('Failed on %s: %s', vid, exc)

    print_aggregate_stats(all_stats)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.info('Interrupted')
    except Exception as exc:
        LOGGER.exception('Fatal: %s', exc)
        raise SystemExit(1)
