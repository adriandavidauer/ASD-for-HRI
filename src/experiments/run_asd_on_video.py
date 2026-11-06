"""Run ASD on a single video and persist its per-frame predictions."""

import argparse
import csv
import logging
import os
import time

from asd4hri.asd import ASD

import cv2
import numpy as np
from paz.abstract.messages import Box2D

from .helpers import setup_logging

LOGGER = logging.getLogger('uniTalk_ASD')

def parse_args():
    p = argparse.ArgumentParser(
        description='Run ASD on a single video and write a predictions CSV')
    p.add_argument('--video',          required=True,
                   help='Path to the input video file')
    p.add_argument('--predictions',    default=None,
                   help='Output predictions CSV path '
                        '(default: results/<video_id>_results.csv)')
    p.add_argument('--aggregate_time', default='results/aggregate_time.csv',
                   help='Aggregate time CSV to append to '
                        '(default: results/aggregate_time.csv)')
    p.add_argument('--video_id',       default=None,
                   help='Override the video_id used in logs and the timing row '
                        '(defaults to the input video basename)')
    p.add_argument('--log_file',       default=None,
                   help='Override the auto-generated log file path')
    p.add_argument('--architecture',   default='CNN2Plus1D_Light',
                   help='Architecture used for the model')
    p.add_argument('--verbose', '-v',  action='store_true',
                   help='Also emit INFO-level messages on the console')

    return p.parse_args()

# ── CSV writers ───────────────────────────────────────────────────────────────

_PREDICTION_FIELDS = ['frame_idx', 'timestamp', 'label', 'x1', 'y1', 'x2', 'y2', 'score',
                      'entity_id']
_AGGREGATE_TIME_FIELDS = ['video_id', 'elapsed_seconds', 'frames_processed', 'fps_processed',
                          'width', 'height']

def _write_prediction_rows(writer, frame_idx, timestamp, pred_boxes, width, height, gt=None):
    """Write one row per predicted box with bbox coords normalised to [0, 1].

    With ``gt`` (a list of ``(entity_id, (x1, y1, x2, y2))`` parallel to ``pred_boxes``, from a
    ``--boxes gt`` run) the row keeps the GT entity id and the GT coordinates unchanged, so the
    scoring can join it to the ground truth exactly.
    """
    if gt is not None:
        for pred, (entity_id, box) in zip(pred_boxes, gt):
            label = getattr(pred, 'class_name', '') or ''
            writer.writerow([frame_idx, repr(timestamp), label, *(repr(float(v)) for v in box),
                             f'{float(getattr(pred, "score", 0.0)):.6f}', entity_id])
        return
    for pred in pred_boxes:
        label = getattr(pred, 'class_name', '') or ''
        x1, y1, x2, y2 = pred.coordinates
        writer.writerow([frame_idx, f'{timestamp:.6f}', label,
                         f'{x1 / width:.6f}', f'{y1 / height:.6f}',
                         f'{x2 / width:.6f}', f'{y2 / height:.6f}',
                         f'{float(getattr(pred, "score", 0.0)):.6f}', ''])


def append_aggregate_time(aggregate_time_csv, video_id, elapsed, frames_processed, fps,
                          width=None, height=None):
    """Append one timing row, writing the header only when the file is new.

    ``width``/``height`` are the frame size in pixels, needed for the face-size bins (#75).
    """
    os.makedirs(os.path.dirname(aggregate_time_csv) or '.', exist_ok=True)
    new_file = not os.path.isfile(aggregate_time_csv)
    with open(aggregate_time_csv, 'a', newline='') as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(_AGGREGATE_TIME_FIELDS)
        w.writerow([video_id, f'{elapsed:.3f}', frames_processed, f'{fps:.2f}',
                    width or '', height or ''])


class _GroundTruthFrames:
    """GT boxes of one video, looked up by the timestamp of a video frame (GT-box mode)."""

    def __init__(self, groundtruth, fps):
        by_ts = {}
        for ts, entity_id, box in groundtruth:
            by_ts.setdefault(float(ts), []).append((entity_id, tuple(float(v) for v in box)))
        self.timestamps = np.array(sorted(by_ts))
        self.frames = [by_ts[t] for t in self.timestamps]
        self.tolerance = 0.5 / fps

    def at(self, timestamp):
        """Return ``(gt_timestamp, [(entity_id, box), ...])`` nearest to ``timestamp``.

        Returns ``(None, [])`` if no annotated timestamp is within half a frame.
        """
        if not len(self.timestamps):
            return None, []
        i = int(np.searchsorted(self.timestamps, timestamp))
        best = min((j for j in (i - 1, i) if 0 <= j < len(self.timestamps)),
                   key=lambda j: abs(self.timestamps[j] - timestamp))
        if abs(self.timestamps[best] - timestamp) > self.tolerance:
            return None, []
        return float(self.timestamps[best]), self.frames[best]

    @staticmethod
    def boxes2D(gt_frame, width, height):
        """Convert normalised GT boxes to pixel ``Box2D`` objects and their entity ids."""
        boxes, ids = [], []
        for entity_id, (x1, y1, x2, y2) in gt_frame:
            boxes.append(Box2D((int(round(x1 * width)), int(round(y1 * height)),
                                int(round(x2 * width)), int(round(y2 * height))), 1.0, 'Face'))
            ids.append(entity_id)
        return boxes, ids


# ── pipeline runner ───────────────────────────────────────────────────────────

def run_asd_on_video(video_path,
                    predictions_csv=None,
                    aggregate_time_csv='results/aggregate_time.csv',
                    video_id=None, architecture='CNN2Plus1D_Light',
                    stride=1, averaging_window_size: int = 1,
                    decision_threshold: float = 0.5,
                    weighted: bool = True,
                    max_consecutive_empty: int = 2,
                    groundtruth=None):
    """Run ASD on one video and write predictions + timing rows.

    Args:
        video_path:         path to input .mp4
        predictions_csv:    output path for this video's predictions CSV.
                            Defaults to results/<video_id>_results.csv.
        aggregate_time_csv: timing CSV to append one row to (one row per video).
                            Defaults to results/aggregate_time.csv; pass None to skip.
        video_id:           override video_id used in logs / paths / timing row.
                            Defaults to the video file's basename.
        groundtruth:        GT-box mode (#75): list of ``(timestamp, entity_id, (x1, y1, x2, y2))``
                            with normalised boxes for this video. The detector is skipped and
                            the GT boxes of the nearest annotated timestamp (within half a frame)
                            are classified, each entity with its own buffer. None = use the
                            face detector.

    Returns:
        tuple[int, float]: (frames_processed, elapsed_seconds)
    """
    video_id = video_id or os.path.splitext(os.path.basename(video_path))[0]
    if predictions_csv is None:
        predictions_csv = os.path.join('predictions', f'{video_id}_results.csv')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')

    # timestamps are real seconds (frame_idx / native fps)
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if not native_fps:
        raise RuntimeError(f'Cannot read FPS from video: {video_path}')


    pipeline = ASD(stride=stride, averaging_window_size=averaging_window_size, architecture=architecture, decision_threshold=decision_threshold, weighted=weighted, max_consecutive_empty=max_consecutive_empty)
    os.makedirs(os.path.dirname(predictions_csv) or '.', exist_ok=True)

    gt_index = _GroundTruthFrames(groundtruth, native_fps) if groundtruth is not None else None

    t0 = time.time()
    frame_idx = 0
    width = height = 0

    try:
        with open(predictions_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(_PREDICTION_FIELDS)

        frame_idx_list = []
        timestamp_list = []
        pred_boxes_list = []
        gt_list = []
        while True:
            is_frame_received, frame = cap.read()
            if not is_frame_received:
                LOGGER.warning('Frame not received or End of stream')
                break
            height, width = frame.shape[:2]
            timestamp = frame_idx / native_fps
            gt_frame = None
            try:
                if gt_index is None:
                    output = pipeline(frame)
                else:
                    gt_ts, gt_frame = gt_index.at(timestamp)
                    boxes, ids = gt_index.boxes2D(gt_frame, width, height)
                    output = pipeline(frame, boxes, ids)
                    timestamp = gt_ts if gt_frame else timestamp
                if output is None:
                    frame_idx += 1
                    continue
                pred_boxes = output.get('boxes2D', []) if isinstance(output, dict) else []
            except Exception as exc:
                LOGGER.warning('pipeline_error video=%s frame=%d: %s',
                                video_id, frame_idx, exc)
                pred_boxes = []

            if pred_boxes != []:
                frame_idx_list.append(frame_idx)
                timestamp_list.append(timestamp)
                pred_boxes_list.append(pred_boxes)
                gt_list.append(gt_frame)



            frame_idx += 1
    finally:
        cap.release()

    elapsed = time.time() - t0

    with open(predictions_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        for idx, timestamp, pred_boxes, gt in zip(frame_idx_list, timestamp_list,
                                                  pred_boxes_list, gt_list):
            _write_prediction_rows(writer, idx, timestamp, pred_boxes, width, height, gt)

    if aggregate_time_csv is not None:
        append_aggregate_time(aggregate_time_csv, video_id, elapsed, frame_idx,
                              frame_idx / elapsed if elapsed > 0 else 0.0, width, height)

    return frame_idx, elapsed



def main():
    args = parse_args()
    log_path = setup_logging('uniTalk_ASD', args.verbose, args.log_file)

    try:
        run_asd_on_video(args.video, args.predictions,
                          aggregate_time_csv=args.aggregate_time,
                          video_id=args.video_id, architecture=args.architecture)
    except Exception:
        LOGGER.exception('Failed processing video=%s', args.video)
        raise SystemExit(1)
    LOGGER.info('run complete')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
