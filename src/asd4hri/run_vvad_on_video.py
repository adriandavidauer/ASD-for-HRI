"""Run DetectVVAD on a single video and persist its per-frame predictions.

Dataset-agnostic: it only needs a video file path.
"""

import argparse
import csv
import logging
import os
import time

from .asd import DetectVVAD

import cv2
import paz.pipelines.detection as dt

from helpers import setup_logging

LOGGER = logging.getLogger('VVAD')

def parse_args():
    p = argparse.ArgumentParser(
        description='Run DetectVVAD on a single video and write a predictions CSV')
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
    p.add_argument('--cascade',        default='frontalface_alt2',
                   help='OpenCV Haar cascade name (default: frontalface_alt2)')
    p.add_argument('--cascade_scale',  type=float, default=1.05,
                   help='Haar scaleFactor; lower finds more faces but is slower (default: 1.05)')
    p.add_argument('--cascade_neighbors', type=int, default=3,
                   help='Haar minNeighbors; lower is more permissive (default: 3)')
    p.add_argument('--verbose', '-v',  action='store_true',
                   help='Also emit INFO-level messages on the console')
    return p.parse_args()

# ── CSV writers ───────────────────────────────────────────────────────────────

_PREDICTION_FIELDS = ['frame_idx', 'timestamp', 'label', 'x1', 'y1', 'x2', 'y2', 'score']
_AGGREGATE_TIME_FIELDS = ['video_id', 'elapsed_seconds', 'frames_processed', 'fps_processed']


def append_aggregate_time(aggregate_time_csv, video_id, elapsed, frames_processed,fps):
    """Append one timing row, writing the header only when the file is new."""
    os.makedirs(os.path.dirname(aggregate_time_csv) or '.', exist_ok=True)
    new_file = not os.path.isfile(aggregate_time_csv)
    with open(aggregate_time_csv, 'a', newline='') as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(_AGGREGATE_TIME_FIELDS)
        w.writerow([video_id, f'{elapsed:.3f}', frames_processed, f'{fps:.2f}'])


# ── pipeline runner ───────────────────────────────────────────────────────────

def run_vvad_on_video(video_path,
                      predictions_csv=None,
                      aggregate_time_csv='results/aggregate_time.csv',
                      video_id=None, architecture='CNN2Plus1D_Light', stride=1,
                      cascade='frontalface_alt2', cascade_scale=1.05, cascade_neighbors=3):
    """Run DetectVVAD on one video and write predictions + timing rows.

    Args:
        video_path:         path to input .mp4
        predictions_csv:    output path for this video's predictions CSV.
                            Defaults to results/<video_id>_results.csv.
        aggregate_time_csv: timing CSV to append one row to (one row per video).
                            Defaults to results/aggregate_time.csv; pass None to skip.
        video_id:           override video_id used in logs / paths / timing row.
                            Defaults to the video file's basename.

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


    pipeline = DetectVVAD(stride=stride, averaging_window_size=1, min_frames=25, patience=10,
                          architecture=architecture, cascade=cascade,
                          cascade_scale=cascade_scale, cascade_neighbors=cascade_neighbors)
    os.makedirs(os.path.dirname(predictions_csv) or '.', exist_ok=True)

    t0 = time.time()
    frame_idx = 0
    width = height = 0

    try:
        rows = []
        while True:
            is_frame_received, frame = cap.read()
            if not is_frame_received:
                LOGGER.warning('Frame not received or End of stream')
                break
            height, width = frame.shape[:2]
            try:
                output= pipeline(frame)
                if output is None:
                    continue
                pred_boxes = output.get('boxes2D', []) if isinstance(output, dict) else []
            except Exception as exc:
                LOGGER.warning('pipeline_error video=%s frame=%d: %s',
                                video_id, frame_idx, exc)
                pred_boxes = []

            if pred_boxes:
                rows.append((frame_idx, frame_idx / native_fps, pred_boxes))

            frame_idx += 1
    finally:
        cap.release()

    elapsed = time.time() - t0

    with open(predictions_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(_PREDICTION_FIELDS)
        for idx, timestamp, pred_boxes in rows:
            for pred in pred_boxes:
                x1, y1, x2, y2 = pred.coordinates
                writer.writerow([idx, f'{timestamp:.6f}',
                                 getattr(pred, 'class_name', '') or '',
                                 f'{x1 / width:.6f}', f'{y1 / height:.6f}',
                                 f'{x2 / width:.6f}', f'{y2 / height:.6f}',
                                 f'{float(getattr(pred, "score", 0.0)):.6f}'])

    if aggregate_time_csv is not None:
        append_aggregate_time(aggregate_time_csv, video_id, elapsed, frame_idx,
                              frame_idx / elapsed if elapsed > 0 else 0.0)

    return frame_idx, elapsed



def main():
    args = parse_args()
    log_path = setup_logging('VVAD', args.verbose)

    try:
        run_vvad_on_video(args.video, args.predictions,
                          aggregate_time_csv=args.aggregate_time,
                          video_id=args.video_id, cascade=args.cascade,
                          cascade_scale=args.cascade_scale,
                          cascade_neighbors=args.cascade_neighbors)
    except Exception:
        LOGGER.exception('Failed processing video=%s', args.video)
        raise SystemExit(1)
    LOGGER.info('run complete')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
