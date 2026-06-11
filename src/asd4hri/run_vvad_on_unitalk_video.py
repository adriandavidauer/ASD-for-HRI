"""Run DetectVVAD on a single video and persist its per-frame predictions.
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

LOGGER = logging.getLogger('uniTalk_VVAD')


# ── CSV writers ───────────────────────────────────────────────────────────────

_PREDICTION_FIELDS = ['frame_idx', 'timestamp', 'label', 'x1', 'y1', 'x2', 'y2']
_AGGREGATE_TIME_FIELDS = ['video_id', 'elapsed_seconds', 'frames_processed', 'fps_processed']

def _write_prediction_rows(writer, frame_idx, timestamp, pred_boxes, width, height):
    """Write one row per predicted box with bbox coords normalised to [0, 1].
    """
    for pred in pred_boxes:
        label = getattr(pred, 'class_name', '') or ''
        x1, y1, x2, y2 = pred.coordinates
        writer.writerow([frame_idx, f'{timestamp:.6f}', label,
                         f'{x1 / width:.6f}', f'{y1 / height:.6f}',
                         f'{x2 / width:.6f}', f'{y2 / height:.6f}'])


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
                      video_id=None, architecture='CNN2Plus1D_Light', stride=1):
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


    pipeline = DetectVVAD(stride=stride, averaging_window_size=1, min_frames=25, patience=10,architecture=architecture)
    os.makedirs(os.path.dirname(predictions_csv) or '.', exist_ok=True)

    t0 = time.time()
    frame_idx = 0

    try:
        with open(predictions_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(_PREDICTION_FIELDS)

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

                timestamp = frame_idx / native_fps
                if pred_boxes != []:
                    _write_prediction_rows(writer, frame_idx, timestamp,
                                           pred_boxes, width, height)

                frame_idx += 1
    finally:
        cap.release()

    elapsed = time.time() - t0

    if aggregate_time_csv is not None:
        append_aggregate_time(aggregate_time_csv, video_id, elapsed, frame_idx,native_fps)

    return frame_idx, elapsed


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
    p.add_argument('--verbose', '-v',  action='store_true',
                   help='Also emit INFO-level messages on the console')
    return p.parse_args()


def main():
    args = parse_args()
    log_path = setup_logging('uniTalk_VVAD', args.verbose)

    try:
        run_vvad_on_video(args.video, args.predictions,
                          aggregate_time_csv=args.aggregate_time,
                          video_id=args.video_id)
    except Exception:
        LOGGER.exception('Failed processing video=%s', args.video)
        raise SystemExit(1)
    LOGGER.info('run complete')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
