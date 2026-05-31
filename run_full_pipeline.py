"""
Full UniTalk VVAD evaluation pipeline.

Walks the video_list, and for every (video_id, url) pair downloads the full
video and its per-video annotation CSV on demand (skipping anything already on
disk), runs DetectVVAD, and writes a per-video predictions CSV.  With --debug it
also renders an annotated output video overlaying pipeline predictions on the
ground-truth boxes.
"""

import argparse
import logging
import os
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / 'src'
sys.path.insert(0, str(_SRC))

import cv2

from run_vvad_on_unitalk_video import run_vvad_on_video
from download_uni_talk import download_video_and_csv
from helpers import (
    setup_logging,
    load_annotations,
    load_predictions_csv,
    build_frame_map,
    create_writer,
    draw_annotated_frame,
    read_video_metadata,
    TARGET_FPS,
)
from stats import compute_iou_matrix, match_predictions_to_gt

LOGGER = logging.getLogger('pipeline')

_VIDEO_LIST_URL = (
    'https://raw.githubusercontent.com/plnguyen2908/UniTalk-ASD-code'
    '/main/video_list/val.csv'
)


def parse_args():
    p = argparse.ArgumentParser(
        description='Download UniTalk val set and evaluate VVAD on every video.'
    )
    p.add_argument('--data_dir', default='data',
                   help='Root data directory (default: data/)')
    p.add_argument('--split', default='val',
                   help='Dataset split (default: val)')
    p.add_argument('--no_download', action='store_true',
                   help='Skip all downloads; fail if a video is missing instead of fetching it')
    p.add_argument('--resume', action='store_true',
                   help='Skip videos whose <video_id>_results.csv already exists')
    p.add_argument('--video', default=None,
                   help='Evaluate a single video_id only')
    p.add_argument('--iou_threshold', type=float, default=0.5,
                   help='IoU threshold for GT-prediction matching (default: 0.5)')
    p.add_argument('--predictions_dir', default=None,
                   help='Directory for per-video predictions CSVs '
                        '(default: <data_dir>/predictions)')
    p.add_argument('--output_videos_dir', default=None,
                   help='Directory for --debug annotated videos '
                        '(default: <data_dir>/output_videos/<split>)')
    p.add_argument('--debug', action='store_true',
                   help='Render an annotated output video overlaying pipeline '
                        'predictions on the ground-truth boxes (default: off)')
    p.add_argument('--log_file', default=None,
                   help='Override the auto-generated log file path')
    p.add_argument('--verbose', '-v', action='store_true',
                   help='Also emit INFO-level messages on the console')
    return p.parse_args()


def load_video_list(video_list_path: Path):
    """Return an ordered list of (video_id, youtube_url) from video_list/<split>.csv."""
    items = []
    if not video_list_path.exists():
        return items
    with open(video_list_path) as f:
        for line in f:
            url = line.strip()
            if not url or url == 'Link':
                continue
            vid = url.split('v=')[-1]
            items.append((vid, url))
    return items


# ── debug rendering ───────────────────────────────────────────────────────────

def render_debug_video(video_path, predictions_csv, annots_df,
                       output_video_path, iou_threshold, video_id):
    """Render an annotated video overlaying predictions and ground-truth boxes.

    Loads the per-video annotations (DataFrame) and the predictions CSV, matches
    them per frame, and draws both with helpers.draw_annotated_frame so the
    colours match the evaluation report (green=GT, red=correct, yellow=wrong,
    black=unmatched prediction).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    by_frame = load_predictions_csv(predictions_csv)
    frame_map = build_frame_map(annots_df, TARGET_FPS, width, height, video_id=video_id)

    writer, out_path = create_writer(output_video_path, TARGET_FPS, width, height)
    LOGGER.info('debug video=%s output=%s', video_id, out_path)

    frame_idx = 0
    try:
        while True:
            is_frame_received, frame = cap.read()
            if not is_frame_received:
                break
            pred_boxes = by_frame.get(frame_idx, [])
            gt_boxes = frame_map.get(frame_idx, [])
            iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
            matches = match_predictions_to_gt(pred_boxes, gt_boxes, iou_threshold, iou_matrix)
            draw_annotated_frame(frame, frame_idx, gt_boxes, pred_boxes, matches, iou_matrix)
            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()


# ── pipeline pass ─────────────────────────────────────────────────────────────

def run_pipeline_phase(args, video_list, result_dir):
    """Download data on demand, run DetectVVAD on every video, write predictions.

    With --debug, also renders an annotated output video per processed video.

    Returns:
        tuple[list[str], list[tuple], list[tuple]]:
            (processed, skipped, failed)
    """
    data_dir = Path(args.data_dir)
    video_dir = data_dir / 'videos' / args.split
    csv_dir = data_dir / 'csv'
    
    aggregate_time_csv = result_dir / 'aggregate_time.csv'

    os.makedirs(str(result_dir), exist_ok=True)
    output_videos_dir= None
    if args.debug:
        output_videos_dir = data_dir / 'output_videos'
        os.makedirs(str(output_videos_dir), exist_ok=True)

    LOGGER.info('phase=pipeline videos=%d result_dir=%s aggregate_time=%s '
                'resume=%s no_download=%s debug=%s',
                len(video_list), result_dir, aggregate_time_csv,
                args.resume, args.no_download, args.debug)

    processed = []
    skipped = []
    failed = []

    for i, (vid, url) in enumerate(video_list, 1):
        video_path = video_dir / f'{vid}.mp4'
        csv_path = csv_dir / f'{vid}.csv'
        predictions_csv = result_dir / f'{vid}.csv'

        LOGGER.info('phase=pipeline video=%s index=%d/%d', vid, i, len(video_list))

        # Fetch the video + annotation CSV on demand (download_video_and_csv
        # skips whatever already exists on disk).
        if not args.no_download:
            try:
                download_video_and_csv(vid, url, str(video_dir), str(csv_dir),
                                       split=args.split, download_videos=True)
            except Exception:
                LOGGER.exception('Download failed video=%s', vid)
                skipped.append((vid, 'download failed'))
                continue

        if not video_path.exists():
            LOGGER.warning('Skipping video=%s reason=video_file_missing path=%s',
                           vid, video_path)
            skipped.append((vid, 'video file missing'))
            continue

        # Run the pipeline (unless resuming and predictions already exist).
        if args.resume and predictions_csv.exists():
            LOGGER.info('Reusing predictions video=%s path=%s', vid, predictions_csv)
        else:
            try:
                run_vvad_on_video(
                    str(video_path), str(predictions_csv),
                    aggregate_time_csv=str(aggregate_time_csv), video_id=vid
                )
            except Exception as exc:
                LOGGER.exception('Pipeline failed video=%s', vid)
                failed.append((vid, str(exc)))
                continue
        processed.append(vid)

        # Optional annotated debug video (predictions overlaid on ground truth).
        if args.debug:
            if not csv_path.exists():
                LOGGER.warning('debug skipped video=%s reason=no_annotation_csv', vid)
            else:
                try:
                    annots_df = load_annotations(str(csv_path))
                    out_path = output_videos_dir / f'{vid}_debug.mp4'
                    render_debug_video(str(video_path), str(predictions_csv), annots_df,
                                       str(out_path), args.iou_threshold, vid)
                except Exception:
                    LOGGER.exception('debug video failed video=%s', vid)

    return processed, skipped, failed


# ── summary ───────────────────────────────────────────────────────────────────

def print_run_summary(video_list, processed, skipped, failed, result_dir):
    LOGGER.info('')
    LOGGER.info('=' * 62)
    LOGGER.info('RUN SUMMARY')
    LOGGER.info('=' * 62)
    LOGGER.info('  Total videos found    : %d', len(video_list))
    LOGGER.info('  Successfully evaluated: %d', len(processed))
    LOGGER.info('  Skipped               : %d', len(skipped))
    for vid, reason in skipped:
        LOGGER.info('    %-30s  (%s)', vid, reason)
    LOGGER.info('  Failed                : %d', len(failed))
    for vid, reason in failed:
        LOGGER.info('    %-30s  (%s)', vid, reason[:80])
    LOGGER.info('  Result files in       : %s', result_dir)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    log_path = setup_logging('pipeline', args.verbose)

    data_dir = Path(args.data_dir)
    result_dir = data_dir / 'predictions'

    LOGGER.info('run start data_dir=%s log_file=%s started_at=%s',
                data_dir.resolve(), log_path,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    video_list_path = Path(_ROOT / 'video_list' / 'val.csv')
    if not os.path.exists(video_list_path):
        urllib.request.urlretrieve(_VIDEO_LIST_URL,video_list_path )
    video_list = load_video_list(video_list_path)

    if args.video:
        video_list = [(v, u) for v, u in video_list if v == args.video]
        if not video_list:
            LOGGER.error('video_id %s not found in video_list', args.video)
            raise SystemExit(1)

    # ── Step 1: pipeline pass — downloads on demand, writes predictions ───────
    processed, skipped, failed = run_pipeline_phase(args, video_list, result_dir)

    print_run_summary(video_list, processed, skipped, failed, result_dir)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.info('Interrupted.')
    except Exception as exc:
        LOGGER.exception('Fatal: %s', exc)
        raise SystemExit(1)
