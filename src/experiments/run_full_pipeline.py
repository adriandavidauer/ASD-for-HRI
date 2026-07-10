"""
Full UniTalk VVAD evaluation pipeline.
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

from asd4hri.run_vvad_on_unitalk_video import run_vvad_on_video
from download_uni_talk import download_video
from helpers import (
    setup_logging,
    load_annotations,
    load_predictions_csv,
    build_frame_map,
    create_writer,
    annotate_debug_frame,
)

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
    p.add_argument('--video', default=None,
                   help='Evaluate a single video_id only')
    p.add_argument('--predictions_dir', default="predictions",
                   help='Directory for per-video predictions CSVs '
                        '(default: <data_dir>/predictions)')
    p.add_argument('--output_videos_dir', default=None,
                   help='Directory for --debug annotated videos '
                        '(default: <data_dir>/output_videos/<split>)')
    p.add_argument('--debug', action='store_true',
                   help='Render an annotated output video overlaying pipeline '
                        'predictions on the ground-truth boxes (default: off)')
    p.add_argument('--iou_threshold', type=float, default=0.5,
                   help='IoU threshold for GT-prediction matching in --debug overlays '
                        '(default: 0.5)')
    p.add_argument('--architecture', default='CNN2Plus1D_Light',
                   help='Override the auto-generated log file path')
    p.add_argument('--stride', default=1, 
                   help='Integer. How many frames are between the predictions (computational expansive (low stride) vs high latency (high stride))')
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
     (green=GT, red=correct, yellow=wrong,
    black=unmatched prediction).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps:
        raise RuntimeError(f'Cannot read FPS from video: {video_path}')

    by_frame = load_predictions_csv(predictions_csv)
    frame_map = build_frame_map(annots_df, fps, width, height, video_id=video_id)

    writer, out_path = create_writer(output_video_path, fps, width, height)
    LOGGER.info('debug video=%s output=%s', video_id, out_path)

    frame_idx = 0
    try:
        while True:
            is_frame_received, frame = cap.read()
            if not is_frame_received:
                break
            pred_boxes = by_frame.get(frame_idx, [])
            gt_boxes = frame_map.get(frame_idx, [])
            annotate_debug_frame(frame, frame_idx, pred_boxes, gt_boxes, iou_threshold)
            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()


# ── pipeline pass ─────────────────────────────────────────────────────────────

def run_pipeline_phase(args, video_list, result_dir,architecture, stride):
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
                'no_download=%s debug=%s',
                len(video_list), result_dir, aggregate_time_csv,
                args.no_download, args.debug)


    annots_df = None
    if args.debug:
        annotations_csv = csv_dir / f'{args.split}_orig.csv'
        if annotations_csv.exists():
            annots_df = load_annotations(str(annotations_csv))
        else:
            LOGGER.warning('debug overlay disabled reason=no_annotation_csv path=%s',
                           annotations_csv)

    processed = []
    skipped = []
    failed = []

    for i, (vid, url) in enumerate(video_list, 1):
        video_path = video_dir / f'{vid}.mp4'
        predictions_csv = result_dir / f'{vid}.csv'

        LOGGER.info('phase=pipeline video=%s index=%d/%d', vid, i, len(video_list))

        if not args.no_download:
            try:
                download_video(vid, url, str(video_dir), download_videos=True)
            except Exception:
                LOGGER.exception('Download failed video=%s', vid)
                skipped.append((vid, 'download failed'))
                continue

        if not video_path.exists():
            LOGGER.warning('Skipping video=%s reason=video_file_missing path=%s',
                           vid, video_path)
            skipped.append((vid, 'video file missing'))
            continue


        try:
            run_vvad_on_video(
                str(video_path), str(predictions_csv),
                aggregate_time_csv=str(aggregate_time_csv), video_id=vid, architecture=architecture, stride=stride
            )
        except Exception as exc:
            LOGGER.exception('Pipeline failed video=%s', vid)
            failed.append((vid, str(exc)))
            continue
        processed.append(vid)

        # Optional annotated debug video (predictions overlaid on ground truth).
        if args.debug and annots_df is not None:
            video_annots = annots_df[annots_df['video_id'] == vid]
            if video_annots.empty:
                LOGGER.warning('debug skipped video=%s reason=no_annotations_for_video', vid)
            else:
                try:
                    out_path = output_videos_dir / f'{vid}_debug.mp4'
                    render_debug_video(str(video_path), str(predictions_csv), video_annots,
                                       str(out_path), args.iou_threshold, vid)
                except Exception:
                    LOGGER.exception('debug video failed video=%s', vid)

    return processed, skipped, failed


# ── summary ───────────────────────────────────────────────────────────────────

def print_run_summary(video_list, processed, skipped, failed, result_dir):
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
   
    result_dir = Path(data_dir / args.predictions_dir)

    LOGGER.info('run start data_dir=%s log_file=%s started_at=%s',
                data_dir.resolve(), log_path,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    video_list_path = Path(_ROOT.parent / 'video_list' / 'val.csv')
    if not os.path.exists(video_list_path):
        urllib.request.urlretrieve(_VIDEO_LIST_URL,video_list_path )
    video_list = load_video_list(video_list_path)

    if args.video:
        video_list = [(v, u) for v, u in video_list if v == args.video]
        if not video_list:
            LOGGER.error('video_id %s not found in video_list', args.video)
            raise SystemExit(1)

    processed, skipped, failed = run_pipeline_phase(args, video_list, result_dir, architecture=args.architecture, stride=args.stride)

    print_run_summary(video_list, processed, skipped, failed, result_dir)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.info('Interrupted.')
    except Exception as exc:
        LOGGER.exception('Fatal: %s', exc)
        raise SystemExit(1)

