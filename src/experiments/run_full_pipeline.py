"""
Full VVAD prediction pipeline over a dataset's videos.

UniTalk videos come from YouTube via download_uni_talk; AVA videos come from S3
via AvaDataset. Everything after acquisition is shared.
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

from .run_asd_on_unitalk_video import run_asd_on_video
from .download_uni_talk import download_video
from .helpers import (
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
        description='Run VVAD over every video of a dataset and write predictions.'
    )
    p.add_argument('--data_dir', default='data',
                   help='Root data directory (default: data/)')
    p.add_argument('--dataset', default='unitalk', choices=['unitalk', 'ava'],
                   help='Which dataset to run over (default: unitalk)')
    p.add_argument('--split', default='val',
                   help='Dataset split; UniTalk only (default: val)')
    p.add_argument('--no_download', action='store_true',
                   help='Skip all downloads; fail if a video is missing instead of fetching it')
    p.add_argument('--video', default=None,
                   help='Evaluate a single video_id only')
    p.add_argument('--predictions_dir', default="predictions",
                   help='Directory for per-video predictions CSVs '
                        '(default: <data_dir>/predictions)')
    p.add_argument('--architecture', default='CNN2Plus1D_Light',
                   help='Override the auto-generated log file path')
    p.add_argument('--stride', default=1,
                   help='Integer. How many frames are between the predictions (computational expansive (low stride) vs high latency (high stride))')
    p.add_argument('--verbose', '-v', action='store_true',
                   help='Also emit INFO-level messages on the console')
    return p.parse_args()


def load_video_list(args, data_dir: Path):
    """Return an ordered list of (video_id, video_path, url) for the chosen dataset.

    url is None when the file is already on disk: AvaDataset fetches AVA videos up
    front, whereas UniTalk videos are pulled from YouTube one at a time in the loop.
    """
    if args.dataset == 'ava':
        from asd4hri.ava_dataset import AvaDataset
        root = data_dir / 'ava'
        video_dir = root / 'videos'
        if args.no_download:
            names = sorted(os.listdir(video_dir)) if video_dir.is_dir() else []
        else:
            dataset = AvaDataset(root_dir=str(root))
            names, video_dir = dataset.file_names, Path(dataset.video_dir)
        return [(os.path.splitext(n)[0], video_dir / n, None) for n in names]

    video_dir = data_dir / 'videos' / args.split
    video_list_path = _ROOT.parent / 'video_list' / 'val.csv'
    if not video_list_path.exists():
        os.makedirs(video_list_path.parent, exist_ok=True)
        urllib.request.urlretrieve(_VIDEO_LIST_URL, video_list_path)

    items = []
    with open(video_list_path) as f:
        for line in f:
            url = line.strip()
            if not url or url == 'Link':
                continue
            vid = url.split('v=')[-1]
            items.append((vid, video_dir / f'{vid}.mp4', url))
    return items


# ── pipeline pass ─────────────────────────────────────────────────────────────

def run_pipeline_phase(args, video_list, result_dir,architecture, stride):
    """Download data on demand, run DetectVVAD on every video, write predictions.

    Returns:
        tuple[list[str], list[tuple], list[tuple]]:
            (processed, skipped, failed)
    """
    aggregate_time_csv = result_dir / 'aggregate_time.csv'

    os.makedirs(str(result_dir), exist_ok=True)

    LOGGER.info('phase=pipeline dataset=%s videos=%d result_dir=%s aggregate_time=%s '
                'no_download=%s', args.dataset, len(video_list), result_dir,
                aggregate_time_csv, args.no_download)

    processed = []
    skipped = []
    failed = []

    for i, (vid, video_path, url) in enumerate(video_list, 1):
        predictions_csv = result_dir / f'{vid}.csv'

        LOGGER.info('phase=pipeline video=%s index=%d/%d', vid, i, len(video_list))

        if url is not None and not args.no_download:
            try:
                download_video(vid, url, str(video_path.parent), download_videos=True)
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
            run_asd_on_video(
                str(video_path), str(predictions_csv),
                aggregate_time_csv=str(aggregate_time_csv), video_id=vid, architecture=architecture, stride=stride
            )
        except Exception as exc:
            LOGGER.exception('Pipeline failed video=%s', vid)
            failed.append((vid, str(exc)))
            continue
        processed.append(vid)

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

    video_list = load_video_list(args, data_dir)

    if args.video:
        video_list = [item for item in video_list if item[0] == args.video]
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

