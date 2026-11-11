"""Full ASD prediction pipeline over a dataset's videos.

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

from .run_asd_on_video import run_asd_on_video
from .download_uni_talk import build_groundtruth_csv, download_video
from .helpers import setup_logging

LOGGER = logging.getLogger('pipeline')

_VIDEO_LIST_URL = (
    'https://raw.githubusercontent.com/plnguyen2908/UniTalk-ASD-code'
    '/main/video_list/val.csv'
)


def build_parser():
    """Return the argument parser for the prediction flags.

    Returns:
        argparse.ArgumentParser: The parser.
    """
    p = argparse.ArgumentParser(
        description='Run ASD over every video of a dataset and write predictions.'
    )
    p.add_argument('--data_dir', default='data',
                   help='Root data directory (default: data/)')
    p.add_argument('--dataset', default='unitalk', choices=['unitalk', 'ava', 'wasd', 'easycom'],
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
    p.add_argument('--av_window', type=int, default=25,
                   help="Frames per prediction for the audio-visual models 'AV:<model>' (#80)")
    p.add_argument('--audio_fallback', default='noise', choices=['noise', 'visual'],
                   help='What the audio-visual models get without audio (#40)')
    p.add_argument('--no_audio', action='store_true',
                   help='Ignore the audio track (tests the audio fallback)')
    p.add_argument('--causal_mode', default='streaming', choices=['streaming', 'window'],
                   help='Only for LipShapeCausal/FaceShapeCausal: one LSTM step per frame '
                        '(streaming) or the model on the last --window frames (window); see #45')
    p.add_argument('--window', type=int, default=38,
                   help='Frames per prediction for --causal_mode window (default: 38)')
    p.add_argument('--shape_passes', type=int, default=1,
                   help='dlib passes per face for the landmark models; 2 fits loose boxes '
                        'better (#92)')
    p.add_argument('--shape_box_mode', default='none',
                   choices=['none', 'fixed', 'previous', 'fixed+previous'],
                   help="Start box of dlib's first pass for the landmark models: the whole crop, "
                        'the typical landmark box of the box source, the previous frame\'s '
                        'landmarks, or both (#105)')
    p.add_argument('--boxes', default='detector', choices=['detector', 'gt'],
                   help='Face boxes to classify: the face detector (end-to-end protocol) or the '
                        'ground-truth boxes (literature protocol, detector skipped); see #75')
    p.add_argument('--verbose', '-v', action='store_true',
                   help='Also emit INFO-level messages on the console')
    return p


def load_groundtruth_boxes(groundtruth_csv):
    """Read an AVA-format GT CSV, with or without a header row.

    Args:
        groundtruth_csv: The CSV.

    Returns:
        dict: video_id -> list of ``(timestamp, entity_id, (x1, y1, x2, y2))``.
    """
    import csv
    by_video = {}
    with open(groundtruth_csv, newline='') as f:
        for row in csv.reader(f):
            if not row or row[0] == 'video_id':
                continue
            by_video.setdefault(row[0], []).append(
                (float(row[1]), row[7], tuple(float(v) for v in row[2:6])))
    return by_video


def load_video_list(args, data_dir: Path):
    """Return the ordered list of videos of the chosen dataset.

    AVA downloads every video and the ground-truth CSV up front, so ``url`` is None. UniTalk
    (fall-through branch) gives the YouTube link; ``run_pipeline_phase`` downloads each missing
    video on demand, and ``csv/<split>_orig.csv`` is built here if missing. Nothing is downloaded
    with ``--no_download``.

    Args:
        args: Parsed flags.
        data_dir: Root data directory.

    Returns:
        list[tuple]: ``(video_id, video_path, url)`` per video.
    """
    if args.dataset == 'easycom':
        from .easycom_dataset import EasyComDataset
        dataset = EasyComDataset(str(data_dir / 'easycom'), split=args.split)
        return [(vid, Path(path), url) for vid, path, url in dataset.video_list()]

    if args.dataset == 'wasd':
        from .wasd_dataset import WasdDataset
        dataset = WasdDataset(str(data_dir / 'wasd'), split=args.split,
                              download=not args.no_download)
        return [(vid, Path(path), url) for vid, path, url in dataset.video_list()]

    if args.dataset == 'ava':
        from .ava_dataset import AvaDataset
        root = data_dir / 'ava'
        video_dir = root / 'videos'
        if args.no_download:
            names = sorted(n for n in os.listdir(video_dir) if (
                root / 'annotations' / f'{os.path.splitext(n)[0]}-activespeaker.csv').exists()
            ) if video_dir.is_dir() else []
        else:
            dataset = AvaDataset(root_dir=str(root), log_dir=str(data_dir / 'logs'))
            names, video_dir = dataset.file_names, Path(dataset.video_dir)
        return [(os.path.splitext(n)[0], video_dir / n, None) for n in names]

    video_dir = data_dir / 'videos' / args.split
    video_list_path = _ROOT.parent / 'experiments' / 'video_list' / 'val.csv'
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

    groundtruth_csv = data_dir / 'csv' / f'{args.split}_orig.csv'
    if not args.no_download and not groundtruth_csv.exists():
        build_groundtruth_csv([vid for vid, _, _ in items], str(groundtruth_csv.parent), args.split)
    return items


# ── pipeline pass ─────────────────────────────────────────────────────────────

def _frame_crop(dataset, video_path):
    """Region of the frame to use: EasyCom videos carry a participant banner on the left.

    Args:
        dataset: Dataset name.
        video_path: The video.

    Returns:
        tuple or None: ``(x0, y0, x1, y1)``, or None to use the whole frame.
    """
    if dataset == 'easycom':
        from .easycom_dataset import EasyComDataset
        return EasyComDataset.frame_crop(video_path)
    return None


def annotated_range(gt_rows, warmup_s=1.0, tail_s=0.5):
    """Time span of a video's annotations, widened by ``warmup_s`` for the temporal buffers.

    Only this span is processed, e.g. the annotated 15 minutes of an AVA movie.

    Args:
        gt_rows: ``(timestamp, entity_id, box)`` rows of the video, or None.
        warmup_s: Seconds before the first annotation.
        tail_s: Seconds after the last annotation.

    Returns:
        tuple or None: ``(start_s, end_s)``, or None (= whole video) without annotations.
    """
    if not gt_rows:
        return None
    times = [row[0] for row in gt_rows]
    return max(0.0, min(times) - warmup_s), max(times) + tail_s


def run_pipeline_phase(args, video_list, result_dir, architecture, stride, groundtruth_csv=None):
    """Download data on demand, run ASD on every video, write predictions.

    With ``args.boxes == 'gt'`` the GT boxes from ``groundtruth_csv`` are classified instead of
    detected faces.

    Args:
        args: Parsed flags.
        video_list: Output of ``load_video_list``.
        result_dir: Folder for the prediction CSVs.
        architecture: Classifier architecture.
        stride: Frames between predictions.
        groundtruth_csv: GT CSV; needed for ``--boxes gt`` and to limit the processed time span.

    Returns:
        tuple[list, list, list]: ``(processed, skipped, failed)``.

    Raises:
        ValueError: If ``--boxes gt`` is used without a ground-truth CSV.
    """
    aggregate_time_csv = result_dir / 'aggregate_time.csv'

    os.makedirs(str(result_dir), exist_ok=True)

    LOGGER.info('phase=pipeline dataset=%s videos=%d result_dir=%s aggregate_time=%s '
                'no_download=%s', args.dataset, len(video_list), result_dir,
                aggregate_time_csv, args.no_download)

    processed = []
    skipped = []
    failed = []

    gt_boxes = None
    all_gt = load_groundtruth_boxes(groundtruth_csv) \
        if groundtruth_csv is not None and Path(groundtruth_csv).is_file() else {}
    if getattr(args, 'boxes', 'detector') == 'gt':
        if not all_gt:
            raise ValueError('--boxes gt needs the ground-truth CSV')
        gt_boxes = all_gt

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
                aggregate_time_csv=str(aggregate_time_csv), video_id=vid, architecture=architecture,
                stride=int(stride),
                groundtruth=None if gt_boxes is None else gt_boxes.get(vid, []),
                av_window=getattr(args, 'av_window', 25),
                audio_fallback=getattr(args, 'audio_fallback', 'noise'),
                use_audio=not getattr(args, 'no_audio', False),
                causal_mode=getattr(args, 'causal_mode', 'streaming'),
                window=getattr(args, 'window', 38),
                shape_passes=getattr(args, 'shape_passes', 1),
                shape_box_mode=getattr(args, 'shape_box_mode', 'none'),
                time_range=annotated_range(all_gt.get(vid)),
                frame_crop=_frame_crop(args.dataset, video_path)
            )
        except Exception as exc:
            LOGGER.exception('Pipeline failed video=%s', vid)
            failed.append((vid, str(exc)))
            continue
        processed.append(vid)

    return processed, skipped, failed


# ── summary ───────────────────────────────────────────────────────────────────

def print_run_summary(video_list, processed, skipped, failed, result_dir):
    """Log how many videos were processed, skipped and failed.

    Args:
        video_list: All videos of the run.
        processed: Ids of the processed videos.
        skipped: ``(video_id, reason)`` of the skipped videos.
        failed: ``(video_id, reason)`` of the failed videos.
        result_dir: Folder with the prediction CSVs.
    """
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
    """Command-line entry point.

    Raises:
        SystemExit: If a requested video is not found or processing fails.
    """
    args = build_parser().parse_args()
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

