"""
Full UniTalk VVAD evaluation pipeline.

Steps:
  1. Ensure annotation CSV is present (downloads if missing via src/dowload_uni_talk.py)
  2. Evaluate VVAD on each video in <data_dir>/videos/val/ — missing videos are
     downloaded automatically from YouTube via yt-dlp before evaluation
  3. Write per-video result files to <data_dir>/results/<video_id>_results.csv
  4. Write aggregate results to <data_dir>/results/aggregate_results.csv

Usage:
    # Auto mode: download missing videos on demand, then evaluate
    python run_full_pipeline.py --data_dir data/

    # Evaluate only – skip any downloads (videos must already be present)
    python run_full_pipeline.py --data_dir data/ --no_download

    # Resume an interrupted run (skips videos whose result file already exists)
    python run_full_pipeline.py --data_dir data/ --resume

    # Single video (useful for testing)
    python run_full_pipeline.py --data_dir data/ --video qv3-HaaxGUc
"""

import argparse
import csv
import logging
import os
import subprocess
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

# ── make src/ importable ──────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
_SRC  = _ROOT / 'src'
sys.path.insert(0, str(_SRC))

from run_vvad_on_unitalk_video import (     # noqa: E402
    load_annotations,
    evaluate_video,
    print_video_stats,
    print_aggregate_stats,
    setup_logging,
    CONSECUTIVE_FRAME_THRESHOLD,
)

LOGGER = logging.getLogger('pipeline')

_VIDEO_LIST_URL = (
    'https://raw.githubusercontent.com/plnguyen2908/UniTalk-ASD-code'
    '/main/video_list/val.csv'
)


# ── CSV result writers ────────────────────────────────────────────────────────

_PER_VIDEO_FIELDS = [
    'video_id', 'elapsed_s',
    'total_video_frames', 'frames_fed_to_pipeline', 'frames_with_predictions',
    'frames_no_predictions', 'no_pred_frames_with_gt', 'no_pred_frames_without_gt',
    'total_predictions', 'preds_on_annot_frames',
    'correct_predictions', 'prediction_accuracy_pct',
    'speaking_predictions', 'correct_speaking', 'fp_speaking', 'fn_speaking',
    'precision_speaking', 'recall_speaking', 'f1_speaking',
    'not_speaking_predictions', 'correct_not_speaking', 'fp_not_speaking', 'fn_not_speaking',
    'precision_not_speaking', 'recall_not_speaking', 'f1_not_speaking',
    'total_annotated_frames', 'annot_frames_no_predictions', 'missed_detection_pct',
    'gt_entities', 'detected_entities', 'entity_detection_rate_pct',
    f'entities_below_{CONSECUTIVE_FRAME_THRESHOLD}_consec_frames',
    'correct_timestamps', 'timestamp_accuracy_pct',
]


def _per_video_row(video_id, stats, elapsed):
    below = stats.entities_below_consecutive_threshold()
    no_pred_total = stats.frames_fed_to_pipeline - stats.frames_with_predictions
    missed_pct = 100.0 * stats.annot_frames_no_predictions / max(1, stats.total_annotated_frames)
    return {
        'video_id': video_id,
        'elapsed_s': round(elapsed, 2),
        'total_video_frames': stats.total_video_frames,
        'frames_fed_to_pipeline': stats.frames_fed_to_pipeline,
        'frames_with_predictions': stats.frames_with_predictions,
        'frames_no_predictions': no_pred_total,
        'no_pred_frames_with_gt': stats.annot_frames_no_predictions,
        'no_pred_frames_without_gt': stats.no_pred_frames_without_gt,
        'total_predictions': stats.total_predictions,
        'preds_on_annot_frames': stats.preds_on_annot_frames,
        'correct_predictions': stats.correct_predictions,
        'prediction_accuracy_pct': round(stats.prediction_accuracy_pct, 4),
        'speaking_predictions': stats.speaking_predictions,
        'correct_speaking': stats.correct_speaking,
        'fp_speaking': stats.fp_speaking,
        'fn_speaking': stats.fn_speaking,
        'precision_speaking': round(stats.precision_speaking, 4),
        'recall_speaking': round(stats.recall_speaking, 4),
        'f1_speaking': round(stats.f1_speaking, 4),
        'not_speaking_predictions': stats.not_speaking_predictions,
        'correct_not_speaking': stats.correct_not_speaking,
        'fp_not_speaking': stats.fp_not_speaking,
        'fn_not_speaking': stats.fn_not_speaking,
        'precision_not_speaking': round(stats.precision_not_speaking, 4),
        'recall_not_speaking': round(stats.recall_not_speaking, 4),
        'f1_not_speaking': round(stats.f1_not_speaking, 4),
        'total_annotated_frames': stats.total_annotated_frames,
        'annot_frames_no_predictions': stats.annot_frames_no_predictions,
        'missed_detection_pct': round(missed_pct, 4),
        'gt_entities': len(stats.gt_entities),
        'detected_entities': len(stats.detected_entities),
        'entity_detection_rate_pct': round(100.0 * stats.entity_detection_rate, 4),
        f'entities_below_{CONSECUTIVE_FRAME_THRESHOLD}_consec_frames': len(below),
        'correct_timestamps': stats.correct_timestamps,
        'timestamp_accuracy_pct': round(stats.timestamp_accuracy_pct, 4),
    }


def write_result_csv(result_dir, video_id, stats, elapsed):
    """Write per-video evaluation stats to <result_dir>/<video_id>_results.csv."""
    os.makedirs(result_dir, exist_ok=True)
    path = os.path.join(result_dir, f'{video_id}_results.csv')
    row = _per_video_row(video_id, stats, elapsed)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_PER_VIDEO_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    LOGGER.info('Results written to %s', path)


def write_aggregate_result_csv(result_dir, all_stats, video_ids):
    """Write all per-video rows plus an aggregate row to <result_dir>/aggregate_results.csv."""
    if not all_stats:
        return
    os.makedirs(result_dir, exist_ok=True)
    path = os.path.join(result_dir, 'aggregate_results.csv')

    csp  = sum(s.correct_speaking        for s in all_stats)
    fpsp = sum(s.fp_speaking             for s in all_stats)
    fnsp = sum(s.fn_speaking             for s in all_stats)
    cnsp = sum(s.correct_not_speaking    for s in all_stats)
    fpns = sum(s.fp_not_speaking         for s in all_stats)
    fnns = sum(s.fn_not_speaking         for s in all_stats)
    cp   = sum(s.correct_predictions     for s in all_stats)
    pa   = sum(s.preds_on_annot_frames   for s in all_stats)
    ct   = sum(s.correct_timestamps      for s in all_stats)
    tf   = sum(s.total_annotated_frames  for s in all_stats)
    det  = sum(len(s.detected_entities)  for s in all_stats)
    tot  = sum(len(s.gt_entities)        for s in all_stats)
    mnd  = sum(s.annot_frames_no_predictions for s in all_stats)
    below_count = sum(len(s.entities_below_consecutive_threshold()) for s in all_stats)

    p_sp  = csp  / max(1, csp  + fpsp)
    r_sp  = csp  / max(1, csp  + fnsp)
    f1_sp = 2 * p_sp * r_sp / max(1e-9, p_sp + r_sp)
    p_ns  = cnsp / max(1, cnsp + fpns)
    r_ns  = cnsp / max(1, cnsp + fnns)
    f1_ns = 2 * p_ns * r_ns / max(1e-9, p_ns + r_ns)

    agg_row = {
        'video_id': 'AGGREGATE',
        'elapsed_s': '',
        'total_video_frames': sum(s.total_video_frames for s in all_stats),
        'frames_fed_to_pipeline': sum(s.frames_fed_to_pipeline for s in all_stats),
        'frames_with_predictions': sum(s.frames_with_predictions for s in all_stats),
        'frames_no_predictions': sum(
            s.frames_fed_to_pipeline - s.frames_with_predictions for s in all_stats
        ),
        'no_pred_frames_with_gt': mnd,
        'no_pred_frames_without_gt': sum(s.no_pred_frames_without_gt for s in all_stats),
        'total_predictions': sum(s.total_predictions for s in all_stats),
        'preds_on_annot_frames': pa,
        'correct_predictions': cp,
        'prediction_accuracy_pct': round(100.0 * cp / max(1, pa), 4),
        'speaking_predictions': sum(s.speaking_predictions for s in all_stats),
        'correct_speaking': csp,
        'fp_speaking': fpsp,
        'fn_speaking': fnsp,
        'precision_speaking': round(p_sp, 4),
        'recall_speaking': round(r_sp, 4),
        'f1_speaking': round(f1_sp, 4),
        'not_speaking_predictions': sum(s.not_speaking_predictions for s in all_stats),
        'correct_not_speaking': cnsp,
        'fp_not_speaking': fpns,
        'fn_not_speaking': fnns,
        'precision_not_speaking': round(p_ns, 4),
        'recall_not_speaking': round(r_ns, 4),
        'f1_not_speaking': round(f1_ns, 4),
        'total_annotated_frames': tf,
        'annot_frames_no_predictions': mnd,
        'missed_detection_pct': round(100.0 * mnd / max(1, tf), 4),
        'gt_entities': tot,
        'detected_entities': det,
        'entity_detection_rate_pct': round(100.0 * det / max(1, tot), 4),
        f'entities_below_{CONSECUTIVE_FRAME_THRESHOLD}_consec_frames': below_count,
        'correct_timestamps': ct,
        'timestamp_accuracy_pct': round(100.0 * ct / max(1, tf), 4),
    }

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_PER_VIDEO_FIELDS)
        writer.writeheader()
        for vid, s in zip(video_ids, all_stats):
            writer.writerow(_per_video_row(vid, s, 0))
        writer.writerow(agg_row)
    LOGGER.info('Aggregate results written to %s', path)


# ── video-list bootstrap ──────────────────────────────────────────────────────

def fetch_video_list(dest: Path = _ROOT / 'video_list' / 'val.csv') -> None:
    """Download video_list/val.csv from the UniTalk GitHub repo.

    Creates the video_list/ directory if it does not exist, then fetches the
    raw CSV from GitHub and writes it to *dest*.  Skips the download if the
    file is already present.

    Args:
        dest: Destination path for the CSV (default: <project_root>/video_list/val.csv).
    """
    if dest.exists():
        LOGGER.info('video_list already present: %s', dest)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info('Fetching video list from %s', _VIDEO_LIST_URL)
    try:
        urllib.request.urlretrieve(_VIDEO_LIST_URL, dest)
        LOGGER.info('Saved video list to %s  (%d bytes)', dest, dest.stat().st_size)
    except Exception as exc:
        LOGGER.error('Failed to download video list: %s', exc)
        raise


# ── per-video YouTube download ────────────────────────────────────────────────

def _load_url_map(video_list_path: Path) -> dict:
    """Return {video_id: youtube_url} from video_list/val.csv."""
    url_map = {}
    if not video_list_path.exists():
        return url_map
    with open(video_list_path) as f:
        for line in f:
            url = line.strip()
            if not url or url == 'Link':
                continue
            vid = url.split('v=')[-1]
            url_map[vid] = url
    return url_map


def _ensure_video(vid: str, url: str, video_dir: Path) -> bool:
    """Download a single YouTube video via yt-dlp if not already present.

    Returns True when the file exists after the attempt.
    """
    target = video_dir / f'{vid}.mp4'
    if target.exists():
        return True

    os.makedirs(str(video_dir), exist_ok=True)
    LOGGER.info('Video not found locally — downloading %s', vid)
    result = subprocess.run([
        'yt-dlp',
        '--extractor-args', 'youtube:player_client=tv_embedded',
        '-f', 'bestvideo[vcodec^=avc1]+bestaudio/bestvideo[ext=mp4]+bestaudio/best[ext=mp4]/best',
        '--merge-output-format', 'mp4',
        '-o', str(video_dir / '%(id)s.%(ext)s'),
        url,
    ])
    if result.returncode != 0:
        LOGGER.warning('yt-dlp failed for %s', vid)
        return False
    return target.exists()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Download UniTalk val set and evaluate VVAD on every video.'
    )
    p.add_argument('--data_dir', default='data',
                   help='Root data directory (default: data/)')
    p.add_argument('--no_download', action='store_true',
                   help='Skip all downloads; fail if a video is missing instead of fetching it')
    p.add_argument('--resume', action='store_true',
                   help='Skip videos whose <video_id>_results.csv already exists')
    p.add_argument('--video', default=None,
                   help='Evaluate a single video_id only (skips aggregate output)')
    p.add_argument('--iou_threshold', type=float, default=0.5,
                   help='IoU threshold for GT-prediction matching (default: 0.5)')
    p.add_argument('--output_dir', default=None,
                   help='Directory for annotated output videos '
                        '(default: <data_dir>/output_videos/val)')
    p.add_argument('--enable_logging', action='store_true',
                   help='Write per-video debug log files under logs/')
    return p.parse_args()


# ── download step ─────────────────────────────────────────────────────────────

def run_download(data_dir: Path, download_videos: bool):
    """Run src/dowload_uni_talk.py from the project root (it needs video_list/ nearby)."""
    script = _SRC / 'dowload_uni_talk.py'
    if not script.exists():
        LOGGER.error('Download script not found: %s', script)
        sys.exit(1)

    cmd = [sys.executable, str(script), '--save_path', str(data_dir.resolve())]
    if download_videos:
        cmd.append('--download_videos')

    LOGGER.info('=' * 62)
    LOGGER.info('STEP 1 — Downloading UniTalk val split')
    LOGGER.info('Command : %s', ' '.join(cmd))
    LOGGER.info('Working : %s', _ROOT)
    LOGGER.info('=' * 62)

    result = subprocess.run(cmd, cwd=str(_ROOT))
    if result.returncode != 0:
        LOGGER.error('Download failed (exit %d). Aborting.', result.returncode)
        sys.exit(result.returncode)
    LOGGER.info('Download complete.')


# ── per-video evaluation loop ─────────────────────────────────────────────────

def run_evaluation(args):
    data_dir   = Path(args.data_dir)
    result_dir = data_dir / 'results'
    csv_path   = data_dir / 'csv' / 'val_orig.csv'
    video_dir  = data_dir / 'videos' / 'val'
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / 'output_videos' / 'val'

    if not csv_path.exists():
        LOGGER.error('Annotation CSV not found: %s', csv_path)
        LOGGER.error('Remove --no_download or check that data_dir is correct.')
        sys.exit(1)

    url_map = _load_url_map(_ROOT / 'video_list' / 'val.csv')

    all_annots = load_annotations(str(csv_path))
    video_ids  = [args.video] if args.video else sorted(all_annots)

    LOGGER.info('')
    LOGGER.info('=' * 62)
    LOGGER.info('STEP 2 — Evaluating VVAD on %d video(s)', len(video_ids))
    LOGGER.info('  data_dir   : %s', data_dir)
    LOGGER.info('  results    : %s', result_dir)
    LOGGER.info('  output_dir : %s', output_dir)
    LOGGER.info('  resume     : %s', args.resume)
    LOGGER.info('  no_download: %s', args.no_download)
    LOGGER.info('=' * 62)

    os.makedirs(str(output_dir), exist_ok=True)

    all_stats      = []
    processed_vids = []
    skipped        = []
    failed         = []

    for i, vid in enumerate(video_ids, 1):
        video_path  = video_dir / f'{vid}.mp4'
        result_file = result_dir / f'{vid}_results.csv'
        output_path = output_dir / f'{vid}_vvad.mp4'

        LOGGER.info('')
        LOGGER.info('── [%d/%d] %s %s', i, len(video_ids), vid, '─' * max(0, 40 - len(vid)))

        if not video_path.exists():
            if args.no_download:
                LOGGER.warning('Video file not found, skipping: %s', video_path)
                skipped.append((vid, 'video file missing'))
                continue
            url = url_map.get(vid)
            if not url:
                LOGGER.warning('No YouTube URL found for %s — cannot download, skipping.', vid)
                skipped.append((vid, 'no URL in video list'))
                continue
            if not _ensure_video(vid, url, video_dir):
                LOGGER.warning('Download failed for %s, skipping.', vid)
                skipped.append((vid, 'download failed'))
                continue

        annots = all_annots.get(vid, [])
        if not annots:
            LOGGER.warning('No GT annotations for %s, skipping.', vid)
            skipped.append((vid, 'no annotations'))
            continue

        if args.resume and result_file.exists():
            LOGGER.info('Result file already exists — skipping (--resume). %s', result_file)
            skipped.append((vid, 'already done'))
            continue

        try:
            stats, actual_out, elapsed = evaluate_video(
                str(video_path), annots, args.iou_threshold, str(output_path)
            )
            print_video_stats(vid, str(video_path), actual_out, stats, elapsed)
            write_result_csv(str(result_dir), vid, stats, elapsed)
            all_stats.append(stats)
            processed_vids.append(vid)
        except Exception as exc:
            LOGGER.exception('Error processing %s: %s', vid, exc)
            failed.append((vid, str(exc)))

    return all_stats, processed_vids, skipped, failed


# ── summary ───────────────────────────────────────────────────────────────────

def print_run_summary(video_ids, processed_vids, skipped, failed, result_dir):
    LOGGER.info('')
    LOGGER.info('=' * 62)
    LOGGER.info('RUN SUMMARY')
    LOGGER.info('=' * 62)
    LOGGER.info('  Total videos found    : %d', len(video_ids))
    LOGGER.info('  Successfully evaluated: %d', len(processed_vids))
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
    setup_logging(to_file=args.enable_logging)

    data_dir   = Path(args.data_dir)
    result_dir = data_dir / 'results'

    LOGGER.info('UniTalk VVAD full pipeline  |  %s', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    LOGGER.info('data_dir : %s', data_dir.resolve())

    # ── Step 0: Ensure video_list/val.csv is present ──────────────────────────
    fetch_video_list()

    # ── Step 1: Ensure annotation CSV is present ───────────────────────────────
    # The annotation CSV is required before evaluation; videos are downloaded
    # on demand per-video in the evaluation loop (unless --no_download is set).
    csv_path = data_dir / 'csv' / 'val_orig.csv'
    if not csv_path.exists():
        if args.no_download:
            LOGGER.error('Annotation CSV not found and --no_download is set: %s', csv_path)
            raise SystemExit(1)
        LOGGER.info('Annotation CSV missing — downloading dataset metadata.')
        run_download(data_dir, download_videos=False)
    else:
        LOGGER.info('Annotation CSV already present — skipping metadata download.')

    # ── Step 2 & 3: Evaluate + write per-video result files ───────────────────
    all_stats, processed_vids, skipped, failed = run_evaluation(args)

    # ── Step 4: Aggregate ─────────────────────────────────────────────────────
    if all_stats:
        LOGGER.info('')
        LOGGER.info('=' * 62)
        LOGGER.info('STEP 3 — Aggregate statistics (%d video(s))', len(all_stats))
        LOGGER.info('=' * 62)
        print_aggregate_stats(all_stats)
        write_aggregate_result_csv(str(result_dir), all_stats, processed_vids)
    else:
        LOGGER.warning('No videos were successfully evaluated — no aggregate file written.')

    # ── Final summary ─────────────────────────────────────────────────────────
    csv_path  = data_dir / 'csv' / 'val_orig.csv'
    all_annots = {}
    if csv_path.exists():
        all_annots = load_annotations(str(csv_path))
    video_ids = [args.video] if args.video else sorted(all_annots)
    print_run_summary(video_ids, processed_vids, skipped, failed, result_dir)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.info('Interrupted.')
    except Exception as exc:
        LOGGER.exception('Fatal: %s', exc)
        raise SystemExit(1)
