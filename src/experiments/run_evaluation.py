"""End-to-end ASD evaluation: run predictions over a dataset, then calculates stats.
"""

import logging
from datetime import datetime
from pathlib import Path

from .helpers import setup_logging
from .run_full_pipeline import build_parser, load_video_list, print_run_summary, run_pipeline_phase
from .stats import TIMESTAMP_TOLERANCE_MS, run_stats

LOGGER = logging.getLogger('evaluation')


def parse_args():
    """Parse the prediction flags plus the scoring flags."""
    p = build_parser()
    p.description = 'Run ASD over every video of a dataset, then score the predictions.'
    p.add_argument('--groundtruth_csv', default='csv/val_orig.csv',
                   help='Master ground-truth CSV, relative to --data_dir (default: csv/val_orig.csv)')
    p.add_argument('--stats_dir', default=None,
                   help='Directory for the stats CSVs, relative to --data_dir '
                        '(default: stats/<predictions_dir>)')
    p.add_argument('--iou_threshold', type=float, default=0.5,
                   help='Minimum IoU to match a prediction to a GT box (default: 0.5)')
    p.add_argument('--timestamp_tolerance_ms', type=float, default=TIMESTAMP_TOLERANCE_MS,
                   help='Max |pred.ts - gt.ts| (ms) to align a prediction to a GT frame')
    p.add_argument('--workers', type=int, default=None,
                   help='Parallel worker processes for scoring (default: all CPUs)')
    p.add_argument('--skip_stats', action='store_true',
                   help='Only write predictions; do not score them')
    return p.parse_args()


def main():
    """Run the prediction phase, then the stats phase unless --skip_stats."""
    args = parse_args()
    log_path = setup_logging('evaluation', args.verbose)

    data_dir = Path(args.data_dir)
    result_dir = data_dir / args.predictions_dir

    LOGGER.info('run start data_dir=%s log_file=%s started_at=%s',
                data_dir.resolve(), log_path,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    video_list = load_video_list(args, data_dir)

    if args.video:
        video_list = [item for item in video_list if item[0] == args.video]
        if not video_list:
            LOGGER.error('video_id %s not found in video_list', args.video)
            raise SystemExit(1)

    processed, skipped, failed = run_pipeline_phase(args, video_list, result_dir,
                                                    architecture=args.architecture,
                                                    stride=args.stride)
    print_run_summary(video_list, processed, skipped, failed, result_dir)

    if args.skip_stats:
        return

    stats_dir = data_dir / (args.stats_dir or Path('stats') / args.predictions_dir)
    run_stats(str(result_dir), str(data_dir / args.groundtruth_csv), str(stats_dir),
              video=args.video, iou_threshold=args.iou_threshold,
              timestamp_tolerance_ms=args.timestamp_tolerance_ms,
              workers=args.workers, verbose=args.verbose,
              log_dir=str(data_dir / 'logs_stats'))
    LOGGER.info('Stats files in        : %s', stats_dir)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.info('Interrupted.')
    except Exception as exc:
        LOGGER.exception('Fatal: %s', exc)
        raise SystemExit(1)
