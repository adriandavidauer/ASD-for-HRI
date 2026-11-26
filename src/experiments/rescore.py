r"""Rescore finished benchmark runs from their saved predictions (#118, #127).

New metrics (early-track mAP, the detector-cost breakdown) don't need the models to run again:
every run keeps its predictions. This recomputes ``stats/`` of each run from them, keeps the
run's own settings, version and timing, and notes the code that rescored it under
``rescored``. A carried baseline therefore gets the new metrics without being rerun.

It also fills in the detector-cost breakdown (``detector_cost``): for every detector run, the
GT-box run of the same configuration is scored as if it had missed the faces the detector run
missed, which splits the drop to the full pipeline into faces not found, crop shape and extra
detections.

Usage (in the image, from ``/app``)::

    python -m src.experiments.rescore --data_dir /app/data --tag v0.0.0
"""

import argparse
import json
import logging
import os
from pathlib import Path

from .run_evaluation import _DEFAULT_GROUNDTRUTH, code_version
from .stats import EARLY_WINDOWS, load_box_scores, map_without_faces, run_stats

LOGGER = logging.getLogger('rescore')

# settings that make two runs the same configuration, apart from their boxes
CONFIG_KEYS = ('dataset', 'split', 'architecture', 'stride', 'av_window', 'audio_fallback',
               'audio', 'causal_mode', 'window', 'shape_passes', 'shape_start_box',
               'audio_gate', 'gate_mode')
CONFIG_DEFAULTS = {'shape_passes': 1, 'shape_start_box': 'none', 'audio_gate': 'none',
                   'gate_mode': 'product'}
# what a run's summary.json keeps from its own run (the metrics are recomputed)
RUN_INFO_KEYS = CONFIG_KEYS + ('boxes', 'host', 'machine', 'version', 'git_commit',
                               'started_at')


def _summary(run_dir):
    """The run's ``summary.json``.

    Returns:
        dict | None: The summary, None if the run has none.
    """
    path = Path(run_dir) / 'stats' / 'summary.json'
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


def _config(summary):
    """The configuration of a run, apart from its boxes.

    Returns:
        tuple: The values of ``CONFIG_KEYS``.
    """
    return tuple(summary.get(k, CONFIG_DEFAULTS.get(k)) for k in CONFIG_KEYS)


def rescore(run_dir, data_dir, workers=None):
    """Recompute the stats of one run from its predictions.

    Args:
        run_dir: The run folder (``predictions/`` and ``stats/``).
        data_dir: The data folder the GT CSVs are relative to.
        workers: Parallel scoring processes.

    Returns:
        dict: The new summary.

    Raises:
        FileNotFoundError: If the run has no summary or no predictions.
    """
    run_dir, data_dir = Path(run_dir), Path(data_dir)
    old = _summary(run_dir)
    if old is None or not (run_dir / 'predictions').is_dir():
        raise FileNotFoundError(f'{run_dir}: needs stats/summary.json and predictions/')
    info = {k: old[k] for k in RUN_INFO_KEYS if k in old}
    version, commit = code_version()
    info['rescored'] = {'version': version, 'git_commit': commit}
    gt_csv = data_dir / _DEFAULT_GROUNDTRUTH[old['dataset']].format(split=old['split'])
    stats_dir = run_dir / 'stats'
    categories = stats_dir / 'categories.csv'
    return run_stats(str(run_dir / 'predictions'), str(gt_csv), str(stats_dir),
                     workers=workers, log_dir=str(data_dir / 'logs_stats'), run_info=info,
                     categories_csv=str(categories) if categories.is_file() else None)


def detector_cost(benchmark_dir, tag=''):
    """Split each detector run's drop from GT boxes to the full pipeline into its causes (#134).

    The partner is the GT-box run of the same configuration (``CONFIG_KEYS``), of the same tag
    if there is one, otherwise the latest. Four mAPs, each one step closer to the full pipeline,
    so that the differences add up to the whole drop:

    ``map_gt``
        The partner: GT boxes, all faces.
    ``map_gt_found``
        The partner with a score of 0 for every face the detector run gave no score (missed by
        the detector, or no landmarks / buffer yet). ``map_gt - map_gt_found``: faces not found.
    ``map_no_fp``
        The detector run without its detections that match no face (its own summary). Same
        faces missing as the step before, only the boxes differ: ``map_gt_found - map_no_fp``
        is the crop shape.
    ``map``
        The full pipeline. ``map_no_fp - map``: extra detections.

    The same four exist per early-track window (``<step>_first_<w>``). Writes them, with
    ``gt_partner``, into the detector run's ``summary.json``.

    Args:
        benchmark_dir: The benchmark folder.
        tag: Only the detector runs of this tag; all if empty.

    Returns:
        int: Number of detector runs updated.
    """
    runs = {}
    for path in sorted(Path(benchmark_dir).glob('*/stats/summary.json')):
        with open(path) as f:
            runs[path.parent.parent] = json.load(f)
    partners = {}
    for run, s in runs.items():
        if s.get('boxes') == 'gt':
            partners.setdefault(_config(s), []).append(run)
    updated = 0
    for run, s in runs.items():
        if s.get('boxes') != 'detector' or (tag and not run.name.startswith(f'{tag}_')):
            continue
        candidates = partners.get(_config(s), [])
        if not candidates:
            continue
        same_tag = [c for c in candidates if tag and c.name.startswith(f'{tag}_')]
        partner = max(same_tag or candidates, key=lambda c: runs[c].get('started_at', ''))
        det_rows, gt_rows = (load_box_scores(r / 'stats') for r in (run, partner))
        if det_rows is None or gt_rows is None or 'map_no_fp' not in s:
            LOGGER.warning('%s: scored before #134, rescore it and its partner first', run.name)
            continue
        scored = det_rows.loc[det_rows['has_score'], ['video_id', 'frame_timestamp', 'entity_id']]
        found = map_without_faces(gt_rows, scored)
        p = runs[partner]
        s.pop('map_gt_on_detected', None)            # the #127 split, replaced by this one
        s.update({'gt_partner': partner.name, 'map_gt': p.get('map'),
                  'map_gt_found': found['map']})
        for _sec, w in EARLY_WINDOWS:
            s[f'map_gt_first_{w}'] = p.get(f'map_first_{w}')
            s[f'map_gt_found_first_{w}'] = found[f'map_first_{w}']
        with open(run / 'stats' / 'summary.json', 'w') as f:
            json.dump(s, f, indent=2, default=str)
        updated += 1
    return updated


def main(argv=None):
    """Command line entry point."""
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--data_dir', default='data')
    p.add_argument('--tag', default='', help='only the runs of this tag')
    p.add_argument('--runs', nargs='*', help='only these run folders (names)')
    p.add_argument('--workers', type=int, default=None)
    p.add_argument('--cost_only', action='store_true', help='only the detector-cost breakdown')
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    benchmark_dir = Path(args.data_dir) / 'benchmark'
    if not args.cost_only:
        for run in sorted(benchmark_dir.iterdir()):
            if not (run / 'stats' / 'summary.json').is_file():
                continue
            if args.tag and not run.name.startswith(f'{args.tag}_'):
                continue
            if args.runs and run.name not in args.runs:
                continue
            if not (run / 'predictions').is_dir():
                LOGGER.warning('%s: no predictions, skipped', run.name)
                continue
            LOGGER.info('rescore %s', run.name)
            rescore(run, args.data_dir, args.workers)
    LOGGER.info('detector cost: %d runs', detector_cost(benchmark_dir, args.tag))


if __name__ == '__main__':
    main()
