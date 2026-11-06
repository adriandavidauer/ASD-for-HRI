r"""Run the benchmark matrix and collect one comparable table (#83).

Every combination of ``--architectures`` x ``--datasets`` x ``--boxes`` is one run of
``run_evaluation``. Its predictions go to ``<data_dir>/benchmark/<run>/predictions`` and its
scores to ``<data_dir>/benchmark/<run>/stats``; a run whose ``stats/summary.json`` exists is
skipped, so an interrupted benchmark continues where it stopped, and the same command can run on
the cluster and on the Pi (results are kept apart by ``--tag``, e.g. ``gpu`` / ``pi``).

At the end all ``summary.json`` files under ``<data_dir>/benchmark`` are collected into
``benchmark.csv`` and ``benchmark.tex``.

Example::

    python -m src.experiments.run_benchmark --data_dir /app/data --no_download --tag gpu \\
        --architectures LipShapeCausal CNN2Plus1D_Light AV:LR-ASD AV:TalkNet \\
        --datasets ava unitalk wasd easycom --boxes gt detector

Extra arguments after ``--`` are passed on to every run, e.g. ``-- --stride 1``.
"""

import argparse
import csv
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

LOGGER = logging.getLogger('benchmark')

# default split per dataset (the one with public labels that the papers report)
DEFAULT_SPLIT = {'ava': 'val', 'unitalk': 'val', 'wasd': 'val', 'easycom': 'test'}

TABLE_COLUMNS = [
    ('dataset', 'Dataset', '{}'), ('architecture', 'Model', '{}'), ('boxes', 'Boxes', '{}'),
    ('map', 'mAP', '{:.1%}'), ('auc', 'AUC', '{:.3f}'), ('f1', 'F1', '{:.3f}'),
    ('balanced_accuracy', 'Bal. acc.', '{:.3f}'), ('det_recall', 'Det. recall', '{:.1%}'),
    ('map_small', 'mAP small', '{:.1%}'), ('fps', 'FPS', '{:.1f}'),
]


def run_name(tag, dataset, architecture, boxes, extra=()):
    """Folder name of one run, e.g. ``gpu_ava_AV-LR-ASD_gt``; extra flags are appended.

    Args:
        tag: Prefix such as 'gpu' or 'pi' (may be empty).
        dataset: Dataset name.
        architecture: Architecture name.
        boxes: 'gt' or 'detector'.
        extra: Extra flags passed to every run.

    Returns:
        str: A filesystem-safe name.
    """
    parts = [tag, dataset, architecture, boxes] + [a.lstrip('-') for a in extra]
    return re.sub(r'[^A-Za-z0-9_.-]+', '-', '_'.join(p for p in parts if p))


def build_command(args, dataset, architecture, boxes, run_dir):
    """The ``run_evaluation`` command line of one run (paths relative to --data_dir).

    Args:
        args: Parsed runner flags.
        dataset: Dataset name.
        architecture: Architecture name.
        boxes: 'gt' or 'detector'.
        run_dir: Output folder of the run.

    Returns:
        list[str]: The command.
    """
    rel = os.path.relpath(run_dir, args.data_dir)
    cmd = [sys.executable, '-m', 'src.experiments.run_evaluation',
           '--data_dir', args.data_dir, '--dataset', dataset,
           '--split', DEFAULT_SPLIT.get(dataset, 'val'),
           '--architecture', architecture, '--boxes', boxes,
           '--predictions_dir', f'{rel}/predictions', '--stats_dir', f'{rel}/stats']
    if args.no_download:
        cmd.append('--no_download')
    return cmd + list(args.extra)


def collect(benchmark_dir):
    """Read every ``*/stats/summary.json`` under ``benchmark_dir``.

    Args:
        benchmark_dir: The benchmark folder.

    Returns:
        list[dict]: One summary per run, with its folder name as ``run``.
    """
    rows = []
    for path in sorted(Path(benchmark_dir).glob('*/stats/summary.json')):
        with open(path) as f:
            summary = json.load(f)
        summary['run'] = path.parent.parent.name
        rows.append(summary)
    return rows


def write_csv(rows, path):
    """Write all summaries as CSV; every key is a column, nested values as JSON.

    Args:
        rows: Summaries from ``collect``.
        path: Output CSV.
    """
    keys = []
    for row in rows:
        keys += [k for k in row if k not in keys]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def _cell(value, fmt):
    """Format one LaTeX table cell.

    Args:
        value: The value, or None.
        fmt: ``str.format`` pattern.

    Returns:
        str: The cell text, '--' for missing values.
    """
    if value is None or value == '':
        return '--'
    try:
        return fmt.format(value).replace('%', r'\%')
    except (ValueError, TypeError):
        return str(value)


def write_latex(rows, path):
    """Write a booktabs table with the headline columns, grouped by dataset.

    Args:
        rows: Summaries from ``collect``.
        path: Output .tex file.
    """
    rows = sorted(rows, key=lambda r: (r.get('dataset', ''), r.get('boxes', ''),
                                       r.get('architecture', '')))
    lines = [r'\begin{tabular}{' + 'l' * 3 + 'r' * (len(TABLE_COLUMNS) - 3) + '}', r'\toprule',
             ' & '.join(h for _k, h, _f in TABLE_COLUMNS) + r' \\', r'\midrule']
    previous = None
    for row in rows:
        if previous is not None and row.get('dataset') != previous:
            lines.append(r'\midrule')
        previous = row.get('dataset')
        lines.append(' & '.join(_cell(row.get(k), f).replace('_', r'\_')
                                for k, _h, f in TABLE_COLUMNS) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main(argv=None):
    """Run the matrix, then collect the table.

    Args:
        argv: Command-line arguments (default: ``sys.argv[1:]``).

    Raises:
        SystemExit: If a run failed.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = argv[argv.index('--') + 1:] if '--' in argv else []
    argv = argv[:argv.index('--')] if '--' in argv else argv
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--data_dir', default='data')
    p.add_argument('--tag', default='', help='Prefix for the run folders, e.g. gpu or pi')
    p.add_argument('--architectures', nargs='+', default=['CNN2Plus1D_Light'])
    p.add_argument('--datasets', nargs='+', default=['unitalk'],
                   choices=['ava', 'unitalk', 'wasd', 'easycom'])
    p.add_argument('--boxes', nargs='+', default=['gt', 'detector'], choices=['gt', 'detector'])
    p.add_argument('--no_download', action='store_true')
    p.add_argument('--collect_only', action='store_true', help='Only rebuild the table')
    p.add_argument('--dry_run', action='store_true', help='Print the commands only')
    args = p.parse_args(argv)
    args.extra = extra
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    benchmark_dir = Path(args.data_dir) / 'benchmark'
    failed = []
    if not args.collect_only:
        for dataset in args.datasets:
            for architecture in args.architectures:
                for boxes in args.boxes:
                    run_dir = benchmark_dir / run_name(args.tag, dataset, architecture, boxes,
                                                       extra)
                    if (run_dir / 'stats' / 'summary.json').is_file():
                        LOGGER.info('skip (done): %s', run_dir.name)
                        continue
                    cmd = build_command(args, dataset, architecture, boxes, run_dir)
                    LOGGER.info('run: %s', ' '.join(cmd))
                    if args.dry_run:
                        continue
                    if subprocess.run(cmd).returncode != 0:
                        failed.append(run_dir.name)

    rows = collect(benchmark_dir)
    if rows:
        write_csv(rows, benchmark_dir / 'benchmark.csv')
        write_latex(rows, benchmark_dir / 'benchmark.tex')
        LOGGER.info('table: %s (%d runs)', benchmark_dir / 'benchmark.csv', len(rows))
    if failed:
        LOGGER.error('failed runs: %s', ', '.join(failed))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
