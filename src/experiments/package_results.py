"""Package evaluation outputs into zip files for upload to a GitHub release.

Standalone on purpose, like stats.py: stdlib only and no imports from the repository, so it
runs on the host next to the data. One zip per predictions dir holds the run's prediction CSVs
(per-video CSVs and ``aggregate_time.csv``) and its stats CSVs (per-video detail CSVs and
``aggregate_results.csv``), stored under the same paths as in ``data_dir``.
"""

import argparse
import zipfile
from pathlib import Path

STATS_ROOT = 'stats'                    # run_evaluation writes stats to <data_dir>/stats/<predictions_dir>
REQUIRED_PREDICTIONS = 'aggregate_time.csv'
REQUIRED_STATS = 'aggregate_results.csv'


def parse_args(argv=None):
    """Parse the command-line flags.

    Args:
        argv: Arguments to parse; ``None`` reads ``sys.argv``.

    Returns:
        argparse.Namespace: The parsed flags.
    """
    p = argparse.ArgumentParser(description='Zip prediction and stats CSVs of evaluation runs '
                                            'for upload to a GitHub release')
    p.add_argument('--data_dir', required=True,
                   help='Root data directory the evaluation wrote to')
    runs = p.add_mutually_exclusive_group(required=True)
    runs.add_argument('--predictions_dir', action='append',
                      help='Predictions folder relative to --data_dir; repeat for several runs')
    runs.add_argument('--all', action='store_true',
                      help=f'Package every predictions folder that has a matching {STATS_ROOT}/ folder')
    p.add_argument('--out_dir', default='release_assets',
                   help='Directory for the zip files (default: release_assets)')
    p.add_argument('--label', default=None,
                   help='Tag for the zip names, e.g. the hardware: gpu or pi')
    p.add_argument('--force', action='store_true',
                   help='Overwrite existing zip files')
    return p.parse_args(argv)


def discover_runs(data_dir):
    """Find every predictions folder in ``data_dir`` that has a matching stats folder.

    Args:
        data_dir: Root data directory.

    Returns:
        list[str]: Sorted predictions folder names, relative to ``data_dir``.

    Raises:
        FileNotFoundError: If no predictions folder has a matching stats folder.
    """
    data_dir = Path(data_dir)
    runs = sorted(d.name for d in (data_dir / STATS_ROOT).glob('*')
                  if d.is_dir() and (data_dir / d.name).is_dir())
    if not runs:
        raise FileNotFoundError(f'no predictions folder with a matching stats folder in {data_dir}')
    return runs


def collect_files(data_dir, predictions_dir):
    """List the CSVs of one run, checking that its aggregate files exist.

    Args:
        data_dir: Root data directory.
        predictions_dir: Predictions folder relative to ``data_dir``.

    Returns:
        list[pathlib.Path]: Sorted CSV paths relative to ``data_dir``.

    Raises:
        ValueError: If ``predictions_dir`` is absolute.
        FileNotFoundError: If a folder or an aggregate CSV is missing.
    """
    if Path(predictions_dir).is_absolute():
        raise ValueError(f'--predictions_dir must be relative to --data_dir: {predictions_dir}')
    data_dir = Path(data_dir)
    predictions = data_dir / predictions_dir
    stats = data_dir / STATS_ROOT / predictions_dir
    if not predictions.is_dir():
        raise FileNotFoundError(f'predictions folder not found: {predictions}')
    if not stats.is_dir():
        # run_evaluation used to put the stats of an absolute --predictions_dir into it (#67)
        raise FileNotFoundError(f'stats folder not found: {stats}. Runs made with an absolute '
                                '--predictions_dir before #67 have their stats CSVs inside '
                                'the predictions folder')
    files = []
    for folder, required in ((predictions, REQUIRED_PREDICTIONS), (stats, REQUIRED_STATS)):
        if not (folder / required).is_file():
            raise FileNotFoundError(f'{required} not found in {folder}')
        files += sorted(p.relative_to(data_dir) for p in folder.glob('*.csv'))
    return files


def zip_name(predictions_dir, label=None):
    """Build the zip file name of one run.

    Args:
        predictions_dir: Predictions folder relative to the data directory.
        label: Optional tag such as ``gpu`` or ``pi``.

    Returns:
        str: ``results_[<label>_]<predictions_dir>.zip``, with ``/`` replaced by ``_``.
    """
    parts = [label] if label else []
    return '_'.join(['results', *parts, *Path(predictions_dir).parts]) + '.zip'


def package_results(data_dir, predictions_dirs, out_dir, label=None, force=False):
    """Write one zip per predictions folder, after checking every run first.

    Args:
        data_dir: Root data directory.
        predictions_dirs: Predictions folders relative to ``data_dir``.
        out_dir: Directory for the zip files; created if missing.
        label: Optional tag for the zip names.
        force: Overwrite existing zip files.

    Returns:
        list[pathlib.Path]: The written zip files.

    Raises:
        FileExistsError: If a zip exists and ``force`` is not set.
    """
    out_dir = Path(out_dir)
    jobs = [(out_dir / zip_name(p, label), collect_files(data_dir, p)) for p in predictions_dirs]
    for zip_path, _ in jobs:
        if zip_path.exists() and not force:
            raise FileExistsError(f'{zip_path} exists; pass --force to overwrite')

    out_dir.mkdir(parents=True, exist_ok=True)
    for zip_path, files in jobs:
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                zf.write(Path(data_dir) / f, f.as_posix())
        print(f'{zip_path}: {len(files)} files')
    return [zip_path for zip_path, _ in jobs]


def main(argv=None):
    """Command-line entry point: parse flags and write the zip files.

    Args:
        argv: Arguments to parse; ``None`` reads ``sys.argv``.

    Raises:
        SystemExit: If a predictions dir is absolute, a folder or file is missing, or a zip
            exists without ``--force``.
    """
    args = parse_args(argv)
    try:
        runs = discover_runs(args.data_dir) if args.all else args.predictions_dir
        package_results(args.data_dir, runs, args.out_dir, args.label, args.force)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        raise SystemExit(f'error: {exc}') from exc


if __name__ == '__main__':
    main()
