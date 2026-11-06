"""Tests for the release packaging script in experiments/package_results.py."""

import zipfile

import pytest

from experiments.package_results import main

RUN_FILES = [
    'predictions_a/vid1.csv',
    'predictions_a/aggregate_time.csv',
    'stats/predictions_a/vid1_detail.csv',
    'stats/predictions_a/aggregate_results.csv',
]


@pytest.fixture
def data_dir(tmp_path):
    """Build a fake data dir with one complete run plus files that must stay out of the zip.

    Args:
        tmp_path: Pytest's per-test temporary directory.

    Returns:
        pathlib.Path: The fake data dir.
    """
    root = tmp_path / 'data'
    for name in RUN_FILES + ['predictions_a/vid1.mp4', 'videos/val/vid1.mp4',
                             'logs_stats/unitalk_stats.log']:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(name)
    return root


def run(data_dir, out_dir, *flags):
    """Call the script's entry point.

    Args:
        data_dir: The fake data dir.
        out_dir: Where the zips go.
        *flags: Extra command-line flags.
    """
    main(['--data_dir', str(data_dir), '--out_dir', str(out_dir), *flags])


def test_zip_holds_run_csvs_under_relative_paths(data_dir, tmp_path):
    """The zip is named after the label and run and holds only that run's CSVs."""
    run(data_dir, tmp_path / 'out', '--predictions_dir', 'predictions_a', '--label', 'gpu')
    with zipfile.ZipFile(tmp_path / 'out' / 'results_gpu_predictions_a.zip') as zf:
        assert sorted(zf.namelist()) == sorted(RUN_FILES)
        assert {i.compress_type for i in zf.infolist()} == {zipfile.ZIP_DEFLATED}
        assert zf.read('stats/predictions_a/aggregate_results.csv').decode() == RUN_FILES[3]


def test_all_finds_runs_with_stats(data_dir, tmp_path):
    """--all packages runs that have a stats folder and skips those without one."""
    (data_dir / 'predictions_no_stats').mkdir()
    run(data_dir, tmp_path / 'out', '--all')
    assert [p.name for p in (tmp_path / 'out').iterdir()] == ['results_predictions_a.zip']


@pytest.mark.parametrize('missing, message', [
    ('predictions_a', 'folder not found'),
    ('predictions_a/aggregate_time.csv', 'aggregate_time.csv not found'),
    ('stats/predictions_a', 'folder not found'),
    ('stats/predictions_a/aggregate_results.csv', 'aggregate_results.csv not found'),
])
def test_missing_inputs_fail(data_dir, tmp_path, missing, message):
    """A missing folder or aggregate CSV stops the run before any zip is written.

    Args:
        data_dir: The fake data dir.
        tmp_path: Pytest's per-test temporary directory.
        missing: Path to delete, relative to the data dir.
        message: Expected part of the error.
    """
    path = data_dir / missing
    if path.is_dir():
        for child in path.iterdir():
            child.unlink()
        path.rmdir()
    else:
        path.unlink()
    with pytest.raises(SystemExit, match=message):
        run(data_dir, tmp_path / 'out', '--predictions_dir', 'predictions_a')
    assert not (tmp_path / 'out').exists()


def test_absolute_predictions_dir_fails(data_dir, tmp_path):
    """An absolute --predictions_dir is rejected instead of being looked up outside data_dir."""
    with pytest.raises(SystemExit, match='must be relative to --data_dir'):
        run(data_dir, tmp_path / 'out', '--predictions_dir', str(data_dir / 'predictions_a'))


def test_missing_stats_folder_hints_at_old_runs(data_dir, tmp_path):
    """Stats left inside the predictions folder by old runs are not picked up silently."""
    (data_dir / 'predictions_old').mkdir()
    (data_dir / 'predictions_old' / 'aggregate_results.csv').touch()
    with pytest.raises(SystemExit, match='absolute --predictions_dir before #67'):
        run(data_dir, tmp_path / 'out', '--predictions_dir', 'predictions_old')


def test_all_without_runs_fails(tmp_path):
    """--all fails when no predictions folder has stats."""
    with pytest.raises(SystemExit, match='no predictions folder'):
        run(tmp_path, tmp_path / 'out', '--all')


def test_existing_zip_needs_force(data_dir, tmp_path):
    """An existing zip is kept unless --force is given, then it is rewritten."""
    zip_path = tmp_path / 'out' / 'results_predictions_a.zip'
    zip_path.parent.mkdir()
    zip_path.write_text('old')
    with pytest.raises(SystemExit, match='--force'):
        run(data_dir, tmp_path / 'out', '--predictions_dir', 'predictions_a')
    assert zip_path.read_text() == 'old'

    run(data_dir, tmp_path / 'out', '--predictions_dir', 'predictions_a', '--force')
    assert zipfile.is_zipfile(zip_path)
