"""Tests for the benchmark runner (#83)."""

import json

from src.experiments import run_benchmark as rb


def test_run_name_is_filesystem_safe():
    """Run names contain no ':' or spaces, extra flags are appended."""
    assert rb.run_name('gpu', 'ava', 'AV:LR-ASD', 'gt') == 'gpu_ava_AV-LR-ASD_gt'
    assert rb.run_name('', 'wasd', 'LipShape', 'detector', ['--stride', '2']) == \
        'wasd_LipShape_detector_stride_2'


def test_dry_run_builds_one_command_per_combination(tmp_path, caplog):
    """One command per model x dataset x protocol, with the dataset's split."""
    caplog.set_level('INFO')
    rb.main(['--data_dir', str(tmp_path), '--architectures', 'A', 'B', '--datasets', 'ava',
             'easycom', '--boxes', 'gt', '--dry_run', '--', '--stride', '1'])
    runs = [r.message for r in caplog.records if r.message.startswith('run: ')]
    assert len(runs) == 4
    easycom = [r for r in runs if '--dataset easycom' in r][0]
    assert '--split test' in easycom and easycom.endswith('--stride 1')


def test_done_runs_are_skipped_and_collected(tmp_path, caplog):
    """Runs with a summary are skipped and end up in the CSV and LaTeX table."""
    caplog.set_level('INFO')
    for name, m in (('ava_A_gt', 0.9), ('unitalk_A_gt', 0.5)):
        stats = tmp_path / 'benchmark' / name / 'stats'
        stats.mkdir(parents=True)
        (stats / 'summary.json').write_text(json.dumps(
            {'dataset': name.split('_')[0], 'architecture': 'A', 'boxes': 'gt', 'map': m,
             'fps': 12.5, 'det_recall': None, 'map_by_category': {'OC': 0.9}}))
    rb.main(['--data_dir', str(tmp_path), '--architectures', 'A', '--datasets', 'ava',
             'unitalk', '--boxes', 'gt'])
    assert sum('skip (done)' in r.message for r in caplog.records) == 2
    csv_text = (tmp_path / 'benchmark' / 'benchmark.csv').read_text()
    assert 'ava_A_gt' in csv_text and '"{""OC"": 0.9}"' in csv_text
    tex = (tmp_path / 'benchmark' / 'benchmark.tex').read_text()
    assert r'90.0\%' in tex and r'\midrule' in tex and '--' in tex
