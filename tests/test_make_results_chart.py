"""Tests for the interactive results page (#74)."""

import json

from src.experiments.make_results_chart import build, settings
from src.experiments.run_benchmark import write_csv


def _summary(arch, boxes='gt', machine='x86_64', started='2026-11-01T00:00', **extra):
    """A ``summary.json`` as ``run_benchmark`` collects it.

    Returns:
        dict: The summary.
    """
    row = {'dataset': 'unitalk', 'split': 'val', 'architecture': arch, 'boxes': boxes,
           'stride': 1, 'av_window': 25, 'audio': True, 'causal_mode': 'streaming',
           'window': 38, 'machine': machine, 'started_at': started, 'videos': 18, 'map': 0.5,
           'f1': 0.4, 'coverage': 0.99, 'fps': 100.0,
           'stage_ms_per_frame': {'read': 1.0, 'model': 4.0, 'frame': 10.0}, 'run': arch}
    row.update(extra)
    return row


def _embedded(html):
    """The data embedded in the page.

    Returns:
        dict: The data.
    """
    start = html.index('const D = ') + len('const D = ')
    return json.loads(html[start:html.index(';\n', start)].replace('<\\/', '</'))


def test_page_holds_all_versions_and_runs(tmp_path):
    """Runs keep their version, Pi runs are marked, the frame time comes from the stages."""
    write_csv([_summary('LipShapeCausal'), _summary('LipShapeCausal', machine='aarch64'),
               _summary('AV:LR-ASD', audio=False)], tmp_path / 'a.csv')
    write_csv([_summary('LipShapeCausal', shape_passes=2)], tmp_path / 'b.csv')
    out = tmp_path / 'page.html'
    build([('0.0.0', tmp_path / 'a.csv'), ('0.1.0', tmp_path / 'b.csv')], out,
          notes={'0.1.0': 'fine-tuned'}, banner='preview')
    html = out.read_text()
    assert html.startswith('<!doctype html>') and '<!--BODY-->' not in html
    data = _embedded(html)
    assert data['versions'] == ['0.0.0', '0.1.0'] and data['notes'] == {'0.1.0': 'fine-tuned'}
    runs = data['runs']
    assert [r['v'] for r in runs] == ['0.0.0'] * 3 + ['0.1.0']
    assert sorted(r['device'] for r in runs if r['v'] == '0.0.0') == ['gpu', 'gpu', 'pi']
    assert [r['variant'] for r in runs if r['model'] == 'AV:LR-ASD'] == ['face only']
    assert runs[0]['ms'] == 10.0 and runs[0]['stages'] == {'read': 1.0, 'model': 4.0}
    assert runs[-1]['settings'] == 'streaming, 2 dlib passes'


def test_duplicate_configuration_keeps_the_later_run(tmp_path):
    """Two runs of one configuration in one version: the later one is used."""
    write_csv([_summary('LipShapeCausal', started='2026-11-02T00:00', map=0.3),
               _summary('LipShapeCausal', started='2026-11-01T00:00', map=0.9)], tmp_path / 'a.csv')
    out = tmp_path / 'page.html'
    data = build([('0.0.0', tmp_path / 'a.csv')], out, fragment=True)
    assert [r['map'] for r in data['runs']] == [0.3]
    assert '<html' not in out.read_text()


def test_settings_name_what_changes_results():
    """Defaults are left out; window mode, stride and dlib passes are named."""
    base = {'architecture': 'LipShapeCausal', 'causal_mode': 'streaming', 'stride': '1',
            'shape_passes': '1', 'audio': 'True'}
    assert settings(base) == 'streaming'
    assert settings(dict(base, causal_mode='window', window='38', stride='2')) == 'window 38, stride 2'
    assert settings(dict(base, architecture='AV:TalkNet', audio='False')) == 'streaming, no audio'


def test_version_column_wins_over_the_label(tmp_path):
    """A run's own version is used; 'v0.0.0' and '0.0.0' are the same version."""
    write_csv([_summary('LipShapeCausal', version='0.0.0'),
               _summary('FaceShapeCausal', version='0.0.1-dev'),
               _summary('CNN2Plus1D_Light')], tmp_path / 'a.csv')
    data = build([('v0.0.0', tmp_path / 'a.csv')], tmp_path / 'page.html')
    assert {r['model']: r['v'] for r in data['runs']} == {
        'LipShapeCausal': 'v0.0.0', 'FaceShapeCausal': '0.0.1-dev', 'CNN2Plus1D_Light': 'v0.0.0'}
    assert data['versions'] == ['v0.0.0', '0.0.1-dev']


def test_pi_csv_rows_are_pi_runs(tmp_path):
    """Rows of benchmark_pi.csv are Pi runs whatever their machine column says."""
    write_csv([_summary('LipShapeCausal')], tmp_path / 'a.csv')
    write_csv([_summary('LipShapeCausal', machine='')], tmp_path / 'pi.csv')
    data = build([('v0.0.0', tmp_path / 'a.csv')], tmp_path / 'page.html',
                 pi={'v0.0.0': tmp_path / 'pi.csv'})
    assert sorted(r['device'] for r in data['runs']) == ['gpu', 'pi']


def test_pi_timing_only_from_the_same_configuration(tmp_path):
    """A run gets Pi timing only from the Pi run of exactly its configuration.

    A Pi run of another dataset, version or setting is not used.
    """
    pi_ms = {'stage_ms_per_frame': {'frame': 100.0}}
    write_csv([_summary('LipShapeCausal'), _summary('LipShapeCausal', dataset='wasd'),
               _summary('FaceShapeCausal'), _summary('CNN2Plus1D_Light')], tmp_path / 'a.csv')
    write_csv([_summary('LipShapeCausal', **pi_ms),                      # matches unitalk Lip
               _summary('FaceShapeCausal', dataset='ava', **pi_ms),      # other dataset
               _summary('CNN2Plus1D_Light', stride=2, **pi_ms)], tmp_path / 'pi.csv')
    write_csv([_summary('FaceShapeCausal')], tmp_path / 'b.csv')
    data = build([('v0.0.0', tmp_path / 'a.csv'), ('v0.1.0', tmp_path / 'b.csv')],
                 tmp_path / 'page.html', pi={'v0.0.0': tmp_path / 'pi.csv'})
    timed = {(r['v'], r['dataset'], r['model']): r.get('pi', {}).get('ms')
             for r in data['runs'] if r['device'] == 'gpu'}
    assert timed == {('v0.0.0', 'unitalk', 'LipShapeCausal'): 100.0,
                     ('v0.0.0', 'wasd', 'LipShapeCausal'): None,
                     ('v0.0.0', 'unitalk', 'FaceShapeCausal'): None,
                     ('v0.0.0', 'unitalk', 'CNN2Plus1D_Light'): None,
                     ('v0.1.0', 'unitalk', 'FaceShapeCausal'): None}
