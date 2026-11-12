"""Tests for collecting the release results for the results page (#74)."""

import json

from src.experiments import release_results
from src.experiments.release_results import collect, select_releases, semver_key
from src.experiments.run_benchmark import write_csv


def _release(tag, *assets, draft=False):
    """A release as the GitHub API lists it.

    Returns:
        dict: The release.
    """
    return {'tag_name': tag, 'draft': draft, 'assets': [{'name': a} for a in assets]}


def _summary(arch, machine='x86_64', **extra):
    """A minimal ``summary.json`` row.

    Returns:
        dict: The row.
    """
    row = {'dataset': 'ava', 'architecture': arch, 'boxes': 'gt', 'machine': machine,
           'started_at': '2026-11-01T00:00', 'map': 0.5, 'fps': 100.0,
           'stage_ms_per_frame': {'model': 4.0, 'frame': 10.0}, 'run': arch}
    row.update(extra)
    return row


def test_semver_key_orders_pre_releases_before_releases():
    """v1.0.0-rc1 < v1.0.0 < v1.1.0 < v10.0.0; other tags are not versions."""
    tags = ['v10.0.0', 'v1.0.0', 'v1.1.0', 'v1.0.0-rc1']
    assert sorted(tags, key=semver_key) == ['v1.0.0-rc1', 'v1.0.0', 'v1.1.0', 'v10.0.0']
    assert semver_key('models') is None and semver_key('v1.0') is None


def test_select_releases_keeps_published_versions_in_order():
    """The models release and drafts are left out; the assets are reported per release."""
    releases = [_release('v0.1.0', 'benchmark.csv', 'benchmark_pi.csv'),
                _release('models', 'LipShapeCausal.keras'),
                _release('v0.0.0', 'benchmark.csv'),
                _release('v0.2.0', 'benchmark.csv', draft=True),
                _release('v0.0.1')]
    assert select_releases(releases) == [
        {'tag': 'v0.0.0', 'csv': True, 'pi': False},
        {'tag': 'v0.0.1', 'csv': False, 'pi': False},
        {'tag': 'v0.1.0', 'csv': True, 'pi': True}]


def _fake_github(monkeypatch, tmp_path, releases):
    """Serve ``releases`` and their CSV assets instead of the GitHub CLI.

    Returns:
        list: The downloads, as ``(tag, asset)``.
    """
    assets = tmp_path / 'assets'
    assets.mkdir()
    for r in releases:
        for a in r['assets']:
            name = a['name']
            rows = [_summary('LipShapeCausal', map=0.3 + 0.1 * len(r['tag_name']))]
            if name == 'benchmark_pi.csv':
                rows = [_summary('LipShapeCausal', machine='', dataset='ava', fps=10.0,
                                 stage_ms_per_frame={'frame': 100.0})]
            write_csv(rows, assets / f"{r['tag_name']}-{name}")
    downloads = []

    def fake_download(repo, tag, name, dest):
        """Copy the fake asset like ``gh release download`` would.

        Returns:
            Path: The file.
        """
        downloads.append((tag, name))
        dest.mkdir(parents=True, exist_ok=True)
        (dest / name).write_bytes((assets / f'{tag}-{name}').read_bytes())
        return dest / name

    monkeypatch.setattr(release_results, 'list_releases', lambda repo: releases)
    monkeypatch.setattr(release_results, 'download', fake_download)
    return downloads


def test_collect_builds_one_version_per_release_with_results(tmp_path, monkeypatch):
    """Releases without benchmark.csv are skipped and logged; the Pi CSV gives Pi runs."""
    releases = [_release('v0.0.0', 'benchmark.csv', 'benchmark_pi.csv'), _release('v0.0.1'),
                _release('v0.1.0', 'benchmark.csv'), _release('models', 'x.keras')]
    downloads = _fake_github(monkeypatch, tmp_path, releases)
    logs = []
    out = tmp_path / 'results.html'
    result = collect('o/r', tmp_path / 'work', out, log=logs.append)
    assert result == {'versions': ['v0.0.0', 'v0.1.0'], 'latest': 'v0.1.0', 'built': True}
    assert downloads == [('v0.0.0', 'benchmark.csv'), ('v0.0.0', 'benchmark_pi.csv'),
                         ('v0.1.0', 'benchmark.csv')]
    assert any('v0.0.1: no benchmark.csv' in m for m in logs)
    html = out.read_text()
    data = json.loads(html[html.index('const D = ') + 10:html.index(';\n', html.index('const D = '))])
    assert data['versions'] == ['v0.0.0', 'v0.1.0']
    assert sorted((r['v'], r['device']) for r in data['runs']) == [
        ('v0.0.0', 'gpu'), ('v0.0.0', 'pi'), ('v0.1.0', 'gpu')]


def test_collect_without_results_builds_nothing(tmp_path, monkeypatch):
    """No release has benchmark.csv yet: no page, but the latest version tag is known."""
    _fake_github(monkeypatch, tmp_path, [_release('v0.0.0'), _release('models', 'x.keras')])
    out = tmp_path / 'results.html'
    assert collect('o/r', tmp_path / 'work', out, log=lambda m: None) == {
        'versions': [], 'latest': 'v0.0.0', 'built': False}
    assert not out.exists()


def test_main_writes_the_github_outputs(tmp_path, monkeypatch):
    """``built`` and ``latest`` go to $GITHUB_OUTPUT for the later workflow steps."""
    _fake_github(monkeypatch, tmp_path, [_release('v0.0.0', 'benchmark.csv')])
    gh_out = tmp_path / 'gh_output'
    monkeypatch.setenv('GITHUB_OUTPUT', str(gh_out))
    release_results.main(['--repo', 'o/r', '-o', str(tmp_path / 'r.html'),
                          '--workdir', str(tmp_path / 'w')])
    assert gh_out.read_text() == 'built=true\nlatest=v0.0.0\n'
