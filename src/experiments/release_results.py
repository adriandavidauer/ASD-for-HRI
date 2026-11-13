"""Collect the benchmark results of all ``v*`` releases and build the results page (#74).

Run by ``.github/workflows/results_chart.yml``. For every published release whose tag is a
version (``v0.0.0``, ``v1.2.0-rc1``, ...), oldest first, it downloads the release assets

* ``benchmark.csv``: the ``run_benchmark`` table of that version (required; a release without
  it is skipped)
* ``benchmark_pi.csv``: the same table of the Raspberry Pi runs (optional; gives the Pi fps)

and writes one page with ``make_results_chart``, one version per release, labelled by its tag.
Nothing is written to the repository; the page goes to the release and the docs site.

Usage (needs the ``gh`` CLI and ``GH_TOKEN``)::

    python -m src.experiments.release_results --repo owner/name -o results.html
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from .make_results_chart import build

ASSET = 'benchmark.csv'
PI_ASSET = 'benchmark_pi.csv'
_SEMVER = re.compile(r'^v(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?$')


def semver_key(tag):
    """Sort key of a version tag; a pre-release sorts before its release.

    Args:
        tag: Release tag, e.g. ``v1.2.0`` or ``v1.2.0-rc1``.

    Returns:
        tuple or None: The key, None when the tag is not a version.
    """
    m = _SEMVER.match(tag)
    if not m:
        return None
    major, minor, patch, pre = m.groups()
    return int(major), int(minor), int(patch), 0 if pre else 1, pre or ''


def select_releases(releases):
    """The published version releases, oldest first, with the assets they carry.

    Args:
        releases: Release objects as the GitHub API lists them (``tag_name``, ``draft``,
            ``assets[].name``).

    Returns:
        list[dict]: ``{'tag', 'csv', 'pi'}`` per release; ``csv``/``pi`` say whether
        ``benchmark.csv`` / ``benchmark_pi.csv`` are attached.
    """
    out = []
    for r in releases:
        if r.get('draft') or semver_key(r.get('tag_name', '')) is None:
            continue
        names = {a['name'] for a in r.get('assets', [])}
        out.append({'tag': r['tag_name'], 'csv': ASSET in names, 'pi': PI_ASSET in names})
    return sorted(out, key=lambda r: semver_key(r['tag']))


def _gh(*args):
    """Run the GitHub CLI.

    Returns:
        str: Its standard output.
    """
    return subprocess.run(['gh', *args], check=True, capture_output=True, text=True).stdout


def list_releases(repo):
    """All releases of a repository.

    Args:
        repo: ``owner/name``.

    Returns:
        list[dict]: The releases.
    """
    pages = _gh('api', '--paginate', '--slurp', f'repos/{repo}/releases')
    return [r for page in json.loads(pages) for r in page]


def download(repo, tag, name, dest):
    """Download one release asset.

    Args:
        repo: ``owner/name``.
        tag: Release tag.
        name: Asset name.
        dest: Folder to put it in.

    Returns:
        Path: The downloaded file.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    _gh('release', 'download', tag, '--repo', repo, '--pattern', name, '--dir', str(dest), '--clobber')
    return dest / name


def collect(repo, workdir, out, log=print):
    """Download the results of all version releases and build the page.

    Args:
        repo: ``owner/name``.
        workdir: Folder for the downloads.
        out: The HTML file to write.
        log: Where the progress goes.

    Returns:
        dict: ``{'versions': [...], 'latest': tag or None, 'built': bool}``; ``latest`` is the
        newest version release, also when it has no results.
    """
    releases = select_releases(list_releases(repo))
    versions, pi = [], {}
    for r in releases:
        if not r['csv']:
            log(f"{r['tag']}: no {ASSET}, skipped")
            continue
        folder = Path(workdir) / r['tag']
        versions.append((r['tag'], download(repo, r['tag'], ASSET, folder)))
        if r['pi']:
            pi[r['tag']] = download(repo, r['tag'], PI_ASSET, folder)
        log(f"{r['tag']}: {ASSET}" + (f' + {PI_ASSET}' if r['pi'] else ''))
    latest = releases[-1]['tag'] if releases else None
    if not versions:
        log(f'no version release has {ASSET}; nothing to build')
        return {'versions': [], 'latest': latest, 'built': False}
    build(versions, out, pi=pi)
    log(f'{out}: {len(versions)} versions')
    return {'versions': [v for v, _ in versions], 'latest': latest, 'built': True}


def main(argv=None):
    """Command line entry point; writes ``built`` and ``latest`` to ``$GITHUB_OUTPUT``."""
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--repo', required=True, help='owner/name')
    p.add_argument('-o', '--out', default='results.html')
    p.add_argument('--workdir', default='release-results')
    args = p.parse_args(argv)
    result = collect(args.repo, args.workdir, args.out, log=lambda m: print(m, file=sys.stderr))
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            f.write(f"built={'true' if result['built'] else 'false'}\n")
            f.write(f"latest={result['latest'] or ''}\n")
    print(json.dumps(result))


if __name__ == '__main__':
    main()
