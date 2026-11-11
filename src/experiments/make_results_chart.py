r"""Build the interactive results page from the ``benchmark.csv`` of several versions (#74).

Every release carries its benchmark results as a ``benchmark.csv`` asset (written by
``run_benchmark``). This script reads one CSV per version, in version order, and writes one
self-contained HTML page (Plotly from a CDN, the data inline):

1. speed vs. accuracy: metric over end-to-end ms per frame, full pipeline (circles) and
   ground-truth boxes (squares) switchable, with a version slider, trails and an optional
   Pareto front ("best trade-offs")
2. the metric of every configuration over the versions
3. where the time goes: the pipeline stages per configuration
4. a filterable table of all runs, with a protocol column

A configuration is ``(dataset, model, boxes, variant)`` and is matched across versions by that
key. Pi runs give the Pi fps of a configuration: they come from the version's
``benchmark_pi.csv`` (``--pi``), or are runs on an ``aarch64`` machine in ``benchmark.csv``.
They usually cover only a few videos, so their accuracy isn't plotted. A run gets the Pi
timing only of a Pi run of the same version, dataset, architecture, settings and boxes; without
one it has no Pi timing (it is left out of the Pi-timing views, and the page says how many runs
that are). No other run stands in for a missing one.

The version of a run is its ``version`` column (written by ``run_evaluation`` from the image's
``ASD4HRI_VERSION``) when present, otherwise the label given on the command line; a leading
``v`` is ignored when comparing the two.

Release assets (see ``release_results``): ``benchmark.csv`` (required) and
``benchmark_pi.csv`` (optional) per ``v*`` release.

Usage::

    python -m src.experiments.make_results_chart \
        --version v0.0.0=benchmark-0.0.0.csv --pi v0.0.0=benchmark_pi-0.0.0.csv \
        --version v0.1.0=benchmark-0.1.0.csv -o results.html

No result data belongs in the repository: the CSVs come from the release assets.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

TEMPLATE = Path(__file__).with_name('results_chart.html')
PI_MACHINES = {'aarch64', 'arm64'}
METRICS = ['map', 'f1', 'map_small', 'coverage', 'det_recall', 'auc', 'balanced_accuracy']


def family(architecture):
    """Model family used for the colours.

    Args:
        architecture: ``--architecture`` of the run.

    Returns:
        str: 'landmarks', 'pixels' or 'av'.
    """
    if architecture.startswith('AV:'):
        return 'av'
    if 'Shape' in architecture:
        return 'landmarks'
    return 'pixels'


def _num(value):
    """A CSV cell as float, or None when it's empty or not a number.

    Returns:
        float or None: The value.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _flag(value, default=True):
    """A CSV cell as bool (``True``/``False`` as written by ``run_benchmark``).

    Returns:
        bool: The value, ``default`` when empty.
    """
    if value in (None, ''):
        return default
    return str(value).strip().lower() in ('true', '1', 'yes')


def settings(row):
    """The settings of a run that change its result or speed, as a short string.

    Two runs of the same configuration with equal settings must give the same timing, so a
    Pi timing is reused for a later version only when its settings are equal.

    Args:
        row: One CSV row.

    Returns:
        str: E.g. ``'streaming, 2 dlib passes'``.
    """
    parts = [row.get('causal_mode') or 'streaming']
    if parts[0] == 'window':
        parts[0] = f"window {row.get('window')}"
    if (row.get('stride') or '1') not in ('1', '1.0'):
        parts.append(f"stride {row['stride']}")
    if row.get('architecture', '').startswith('AV:') and row.get('av_window') not in (None, '', '25'):
        parts.append(f"AV window {row['av_window']}")
    passes = row.get('shape_passes') or '1'
    if 'Shape' in row.get('architecture', '') and passes not in ('1', '1.0'):
        parts.append(f'{passes} dlib passes')
    if not _flag(row.get('audio')):
        parts.append('no audio')
    return ', '.join(parts)


def variant(row):
    """Part of the configuration key besides model and boxes: AV models without audio.

    Args:
        row: One CSV row.

    Returns:
        str: '' or 'face only'.
    """
    if row.get('architecture', '').startswith('AV:') and not _flag(row.get('audio')):
        return 'face only'
    return ''


def run_version(row, label):
    """The version of a run: its ``version`` column, else the label of its CSV.

    Args:
        row: One CSV row.
        label: Version label of the CSV.

    Returns:
        str: The version; ``label`` when both name the same version (``v`` prefix ignored).
    """
    own = (row.get('version') or '').strip()
    if not own or own.lstrip('v') == label.lstrip('v'):
        return label
    return own


def record(row, version, pi=False):
    """One run of the page.

    Args:
        row: One CSV row.
        version: Version label.
        pi: The row comes from a Pi CSV.

    Returns:
        dict: The fields the page uses.
    """
    stages = row.get('stage_ms_per_frame') or '{}'
    stages = json.loads(stages) if isinstance(stages, str) else stages
    frame_ms = stages.get('frame') or (1000.0 / _num(row['fps']) if _num(row.get('fps')) else None)
    out = {
        'v': version, 'dataset': row['dataset'], 'model': row['architecture'],
        'family': family(row['architecture']), 'boxes': row['boxes'], 'variant': variant(row),
        'settings': settings(row), 'device': 'pi' if pi or row.get('machine') in PI_MACHINES else 'gpu',
        'videos': _num(row.get('videos')), 'ms': frame_ms, 'fps': _num(row.get('fps')),
        'stages': {k: v for k, v in stages.items() if k != 'frame'}, 'run': row.get('run', ''),
        'started': row.get('started_at', ''),
    }
    for m in METRICS:
        out[m] = _num(row.get(m))
    return out


def load(versions, pi=None):
    """Read the CSVs and drop duplicate configurations within a version (the latest run wins).

    Args:
        versions: ``[(label, csv path), ...]`` in version order.
        pi: ``{label: csv path}`` of the Pi runs of a version.

    Returns:
        list[dict]: The runs.
    """
    runs = []
    files = []
    for label, path in versions:
        files.append((label, path, False))
        if pi and label in pi:
            files.append((label, pi[label], True))
    for label, path, is_pi in files:
        with open(path, newline='') as f:
            rows = [record(r, run_version(r, label), is_pi) for r in csv.DictReader(f)]
        latest = {}
        for r in rows:
            key = (r['v'], r['device'], r['dataset'], r['model'], r['boxes'], r['variant'])
            if key in latest:
                print(f'{label}: two runs of {key}, keeping the later one', file=sys.stderr)
                if r['started'] < latest[key]['started']:
                    continue
            latest[key] = r
        runs += latest.values()
    return runs


def attach_pi(runs):
    """Give every non-Pi run the timing of its Pi run, if there is one.

    The Pi run must have the same version, dataset, architecture, boxes, variant and settings;
    there is no fallback to another dataset or version.

    Args:
        runs: The runs (``load``); changed in place.

    Returns:
        list[dict]: The same runs; a matched run has ``pi = {'ms', 'stages', 'run'}``.
    """
    def key(r):
        """Identity of a configuration on one device.

        Returns:
            tuple: The key.
        """
        return r['v'], r['dataset'], r['model'], r['boxes'], r['variant'], r['settings']
    pi = {key(r): r for r in runs if r['device'] == 'pi'}
    for r in runs:
        if r['device'] != 'pi' and key(r) in pi:
            p = pi[key(r)]
            r['pi'] = {'ms': p['ms'], 'stages': p['stages'], 'run': p['run']}
    return runs


def build(versions, out_path, notes=None, banner='', fragment=False, title='ASD-for-HRI Results',
          pi=None):
    """Write the page.

    Args:
        versions: ``[(label, csv path), ...]`` in version order.
        out_path: The HTML file to write.
        notes: ``{label: text}`` shown under the version slider.
        banner: Text of a notice at the top (e.g. that the data is a preview).
        fragment: Leave out ``<!doctype>``, ``<html>``, ``<head>`` and ``<body>`` (for hosts that
            wrap the page themselves).
        title: Page title.
        pi: ``{label: csv path}`` of the Pi runs of a version (``benchmark_pi.csv``).

    Returns:
        dict: The data embedded in the page.
    """
    runs = attach_pi(load(versions, pi))
    order = []
    for v in [label for label, _ in versions] + [r['v'] for r in runs]:
        if v not in order and any(r['v'] == v for r in runs):
            order.append(v)
    data = {'versions': order, 'notes': notes or {}, 'banner': banner, 'runs': runs}
    html = TEMPLATE.read_text()
    payload = json.dumps(data, separators=(',', ':')).replace('</', '<\\/')
    html = html.replace('__TITLE__', title).replace('/*__DATA__*/null', payload)
    if not fragment:
        head, body = html.split('<!--BODY-->', 1)
        html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1, '
                'viewport-fit=cover">\n' + head + '</head>\n<body>\n' + body + '</body>\n</html>\n')
    else:
        html = html.replace('<!--BODY-->', '')
    Path(out_path).write_text(html)
    return data


def _pairs(values, what):
    """Split ``label=value`` arguments.

    Returns:
        list[tuple]: ``(label, value)`` pairs.

    Raises:
        SystemExit: If an argument has no ``=``.
    """
    out = []
    for v in values or []:
        if '=' not in v:
            raise SystemExit(f'{what} must look like label=value, got {v!r}')
        out.append(tuple(v.split('=', 1)))
    return out


def main(argv=None):
    """Command line entry point."""
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--version', action='append', required=True, metavar='LABEL=CSV',
                   help='one benchmark.csv per version, oldest first')
    p.add_argument('--pi', action='append', metavar='LABEL=CSV',
                   help='benchmark_pi.csv of a version: its Pi runs (timing)')
    p.add_argument('--note', action='append', metavar='LABEL=TEXT', help='note for a version')
    p.add_argument('--banner', default='', help='notice at the top of the page')
    p.add_argument('--title', default='ASD-for-HRI Results')
    p.add_argument('--fragment', action='store_true', help='no <html>/<head>/<body> wrapper')
    p.add_argument('-o', '--out', default='results.html')
    args = p.parse_args(argv)
    data = build(_pairs(args.version, '--version'), args.out, dict(_pairs(args.note, '--note')),
                 args.banner, args.fragment, args.title, dict(_pairs(args.pi, '--pi')))
    print(f"{args.out}: {len(data['runs'])} runs, versions {', '.join(data['versions'])}")


if __name__ == '__main__':
    main()
