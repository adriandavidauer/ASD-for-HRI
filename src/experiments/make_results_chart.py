r"""Build the interactive results page from the ``benchmark.csv`` of several versions (#74).

Every release carries its benchmark results as a ``benchmark.csv`` asset (written by
``run_benchmark``). This script reads one CSV per version, in version order, and writes one
self-contained HTML page (Plotly from a CDN, the data inline):

1. speed vs. accuracy: metric over ms per frame, with a version slider, trails and the Pareto front
2. the metric of every configuration over the versions
3. where the time goes: the pipeline stages per configuration
4. a filterable table of all runs

A configuration is ``(dataset, model, boxes, variant)`` and is matched across versions by that
key. Runs on an ``aarch64`` machine are Pi runs: they give the Pi timing of a ``(model, boxes,
variant)`` (they usually cover only a few videos, so their accuracy isn't used for the plot).

Usage::

    python -m src.experiments.make_results_chart \
        --version 0.0.0=benchmark-0.0.0.csv --version 0.1.0=benchmark-0.1.0.csv -o results.html

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


def record(row, version):
    """One run of the page.

    Args:
        row: One CSV row.
        version: Version label.

    Returns:
        dict: The fields the page uses.
    """
    stages = row.get('stage_ms_per_frame') or '{}'
    stages = json.loads(stages) if isinstance(stages, str) else stages
    frame_ms = stages.get('frame') or (1000.0 / _num(row['fps']) if _num(row.get('fps')) else None)
    out = {
        'v': version, 'dataset': row['dataset'], 'model': row['architecture'],
        'family': family(row['architecture']), 'boxes': row['boxes'], 'variant': variant(row),
        'settings': settings(row), 'device': 'pi' if row.get('machine') in PI_MACHINES else 'gpu',
        'videos': _num(row.get('videos')), 'ms': frame_ms, 'fps': _num(row.get('fps')),
        'stages': {k: v for k, v in stages.items() if k != 'frame'}, 'run': row.get('run', ''),
        'started': row.get('started_at', ''),
    }
    for m in METRICS:
        out[m] = _num(row.get(m))
    return out


def load(versions):
    """Read the CSVs and drop duplicate configurations within a version (the latest run wins).

    Args:
        versions: ``[(label, csv path), ...]`` in version order.

    Returns:
        list[dict]: The runs.
    """
    runs = []
    for label, path in versions:
        with open(path, newline='') as f:
            rows = [record(r, label) for r in csv.DictReader(f)]
        latest = {}
        for r in rows:
            key = (r['device'], r['dataset'], r['model'], r['boxes'], r['variant'])
            if key in latest:
                print(f'{label}: two runs of {key}, keeping the later one', file=sys.stderr)
                if r['started'] < latest[key]['started']:
                    continue
            latest[key] = r
        runs += latest.values()
    return runs


def build(versions, out_path, notes=None, banner='', fragment=False, title='ASD-for-HRI Results'):
    """Write the page.

    Args:
        versions: ``[(label, csv path), ...]`` in version order.
        out_path: The HTML file to write.
        notes: ``{label: text}`` shown under the version slider.
        banner: Text of a notice at the top (e.g. that the data is a preview).
        fragment: Leave out ``<!doctype>``, ``<html>``, ``<head>`` and ``<body>`` (for hosts that
            wrap the page themselves).
        title: Page title.

    Returns:
        dict: The data embedded in the page.
    """
    data = {'versions': [label for label, _ in versions], 'notes': notes or {}, 'banner': banner,
            'runs': load(versions)}
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
    p.add_argument('--note', action='append', metavar='LABEL=TEXT', help='note for a version')
    p.add_argument('--banner', default='', help='notice at the top of the page')
    p.add_argument('--title', default='ASD-for-HRI Results')
    p.add_argument('--fragment', action='store_true', help='no <html>/<head>/<body> wrapper')
    p.add_argument('-o', '--out', default='results.html')
    args = p.parse_args(argv)
    data = build(_pairs(args.version, '--version'), args.out, dict(_pairs(args.note, '--note')),
                 args.banner, args.fragment, args.title)
    print(f"{args.out}: {len(data['runs'])} runs, versions {', '.join(data['versions'])}")


if __name__ == '__main__':
    main()
