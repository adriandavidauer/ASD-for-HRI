"""Tests for the detector-cost breakdown in ``src/experiments/rescore.py`` (#134)."""

import json

import numpy as np
from src.experiments import stats
from src.experiments.rescore import detector_cost

POS, NEG = 'SPEAKING_AUDIBLE', 'NOT_SPEAKING'


def _gt_rows(n_frames=100):
    """Two entities, 'v:1' speaks in every other frame.

    Returns:
        list[dict]: GT rows.
    """
    rows = []
    for f in range(n_frames):
        ts = round(f * 0.04, 2)
        rows.append({'frame_timestamp': ts, 'bbox': (0.1, 0.1, 0.3, 0.4), 'ava_label': NEG,
                     'asd_label': 'not-speaking', 'entity_id': 'v:0'})
        lab = POS if f % 2 else NEG
        rows.append({'frame_timestamp': ts, 'bbox': (0.6, 0.1, 0.8, 0.4), 'ava_label': lab,
                     'asd_label': 'x', 'entity_id': 'v:1'})
    return rows


def _write_run(root, name, gt, score, boxes, drop=lambda r: False, extra=()):
    """Score made-up predictions into ``<root>/<name>/stats``."""
    by_ts = {}
    for r in gt:
        if drop(r):
            continue
        s = score(r)
        by_ts.setdefault(r['frame_timestamp'], []).append(stats._PredBox(
            r['bbox'], 'x', s, r['entity_id'] if boxes == 'gt' else None))
    for ts, box, s in extra:
        by_ts.setdefault(ts, []).append(stats._PredBox(box, 'x', s, None))
    protocol, gt_df, fp, n = stats.protocol_rows('v', stats.GroundTruthIndex(gt), by_ts, 0.02)
    info = {'dataset': 'ava', 'split': 'val', 'architecture': 'M', 'boxes': boxes,
            'started_at': '2026-01-01'}
    stats.write_protocol_results(str(root / name / 'stats'), [('v', protocol, gt_df, fp, n)],
                                 run_info=info)


def test_detector_cost_steps_split_the_drop_into_its_causes(tmp_path):
    """Faces not found, crop shape and extra detections, each in its own step."""
    gt = _gt_rows()
    rng = np.random.default_rng(0)
    noisy = {}

    def gt_score(r):
        # good, but not perfect, on GT boxes
        return noisy.setdefault((r['frame_timestamp'], r['entity_id']),
                                float(0.6 * (r['ava_label'] == POS) + 0.5 * rng.random()))

    _write_run(tmp_path, 't_ava_M_gt', gt, gt_score, 'gt')
    missed = lambda r: r['entity_id'] == 'v:1' and r['frame_timestamp'] < 1.0  # noqa: E731
    extra = [(round(f * 0.04, 2), (0.4, 0.5, 0.55, 0.9), 0.99) for f in range(60)]
    # the detector run scores perfectly where it scores: its crops are better than the GT boxes
    _write_run(tmp_path, 't_ava_M_detector', gt, lambda r: float(r['ava_label'] == POS),
               'detector', drop=missed, extra=extra)
    assert detector_cost(tmp_path, 't') == 1
    s = json.loads((tmp_path / 't_ava_M_detector' / 'stats' / 'summary.json').read_text())
    assert s['gt_partner'] == 't_ava_M_gt'
    assert s['map_gt'] > s['map_gt_found']           # faces not found cost
    assert s['map_no_fp'] > s['map_gt_found']        # the better crop gains
    assert s['map_no_fp'] > s['map']                 # extra detections cost
    assert s['map_gt_found_first_1s'] < s['map_gt_first_1s']   # the misses are early
    assert s['map_gt_found_first_2s'] > s['map_gt_found_first_0.5s']
    assert 'map_gt_on_detected' not in s
