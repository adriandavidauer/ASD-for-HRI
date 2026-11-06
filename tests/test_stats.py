"""Tests for the comparable scoring protocols in ``src/experiments/stats.py`` (#75)."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_STATS = Path(__file__).resolve().parent.parent / 'src' / 'experiments' / 'stats.py'
_spec = importlib.util.spec_from_file_location('asd_stats', _STATS)
stats = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stats)

POS, NEG = 'SPEAKING_AUDIBLE', 'NOT_SPEAKING'


def _gt_rows(n_frames=50, seed=0):
    """Two entities per frame, entity 'v:1' speaks in every other frame."""
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_frames):
        ts = round(f * 0.04, 2)
        rows.append({'frame_timestamp': ts, 'bbox': (0.1, 0.1, 0.3, 0.4), 'ava_label': NEG,
                     'asd_label': 'not-speaking', 'entity_id': 'v:0'})
        lab = POS if f % 2 else NEG
        rows.append({'frame_timestamp': ts, 'bbox': (0.6, 0.1, 0.8, 0.4), 'ava_label': lab,
                     'asd_label': 'speaking' if lab == POS else 'not-speaking',
                     'entity_id': 'v:1'})
    rng.shuffle(rows)
    return rows


def _preds(gt_rows, score_fn, with_entity, drop=lambda r: False, extra=()):
    """Prediction boxes per timestamp built from the GT rows."""
    by_ts = {}
    for r in gt_rows:
        if drop(r):
            continue
        s = score_fn(r)
        by_ts.setdefault(r['frame_timestamp'], []).append(stats._PredBox(
            r['bbox'], 'speaking' if s > 0.5 else 'not-speaking', s,
            r['entity_id'] if with_entity else None))
    for ts, box, s in extra:
        by_ts.setdefault(ts, []).append(stats._PredBox(box, 'speaking', s, None))
    return by_ts


def _metrics(gt_rows, preds, width=None):
    protocol, gt_df, fp, n_det = stats.protocol_rows(
        'v', stats.GroundTruthIndex(gt_rows), preds, 0.02, 0.5, width)
    return protocol, stats.protocol_metrics(gt_df, fp, n_det, protocol)


def test_gt_protocol_perfect_scores_give_map_1():
    gt = _gt_rows()
    protocol, m = _metrics(gt, _preds(gt, lambda r: float(r['ava_label'] == POS), True))
    assert protocol == 'gt'
    assert m['map'] == pytest.approx(1.0)
    assert m['coverage'] == pytest.approx(1.0)
    assert m['f1'] == pytest.approx(1.0)


def test_gt_protocol_constant_scores_give_positive_rate():
    gt = _gt_rows(400)
    _, m = _metrics(gt, _preds(gt, lambda r: 0.3, True))
    pos_rate = m['gt_positives'] / m['gt_boxes']
    assert m['map'] == pytest.approx(pos_rate, abs=0.02)
    assert m['auc'] == pytest.approx(0.5)


def test_gt_protocol_missing_predictions_get_score_zero():
    gt = _gt_rows()
    _, m = _metrics(gt, _preds(gt, lambda r: float(r['ava_label'] == POS), True,
                               drop=lambda r: r['frame_timestamp'] < 0.5))
    assert m['coverage'] < 1.0
    assert m['map'] < 1.0


def test_detector_protocol_perfect_detector_equals_gt_protocol():
    gt = _gt_rows()
    score = lambda r: 0.9 if r['ava_label'] == POS else 0.2  # noqa: E731
    protocol, m = _metrics(gt, _preds(gt, score, False))
    assert protocol == 'detector'
    assert m['map'] == pytest.approx(1.0)
    assert m['det_recall'] == pytest.approx(1.0)
    assert m['det_precision'] == pytest.approx(1.0)
    assert m['dropped_faces'] == 0


def test_detector_protocol_dropped_faces_lower_map_and_recall():
    gt = _gt_rows()
    score = lambda r: 0.9 if r['ava_label'] == POS else 0.2  # noqa: E731
    dropped = lambda r: r['entity_id'] == 'v:1' and r['frame_timestamp'] < 1.0  # noqa: E731
    _, m = _metrics(gt, _preds(gt, score, False, drop=dropped))
    assert m['dropped_faces'] == 25
    assert m['det_recall'] == pytest.approx(75 / 100)
    # missed positives are never retrieved -> recall of the ranking stops below 1
    assert m['map'] < 1.0


def test_detector_protocol_false_detections_count_as_false_positives():
    gt = _gt_rows()
    score = lambda r: 0.9 if r['ava_label'] == POS else 0.2  # noqa: E731
    extra = [(round(f * 0.04, 2), (0.85, 0.5, 0.95, 0.7), 0.95) for f in range(50)]
    _, m = _metrics(gt, _preds(gt, score, False, extra=extra))
    assert m['det_precision'] == pytest.approx(100 / 150)
    assert m['fp'] == 50
    assert m['map'] < 1.0


def test_low_iou_detection_does_not_match():
    gt = _gt_rows(10)
    preds = {}
    for r in gt:
        x1, y1, x2, y2 = r['bbox']
        shifted = (x1 + 0.12, y1, x2 + 0.12, y2)  # IoU 0.25 with its GT box
        preds.setdefault(r['frame_timestamp'], []).append(
            stats._PredBox(shifted, 'speaking', 0.9, None))
    _, m = _metrics(gt, preds)
    assert m['det_recall'] == pytest.approx(0.0)


def test_face_size_bins():
    assert stats.face_size_bin(10) == 'small'
    assert stats.face_size_bin(64) == 'medium'
    assert stats.face_size_bin(200) == 'large'
    assert stats.face_size_bin(None) == ''
    gt = _gt_rows()
    # boxes are 0.2 wide -> 128 px at width 640 -> 'large'; 96 px at 480 -> 'medium'
    _, m = _metrics(gt, _preds(gt, lambda r: 0.5, False), width=480)
    assert m['n_medium'] == 100 and m['n_small'] == 0


def test_official_script_agrees_on_written_ava_csvs(tmp_path):
    """The AVA CSVs of a gt run, scored by the official script, give the reported mAP."""
    gt = _gt_rows(100)
    rng = np.random.default_rng(1)
    preds = _preds(gt, lambda r: float(rng.random()) * 0.5 + 0.5 * (r['ava_label'] == POS), True)
    protocol, gt_df, fp, n_det = stats.protocol_rows(
        'v', stats.GroundTruthIndex(gt), preds, 0.02)
    summary = stats.write_protocol_results(str(tmp_path), [('v', protocol, gt_df, fp, n_det)])
    ava = stats._ava_eval()
    cols = ['video_id', 'frame_timestamp', 'entity_box_x1', 'entity_box_y1',
            'entity_box_x2', 'entity_box_y2', 'label', 'entity_id']
    g = ava.load_csv(str(tmp_path / 'ava_groundtruth.csv'), cols)
    p = ava.load_csv(str(tmp_path / 'ava_predictions.csv'), cols + ['score'])
    precision, recall = ava.calculate_precision_recall(ava.merge_groundtruth_and_predictions(g, p))
    assert summary['map'] == pytest.approx(ava.compute_average_precision(precision, recall))
    assert json.loads((tmp_path / 'summary.json').read_text())['protocol'] == 'gt'
    with open(tmp_path / 'protocol_results.csv') as f:
        assert [r['video_id'] for r in csv.DictReader(f)] == ['v', 'pooled']


def test_old_prediction_files_without_score_column(tmp_path):
    path = tmp_path / 'p.csv'
    path.write_text('frame_idx,timestamp,label,x1,y1,x2,y2\n'
                    '0,0.0,speaking,0,0,1,1\n0,0.0,not-speaking,0,0,1,1\n'
                    '0,0.0,No Prediction yet,0,0,1,1\n')
    scores = [b.score for b in stats.load_predictions_csv(str(path))[0.0]]
    assert scores == [1.0, 0.0, -1.0]
