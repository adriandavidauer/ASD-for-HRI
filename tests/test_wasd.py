"""Tests for the WASD adapter (#81) and per-category scoring."""

import csv

import numpy as np
import pytest

from src.experiments.wasd_dataset import WasdDataset
from tests.test_stats import stats, _gt_rows, _preds, POS


def _fixture(tmp_path):
    (tmp_path / 'csv').mkdir()
    with open(tmp_path / 'csv' / 'val_orig.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['video_id', 'frame_timestamp', 'entity_box_x1', 'entity_box_y1',
                    'entity_box_x2', 'entity_box_y2', 'label', 'entity_id'])
        for vid in ('abc_10-40', 'abc_50-80', 'x_y_0-30'):
            w.writerow([vid, 0, 0.1, 0.1, 0.2, 0.2, 'NOT_SPEAKING', f'{vid}:0'])
    with open(tmp_path / 'csv' / 'wasd_metadata.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Video', 'Category'])
        w.writerow(['abc_10-40', 4])
        w.writerow(['abc_90-120', 4])
        w.writerow(['x_y_0-30', 1])
    return WasdDataset(str(tmp_path), download=False)


def test_parse_video_id_keeps_underscores_in_youtube_id():
    assert WasdDataset.parse_video_id('x_y_0-30') == ('x_y', 0, 30)
    assert WasdDataset.parse_video_id('0BRxm7G1acw_1419-1449') == ('0BRxm7G1acw', 1419, 1449)


def test_categories_from_metadata_with_source_fallback(tmp_path):
    ds = _fixture(tmp_path)
    assert ds.video_ids == ['abc_10-40', 'abc_50-80', 'x_y_0-30']
    # abc_50-80 is not in the metadata -> category of its source video's other clips
    assert ds.categories() == {'abc_10-40': 'SS', 'abc_50-80': 'SS', 'x_y_0-30': 'SI'}
    ds.write_categories_csv(str(tmp_path / 'cat.csv'))
    assert stats.load_categories(str(tmp_path / 'cat.csv'))['x_y_0-30'] == 'SI'


def test_protocol_results_have_category_rows(tmp_path):
    per_video = []
    for vid, good in (('a', True), ('b', False)):
        gt = _gt_rows(40, seed=1)
        score = (lambda r: float(r['ava_label'] == POS)) if good else (lambda r: 0.5)
        protocol, gt_df, fp, n = stats.protocol_rows(
            vid, stats.GroundTruthIndex(gt), _preds(gt, score, True), 0.02)
        per_video.append((vid, protocol, gt_df, fp, n))
    summary = stats.write_protocol_results(str(tmp_path), per_video,
                                           categories={'a': 'OC', 'b': 'SS'})
    assert summary['map_by_category']['OC'] == pytest.approx(1.0)
    assert summary['map_by_category']['SS'] < 1.0
    with open(tmp_path / 'protocol_results.csv') as f:
        ids = [r['video_id'] for r in csv.DictReader(f)]
    assert ids == ['a', 'b', 'category:OC', 'category:SS', 'pooled']
