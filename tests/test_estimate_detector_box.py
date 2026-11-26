"""Tests for the estimation of dlib's detector box in the face crop (#132)."""

from pathlib import Path

import cv2
import dlib
import numpy as np
import pytest

from asd4hri.processors import GetShapeFeatures
from src.model_training import estimate_detector_box as e

VIDEO = Path(__file__).parent / '001.avi'


def test_tight_box_in_a_loose_crop_matches_despite_low_iou():
    """A dlib box well inside a loose GT crop is matched although its IoU is below 0.5."""
    region, det = (0, 0, 200, 200), (50, 70, 150, 170, 1.0)
    assert e.match([region], [det]) == [(0, 0)]


@pytest.mark.parametrize('det', [
    (90, 90, 120, 120, 1.0),     # a small background face: side < MIN_SIDE of the crop
    (150, 50, 260, 160, 1.0),    # mostly outside the crop
])
def test_unrelated_boxes_do_not_match(det):
    """Too small or mostly outside the crop: no match."""
    assert e.match([(0, 0, 200, 200)], [det]) == []


def test_matching_is_one_to_one_by_iou():
    """Two crops that both contain a face: each face goes to the crop it overlaps most."""
    regions = [(0, 0, 200, 200), (90, 20, 250, 180)]
    dets = [(120, 50, 200, 130, 1.0), (20, 50, 100, 130, 1.0)]
    assert e.match(regions, dets) == [(0, 1), (1, 0)]
    assert len(e.match(regions[:1], dets[:1] * 2)) == 1


def test_fractions_give_back_the_dlib_rectangle_in_GetShapeFeatures(monkeypatch):
    """Fractions of a crop turn back into the same dlib box inside that crop."""
    region, det = (40, 30, 240, 230), (90, 100, 190, 200)
    fractions = e.as_fractions(region, det)
    from asd4hri import processors
    monkeypatch.setitem(processors.SHAPE_START_BOXES, 'test', tuple(fractions))
    features, rects = GetShapeFeatures(start_box='test'), []
    predictor = features.shape_predictor
    features.shape_predictor = lambda img, rect: (rects.append(rect), predictor(img, rect))[1]
    features([np.zeros((200, 200, 3), np.uint8)])
    assert rects == [dlib.rectangle(50, 70, 150, 170)]


def test_summarize_gives_median_spread_and_count():
    """Median, quartiles and n of the matched boxes; empty input gives n = 0."""
    out = e.summarize([[0.1, 0.2, 0.8, 0.9], [0.2, 0.3, 0.9, 1.0], [0.3, 0.4, 1.0, 1.1]])
    assert out['box'] == [0.2, 0.3, 0.9, 1.0] and out['n'] == 3
    assert out['iqr'][0] == [0.15, 0.25]
    assert e.summarize([])['n'] == 0


def test_sample_timestamps_spread_over_the_video():
    """At most k annotated timestamps, first and last included."""
    gt = [(t / 10, 'a', (0, 0, 1, 1)) for t in range(100)] * 2
    stamps = e.sample_timestamps(gt, 5)
    assert len(stamps) == 5 and stamps[0] == 0.0 and stamps[-1] == 9.9
    assert len(e.sample_timestamps(gt[:3], 5)) == 3


def test_matched_fractions_skip_small_crops():
    """Crops smaller than MIN_SIZE are not used."""
    rec = {'gt': [(0, 0, 200, 200), (300, 0, 330, 30)],
           'dlib': [(50, 70, 150, 170, 1.0), (302, 2, 328, 28, 1.0)]}
    fractions = e.matched_fractions([rec], 'gt')
    assert np.allclose(fractions, [[0.25, 0.35, 0.75, 0.85]])


def test_process_video_records_regions_detections_and_proxy_errors():
    """One record per sampled frame, with the pipeline's crop regions and dlib's boxes."""
    fps = cv2.VideoCapture(str(VIDEO)).get(cv2.CAP_PROP_FPS)
    gt = [(round(i / fps, 3), 'a', (0.1, 0.1, 0.6, 0.7)) for i in range(4)]
    e._init(False, ('none', 'ava_gt'))
    records = e.process_video(('ava', '001', str(VIDEO), None, gt, [gt[0][0], gt[3][0]]))
    assert len(records) == 2
    for rec in records:
        h, w = rec['size']
        x0, y0, x1, y1 = rec['gt'][0]
        assert x1 - x0 == y1 - y0 or x0 == 0 or y0 == 0 or x1 == w or y1 == h  # square or clipped
        assert all(len(d) == 5 for d in rec['dlib'])
        for face in rec['proxy']:
            assert set(face['errors']['p2']) == {'none', 'ava_gt'}
    report = e.proxy_report(records)
    assert report['p2']['none']['n'] == sum(len(r['proxy']) for r in records)


@pytest.mark.parametrize('dataset, boxes, expected', [
    ('ava', 'gt', 'ava'), ('wasd', 'gt', 'wasd'), ('easycom', 'gt', 'easycom'),
    ('ava', 'detector', 'YuNet'), ('unitalk', 'detector', 'YuNet'),
])
def test_adjusted_start_box_is_the_constant_of_the_box_source(dataset, boxes, expected):
    """GT boxes use their dataset's constant, detector boxes the YuNet one."""
    from asd4hri.processors import SHAPE_START_BOXES, adjusted_start_box
    assert adjusted_start_box(dataset, boxes) == expected
    x0, y0, x1, y1 = SHAPE_START_BOXES[expected]
    assert 0 <= x0 < x1 and 0 <= y0 < y1


def test_no_adjusted_start_box_for_unitalk_gt_boxes():
    """UniTalk's train split was not available for the estimation: no silent fallback."""
    from asd4hri.processors import adjusted_start_box
    with pytest.raises(ValueError):
        adjusted_start_box('unitalk', 'gt')


@pytest.mark.parametrize('flag, dataset, boxes, expected', [
    ('adjusted', 'wasd', 'gt', 'wasd'), ('adjusted', 'wasd', 'detector', 'YuNet'),
    ('none', 'wasd', 'gt', 'none'), ('ava_gt', 'ava', 'gt', 'ava_gt'),
])
def test_run_full_pipeline_resolves_the_adjusted_flag(flag, dataset, boxes, expected):
    """--shape_start_box adjusted picks the run's constant; other values pass through."""
    from src.experiments.run_full_pipeline import build_parser, resolve_start_box
    args = build_parser().parse_args(['--dataset', dataset, '--boxes', boxes,
                                      '--shape_start_box', flag])
    assert resolve_start_box(args) == expected
