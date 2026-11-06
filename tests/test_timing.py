"""Tests for the per-stage timing (#76)."""

import time

import numpy as np
import pytest

from asd4hri.timing import StageTimer


def test_stage_timer_records_calls_and_items():
    timer = StageTimer()
    for n in (1, 2, 3):
        with timer.stage('model', n):
            time.sleep(0.002)
    with timer.stage('detect'):
        pass
    rows = {r['stage']: r for r in timer.summary()}
    assert list(rows) == ['model', 'detect']
    assert rows['model']['calls'] == 3
    assert rows['model']['items'] == 6
    assert rows['model']['mean_ms'] >= 2.0
    assert rows['model']['ms_per_item'] == pytest.approx(rows['model']['total_s'] * 1000 / 6)
    assert rows['detect']['items'] is None and rows['detect']['ms_per_item'] is None
    assert rows['model']['median_ms'] <= rows['model']['p95_ms'] <= rows['model']['max_ms']


def test_stage_timer_disabled_records_nothing():
    timer = StageTimer(enabled=False)
    with timer.stage('model', 1):
        pass
    assert timer.summary() == []


def test_stage_timer_records_on_exception():
    timer = StageTimer()
    with pytest.raises(RuntimeError):
        with timer.stage('model'):
            raise RuntimeError
    assert timer.summary()[0]['calls'] == 1


def test_asd_stages_sum_to_frame_time():
    """The inner stages of ASD account for (almost) all of the frame time."""
    from pathlib import Path
    import cv2
    from asd4hri.asd import ASD
    asd = ASD(architecture='LipShape', stride=1, averaging_window_size=1)
    video = cv2.VideoCapture(str(Path(__file__).parent / '001.avi'))
    for _ in range(45):
        ok, frame = video.read()
        assert ok
        asd(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    rows = {r['stage']: r for r in asd.timer.summary()}
    assert {'frame', 'detect', 'track', 'crop', 'preprocess', 'buffer', 'model',
            'average'} <= set(rows)
    inner = sum(r['total_s'] for k, r in rows.items() if k != 'frame')
    assert inner <= rows['frame']['total_s']
    assert inner >= 0.9 * rows['frame']['total_s']
    assert rows['frame']['calls'] == 45
