"""Tests for the audio-visual models in the pipeline (#40, #80)."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('python_speech_features')

from asd4hri.av import AVClassifier, face_crop, load_audio, mfcc_window  # noqa: E402

VIDEO = Path(__file__).parent / '001.avi'


def test_mfcc_window_has_four_rows_per_frame():
    audio = (np.random.default_rng(0).normal(0, 1000, 16000 * 3)).astype(np.int16)
    for fps in (25.0, 30.0):
        assert mfcc_window(audio, 0.5, 20, fps).shape == (80, 13)
    # past the end of the audio: wrapped / zero-filled, never a wrong shape
    assert mfcc_window(audio, 2.9, 25, 25.0).shape == (100, 13)


def test_face_crop_is_grey_112_and_clips_to_the_image():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    assert face_crop(image, (150, 50, 260, 160)).shape == (112, 112)
    assert face_crop(image, (300, 300, 310, 310)).shape == (112, 112)


def test_load_audio_from_test_video():
    audio = load_audio(VIDEO)
    assert audio.dtype == np.int16


@pytest.mark.parametrize('fallback', ['noise', 'visual'])
def test_av_classifier_scores_every_face_and_frame(fallback):
    clf = AVClassifier('LR-ASD', window=10, audio_fallback=fallback, device='cpu')
    clf.set_audio(None, 25.0)                      # no audio -> fallback path
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    for t in range(12):
        ids = ['a', 'b'] if t < 8 else ['b']
        boxes = [(10, 10, 90, 90), (150, 20, 230, 100)][:len(ids)] if t < 8 \
            else [(150, 20, 230, 100)]
        out = clf(image, boxes, ids, t / 25.0)
        assert len(out) == len(ids)
        assert all(0.0 <= s <= 1.0 for s in out)
    assert len(clf.tracks['b']['faces']) == 10      # window caps the buffer


def test_av_classifier_with_real_audio_uses_av_backend():
    clf = AVClassifier('LR-ASD', window=5, device='cpu')
    audio = (np.random.default_rng(1).normal(0, 3000, 16000)).astype(np.int16)
    clf.set_audio(audio, 25.0)
    image = np.full((120, 120, 3), 128, dtype=np.uint8)
    out = [clf(image, [(0, 0, 100, 100)], ['x'], t / 25.0)[0] for t in range(5)]
    assert all(s is not None for s in out)
