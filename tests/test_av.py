"""Tests for the audio-visual models in the pipeline (#40, #80)."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('python_speech_features')
pytest.importorskip('pyasd')

from asd4hri.av import (AV_MODELS, AVClassifier, AVModel, face_crop, load_audio,  # noqa: E402
                        mfcc_window)

VIDEO = Path(__file__).parent / '001.avi'
# scores of the previous implementation (the authors' repos as git submodules) on fixed inputs
REFERENCE = Path(__file__).parent / 'av_reference.npz'


def _reference_inputs():
    """The fixed inputs the reference scores were computed on.

    Returns:
        tuple: ``(video (2, 25, 112, 112) uint8, audio (2, 100, 13) float32)``.
    """
    rng = np.random.default_rng(0)
    video = rng.integers(0, 255, (2, 25, 112, 112), dtype=np.uint8)
    video = ((video.astype(np.float32) + np.roll(video, 1, -1)) / 2).astype(np.uint8)
    audio = rng.normal(0, 10, (2, 100, 13)).astype(np.float32)
    return video, audio


@pytest.mark.parametrize('name', ['LR-ASD', 'TalkNet'])
def test_pyasd_models_match_the_original_repos(name):
    """With the AVA weights, pyasd gives the scores of the authors' code, with and without audio."""
    video, audio = _reference_inputs()
    ref = np.load(REFERENCE)
    model = AVModel(name, device='cpu')
    np.testing.assert_allclose(model.scores(video, audio), ref[f'{name}_av'], atol=1e-5)
    np.testing.assert_allclose(model.scores(video, None), ref[f'{name}_v'], atol=1e-5)


@pytest.mark.parametrize('name', list(AV_MODELS))
def test_every_av_model_loads_and_scores_every_frame(name):
    """Every AV model loads its AVA weights and returns one probability per frame."""
    video, audio = _reference_inputs()
    for deploy in (False, True):
        out = AVModel(name, device='cpu', deploy=deploy).scores(video[:, :8], audio[:, :32])
        assert out.shape == (2, 8)
        assert np.all((out >= 0) & (out <= 1))


def test_mfcc_window_has_four_rows_per_frame():
    """MFCC windows have 4 rows per video frame at any fps, also past the end of the audio."""
    audio = (np.random.default_rng(0).normal(0, 1000, 16000 * 3)).astype(np.int16)
    for fps in (25.0, 30.0):
        assert mfcc_window(audio, 0.5, 20, fps).shape == (80, 13)
    # past the end of the audio: wrapped / zero-filled, never a wrong shape
    assert mfcc_window(audio, 2.9, 25, 25.0).shape == (100, 13)


def test_face_crop_is_grey_112_and_clips_to_the_image():
    """Face crops are grey 112x112, also for boxes outside the image."""
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    assert face_crop(image, (150, 50, 260, 160)).shape == (112, 112)
    assert face_crop(image, (300, 300, 310, 310)).shape == (112, 112)


def test_load_audio_from_test_video():
    """The audio track of a video decodes to int16."""
    audio = load_audio(VIDEO)
    assert audio.dtype == np.int16


@pytest.mark.parametrize('fallback', ['noise', 'visual'])
def test_av_classifier_scores_every_face_and_frame(fallback):
    """Every face gets a score in every frame; the window caps the buffer."""
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
    """With audio, every frame of a face gets a score from the audio-visual back end."""
    clf = AVClassifier('LR-ASD', window=5, device='cpu')
    audio = (np.random.default_rng(1).normal(0, 3000, 16000)).astype(np.int16)
    clf.set_audio(audio, 25.0)
    image = np.full((120, 120, 3), 128, dtype=np.uint8)
    out = [clf(image, [(0, 0, 100, 100)], ['x'], t / 25.0)[0] for t in range(5)]
    assert all(s is not None for s in out)
