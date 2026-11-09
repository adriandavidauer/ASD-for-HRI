"""Tests for CAST-ASD in the pipeline (#80)."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('cast_asd')
pytest.importorskip('moshi')

from asd4hri.av import AVClassifier  # noqa: E402
from asd4hri.cast import CASTModel  # noqa: E402

# target scores of CAST-ASD's own validation path (collate_scenes with score_targets_only=True,
# then AudioVisualASD.forward) on the inputs below, computed with the pinned cast_asd commit
REFERENCE = Path(__file__).parent / 'cast_reference.npz'


def _cast_inputs():
    """Two scenes of 25 frames: three tracks, and a target alone; 1 s of audio each.

    Returns:
        tuple: ``(scenes, waveforms)``: a list of ``(tracks, 25, 112, 112)`` uint8 arrays and
        ``(2, 16000)`` float32 audio (not yet loudness-normalised).
    """
    rng = np.random.default_rng(0)
    scenes = []
    for tracks in (3, 1):
        faces = rng.integers(0, 255, (tracks, 25, 112, 112), dtype=np.uint8)
        scenes.append(((faces.astype(np.float32) + np.roll(faces, 1, -1)) / 2).astype(np.uint8))
    t = np.arange(16000) / 16000
    waveforms = np.stack([0.1 * np.sin(2 * np.pi * 220 * t) * (t > 0.4),
                          rng.normal(0, 0.05, 16000)]).astype(np.float32)
    return scenes, waveforms


@pytest.fixture(scope='module')
def model():
    """CAST-ASD with its AVA weights on the CPU.

    Returns:
        CASTModel: The model.
    """
    return CASTModel(device='cpu')


def test_cast_matches_its_own_validation_path(model):
    """Our scene batch gives the target scores of CAST-ASD's collate + forward."""
    scenes, waveforms = _cast_inputs()
    ref = np.load(REFERENCE)['scores']
    np.testing.assert_allclose(model.scores(scenes, waveforms), ref, atol=1e-4)


def test_cast_is_causal(model, monkeypatch):
    """Changing the last frame and its audio leaves the earlier scores of the network unchanged.

    The loudness normalisation is switched off: its gain comes from the whole window, so it does
    change earlier frames, but the pipeline only uses the newest frame, whose gain needs no
    later audio.
    """
    monkeypatch.setattr(model, 'normalize', lambda wave: (wave, 1.0))
    scenes, waveforms = _cast_inputs()
    before = model.scores(scenes, waveforms)
    scenes = [s.copy() for s in scenes]
    for s in scenes:
        s[:, -1] = 255 - s[:, -1]
    waveforms = waveforms.copy()
    waveforms[:, -640:] = 0.3
    after = model.scores(scenes, waveforms)
    np.testing.assert_allclose(after[:, :-1], before[:, :-1], atol=1e-4)


def test_av_classifier_with_cast_picks_scene_mates():
    """A buffered face is a scene-mate when it shares more than half of the target's frames."""
    clf = AVClassifier('CAST-ASD', window=8, device='cpu')
    rng = np.random.default_rng(1)
    clf.set_audio(rng.normal(0, 2000, 16000).astype(np.int16), fps=25)
    tracks = {}                                   # frame -> sorted track counts of its scenes
    original = clf.model.scores
    def spy(scenes, waves):
        tracks.setdefault(clf.frame, []).extend(len(s) for s in scenes)
        return original(scenes, waves)
    clf.model.scores = spy
    image = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    boxes = {'a': (10, 10, 90, 90), 'b': (120, 20, 200, 100), 'c': (220, 30, 300, 110)}
    for i in range(8):
        ids = ['a', 'b', 'c'] if i >= 6 else ['a', 'b']    # c joins at frame 6
        out = clf(image, [boxes[k] for k in ids], ids=ids, timestamp=i / 25)
        assert len(out) == len(ids) and all(0 <= p <= 1 for p in out)
    assert sorted(tracks[1]) == [2, 2]            # frame 0: a and b together
    # frame 7: c shares 2 of the 8 frames of a and b (no mate for them), but both of its own
    assert sorted(tracks[8]) == [2, 2, 3]
