"""Tests for the voice activity gate of the visual-only models (#103)."""

from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip('webrtcvad')

from asd4hri.asd import ASD  # noqa: E402
from asd4hri.av import load_audio  # noqa: E402
from asd4hri.vad import SAMPLE_RATE, VoiceActivity, gate  # noqa: E402

VIDEO = Path(__file__).parent / '001.avi'


def test_gate_modes():
    """Product, minimum and hard gate; the hard gate keeps the order in silent frames."""
    assert gate(0.8, 0.5, 'product') == pytest.approx(0.4)
    assert gate(0.8, 0.5, 'min') == 0.5
    assert gate(0.8, 0.6, 'hard') == 0.8
    assert gate(0.8, 0.4, 'hard') == pytest.approx(0.08)
    assert gate(0.8, 0.4, 'hard') > gate(0.7, 0.4, 'hard')
    with pytest.raises(ValueError):
        gate(0.8, 0.5, 'max')


@pytest.mark.parametrize('backend', ['webrtc', 'silero'])
def test_voice_activity_is_causal_and_finds_speech(backend):
    """The probability at t only depends on the audio before t; speech scores above silence."""
    speech = load_audio(VIDEO)
    assert len(speech) > SAMPLE_RATE
    audio = np.concatenate([np.zeros(SAMPLE_RATE * 2, np.int16), speech])
    times = np.arange(0.1, len(audio) / SAMPLE_RATE, 0.04)
    vad = VoiceActivity(backend)
    vad.set_audio(audio)
    probs = np.array([vad(t) for t in times])
    assert 0 <= probs.min() and probs.max() <= 1
    assert probs[times < 1.9].max() < 0.5
    assert probs[times > 2.5].mean() > probs[times < 1.9].mean() + 0.2

    # change everything after t = 2.5 s: nothing before it may change
    changed = audio.copy()
    changed[int(2.5 * SAMPLE_RATE):] = 0
    vad.set_audio(changed)
    again = np.array([vad(t) for t in times])
    np.testing.assert_allclose(again[times <= 2.5], probs[times <= 2.5])


def test_voice_activity_without_audio_and_after_a_seek():
    """No audio gives None (no gating); a seek only processes the warm-up before it."""
    vad = VoiceActivity('webrtc')
    assert vad(1.0) is None
    vad.set_audio(np.zeros(SAMPLE_RATE * 600, np.int16))
    assert vad(500.0) == 0.0
    assert vad.position <= 500 * SAMPLE_RATE and vad.position > 498 * SAMPLE_RATE


def test_asd_gates_the_scores_of_all_faces():
    """Silence gates every score to zero with 'product'; without audio the scores stay."""
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = [cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB) for _ in range(10)]
    h, w = frames[0].shape[:2]

    def run(audio_gate, audio):
        """Scores of the last frame with two fixed boxes.

        Returns:
            list: The scores.
        """
        from paz.abstract.messages import Box2D
        asd = ASD(architecture='LipShapeCausal', audio_gate=audio_gate)
        assert asd.needs_audio == (audio_gate != 'none')
        asd.set_audio(audio, fps)
        for i, frame in enumerate(frames):
            boxes = [Box2D([w // 8, h // 8, w * 5 // 8, h * 7 // 8], 1.0, 'face'),
                     Box2D([w // 4, h // 6, w * 3 // 4, h * 5 // 6], 1.0, 'face')]
            out = asd(frame, boxes, ['a', 'b'], timestamp=i / fps)
        return [b.score for b in out['boxes2D']]

    plain = run('none', None)
    assert all(s >= 0 for s in plain)
    assert run('webrtc', None) == plain
    assert run('webrtc', np.zeros(SAMPLE_RATE, np.int16)) == [0.0, 0.0]
    with pytest.raises(ValueError):
        ASD(architecture='LipShapeCausal', audio_gate='loud')
