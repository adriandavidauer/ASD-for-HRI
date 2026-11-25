"""Voice activity detection as a cheap audio gate for the visual-only models (#103, stage 1).

One causal speech probability per video frame, computed once per frame and shared by all
faces. The visual score of every face is then combined with it (``gate``): a face that moves
its mouth while nobody speaks gets a low score. The VAD cannot tell which face speaks.

Two back ends:

* ``webrtc``: the WebRTC VAD (C, ``webrtcvad-wheels``), one speech/no-speech decision per
  30 ms of audio. Effectively free.
* ``silero``: Silero VAD v6 (MIT, ONNX, 2.3 MB) on onnxruntime, one probability per 32 ms.

Both only use audio up to the current frame. The probability of a frame is the mean of the
decisions of the last ``smoothing_s`` seconds. On six AVA val videos (frame-level AP of "someone
on screen speaks", base rate 0.34) Silero reaches 0.62 without smoothing and WebRTC 0.45 at
aggressiveness 3 and 0.1 s, at 0.14 and 0.01 ms per 25 fps frame on one workstation core.
"""

from collections import deque

import numpy as np

SAMPLE_RATE = 16000
VAD_OPTIONS = ['none', 'webrtc', 'silero']
GATE_MODES = ['product', 'min', 'hard']
SMOOTHING_S = {'webrtc': 0.1, 'silero': 0.0}     # best frame-level AP on AVA val

SILERO = {
    'file': 'silero_vad_v6.2.3.onnx',
    'url': 'https://raw.githubusercontent.com/snakers4/silero-vad/'
           '5cd7945676eb32225748052e2e6a0580e4686a08/src/silero_vad/data/silero_vad.onnx',
    'sha256': '1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3',
}
_SILERO_CONTEXT = 64        # samples of the previous chunk that Silero v5/v6 expects in front
_WARMUP_S = 1.0             # after a seek, the VAD starts this long before the requested time


class _WebRTC:
    """WebRTC VAD on 30 ms chunks.

    Args:
        aggressiveness: 0 (keeps most speech) to 3 (drops most non-speech).
    """

    chunk = 480

    def __init__(self, aggressiveness=3):
        """Create the C VAD."""
        self.aggressiveness = aggressiveness
        self.reset()

    def reset(self):
        """Start a fresh C VAD (it adapts to the noise level, so it has state)."""
        import webrtcvad
        self.vad = webrtcvad.Vad(self.aggressiveness)

    def __call__(self, samples):
        """Speech decision of one chunk.

        Args:
            samples: ``chunk`` int16 samples.

        Returns:
            float: 1.0 for speech, else 0.0.
        """
        return float(self.vad.is_speech(samples.tobytes(), SAMPLE_RATE))


class _Silero:
    """Silero VAD (ONNX) on 32 ms chunks, single-threaded like on the Pi."""

    chunk = 512

    def __init__(self):
        """Load the ONNX model (downloaded and checked on first use)."""
        import onnxruntime

        from .av import fetch
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            str(fetch(SILERO)), options, providers=['CPUExecutionProvider'])
        self.sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self):
        """Clear the recurrent state and the context."""
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros(_SILERO_CONTEXT, dtype=np.float32)

    def __call__(self, samples):
        """Speech probability of one chunk.

        Args:
            samples: ``chunk`` int16 samples.

        Returns:
            float: The probability.
        """
        x = np.concatenate([self.context, samples.astype(np.float32) / 32768.0])
        out, self.state = self.session.run(
            None, {'input': x[None], 'state': self.state, 'sr': self.sr})
        self.context = x[-_SILERO_CONTEXT:]
        return float(out[0, 0])


class VoiceActivity:
    """Causal speech probability of an audio track at any time.

    Args:
        backend: 'webrtc' or 'silero'.
        smoothing_s: The probability of a frame is the mean over the chunks of this many
            seconds before it; None takes the best value on AVA val (``SMOOTHING_S``).
        aggressiveness: Only for 'webrtc', see ``_WebRTC``.
    """

    def __init__(self, backend='silero', smoothing_s=None, aggressiveness=3):
        """Create the back end.

        Raises:
            ValueError: For an unknown back end.
        """
        if backend == 'webrtc':
            self.vad = _WebRTC(aggressiveness)
        elif backend == 'silero':
            self.vad = _Silero()
        else:
            raise ValueError(f"VAD back end must be 'webrtc' or 'silero', not {backend}")
        self.backend = backend
        if smoothing_s is None:
            smoothing_s = SMOOTHING_S[backend]
        self.history = deque(maxlen=max(1, round(smoothing_s * SAMPLE_RATE / self.vad.chunk)))
        self.set_audio(None)

    def set_audio(self, audio):
        """Start on a new audio track.

        Args:
            audio: int16 mono samples at 16 kHz, or None / empty for no audio.
        """
        self.audio = np.zeros(0, np.int16) if audio is None else np.asarray(audio, np.int16)
        self.position = 0
        self.history.clear()
        self.vad.reset()

    def __call__(self, timestamp):
        """Speech probability at ``timestamp``, from the audio before it.

        Runs the VAD on every complete chunk up to ``timestamp`` that it has not seen yet. A
        jump forward (a seek) restarts it ``_WARMUP_S`` before ``timestamp``.

        Args:
            timestamp: Time in seconds.

        Returns:
            float or None: The probability, or None without audio there (no gating).
        """
        end = min(int(timestamp * SAMPLE_RATE), len(self.audio))
        chunk = self.vad.chunk
        if end < self.position or end - self.position > (_WARMUP_S + 1) * SAMPLE_RATE:
            self.position = max(0, end - int(_WARMUP_S * SAMPLE_RATE))
            self.history.clear()
            self.vad.reset()
        while self.position + chunk <= end:
            self.history.append(self.vad(self.audio[self.position:self.position + chunk]))
            self.position += chunk
        return float(np.mean(self.history)) if self.history else None


def gate(score, speech, mode='product', threshold=0.5, floor=0.1):
    """Combine a face's visual score with the frame's speech probability.

    Args:
        score: Visual speaking score in [0, 1].
        speech: Speech probability of the frame in [0, 1].
        mode: 'product' (``score * speech``), 'min' (``min(score, speech)``) or 'hard'
            (``score``, times ``floor`` when ``speech < threshold``).
        threshold: Only for 'hard'.
        floor: Only for 'hard'; keeps the order of the faces in silent frames.

    Returns:
        float: The gated score.

    Raises:
        ValueError: For an unknown mode.
    """
    if mode == 'product':
        return score * speech
    if mode == 'min':
        return min(score, speech)
    if mode == 'hard':
        return score if speech >= threshold else score * floor
    raise ValueError(f'gate mode must be one of {GATE_MODES}, not {mode}')
