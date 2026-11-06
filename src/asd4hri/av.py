"""Audio-visual ASD models (TalkNet family) inside the asd4hri pipeline (#40, #80).

TalkNet, LR-ASD, Light-ASD and LRX-ASD share one input format:

* video: grey 112x112 face crops of one face track, ``(B, T, 112, 112)`` uint8;
* audio: 13 MFCCs with 4 audio frames per video frame, ``(B, 4T, 13)``. As in TalkNet's AVA
  data loader, the MFCC window and step are scaled by ``25 / fps``, so the video does not have
  to be resampled to 25 fps.

The models are PyTorch and live in their own repositories (git submodules under ``src/``).
They are imported by path and need the optional extra ``pip install asd4hri[av]``.
``AVClassifier`` buffers the crops of every face and predicts the newest frame from the last
``window`` frames plus the audio of the same span (no look-ahead beyond the current frame).
"""

import importlib
import subprocess
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from paz.abstract import Processor

from .timing import NULL_TIMER

_SRC = Path(__file__).resolve().parent.parent
_MODEL_DIR = _SRC / 'models'

# name -> how to build it. 'repo' is the submodule directory, 'net' the model class,
# 'cross_attention' whether the TalkNet cross-attention step runs between front end and back end.
AV_MODELS = {
    'LR-ASD': {
        'repo': 'LR-ASD', 'net': ('model.Model', 'ASD_Model'), 'cross_attention': False,
        'weights': ('file', 'weight/pretrain_AVA.model'),
    },
    'TalkNet': {
        'repo': 'TalkNet-ASD', 'net': ('model.talkNetModel', 'talkNetModel'),
        'cross_attention': True,
        'weights': ('gdrive', '1NVIkksrD3zbxbDuDbPc_846bLfPSZcZm', 'TalkNet_pretrain_AVA.model'),
    },
}
AV_PREFIX = 'AV:'


def av_architectures():
    """Architecture names of the AV models for ``--architecture``, e.g. 'AV:LR-ASD'."""
    return [AV_PREFIX + name for name in AV_MODELS]


# ── audio ────────────────────────────────────────────────────────────────────

def load_audio(video_path, sample_rate=16000):
    """Decode the audio track of a video to mono int16 at ``sample_rate`` with ffmpeg.

    Returns:
        numpy.ndarray: int16 samples; empty if the video has no audio stream.
    """
    cmd = ['ffmpeg', '-v', 'error', '-i', str(video_path), '-vn', '-ac', '1', '-ar',
           str(sample_rate), '-f', 's16le', '-']
    out = subprocess.run(cmd, capture_output=True)
    if out.returncode != 0:
        return np.zeros(0, dtype=np.int16)
    return np.frombuffer(out.stdout, dtype=np.int16)


def mfcc_window(audio, start_s, num_frames, fps, sample_rate=16000):
    """MFCCs for ``num_frames`` video frames starting at ``start_s`` (4 rows per frame).

    Window and step are scaled by ``25 / fps`` as in TalkNet's AVA loader. Missing audio at
    the borders is filled by wrapping, like TalkNet does.

    Returns:
        numpy.ndarray: ``(4 * num_frames, 13)`` float32.
    """
    import python_speech_features
    first = max(0, int(round(start_s * sample_rate)))
    last = int(round((start_s + num_frames / fps) * sample_rate))
    segment = audio[first:last]
    rows = 4 * num_frames
    if len(segment) < int(0.025 * 25 / fps * sample_rate) + 1:
        return np.zeros((rows, 13), dtype=np.float32)
    feats = python_speech_features.mfcc(segment, sample_rate, numcep=13,
                                        winlen=0.025 * 25 / fps, winstep=0.010 * 25 / fps)
    if feats.shape[0] < rows:
        feats = np.pad(feats, ((0, rows - feats.shape[0]), (0, 0)), 'wrap')
    return feats[:rows].astype(np.float32)


# ── faces ────────────────────────────────────────────────────────────────────

def face_crop(image, box):
    """Grey 112x112 crop of one face box, as in TalkNet's AVA preprocessing.

    Args:
        image: RGB frame.
        box: ``(x1, y1, x2, y2)`` in pixels.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    face = image[y1:y2, x1:x2]
    if face.size == 0:
        return np.zeros((112, 112), dtype=np.uint8)
    return cv2.resize(cv2.cvtColor(face, cv2.COLOR_RGB2GRAY), (112, 112))


# ── models ───────────────────────────────────────────────────────────────────

def _import_from_repo(repo_dir, module, attr):
    """Import ``module.attr`` from a model repo whose modules have generic names (model, loss)."""
    repo_dir = str(repo_dir)
    clashing = [m for m in list(sys.modules) if m in ('model', 'loss') or m.startswith('model.')]
    saved = {m: sys.modules.pop(m) for m in clashing}
    sys.path.insert(0, repo_dir)
    try:
        net_cls = getattr(importlib.import_module(module), attr)
        loss = importlib.import_module('loss')
        return net_cls, loss.lossAV, loss.lossV
    finally:
        sys.path.remove(repo_dir)
        for m in [m for m in list(sys.modules) if m in ('model', 'loss') or m.startswith('model.')]:
            sys.modules.pop(m)
        sys.modules.update(saved)


def weights_path(name):
    """Local path of a model's AVA weights, downloading them if needed."""
    spec = AV_MODELS[name]['weights']
    if spec[0] == 'file':
        return _SRC / AV_MODELS[name]['repo'] / spec[1]
    path = _MODEL_DIR / spec[2]
    if not path.is_file():
        import gdown
        path.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id=spec[1], output=str(path), quiet=True)
    return path


class AVModel:
    """One TalkNet-family model, ready for inference.

    Args:
        name: Key of ``AV_MODELS``.
        device: 'cuda', 'cpu' or None (cuda if available).
    """

    def __init__(self, name, device=None):
        """Build the network and load its AVA weights."""
        import torch
        self.torch = torch
        spec = AV_MODELS[name]
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        net_cls, loss_av, loss_v = _import_from_repo(_SRC / spec['repo'], *spec['net'])

        # same attribute names as the repos' training wrappers, so the state dict keys match
        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model, self.lossAV, self.lossV = net_cls(), loss_av(), loss_v()

        self.net = Wrapper()
        state = torch.load(weights_path(name), map_location='cpu')
        own = self.net.state_dict()
        loaded = 0
        for key, value in state.items():
            key = key.replace('module.', '')
            if key in own and own[key].shape == value.shape:
                own[key].copy_(value)
                loaded += 1
        if loaded < len(own) * 0.95:
            raise RuntimeError(f'{name}: only {loaded}/{len(own)} weights matched')
        self.net.to(self.device).eval()
        self.cross_attention = spec['cross_attention']

    def scores(self, video, audio=None):
        """Speaking probabilities for every frame of a batch of equal-length tracks.

        Args:
            video: ``(B, T, 112, 112)`` uint8 grey faces.
            audio: ``(B, 4T, 13)`` MFCCs, or None to use the visual-only back end.

        Returns:
            numpy.ndarray: ``(B, T)`` probabilities.
        """
        torch = self.torch
        m, B, T = self.net.model, video.shape[0], video.shape[1]
        with torch.no_grad():
            v = m.forward_visual_frontend(torch.as_tensor(video, dtype=torch.float32,
                                                          device=self.device))
            if audio is None:
                x = m.forward_visual_backend(v)
                logits = self.net.lossV.FC(x)
            else:
                a = m.forward_audio_frontend(torch.as_tensor(audio, dtype=torch.float32,
                                                             device=self.device))
                if self.cross_attention:
                    a, v = m.forward_cross_attention(a, v)
                x = m.forward_audio_visual_backend(a, v)
                logits = self.net.lossAV.FC(x)
            prob = torch.softmax(logits, dim=-1)[:, 1]
        return prob.reshape(B, T).cpu().numpy()


class AVClassifier(Processor):
    """Score faces with an audio-visual model, one window per face (#40).

    Call ``set_audio`` once per video (or stream). Each call takes the full RGB frame, the face
    boxes, optional entity ids and the frame timestamp, and returns one score per box: the
    probability for the newest frame, predicted from the last ``window`` frames of that face
    and the audio of the same time span. Faces with fewer frames are scored on what is there.

    Args:
        name: Key of ``AV_MODELS``.
        window: Maximum number of frames per prediction (25 = 1 s at 25 fps).
        stride: Predict a face only every ``stride`` frames (None in between).
        max_consecutive_empty: Frames a face may be missing before its buffer is dropped.
        audio_fallback: Without audio: 'noise' feeds low-level white noise to the AV back end,
            'visual' uses the model's visual-only back end.
        timer: Optional ``StageTimer``.
        device: Torch device, None = cuda if available.
    """

    def __init__(self, name, window=25, stride=1, max_consecutive_empty=5,
                 audio_fallback='noise', timer=None, device=None):
        """Load the model."""
        super().__init__()
        if audio_fallback not in ('noise', 'visual'):
            raise ValueError(f"audio_fallback must be 'noise' or 'visual', not {audio_fallback}")
        self.model = AVModel(name, device)
        self.window, self.stride = window, stride
        self.max_consecutive_empty = max_consecutive_empty
        self.audio_fallback = audio_fallback
        self.timer = timer or NULL_TIMER
        self.audio, self.fps = np.zeros(0, dtype=np.int16), 25.0
        self.tracks = {}
        self.rng = np.random.default_rng(0)

    def set_audio(self, audio, fps):
        """Set the audio (int16, 16 kHz mono) and frame rate for the following calls."""
        self.audio = audio if audio is not None else np.zeros(0, dtype=np.int16)
        self.fps = float(fps)
        self.reset()

    def reset(self):
        """Drop all face buffers."""
        self.tracks = {}

    def _audio_for(self, start_s, num_frames):
        if len(self.audio) and start_s * 16000 < len(self.audio):
            return mfcc_window(self.audio, start_s, num_frames, self.fps)
        if self.audio_fallback == 'visual':
            return None
        noise = (self.rng.normal(0, 30, int(num_frames / self.fps * 16000) + 400)
                 .astype(np.int16))
        return mfcc_window(noise, 0.0, num_frames, self.fps)

    def call(self, image, boxes, ids=None, timestamp=None):
        """Score every box of one frame.

        Args:
            image: RGB frame.
            boxes: Face boxes ``(x1, y1, x2, y2)`` in pixels.
            ids: Entity ids, one per box; default = position in the list.
            timestamp: Time of the frame in seconds.

        Returns:
            list: One probability per box, or None where no prediction is due.
        """
        timer, n = self.timer, len(boxes)
        keys = list(ids) if ids is not None else list(range(n))
        with timer.stage('preprocess', n):
            crops = [face_crop(image, b) for b in boxes]
        with timer.stage('buffer', n):
            for key, crop in zip(keys, crops):
                track = self.tracks.setdefault(key, {'faces': deque(maxlen=self.window),
                                                     'times': deque(maxlen=self.window),
                                                     'missed': 0, 'since': self.stride - 1})
                track['faces'].append(crop)
                track['times'].append(timestamp)
                track['missed'] = 0
                track['since'] += 1
            for key in list(self.tracks):
                if key not in keys:
                    self.tracks[key]['missed'] += 1
                    if self.tracks[key]['missed'] >= self.max_consecutive_empty:
                        del self.tracks[key]
        due = [k for k in keys if self.tracks[k]['since'] >= self.stride]
        scores = {}
        if due:
            groups = {}
            for k in due:
                self.tracks[k]['since'] = 0
                groups.setdefault(len(self.tracks[k]['faces']), []).append(k)
            for length, group in groups.items():
                video = np.stack([np.stack(self.tracks[k]['faces']) for k in group])
                with timer.stage('audio', len(group)):
                    audio = [self._audio_for(self.tracks[k]['times'][0] or 0.0, length)
                             for k in group]
                audio = None if any(a is None for a in audio) else np.stack(audio)
                with timer.stage('model', len(group)):
                    out = self.model.scores(video, audio)[:, -1]
                scores.update(zip(group, out.tolist()))
        return [scores.get(k) for k in keys]
