"""Audio-visual ASD models (TalkNet family) inside the asd4hri pipeline (#40, #80).

TalkNet, LR-ASD, Light-ASD and LRX-ASD share one input format:

* video: grey 112x112 face crops of one face track, ``(B, T, 112, 112)`` uint8;
* audio: 13 MFCCs with 4 audio frames per video frame, ``(B, 4T, 13)``. As in TalkNet's AVA
  data loader, the MFCC window and step are scaled by ``25 / fps``, so the video does not have
  to be resampled to 25 fps.

The networks come from the ``pyasd`` package (LRX-ASD repository, pinned in the ``av`` extra:
``pip install asd4hri[av]``). The AVA weights are downloaded on first use into
``$ASD4HRI_WEIGHTS`` (default ``src/models``) and checked against their SHA-256.
``AVClassifier`` buffers the crops of every face and predicts the newest frame from the last
``window`` frames plus the audio of the same span (no look-ahead beyond the current frame).
"""

import hashlib
import os
import subprocess
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from paz.abstract import Processor

from .timing import NULL_TIMER

_SRC = Path(__file__).resolve().parent.parent
_PYASD_RELEASE = 'https://github.com/SebastianJanampa/LRX-ASD/releases/download/models_weight/'

# name -> pyasd architecture and AVA weights. The LR-ASD and Light-ASD files of the pyasd
# release are byte-identical to the authors' AVA weights. pyasd's TalkNet file is a retrained
# model, so TalkNet uses the authors' original checkpoint (the one behind its 92.3 mAP).
AV_MODELS = {
    'LR-ASD': {
        'pyasd': 'lrasd', 'file': 'lr_asd.model', 'url': _PYASD_RELEASE + 'lr_asd.model',
        'sha256': 'e76d2e1d6d0f3b3d64468f055a03e44e074bca2bfc90d4a10af1b7fa4ae27091',
    },
    'TalkNet': {
        'pyasd': 'talknet', 'file': 'TalkNet_pretrain_AVA.model',
        'gdrive': '1NVIkksrD3zbxbDuDbPc_846bLfPSZcZm',
        'sha256': 'a83d7e88c46633ebd06025337512a7d78edd8ebb8981b148b702c02de767a4e0',
    },
    'Light-ASD': {
        'pyasd': 'lightasd', 'file': 'light_asd.model', 'url': _PYASD_RELEASE + 'light_asd.model',
        'sha256': '7ed8c839e7eaa772ca0504e8bd9fc09901e7539561e9ae3465e7671f7a4079ec',
    },
    'LRX-ASD': {
        'pyasd': 'lrxasd', 'file': 'lrx_asd.model', 'url': _PYASD_RELEASE + 'lrx_asd.model',
        'sha256': '0219bb1295e8815feaf221c250bd3a149b14f3c245a77ffed87881a7954fe207',
    },
}
AV_PREFIX = 'AV:'

# parameters that only exist for training (LRX-ASD's DINO head) and may be absent in a file
_TRAINING_ONLY = ('dino_head.', 'center')


def av_architectures():
    """Architecture names of the AV models for ``--architecture``, e.g. 'AV:LR-ASD'.

    Returns:
        list[str]: The names.
    """
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

    Returns:
        numpy.ndarray: ``(112, 112)`` uint8; zeros if the box is outside the image.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    face = image[y1:y2, x1:x2]
    if face.size == 0:
        return np.zeros((112, 112), dtype=np.uint8)
    return cv2.resize(cv2.cvtColor(face, cv2.COLOR_RGB2GRAY), (112, 112))


# ── models ───────────────────────────────────────────────────────────────────

def weights_dir():
    """Folder for the downloaded AV weights: ``$ASD4HRI_WEIGHTS`` or ``src/models``.

    Returns:
        pathlib.Path: The folder.
    """
    return Path(os.environ.get('ASD4HRI_WEIGHTS', _SRC / 'models'))


def _sha256(path):
    """SHA-256 of a file.

    Args:
        path: The file.

    Returns:
        str: Hex digest.
    """
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def weights_path(name):
    """Local path of a model's AVA weights, downloading and checking them if needed.

    Args:
        name: Key of ``AV_MODELS``.

    Returns:
        pathlib.Path: The weights file.

    Raises:
        RuntimeError: If the file does not have the expected SHA-256.
    """
    spec = AV_MODELS[name]
    path = weights_dir() / spec['file']
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + '.part')
        if 'gdrive' in spec:
            import gdown
            gdown.download(id=spec['gdrive'], output=str(tmp), quiet=True)
        else:
            urllib.request.urlretrieve(spec['url'], tmp)
        tmp.replace(path)
    if _sha256(path) != spec['sha256']:
        raise RuntimeError(f'{path}: unexpected SHA-256, delete it to download it again')
    return path


def _rename(key):
    """Map a state-dict key of the original repos (or pyasd) to pyasd's module names.

    The original checkpoints wrap the network as ``model.*`` and call the heads
    ``lossAV`` / ``lossV`` / ``lossA``; pyasd calls them ``av_head`` / ``v_head`` / ``a_head``.

    Args:
        key: State-dict key.

    Returns:
        str: The key in pyasd's naming.
    """
    for prefix in ('module.', 'model.'):
        if key.startswith(prefix):
            key = key[len(prefix):]
    for old, new in (('lossAV.', 'av_head.'), ('lossV.', 'v_head.'), ('lossA.', 'a_head.')):
        if key.startswith(old):
            key = new + key[len(old):]
    return key


class AVModel:
    """One TalkNet-family model from ``pyasd``, ready for inference.

    Args:
        name: Key of ``AV_MODELS``.
        device: 'cuda', 'cpu' or None (cuda if available).
        deploy: Fuse the re-parameterisable branches (LRX-ASD) for speed. This removes the
            visual-only head, so ``scores(video, None)`` is then not available.
    """

    def __init__(self, name, device=None, deploy=False):
        """Build the network and load its AVA weights.

        Raises:
            RuntimeError: If the weights do not fit the network.
        """
        import torch
        from pyasd.engine.model import MODELS
        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = MODELS[AV_MODELS[name]['pyasd']]()
        state = torch.load(weights_path(name), map_location='cpu', weights_only=True)
        if isinstance(state, dict) and 'model' in state and isinstance(state['model'], dict):
            state = state['model']
        state = {_rename(k): v for k, v in state.items()}
        missing, unexpected = self.net.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.startswith(_TRAINING_ONLY)]
        if missing or unexpected:
            raise RuntimeError(f'{name}: weights do not fit, missing {missing[:5]}, '
                               f'unexpected {unexpected[:5]}')
        self.net.eval()
        if deploy and hasattr(self.net, 'deploy'):
            self.net = self.net.deploy()
        self.net.to(self.device)

    def scores(self, video, audio=None):
        """Speaking probabilities for every frame of a batch of equal-length tracks.

        Args:
            video: ``(B, T, 112, 112)`` uint8 grey faces.
            audio: ``(B, 4T, 13)`` MFCCs, or None to use the visual-only back end.

        Returns:
            numpy.ndarray: ``(B, T)`` probabilities.

        Raises:
            RuntimeError: For ``audio=None`` on a deployed model (no visual-only head).
        """
        torch, net = self.torch, self.net
        B, T = video.shape[0], video.shape[1]
        with torch.no_grad():
            v = torch.as_tensor(video, dtype=torch.float32, device=self.device)
            v = (v / 255.0).unsqueeze(2)                      # (B, T, 1, 112, 112) in [0, 1]
            if audio is None:
                if not hasattr(net, 'v_head'):
                    raise RuntimeError('visual-only scores need deploy=False')
                logits = net.v_head(net.forward_visual_backend(net.forward_visual_frontend(v)))
                prob = torch.softmax(logits, dim=-1)[..., 1]
            else:
                a = torch.as_tensor(audio, dtype=torch.float32, device=self.device)
                prob = net(v, a.transpose(1, 2))['pred_scores'][..., 1]   # audio (B, 13, 4T)
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
        """Load the model.

        Raises:
            ValueError: If ``audio_fallback`` is not 'noise' or 'visual'.
        """
        super().__init__()
        if audio_fallback not in ('noise', 'visual'):
            raise ValueError(f"audio_fallback must be 'noise' or 'visual', not {audio_fallback}")
        self.model = AVModel(name, device, deploy=audio_fallback == 'noise')
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
