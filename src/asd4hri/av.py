"""Audio-visual ASD models (TalkNet family) inside the asd4hri pipeline (#40, #80).

TalkNet, LR-ASD, Light-ASD, LRX-ASD and FabuLight share one input format:

* video: grey 112x112 face crops of one face track, ``(B, T, 112, 112)`` uint8;
* audio: 13 MFCCs with 4 audio frames per video frame, ``(B, 4T, 13)``. As in TalkNet's AVA
  data loader, the MFCC window and step are scaled by ``25 / fps``, so the video does not have
  to be resampled to 25 fps.

The networks come from the ``pyasd`` package (LRX-ASD repository, pinned in the ``av`` extra:
``pip install asd4hri[av]``); FabuLight-ASD is pyasd's Light-ASD with FabuLight's WASD
weights. The weights are downloaded on first use into
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
_FABULIGHT = ('https://raw.githubusercontent.com/knowledgetechnologyuhh/FabuLight-ASD/'
              '265c5ce95b55830a2622ce35ae6556dd50289fc0/exps/exp_0.001_0.95_60_2000_Adam_epochStep_'
              '0.0_1.3_linear_0.02_112_F_fabulight_config/model/best_0044.model')
_HF = 'https://huggingface.co/'
_VISUAL_MEAN, _VISUAL_STD = 0.4161, 0.1688        # pyasd's normalisation of the grey crops

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
        # pyasd's TalkNet.forward skips this step of the original model, so ``scores`` runs it
        'cross_attention': True,
    },
    'Light-ASD': {
        'pyasd': 'lightasd', 'file': 'light_asd.model', 'url': _PYASD_RELEASE + 'light_asd.model',
        'sha256': '7ed8c839e7eaa772ca0504e8bd9fc09901e7539561e9ae3465e7671f7a4079ec',
    },
    'LRX-ASD': {
        'pyasd': 'lrxasd', 'file': 'lrx_asd.model', 'url': _PYASD_RELEASE + 'lrx_asd.model',
        'sha256': '0219bb1295e8815feaf221c250bd3a149b14f3c245a77ffed87881a7954fe207',
    },
    # FabuLight-ASD without body pose (MIT): the Light-ASD network trained on WASD instead of
    # AVA, with another std for the crops (0.2380). Same outputs as its own code (tested).
    'FabuLight': {
        'pyasd': 'lightasd', 'file': 'fabulight_wasd.model', 'url': _FABULIGHT,
        'sha256': '40dfff26c90140194c0f6269f6a50840cb9de84e9c5947d830bc8c44863b3845',
        'visual_std': 0.2380,
    },
    # CAST-ASD (MIT): causal scene model, one target face plus up to two faces of the same frames
    # (asd4hri.cast). Audio goes through the frozen Mimi encoder of kyutai (CC-BY 4.0), whose
    # weights are the second file.
    'CAST-ASD': {
        'scene': True, 'file': 'cast_asd_ava.ckpt',
        'url': _HF + 'prathamg007/CAST-ASD/resolve/86b2ec3f412ee1f814401701a866c0a0f65a6ef5/'
                     'weights/ava_cast.ckpt',
        'sha256': 'd1fdcc538ec5fae06b78516cc9301f8dd3500a5dedd0bb0c22c6e39b118fd358',
        'extra': [{
            'file': 'mimi-e351c8d8-checkpoint125.safetensors',
            'url': _HF + 'kyutai/moshiko-pytorch-bf16/resolve/2bfc9ae6e89079a5cc7ed2a68436010d91a3d289/'
                         'tokenizer-e351c8d8-checkpoint125.safetensors',
            'sha256': '09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50',
        }],
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
    """Local path of a model's weights, downloading and checking them (and its extra files).

    Args:
        name: Key of ``AV_MODELS``.

    Returns:
        pathlib.Path: The weights file; the extra files are next to it.
    """
    for extra in AV_MODELS[name].get('extra', []):
        fetch(extra)
    return fetch(AV_MODELS[name])


def fetch(spec):
    """Download one weights file of ``spec`` into ``weights_dir()`` if missing, and check it.

    Args:
        spec: Dict with ``file``, ``sha256`` and ``url`` (or ``gdrive``, a Google Drive id).

    Returns:
        pathlib.Path: The file.

    Raises:
        RuntimeError: If the file does not have the expected SHA-256.
    """
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

    The original checkpoints wrap the network as ``model.*`` (``model.module.*`` with
    DataParallel) and call the heads ``lossAV`` / ``lossV`` / ``lossA``; pyasd calls them
    ``av_head`` / ``v_head`` / ``a_head``. FabuLight names the visual encoder ``faceEncoder``,
    the GRU ``GRUFA`` and the heads ``lossFA`` / ``lossF``.

    Args:
        key: State-dict key.

    Returns:
        str: The key in pyasd's naming.
    """
    parts = [p for p in key.split('.') if p != 'module']
    if parts[0] == 'model':
        parts = parts[1:]
    key = '.'.join(parts)
    for old, new in (('lossAV.', 'av_head.'), ('lossV.', 'v_head.'), ('lossA.', 'a_head.'),
                     ('lossFA.', 'av_head.'), ('lossF.', 'v_head.'),
                     ('faceEncoder.', 'visualEncoder.'), ('GRUFA.', 'GRU.')):
        if key.startswith(old):
            key = new + key[len(old):]
            break
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
        self.cross_attention = AV_MODELS[name].get('cross_attention', False)
        # rescale the crops so that pyasd's normalisation gives the model's own std
        self.visual_scale = _VISUAL_STD / AV_MODELS[name].get('visual_std', _VISUAL_STD)
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
            v = v / 255.0
            if self.visual_scale != 1.0:
                v = _VISUAL_MEAN + (v - _VISUAL_MEAN) * self.visual_scale
            v = v.unsqueeze(2)                                # (B, T, 1, 112, 112)
            if audio is None:
                if not hasattr(net, 'v_head'):
                    raise RuntimeError('visual-only scores need deploy=False')
                logits = net.v_head(net.forward_visual_backend(net.forward_visual_frontend(v)))
                prob = torch.softmax(logits, dim=-1)[..., 1]
            else:
                a = torch.as_tensor(audio, dtype=torch.float32, device=self.device)
                a = net.forward_audio_frontend(a.transpose(1, 2))      # MFCCs as (B, 13, 4T)
                v = net.forward_visual_frontend(v)
                if self.cross_attention:
                    a, v = net.forward_cross_attention(a, v)
                logits = net.av_head(net.forward_audio_visual_backend(a, v))
                prob = torch.softmax(logits, dim=-1)[..., 1]
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
        self.scene = AV_MODELS[name].get('scene', False)
        if self.scene:
            from .cast import CASTModel
            self.model = CASTModel(name, device)
        else:
            self.model = AVModel(name, device, deploy=audio_fallback == 'noise')
        self.window, self.stride = window, stride
        self.max_consecutive_empty = max_consecutive_empty
        self.audio_fallback = audio_fallback
        self.timer = timer or NULL_TIMER
        self.audio, self.fps = np.zeros(0, dtype=np.int16), 25.0
        self.tracks = {}
        self.frame = 0                                    # frame counter, aligns scene-mates
        self.rng = np.random.default_rng(0)

    def set_audio(self, audio, fps):
        """Set the audio (int16, 16 kHz mono) and frame rate for the following calls."""
        self.audio = audio if audio is not None else np.zeros(0, dtype=np.int16)
        self.fps = float(fps)
        self.reset()

    def reset(self):
        """Drop all face buffers."""
        self.tracks = {}
        self.frame = 0

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
                                                     'frames': deque(maxlen=self.window),
                                                     'missed': 0, 'since': self.stride - 1})
                track['faces'].append(crop)
                track['times'].append(timestamp)
                track['frames'].append(self.frame)
                track['missed'] = 0
                track['since'] += 1
            for key in list(self.tracks):
                if key not in keys:
                    self.tracks[key]['missed'] += 1
                    if self.tracks[key]['missed'] >= self.max_consecutive_empty:
                        del self.tracks[key]
        self.frame += 1
        due = [k for k in keys if self.tracks[k]['since'] >= self.stride]
        scores = {}
        if due:
            groups = {}
            for k in due:
                self.tracks[k]['since'] = 0
                groups.setdefault(len(self.tracks[k]['faces']), []).append(k)
            for length, group in groups.items():
                if self.scene:
                    scores.update(self._scene_scores(group, length))
                    continue
                video = np.stack([np.stack(self.tracks[k]['faces']) for k in group])
                with timer.stage('audio', len(group)):
                    audio = [self._audio_for(self.tracks[k]['times'][0] or 0.0, length)
                             for k in group]
                audio = None if any(a is None for a in audio) else np.stack(audio)
                with timer.stage('model', len(group)):
                    out = self.model.scores(video, audio)[:, -1]
                scores.update(zip(group, out.tolist()))
        return [scores.get(k) for k in keys]

    def _scene_scores(self, group, length):
        """Scores of a scene model for targets with ``length`` buffered frames each.

        As in CAST-ASD's AVA data: each target gets up to ``max_tracks - 1`` scene-mates, the
        buffered faces that share more than half of its frames (most shared first, ties by
        id), aligned onto its frames and zero where they are missing.

        Args:
            group: Keys of the targets.
            length: Their number of frames.

        Returns:
            dict: Key -> probability for the target's newest frame.
        """
        timer, scenes, waves = self.timer, [], []
        with timer.stage('buffer', len(group)):
            for k in group:
                own = list(self.tracks[k]['frames'])
                position = {f: i for i, f in enumerate(own)}
                mates = []
                for other, track in self.tracks.items():
                    shared = [(position[f], j) for j, f in enumerate(track['frames'])
                              if f in position]
                    if other != k and len(shared) > 0.5 * length:
                        mates.append((-len(shared), str(other), track, shared))
                mates.sort(key=lambda m: m[:2])
                faces = np.zeros((1 + min(len(mates), self.model.max_tracks - 1), length, 112, 112),
                                 dtype=np.uint8)
                faces[0] = np.stack(self.tracks[k]['faces'])
                for slot, (_, _, track, shared) in enumerate(mates[:len(faces) - 1], start=1):
                    for i, j in shared:
                        faces[slot, i] = track['faces'][j]
                scenes.append(faces)
        with timer.stage('audio', len(group)):
            for k in group:
                start = self.tracks[k]['times'][0] or 0.0
                waves.append(self._waveform_for(start, length))
        with timer.stage('model', len(group)):
            out = self.model.scores(scenes, np.stack(waves))[:, -1]
        return dict(zip(group, out.tolist()))

    def _waveform_for(self, start_s, num_frames):
        """16 kHz float waveform of ``num_frames`` frames from ``start_s``; noise without audio.

        Returns:
            numpy.ndarray: ``(samples,)`` float32 in [-1, 1].
        """
        samples = int(round(num_frames / self.fps * 16000))
        if len(self.audio) and start_s * 16000 < len(self.audio):
            first = max(0, int(round(start_s * 16000)))
            segment = self.audio[first:first + samples]
        else:
            segment = self.rng.normal(0, 30, samples).astype(np.int16)
        wave = np.zeros(samples, dtype=np.float32)
        wave[:len(segment)] = segment.astype(np.float32) / 32768.0
        return wave
