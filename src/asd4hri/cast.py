"""CAST-ASD, the causal scene model of Gupta et al., inside the asd4hri pipeline (#80).

CAST-ASD scores one target face together with up to two other faces of the same frames
("scene-mates") and the raw audio, with zero look-ahead. Its audio front end is the frozen Mimi
encoder (kyutai). The network comes from the ``cast_asd`` package (CAST-ASD repository, pinned in
the ``av`` extra); the AVA checkpoint ``ava_cast.ckpt`` (official AVA val mAP 91.21) and the Mimi
weights are downloaded like the other AV weights (``asd4hri.av.weights_path``).

``AVClassifier`` builds the scenes; this module only runs the network on them. Inputs follow the
validation path of CAST-ASD's AVA recipe (``scene_ava_dm.collate_scenes`` with
``score_targets_only=True``):

* video: grey 112x112 crops, uint8, the target in slot 0;
* audio: one 16 kHz waveform per scene, scaled to -25 dBFS like CAST-ASD's data loader. The
  loader scales a whole face track at once; here the gain comes from the window, which ends at
  the newest frame, so it needs no later audio;
* the key padding mask marks the target's frames as real and the scene-mates' as padding, so the
  scene-mates reach the target only through the CAST block.
"""

import copy
from unittest import mock

import numpy as np

from .av import weights_path

# the model part of CAST-ASD's configs/ava/_base.yaml + configs/ava/cast.yaml
CAST_CONFIG = {
    'audio': {'name': 'mimi'},
    'head': 'binary',
    'cross': {'d_model': 128, 'm_ff': 4, 'n_heads': 8, 'n_layers': 1, 'dropout': 0.1},
    'temporal': {'d_model': 256, 'm_ff': 4, 'n_heads': 8, 'n_layers': 1, 'dropout': 0.1},
    'scene': {'mechanism': 'cast', 'max_tracks': 3, 'n_layers': 1, 'n_heads': 8, 'm_ff': 4,
              'dropout': 0.1, 'gated': True},
}
_ENCODER = 'audio_frontend.encoder.'       # frozen Mimi, not stored in the checkpoint


class CASTModel:
    """CAST-ASD with its AVA weights, ready for inference.

    Args:
        name: Key of ``asd4hri.av.AV_MODELS`` ('CAST-ASD').
        device: 'cuda', 'cpu' or None (cuda if available).
    """

    def __init__(self, name='CAST-ASD', device=None):
        """Build the network, load Mimi and the checkpoint.

        Raises:
            RuntimeError: If the checkpoint does not fit the network.
        """
        import torch
        from cast_asd.data.media import normalize_waveform
        from cast_asd.models.asd import AudioVisualASD
        self.torch, self.normalize = torch, normalize_waveform
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_tracks = CAST_CONFIG['scene']['max_tracks']
        checkpoint = weights_path(name)
        mimi = checkpoint.parent / 'mimi-e351c8d8-checkpoint125.safetensors'
        # cast_asd fetches Mimi from the Hugging Face Hub; hand it the pinned, checked file
        with mock.patch('huggingface_hub.hf_hub_download', lambda *a, **k: str(mimi)):
            self.net = AudioVisualASD(copy.deepcopy(CAST_CONFIG))
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)['state_dict']
        state = {k[len('model.'):]: v for k, v in state.items() if k.startswith('model.')}
        missing, unexpected = self.net.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.startswith(_ENCODER)]
        if missing or unexpected:
            raise RuntimeError(f'{name}: weights do not fit, missing {missing[:5]}, '
                               f'unexpected {unexpected[:5]}')
        self.net.eval().to(self.device)

    def scores(self, scenes, waveforms):
        """Speaking probabilities of the target of every scene, for every frame.

        Args:
            scenes: One ``(tracks, T, 112, 112)`` uint8 array per scene, the target first and
                at most ``max_tracks`` tracks; all scenes have the same ``T``.
            waveforms: ``(scenes, samples)`` float32 16 kHz audio spanning the ``T`` frames.

        Returns:
            numpy.ndarray: ``(scenes, T)`` probabilities of the targets.
        """
        torch, dev = self.torch, self.device
        T = scenes[0].shape[1]
        slots = [len(s) for s in scenes]
        # targets in rows 0..n-1, scene-mates after them, as in collate_scenes
        rows = list(range(len(scenes)))
        mates = [(i, slot) for i, n in enumerate(slots) for slot in range(1, n)]
        visual = np.concatenate([np.stack([s[0] for s in scenes])]
                                + [scenes[i][slot][None] for i, slot in mates])
        index = rows + [i for i, _ in mates]
        slot = [0] * len(rows) + [slot for _, slot in mates]
        track_mask = np.zeros((len(scenes), max(slots)), dtype=bool)
        for i, n in enumerate(slots):
            track_mask[i, :n] = True
        frame_mask = np.zeros((len(index), T), dtype=bool)
        frame_mask[:len(rows)] = True
        with torch.no_grad():
            wave = torch.stack([self.normalize(torch.as_tensor(w)[None])[0] for w in waveforms])
            wave = wave.to(dev)
            n_audio = torch.full((len(scenes),), wave.shape[-1], device=dev)
            scene = {
                'index': torch.as_tensor(index, device=dev),
                'slot': torch.as_tensor(slot, device=dev),
                'mask': torch.as_tensor(track_mask, device=dev),
                'audio_lengths': n_audio,
                'frame_lengths': torch.full((len(scenes),), T, device=dev),
            }
            av, _, _ = self.net(wave, torch.as_tensor(visual, device=dev).float(),
                                n_audio[torch.as_tensor(index, device=dev)],
                                torch.full((len(index),), T, device=dev),
                                torch.as_tensor(frame_mask, device=dev), scene=scene)
            prob = av[:len(rows)].sigmoid()
        return prob.cpu().numpy()
