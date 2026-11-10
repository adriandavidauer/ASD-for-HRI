"""Fine-tune LipShapeCausal / FaceShapeCausal on AVA-ActiveSpeaker frame labels (#92).

Input: the ``<video_id>.npz`` files of ``extract_ava_landmarks``. Each face track (one entity,
consecutive video frames) becomes one sequence with a label per frame. Training batches are
random-length, random-offset crops of the tracks, so the model sees every input length from a
single frame up to ``--max_len``. In the pipeline the streaming model runs over the whole
track, so long crops are part of the training.

About 10% of the videos are held out for early stopping (split by video, fixed seed); AVA val
is not used. The model starts from the VVAD-LRS3 weights in ``src/models`` unless
``--from_scratch`` is given. The best model by held-out per-frame average precision is saved
as ``<out_dir>/<Lip|Face>ShapeCausal.keras``, a drop-in replacement for the released one.

Usage (from the repo root, in the image)::

    python -m src.model_training.finetune_causal_shape_ava /ft/ava_train_landmarks -f LipShape -o out/
"""

import argparse
import json
import zlib
from pathlib import Path

import numpy as np

from asd4hri.models import normalize_shape_frames

LIP = slice(48, 68)
MAX_GAP = 5          # a gap of more frames ends a track (the pipeline's max_consecutive_empty)


def load_tracks(files, feature_type):
    """Split the extracted rows of some videos into per-entity tracks.

    Args:
        files: ``.npz`` files of ``extract_ava_landmarks``.
        feature_type: 'LipShape' (points 48-67) or 'FaceShape' (all 68).

    Returns:
        list: ``(x, y)`` per track: normalised landmarks ``(T, landmarks, 2)`` float32 and
        per-frame labels ``(T,)`` float32, in frame order.
    """
    tracks = []
    for path in files:
        d = np.load(path)
        points = d['landmarks'][:, LIP] if feature_type == 'LipShape' else d['landmarks']
        x_all = normalize_shape_frames(points)
        for entity in np.unique(d['entity']):
            rows = np.flatnonzero(d['entity'] == entity)
            rows = rows[np.argsort(d['frame'][rows], kind='stable')]
            cuts = np.flatnonzero(np.diff(d['frame'][rows]) > MAX_GAP) + 1
            for part in np.split(rows, cuts):
                tracks.append((x_all[part], d['label'][part].astype(np.float32)))
    return tracks


def split_videos(files, held_out=0.1, seed=0):
    """Split the video files into train and held-out by a hash of their name.

    Args:
        files: The ``.npz`` files.
        held_out: Fraction of held-out videos.
        seed: Changes the split.

    Returns:
        tuple: ``(train_files, held_out_files)``.
    """
    def score(p):
        """Position of a file in [0, 1), fixed by its name and the seed.

        Returns:
            float: The position.
        """
        return zlib.crc32(f'{seed}:{Path(p).stem}'.encode()) / 2**32
    # at least one held-out video, the one with the lowest score
    cut = max(held_out, min((score(f) for f in files), default=0) + 1e-12)
    train = [f for f in files if score(f) >= cut]
    return train, [f for f in files if score(f) < cut]


class TrackCrops:
    """Endless batches of random-length, random-offset crops of tracks with per-frame labels.

    Every batch has one length ``L``, drawn log-uniformly from 1 to ``max_len``; its crops come
    from tracks of at least ``L`` frames, picked with probability proportional to the number of
    crops they hold, so every frame is equally likely to be used.

    Args:
        tracks: ``(x, y)`` per track (``load_tracks``).
        batch_size: Crops per batch.
        max_len: Longest crop.
        seed: Seed of the generator.
    """

    def __init__(self, tracks, batch_size=64, max_len=256, seed=0):
        """Sort the tracks by length for the per-length sampling."""
        self.tracks = sorted(tracks, key=lambda t: len(t[1]))
        self.lengths = np.array([len(t[1]) for t in self.tracks])
        self.batch_size = batch_size
        self.max_len = int(min(max_len, self.lengths.max()))
        self.rng = np.random.default_rng(seed)

    def batch(self):
        """One batch.

        Returns:
            tuple: ``(x, y)`` of shapes ``(B, L, landmarks, 2)`` and ``(B, L, 1)``.
        """
        rng = self.rng
        length = int(np.exp(rng.uniform(0, np.log(self.max_len + 1))))
        length = max(1, min(length, self.max_len))
        first = int(np.searchsorted(self.lengths, length))
        weights = (self.lengths[first:] - length + 1).astype(np.float64)
        picks = first + rng.choice(len(weights), self.batch_size, p=weights / weights.sum())
        xb, yb = [], []
        for i in picks:
            x, y = self.tracks[i]
            start = int(rng.integers(0, len(y) - length + 1))
            xb.append(x[start:start + length])
            yb.append(y[start:start + length])
        return np.stack(xb), np.stack(yb)[..., None]

    def __iter__(self):
        """Yield batches forever.

        Yields:
            tuple: ``(x, y)`` (see ``batch``).
        """
        while True:
            yield self.batch()


def predict_tracks(model, tracks, batch_frames=200_000):
    """Causal per-frame scores of whole tracks.

    Tracks of similar length are padded at the end and run together; for a causal model the
    padding after a frame does not change that frame's output.

    Args:
        model: Causal Keras model.
        tracks: ``(x, y)`` per track.
        batch_frames: Upper bound of frames (padded) per model call.

    Returns:
        tuple: ``(scores, labels)``: concatenated per-frame scores and labels.
    """
    order = np.argsort([len(y) for _, y in tracks])
    scores, labels = [], []
    i = 0
    while i < len(order):
        # ascending lengths: the last track of the group is the longest
        n = 1
        while i + n < len(order) and (n + 1) * len(tracks[order[i + n]][1]) <= batch_frames:
            n += 1
        group = order[i:i + n]
        longest = len(tracks[group[-1]][1])
        x = np.zeros((len(group), longest) + tracks[group[0]][0].shape[1:], np.float32)
        for k, j in enumerate(group):
            x[k, :len(tracks[j][1])] = tracks[j][0]
        out = model.predict(x, batch_size=len(group), verbose=0)[..., 0]
        for k, j in enumerate(group):
            scores.append(out[k, :len(tracks[j][1])])
            labels.append(tracks[j][1])
        i += n
    return np.concatenate(scores), np.concatenate(labels)


def main():
    """Fine-tune, keep the best model by held-out per-frame AP, write a JSON report."""
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('landmark_dir', help='folder with the .npz files of extract_ava_landmarks')
    p.add_argument('-f', '--feature_type', choices=['LipShape', 'FaceShape'], default='LipShape')
    p.add_argument('-o', '--out_dir', default='.')
    p.add_argument('-e', '--epochs', type=int, default=60)
    p.add_argument('--steps', type=int, default=1000, help='batches per epoch')
    p.add_argument('-b', '--batch_size', type=int, default=64)
    p.add_argument('--max_len', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--from_scratch', action='store_true')
    p.add_argument('-s', '--seed', type=int, default=42)
    args = p.parse_args()

    import keras
    from sklearn.metrics import average_precision_score

    from asd4hri.asd import load_vvad_classifier
    from asd4hri.models import buildCausalFeatureLSTM

    keras.utils.set_random_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f'{args.feature_type}Causal'
    files = sorted(Path(args.landmark_dir).glob('*.npz'))
    files = [f for f in files if '.part' not in f.name]
    train_files, val_files = split_videos(files)
    train = load_tracks(train_files, args.feature_type)
    val = load_tracks(val_files, args.feature_type)
    num_landmarks = train[0][0].shape[1]
    print(f'{len(train_files)} train videos / {len(train)} tracks / '
          f'{sum(len(y) for _, y in train)} frames; {len(val_files)} held-out videos / '
          f'{len(val)} tracks', flush=True)

    if args.from_scratch:
        model = buildCausalFeatureLSTM(num_landmarks)
    else:
        model, _ = load_vvad_classifier(name)
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(args.lr),
                  metrics=['accuracy'])
    best_path = out_dir / f'{name}.keras'

    def held_out_ap():
        """Per-frame average precision of the current model on the held-out tracks.

        Returns:
            float: The AP.
        """
        scores, labels = predict_tracks(model, val)
        return float(average_precision_score(labels, scores))

    class HeldOutAP(keras.callbacks.Callback):
        """Logs ``val_ap`` after every epoch for the checkpoint and the schedules."""

        def on_epoch_end(self, epoch, logs=None):
            """Add the held-out per-frame AP to the logs."""
            logs['val_ap'] = held_out_ap()

    initial = held_out_ap()
    print(f'held-out AP before fine-tuning: {initial:.4f}', flush=True)
    callbacks = [
        HeldOutAP(),
        keras.callbacks.ReduceLROnPlateau(monitor='val_ap', mode='max', factor=0.3, patience=5,
                                          min_lr=1e-5),
        keras.callbacks.EarlyStopping(monitor='val_ap', mode='max', patience=12,
                                      restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(str(best_path), monitor='val_ap', mode='max',
                                        save_best_only=True),
        keras.callbacks.CSVLogger(str(out_dir / f'{name}_history.csv')),
    ]
    crops = TrackCrops(train, args.batch_size, args.max_len, seed=args.seed)
    model.fit(iter(crops), steps_per_epoch=args.steps, epochs=args.epochs,
              callbacks=callbacks, verbose=2)

    model = keras.models.load_model(str(best_path))
    report = {
        'feature_type': args.feature_type, 'from_scratch': args.from_scratch,
        'params': int(model.count_params()), 'train_videos': len(train_files),
        'held_out_videos': [f.stem for f in val_files],
        'held_out_ap_before': initial, 'held_out_ap': held_out_ap(), 'args': vars(args),
    }
    with open(out_dir / f'{name}_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != 'held_out_videos'}), flush=True)


if __name__ == '__main__':
    main()
