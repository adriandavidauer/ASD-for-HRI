"""Train the causal, variable-length landmark models LipShapeCausal / FaceShapeCausal (#45).

The model predicts speaking / not speaking for every frame from that frame and all frames
before it. Training uses the VVAD-LRS3 landmark features (hdf5 with ``x_train`` of shape
``(N, 38, landmarks, 2)`` and clip labels ``y_train``). Every batch is cut to a random length
(1-38 frames) at a random offset, and the clip label is the target of every frame.

Usage (from the repo root)::

    python -m src.model_training.train_causal_shape /Data/lipFeatures.h5 -f LipShape -o out/
"""

import argparse
import json
import math
import os
from pathlib import Path

import h5py
import keras
import numpy as np
from sklearn.model_selection import train_test_split

from asd4hri.models import buildCausalFeatureLSTM, normalize_shape_frames

LENGTHS = (1, 2, 3, 5, 10, 15, 19, 25, 30, 38)


class RandomLengthBatches(keras.utils.PyDataset):
    """Batches of random-length, random-offset crops with per-frame targets.

    Args:
        x: Normalized samples ``(N, T, landmarks, 2)``.
        y: Clip labels ``(N,)``.
        batch_size: Samples per batch.
        min_len: Shortest crop length.
        seed: Seed for the crop lengths, offsets and shuffling.
    """

    def __init__(self, x, y, batch_size=64, min_len=1, seed=0, **kwargs):
        """Shuffle the samples once; batches are cut on access."""
        super().__init__(**kwargs)
        self.x, self.y = x, y.astype(np.float32)
        self.batch_size, self.min_len = batch_size, min_len
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(len(x))

    def __len__(self):
        """Number of batches per epoch.

        Returns:
            int: The number of batches.
        """
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        """One batch, cut to a random length at a random offset.

        Args:
            idx: Batch index.

        Returns:
            tuple: ``(x, y)`` of shapes ``(B, L, landmarks, 2)`` and ``(B, L, 1)``.
        """
        ids = self.order[idx * self.batch_size:(idx + 1) * self.batch_size]
        full = self.x.shape[1]
        length = int(self.rng.integers(self.min_len, full + 1))
        start = int(self.rng.integers(0, full - length + 1))
        xb = self.x[ids, start:start + length]
        yb = np.repeat(self.y[ids, None, None], length, axis=1)
        return xb, yb

    def on_epoch_end(self):
        """Reshuffle the samples."""
        self.order = self.rng.permutation(len(self.x))


def accuracy_by_length(model, x, y, lengths=LENGTHS):
    """Accuracy of the prediction at the last frame of the first ``n`` frames, per length n.

    Args:
        model: Trained causal model.
        x: Normalised samples ``(N, T, landmarks, 2)``.
        y: Clip labels.
        lengths: Lengths to evaluate.

    Returns:
        dict: length -> accuracy.
    """
    out = {}
    for n in lengths:
        pred = model.predict(x[:, :n], batch_size=512, verbose=0)[:, -1, 0]
        out[n] = float(((pred > 0.5) == (y > 0.5)).mean())
    return out


def per_frame_accuracy(model, x, y):
    """Accuracy over all frames of full-length samples, each frame scored causally.

    Args:
        model: Trained causal model.
        x: Normalised samples ``(N, T, landmarks, 2)``.
        y: Clip labels.

    Returns:
        float: The accuracy.
    """
    pred = model.predict(x, batch_size=512, verbose=0)[..., 0]
    return float(((pred > 0.5) == (y[:, None] > 0.5)).mean())


def main():
    """Train, keep the best model by last-frame validation accuracy, write a JSON report."""
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('path', help='hdf5 file with x_train/y_train (and x_test/y_test)')
    p.add_argument('-f', '--feature_type', choices=['LipShape', 'FaceShape'], default='LipShape')
    p.add_argument('-o', '--out_dir', default='.')
    p.add_argument('-e', '--epochs', type=int, default=200)
    p.add_argument('-b', '--batch_size', type=int, default=64)
    p.add_argument('-s', '--seed', type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)
    keras.utils.set_random_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    with h5py.File(args.path, 'r') as data:
        x = normalize_shape_frames(data['x_train'][:])
        y = data['y_train'][:].astype(np.float32)
        x_test = normalize_shape_frames(data['x_test'][:]) if 'x_test' in data else None
        y_test = data['y_test'][:].astype(np.float32) if 'y_test' in data else None

    # same split as trainModel.py (test_size 0.2, random_state 42) -> comparable to 0.8958
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    model = buildCausalFeatureLSTM(x.shape[2])
    name = f'{args.feature_type}Causal'
    best_path = str(Path(args.out_dir) / f'{name}.keras')

    # validate on full-length clips, scoring the last frame (= the old clip-level protocol)
    class LastFrameAccuracy(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            """Log the validation accuracy of the last frame of full-length clips."""
            pred = self.model.predict(x_val, batch_size=1024, verbose=0)[:, -1, 0]
            logs['val_last_accuracy'] = float(((pred > 0.5) == (y_val > 0.5)).mean())

    callbacks = [
        LastFrameAccuracy(),
        keras.callbacks.ReduceLROnPlateau(monitor='val_last_accuracy', mode='max', factor=0.3,
                                          patience=8, min_lr=1e-5),
        keras.callbacks.EarlyStopping(monitor='val_last_accuracy', mode='max', patience=25,
                                      restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(best_path, monitor='val_last_accuracy', mode='max',
                                        save_best_only=True),
        keras.callbacks.CSVLogger(str(Path(args.out_dir) / f'{name}_history.csv')),
    ]
    model.fit(RandomLengthBatches(x_tr, y_tr, args.batch_size, seed=args.seed),
              epochs=args.epochs, callbacks=callbacks, verbose=2)

    model = keras.models.load_model(best_path)
    report = {
        'feature_type': args.feature_type,
        'params': int(model.count_params()),
        'val_accuracy_by_length': accuracy_by_length(model, x_val, y_val),
        'val_per_frame_accuracy': per_frame_accuracy(model, x_val, y_val),
    }
    if x_test is not None:
        report['test_accuracy_by_length'] = accuracy_by_length(model, x_test, y_test)
        report['test_per_frame_accuracy'] = per_frame_accuracy(model, x_test, y_test)
    with open(Path(args.out_dir) / f'{name}_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
