"""Tests for the AVA fine-tuning of the causal landmark models (#92)."""

import numpy as np

from asd4hri.models import buildCausalFeatureLSTM, normalize_shape_frames
from src.model_training.finetune_causal_shape_ava import (TrackCrops, load_tracks,
                                                          predict_tracks, split_videos)


def _npz(path, rng):
    """Extraction output with two entities; entity b has a gap of 10 frames.

    Args:
        path: File to write.
        rng: Random generator.
    """
    frames = list(range(20)) + list(range(5)) + list(range(15, 30))
    entity = ['a'] * 20 + ['b'] * 5 + ['b'] * 15
    n = len(frames)
    np.savez(path, entity=np.array(entity), frame=np.array(frames, np.int32),
             timestamp=np.array(frames) / 25.0,
             landmarks=rng.integers(0, 100, (n, 68, 2)).astype(np.int16),
             label=(np.arange(n) % 3 == 0).astype(np.uint8), size=np.full(n, 80, np.int16),
             fps=np.float64(25))


def test_load_tracks_splits_entities_at_gaps(tmp_path):
    """One track per entity and run of frames; lips are points 48-67, frame-normalised."""
    _npz(tmp_path / 'v.npz', np.random.default_rng(0))
    tracks = load_tracks([tmp_path / 'v.npz'], 'LipShape')
    assert sorted(len(y) for _, y in tracks) == [5, 15, 20]
    d = np.load(tmp_path / 'v.npz')
    a = [t for t in tracks if len(t[1]) == 20][0]
    np.testing.assert_allclose(a[0], normalize_shape_frames(d['landmarks'][:20, 48:68]))
    np.testing.assert_array_equal(a[1], d['label'][:20])
    assert load_tracks([tmp_path / 'v.npz'], 'FaceShape')[0][0].shape[1:] == (68, 2)


def test_split_videos_is_fixed_and_disjoint():
    """The held-out videos are about 10%, the same on every call, and not in train."""
    files = [f'v{i}.npz' for i in range(200)]
    train, held = split_videos(files)
    assert set(train).isdisjoint(held) and len(train) + len(held) == 200
    assert 8 <= len(held) <= 35
    assert split_videos(files) == (train, held)


def test_track_crops_keep_frames_and_labels_together():
    """Every crop is a contiguous piece of one track, with that piece's labels."""
    tracks = []
    for n in (3, 40, 100):
        x = np.arange(n, dtype=np.float32)[:, None, None] * np.ones((1, 20, 2), np.float32)
        tracks.append((x + 1000 * n, (np.arange(n) + 1000 * n).astype(np.float32)))
    crops = TrackCrops(tracks, batch_size=16, max_len=64, seed=0)
    seen = set()
    for _ in range(50):
        x, y = crops.batch()
        assert x.shape[:2] == y.shape[:2] and y.shape[-1] == 1 and x.shape[1] <= 64
        seen.add(x.shape[1])
        np.testing.assert_array_equal(x[:, :, 0, 0], y[..., 0])
        assert np.all(np.diff(y[..., 0], axis=1) == 1)
    assert min(seen) < 5 and max(seen) > 20


def test_predict_tracks_equals_one_track_at_a_time():
    """Padding at the end does not change the causal scores."""
    model = buildCausalFeatureLSTM(20)
    rng = np.random.default_rng(0)
    tracks = [(rng.normal(size=(n, 20, 2)).astype(np.float32), np.ones(n, np.float32))
              for n in (7, 1, 30, 12)]
    scores, labels = predict_tracks(model, tracks, batch_frames=40)
    order = np.argsort([7, 1, 30, 12])
    expected = np.concatenate([model.predict(tracks[i][0][None], verbose=0)[0, :, 0]
                               for i in order])
    np.testing.assert_allclose(scores, expected, atol=1e-5)
    assert len(labels) == 50
