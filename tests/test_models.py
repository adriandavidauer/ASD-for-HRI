"""Tests for the model architectures."""

# 3rd party imports
import numpy as np
import pytest
from keras.layers import LSTM, Dense

# local imports
from asd4hri.models import buildFeatureLSTM

# end file header
__author__      = 'Adrian Auer'


def test_buildFeatureLSTM():
    """The landmark LSTM has the requested layers and maps a batch of samples to probabilities."""
    model, name = buildFeatureLSTM((38, 20, 2), num_lstm_layers=2, lstm_dims=8, num_dense_layers=2, dense_dims=16)
    assert name == 'FeatureLSTM(38, 20, 2)_2_8_2_16'
    assert sum(isinstance(layer, LSTM) for layer in model.layers) == 2
    assert [layer.units for layer in model.layers if isinstance(layer, Dense)] == [16, 16, 1]
    predictions = model.predict(np.random.default_rng(0).random((3, 38, 20, 2)), verbose=0)
    assert predictions.shape == (3, 1)
    assert np.all((predictions >= 0) & (predictions <= 1))


# ── causal landmark models (#45) ─────────────────────────────────────────────

def test_causal_feature_lstm_variable_length_and_causality():
    """Any input length works, and the output at frame t ignores frames after t."""
    from asd4hri.models import buildCausalFeatureLSTM
    model = buildCausalFeatureLSTM(20)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 12, 20, 2)).astype('float32')
    full = np.asarray(model(x))
    assert full.shape == (2, 12, 1)
    assert np.asarray(model(x[:, :1])).shape == (2, 1, 1)
    x2 = x.copy()
    x2[:, 6:] = rng.normal(size=x2[:, 6:].shape)        # change the future only
    np.testing.assert_allclose(np.asarray(model(x2))[:, :6], full[:, :6], atol=1e-5)


def test_streaming_matches_keras_model():
    """Stepping frame by frame in numpy gives the Keras outputs, per entity."""
    from asd4hri.models import buildCausalFeatureLSTM, StreamingFeatureLSTM
    model = buildCausalFeatureLSTM(20, lstm_dims=8, dense_dims=16)
    rng = np.random.default_rng(1)
    for layer in model.layers:                           # non-trivial BN statistics
        if layer.name == 'bn':
            g, b, m, v = layer.get_weights()
            layer.set_weights([g * 1.5, b + 0.1, m + 0.2, v * 2.0])
    x = rng.normal(size=(2, 9, 20, 2)).astype('float32')
    expected = np.asarray(model(x))[..., 0]
    stream = StreamingFeatureLSTM(model)
    got = np.stack([stream.step(x[:, t], ['a', 'b']) for t in range(9)], axis=1)
    np.testing.assert_allclose(got, expected, atol=1e-4)
    stream.forget(['a'])
    assert 'a' not in stream.states and 'b' in stream.states


def test_normalize_shape_frames_is_per_frame():
    from asd4hri.models import normalize_shape_frames
    x = np.random.default_rng(2).integers(0, 255, (5, 20, 2))
    out = normalize_shape_frames(x)
    np.testing.assert_allclose(out[3], normalize_shape_frames(x[3]), atol=1e-6)
    assert np.abs(out).max() == pytest.approx(1.0)
    np.testing.assert_allclose(out.mean(axis=1), 0, atol=1e-6)
