"""Tests for the model architectures."""

# 3rd party imports
import numpy as np
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
