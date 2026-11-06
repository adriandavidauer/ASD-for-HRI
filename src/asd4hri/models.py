"""Model Architectures for VVAD."""

# System imports



# 3rd party imports

from keras.models import load_model, Sequential
from keras.layers import Dense, Input, LSTM, TimeDistributed, BatchNormalization, Flatten

# local imports


# end file header
__author__      = 'Adrian Auer'





def buildFeatureLSTM(input_shape, num_lstm_layers=1, lstm_dims=32, num_dense_layers=1, dense_dims=512, **kwargs):
        """Build and compile the LSTM classifier used for facial landmark (shape) features.

        Adjusted from https://github.com/adriandavidauer/VVAD/tree/main to rebuild the model with
        weights in Keras 3 format.

        Args:
            input_shape: Shape of one sample, ``(timesteps, num_landmarks, 2)``.
            num_lstm_layers: Number of stacked LSTM layers.
            lstm_dims: Number of units per LSTM layer.
            num_dense_layers: Number of ReLU dense layers after the LSTM layers.
            dense_dims: Number of units per dense layer.
            **kwargs: Ignored.

        Returns:
            tuple: ``(model, model_name)``, the compiled Keras ``Sequential`` model and a name
            encoding its hyperparameters.
        """
        model = Sequential()
        # handels input shape for Keras 3
        model.add(Input(shape=input_shape))        
        model.add(TimeDistributed(
            Flatten(input_shape=(input_shape[-2], input_shape[-1]))))
        if num_lstm_layers > 1:
            for i in range(num_lstm_layers - 1):
                # if not i:
                #     model.add(LSTM(lstm_dims, input_shape=input_shape, return_sequences=True))
                #     model.add(BatchNormalization())
                # else:
                model.add(LSTM(lstm_dims, return_sequences=True))
                model.add(BatchNormalization())

        # if model.layers:
        model.add(LSTM(lstm_dims))
        model.add(BatchNormalization())
        # else:
        #     model.add(LSTM(lstm_dims,input_shape=input_shape))
        #     model.add(BatchNormalization())

        # Add some more dense here
        for i in range(num_dense_layers):
            model.add(Dense(dense_dims, activation='relu'))

        model.add(Dense(1, activation="sigmoid"))
        model.compile(loss="binary_crossentropy",
                      optimizer='sgd',
                      metrics=["accuracy"])

        modelName = 'FeatureLSTM{}_'.format(input_shape) + str(num_lstm_layers) + '_' + str(
            lstm_dims) + '_' + str(num_dense_layers) + '_' + str(dense_dims)
        # model.build(input_shape)
        return model, modelName    

def buildCausalFeatureLSTM(num_landmarks, lstm_dims=32, dense_dims=512, **kwargs):
    """Build the causal (per-frame) landmark LSTM for variable-length input (#45).

    Same layers and size as ``buildFeatureLSTM`` with one LSTM and one dense layer, but the LSTM
    returns its output at every time step and the dense layers are applied per step. The output
    at frame *t* therefore uses only frames 0..t (zero look-ahead), and the input can have any
    length, from a single frame upwards.

    Args:
        num_landmarks: Number of landmarks per frame (20 for lips, 68 for the whole face).
        lstm_dims: Number of LSTM units.
        dense_dims: Number of units of the ReLU dense layer.
        **kwargs: Ignored.

    Returns:
        keras.Model: Input ``(batch, time, num_landmarks, 2)``, output ``(batch, time, 1)``
        speaking probabilities. Compiled with Adam and binary cross-entropy.
    """
    import keras
    from keras import layers

    inputs = keras.Input(shape=(None, num_landmarks, 2))
    x = layers.Reshape((-1, num_landmarks * 2))(inputs)
    x = layers.LSTM(lstm_dims, return_sequences=True, name='lstm')(x)
    x = layers.BatchNormalization(name='bn')(x)
    x = layers.Dense(dense_dims, activation='relu', name='dense')(x)
    outputs = layers.Dense(1, activation='sigmoid', name='out')(x)
    model = keras.Model(inputs, outputs, name=f'CausalFeatureLSTM_{num_landmarks}')
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(1e-3),
                  metrics=['accuracy'])
    return model


def normalize_shape_frames(landmarks):
    """Normalize landmarks frame by frame, using no information from other frames (#45).

    Each frame is centred on the mean of its landmarks and scaled by its largest absolute
    coordinate, so it lies in [-1, 1]. Unlike ``NormalizeShapeSample``, which uses the first
    frame and the maximum over the whole sample, this is causal and independent of the length.

    Args:
        landmarks: Array of shape ``(..., num_landmarks, 2)``.

    Returns:
        numpy.ndarray: Float32 array of the same shape.
    """
    import numpy as np
    x = np.asarray(landmarks, dtype=np.float32)
    x = x - x.mean(axis=-2, keepdims=True)
    scale = np.abs(x).max(axis=(-2, -1), keepdims=True)
    return x / np.maximum(scale, 1e-6)


class StreamingFeatureLSTM:
    """Run a ``buildCausalFeatureLSTM`` model one frame at a time, in numpy (#45).

    Keeps one LSTM state per entity, so each new frame costs one LSTM step regardless of how
    many frames came before. Gives the same outputs as running the Keras model on the whole
    sequence.

    Args:
        model: A trained ``buildCausalFeatureLSTM`` Keras model.
    """

    def __init__(self, model):
        """Copy the weights out of the Keras model."""
        import numpy as np
        lstm = model.get_layer('lstm')
        self.kernel, self.recurrent, self.bias = (np.asarray(w) for w in lstm.get_weights())
        self.units = self.recurrent.shape[0]
        gamma, beta, mean, var = model.get_layer('bn').get_weights()
        eps = model.get_layer('bn').epsilon
        self.bn_scale = gamma / np.sqrt(var + eps)
        self.bn_shift = beta - mean * self.bn_scale
        self.w1, self.b1 = model.get_layer('dense').get_weights()
        self.w2, self.b2 = model.get_layer('out').get_weights()
        self.states = {}

    @staticmethod
    def _sigmoid(x):
        import numpy as np
        return 1.0 / (1.0 + np.exp(-x))

    def step(self, frames, ids):
        """Advance the given entities by one frame.

        Args:
            frames: Array ``(n, num_landmarks, 2)`` of normalized landmarks, one per entity.
            ids: Entity ids, one per frame.

        Returns:
            numpy.ndarray: ``(n,)`` speaking probabilities for these frames.
        """
        import numpy as np
        x = np.asarray(frames, dtype=np.float32).reshape(len(ids), -1)
        u = self.units
        h = np.stack([self.states.get(i, (np.zeros(u, np.float32),) * 2)[0] for i in ids])
        c = np.stack([self.states.get(i, (np.zeros(u, np.float32),) * 2)[1] for i in ids])
        z = x @ self.kernel + h @ self.recurrent + self.bias      # Keras gate order: i, f, c, o
        i_g = self._sigmoid(z[:, :u])
        f_g = self._sigmoid(z[:, u:2 * u])
        c = f_g * c + i_g * np.tanh(z[:, 2 * u:3 * u])
        h = self._sigmoid(z[:, 3 * u:]) * np.tanh(c)
        for k, key in enumerate(ids):
            self.states[key] = (h[k], c[k])
        y = np.maximum(0.0, (h * self.bn_scale + self.bn_shift) @ self.w1 + self.b1)
        return self._sigmoid(y @ self.w2 + self.b2).reshape(-1)

    def forget(self, ids):
        """Drop the state of the given entities (e.g. faces that left the frame)."""
        for key in ids:
            self.states.pop(key, None)

    def reset(self):
        """Drop all states."""
        self.states = {}
