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