'''
Adjusted Version of ClassifyVVAD to use dlib and lib_shape or face_shape model.
'''

# System imports
import os
from pathlib import Path
import bz2
import errno
import os
import urllib.request

# 3rd party imports
import dlib

from paz.models.classification import VVAD_LRS3_LSTM, CNN2Plus1D
from paz.datasets import get_class_names
from paz.pipelines import PreprocessImage
from paz import processors as pr
from paz.abstract import Processor, SequentialProcessor

from keras.models import load_model, Sequential
from keras.layers import Dense, Input, LSTM, TimeDistributed, BatchNormalization, Flatten




import numpy as np
from tqdm import tqdm





# local imports

# end file header
__author__      = 'Adrian Auer'


Average_Options = ['mean', 'weighted']
Architecture_Options = ['VVAD-LRS3-LSTM', 'CNN2Plus1D', 'CNN2Plus1D_Filters', 'CNN2Plus1D_Layers',
                        'CNN2Plus1D_Light', 'LipShape', 'FaceShape']




class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(
            url, filename=output_path, reporthook=t.update_to)

def SHAPE_PREDICTOR_68_FACE_LANDMARKS():
    predictor_path = Path(__file__).absolute().parent.parent / "models" / \
        'shape_predictor_68_face_landmarks.dat'
    compressed_file = Path(predictor_path.parent /
                           (predictor_path.name + '.bz2'))
    if not predictor_path.exists():
        download_url('https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2',
                     compressed_file)

        with open(predictor_path, 'wb') as new_file, bz2.BZ2File(compressed_file, 'rb') as file:
            for data in iter(lambda: file.read(100 * 1024), b''):
                new_file.write(data)

        # remove compressed file
        compressed_file.unlink()

        if predictor_path.exists():
            return predictor_path
        else:  # This case should never happen! is only possible if file is deleted externally
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), predictor_path)
    else:
        return predictor_path
    

def buildFeatureLSTM(input_shape, num_lstm_layers=1, lstm_dims=32, num_dense_layers=1, dense_dims=512, **kwargs):
        """adjusted from https://github.com/adriandavidauer/VVAD/tree/main to rebuild model with weights in Keras 3 format."""
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

class ClassifyVVAD(SequentialProcessor):
    """Visual Voice Activity Detection pipeline for classifying speaking and not speaking from cropped RGB face
    video clips.

    # Arguments
        input_size: Tuple of integers. Input shape to the model in following format: (frames, height, width, channels)
            e.g. (38, 96, 96, 3).
        architecture: String. Name of the architecture to use. Currently supported: 'VVAD-LRS3-LSTM', 'CNN2Plus1D',
            'CNN2Plus1D_Filters', 'CNN2Plus1D_Layers' and 'CNN2Plus1D_Light'
        stride: Integer. How many frames are between the predictions (computational expansive (low update rate) vs
            high latency (high update rate))
        averaging_window_size: Integer. How many predictions are averaged. Set to 1 to disable averaging
        average_type: String. 'mean' or 'weighted'. How the predictions are averaged. Set average to 1 to
            disable averaging
    """
    def __init__(self, input_size=(38, 96, 96, 3), architecture='CNN2Plus1D_Light',
                 stride=38, averaging_window_size=2, average_type='mean'):
        super(ClassifyVVAD, self).__init__()
        assert average_type in Average_Options, f"'{average_type}' is not in {Average_Options}"
        assert architecture in Architecture_Options, f"'{architecture}' is not in {Architecture_Options}"

        if architecture == 'VVAD-LRS3-LSTM':
            self.classifier = VVAD_LRS3_LSTM(weights='VVAD_LRS3')
        elif architecture.startswith('CNN2Plus1D'):
            self.classifier = CNN2Plus1D(weights='VVAD_LRS3',
                                         architecture=str(architecture))
        elif architecture == 'LipShape':
            self.classifier = load_model(str(Path(__file__).absolute().parent.parent / "models" / 'lipFeatureModel.keras'))
            input_size = (38, 20, 2)
        elif architecture == 'FaceShape':
            self.classifier = load_model(str(Path(__file__).absolute().parent.parent / "models" / 'faceFeatureModel.keras'))
            input_size = (38, 68, 2)

        self.class_names = get_class_names('VVAD_LRS3')

        if 'Shape' in architecture:
            # empty preprocess for shape features
            preprocess = SequentialProcessor()
            preprocess.add(GetShapeFeatures())

        else:
            preprocess = PreprocessImage(input_size[1:3], (0.0, 0.0, 0.0))
        self.buffer_images = pr.BufferImages(input_size, stride=stride)
        preprocess.add(self.buffer_images)

        self.add(pr.PredictWithNones(self.classifier, preprocess))

        weighted_mean = (average_type == 'weighted')
        self.avg = pr.AveragePredictions(averaging_window_size, weighted_mean)
        self.add(pr.ControlMap(self.avg, [0], [0]))

        self.add(pr.ControlMap(pr.NoneConverter(), [0], [0]))
        self.add(pr.CopyDomain([0], [1]))
        self.add(pr.ControlMap(pr.FloatToBoolean(), [0], [0]))
        self.add(pr.ControlMap(pr.BooleanToTextMessage(true_message=self.class_names[0], false_message=self.class_names[1]), [0], [0]))
        self.add(pr.WrapOutput(['class_name', 'scores']))
    
    def reset(self):
        """Clear temporal state: clip buffer (BufferImages) and score window (AveragePredictions)."""
        # BufferImages
        self.buffer_images.frames_since_last_update = 0
        self.buffer_images.buffer_index = 0
        self.buffer_images.is_full = False
        if isinstance(self.buffer_images.buffer, np.ndarray):
            self.buffer_images.buffer[...] = 0

        # AveragePredictions
        self.avg.predictions.clear()

class GetShapeFeatures(Processor):
    """Processor to extract shape features from cropped RGB face using dlib's shape predictor."""
    def __init__(self, architecture='FaceShape', shape_predictor_path=None):
        super(GetShapeFeatures, self).__init__()
        if shape_predictor_path is None:
            shape_predictor_path = SHAPE_PREDICTOR_68_FACE_LANDMARKS()
        self.shape_predictor = dlib.shape_predictor(str(shape_predictor_path))
        self.architecture = architecture

    def call(self, image):
        shape = self.predictor(image, dlib.rectangle(
                0, 0, image.shape[1], image.shape[0]))
        if self.architecture == 'LipShape':
            # return only lip landmarks (48-67)
            return shape.parts()[48:68]
        else:
            return shape.parts()


if __name__ == '__main__':
    # load Processor for testing
    test_classiffier = ClassifyVVAD(architecture='LipShape')
    # TODO: run processor for testing