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
from collections import deque


# 3rd party imports
import dlib

from paz.models.classification import VVAD_LRS3_LSTM, CNN2Plus1D
from paz.datasets import get_class_names
from paz.pipelines import PreprocessImage
from paz import processors as pr
from paz.abstract import Processor, SequentialProcessor
from paz.backend.camera import VideoPlayer, Camera
import paz.pipelines.detection as dt





from keras.models import load_model, Sequential
from keras.layers import Dense, Input, LSTM, TimeDistributed, BatchNormalization, Flatten




import numpy as np
from tqdm import tqdm





# local imports
from .processors import *

# end file header
__author__      = 'Adrian Auer'


Architecture_Options = ['VVAD-LRS3-LSTM', 'CNN2Plus1D', 'CNN2Plus1D_Filters', 'CNN2Plus1D_Layers',
                        'CNN2Plus1D_Light', 'LipShape', 'FaceShape']




    

def load_vvad_classifier(architecture):
    """Load the Keras classifier model for the given architecture.
    
    # Arguments
        architecture: String. Name of the architecture to use. Currently supported: 'VVAD-LRS3-LSTM', 'CNN2Plus1D', 'CNN2Plus1D_Filters', 'CNN2Plus1D_Layers', 'CNN2Plus1D_Light', 'LipShape' and 'FaceShape'
    
    # Returns
        tuple: (Keras model for the given architecture, input shape of the model)
    """
    if architecture == 'VVAD-LRS3-LSTM':
        return (VVAD_LRS3_LSTM(weights='VVAD_LRS3'), (38, 96, 96, 3))
    elif architecture.startswith('CNN2Plus1D'):
        return (CNN2Plus1D(weights='VVAD_LRS3', architecture=str(architecture)), (38, 96, 96, 3))
    elif architecture == 'LipShape':
        return (load_model(str(Path(__file__).absolute().parent.parent / "models" / 'paz_LipShape_0.8958.keras')), (38, 20, 2))
    elif architecture == 'FaceShape':
        return (load_model(str(Path(__file__).absolute().parent.parent / "models" / 'faceFeatureModel.keras')), (38, 68, 2))
    else:
        raise ValueError(f"Unsupported architecture: {architecture}. Supported architectures are: {Architecture_Options}")


class ClassifyVVAD(SequentialProcessor):
    """Visual Voice Activity Detection pipeline for classifying speaking and not speaking from cropped RGB face
    video clips.
    Expects a batch of face images - each image representig a different entity. 
    Images will be buffered until the inputsize of the model.
    Returns a batch of (averaged) predictions - each prediction corresponding to the entity in the input image at the same index.

    # Arguments
        input_size: Tuple of integers. Input shape to the model in following format: (frames, height, width, channels)
            e.g. (38, 96, 96, 3).
        architecture: String. Name of the architecture to use. Currently supported: 'VVAD-LRS3-LSTM', 'CNN2Plus1D',
            'CNN2Plus1D_Filters', 'CNN2Plus1D_Layers', 'CNN2Plus1D_Light', 'LipShape' and 'FaceShape'
        stride: Integer. How many frames are between the predictions (computational expansive (low update rate) vs
            high latency (high update rate))
        averaging_window_size: Integer. How many predictions are averaged. Set to 1 to disable averaging
        weigthed: Boolean. 
                If True weigthed by the index+1 (otherwise the first entry will always be zero and single values will be removed) of the value.
                If False just the average over all entries
    """
    def __init__(self, architecture='CNN2Plus1D_Light',
                 stride=38, averaging_window_size=2, weighted=False, max_consecutive_empty=5):
        super(ClassifyVVAD, self).__init__()
        assert architecture in Architecture_Options, f"'{architecture}' is not in {Architecture_Options}"

        classifier, input_size = load_vvad_classifier(architecture)
        # Dummy predict to fully initialize 
        classifier(np.array([np.empty(input_size)]))

        self.class_names = get_class_names('VVAD_LRS3')

        preprocess = SequentialProcessor()

        if 'Shape' in architecture:            
            preprocess.add(GetShapeFeatures(architecture=architecture)) # works on batch of face images not on batch of samples - needs to be done before buffering into a sample
        else:
            preprocess.add(PreprocessImages(input_size[1:3])) # works on batch of images not on batch of samples - needs to be done before buffering into a sample 
        self.buffer_features = BufferFeatures(input_size, stride=stride, max_consecutive_empty=max_consecutive_empty)
        # We buffer the incoming face images or features
        preprocess.add(self.buffer_features)


        if 'Shape' in architecture:            
            preprocess.add(NormalizeShapeSample()) # works on batch of samples

        


        self.add(PredictWithNones(model=classifier, preprocess=preprocess, postprocess=flatten_predictions))

        #  buffer predictions with stride 1 and return_incomplete_samples so so that it returns each time something comes in and max_consecutive_empty is set to stride so that if the model gives no predictions for a stride steps this will not cause the buffer to be reset
        self.buffer_predictions = BufferFeatures((averaging_window_size,), stride=1, max_consecutive_empty=stride, return_incomplete_samples=True) 
        self.add(self.buffer_predictions)

        self.avg = AveragePredictions(weighted=weighted, normalize=True) # normalize to [0, 1] so that the output is always in [0, 1] independent of the averaging_window_size
        self.add(self.avg)
        # is controlMap hindering Batch predictions? Looks like it is only taking the first input and mapping it to the first output.
        # self.add(pr.ControlMap(self.avg, [0], [0]))
        # why convert nones to 0, makes no sese because the output of the model could be zero as well
        # self.add(pr.ControlMap(pr.NoneConverter(), [0], [0]))
        # dont get it
        # self.add(pr.CopyDomain([0], [1]))
        # self.add(pr.ControlMap(pr.FloatToBoolean(), [0], [0]))
        # self.add(pr.ControlMap(pr.BooleanToTextMessage(true_message=self.class_names[0], false_message=self.class_names[1]), [0], [0]))
        # self.add(pr.WrapOutput(['class_name', 'scores']))
    
    def reset(self):
        """Clear temporal state: clip buffer (BufferImages) and score window (AveragePredictions)."""
        # Clear Buffer
        self.buffer_features.reset()

        # AveragePredictions
        self.buffer_predictions.reset()




class ASD(Processor):
    """Active Speaker Detection pipeline.

    # Example
        ``` python
            pipeline = ASD(architecture='LipShape')
            camera = Camera(0)
            player = VideoPlayer((640, 480), pipeline, camera)
            player.run()
        ```

    # Returns
        Dictionary with ``image`` and ``boxes2D``.

    # Returns
        A function that takes an RGB image and outputs the predictions
        as a dictionary with ``keys``: ``image`` and ``boxes2D``.
        The corresponding values of these keys contain the image with the drawn
        inferences and a list of ``paz.abstract.messages.Boxes2D``.
        Note multiple images are needed to produce a prediction.

    # Arguments
        architecture: String. Name of the architecture to use. Currently supported: 'VVAD-LRS3-LSTM', 'CNN2Plus1D',
            'CNN2Plus1D_Filters' and 'CNN2Plus1D_Light', 'LipShape' and 'FaceShape'
        stride: Integer. How many frames are between the predictions (computational expansive (low stride) vs
            high latency (high stride))
        averaging_window_size: Integer. How many predictions are averaged. Set to 1 to disable averaging
        average_type: String. 'mean' or 'weighted'. How the predictions are averaged. Set averaging_window_size to 1 to
            disable averaging
    """

    def __init__(self, architecture='CNN2Plus1D_Light', stride=2, averaging_window_size=3, decision_threshold=0.5,
                 weighted= True, max_consecutive_empty=2, annotate_output=False):
        super(ASD, self).__init__()
        self.annotate_output = annotate_output
        self.offsets = [0,0]
        self.colors = [[0, 255, 0], [255, 0, 0], [0, 0, 0]]
        self.absent_counts = []

        #detection
        self.copy = pr.Copy()
        self.detect = dt.HaarCascadeFrontalFace()
        self.square = SequentialProcessor()
        self.square.add(pr.SquareBoxes2D())
        self.square.add(pr.OffsetBoxes2D(self.offsets))
        self.clip = pr.ClipBoxes2D()
        self.crop = pr.CropBoxes2D()

        
        self.vvad_args = dict(
            stride=stride,
            averaging_window_size=averaging_window_size,
            weighted=weighted,
            architecture=architecture, max_consecutive_empty=max_consecutive_empty
        )
        class_names = get_class_names('VVAD_LRS3')
        corrected_class_names = [class_names[1], class_names[0]] # in PAZ the order is speaking, not-speaking but we need it the other way around.
        self.classifier = AddClassAndScoreToBoxes(ClassifyVVAD(**self.vvad_args), class_names=corrected_class_names, decision_threshold=decision_threshold)
        

        self.class_names = list(get_class_names('VVAD_LRS3'))
        self.class_names.append('No Prediction yet')

        self.draw = pr.DrawBoxes2D(self.class_names, self.colors, True)
        self.wrap = pr.WrapOutput(['image', 'boxes2D'])

    def call(self, image):
        # get the face boxes and crop the faces from the image
        image_copy = self.copy(image)
        boxes2D = self.detect(image_copy)['boxes2D']
        boxes2D = self.square(boxes2D)
        boxes2D = self.clip(image, boxes2D)
        cropped_images = self.crop(image, boxes2D)

        # call classifyVVAD for the whole batch of faces
        boxes2D = self.classifier(cropped_images, boxes2D)

        # only if flag is set
        if self.annotate_output:
            image = self.draw(image, boxes2D)
        return self.wrap(image, boxes2D)

class DetectVVAD(ASD):
    """Deprecated Version of ASD.
    """
    def __init__(self, architecture='CNN2Plus1D_Light', stride=2, averaging_window_size=3,
                 average_type='weighted', offsets=[0,0], colors=[[0, 255, 0], [255, 0, 0]], min_frames=38, patience=5):
        super(DetectVVAD, self).__init__(architecture=architecture, stride=stride,
                                         averaging_window_size=averaging_window_size,
                                         average_type=average_type, offsets=offsets,
                                         colors=colors, max_consecutive_empty=patience)
        raise DeprecationWarning("DetectVVAD is deprecated. Use ASD instead.")
if __name__ == '__main__':
    # load Processor for testing
    #test_classiffier = ClassifyVVAD(architecture='LipShape')
    # run processor for testing
    pipeline = ASD(architecture='LipShape', annotate_output=True)
    camera = Camera(0)
    player = VideoPlayer((640, 480), pipeline, camera)
    player.run()
