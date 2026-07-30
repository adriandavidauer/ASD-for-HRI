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
    """Load the Keras classifier model for the given architecture."""
    assert architecture in Architecture_Options, f"'{architecture}' is not in {Architecture_Options}"

    if architecture == 'VVAD-LRS3-LSTM':
        return VVAD_LRS3_LSTM(weights='VVAD_LRS3')
    elif architecture.startswith('CNN2Plus1D'):
        return CNN2Plus1D(weights='VVAD_LRS3', architecture=str(architecture))
    elif architecture == 'LipShape':
        return load_model(str(Path(__file__).absolute().parent.parent / "models" / 'paz_LipShape_0.8958.keras'))
    elif architecture == 'FaceShape':
        return load_model(str(Path(__file__).absolute().parent.parent / "models" / 'faceFeatureModel.keras'))


class ClassifyVVAD(SequentialProcessor):
    """Visual Voice Activity Detection pipeline for classifying speaking and not speaking from cropped RGB face
    video clips.
    Expects a batch of face images - each image representig a different entity. 
    Images will be buffered until the inputsize of the model.

    # Arguments
        input_size: Tuple of integers. Input shape to the model in following format: (frames, height, width, channels)
            e.g. (38, 96, 96, 3).
        architecture: String. Name of the architecture to use. Currently supported: 'VVAD-LRS3-LSTM', 'CNN2Plus1D',
            'CNN2Plus1D_Filters', 'CNN2Plus1D_Layers', 'CNN2Plus1D_Light', 'LipShape' and 'FaceShape'
        stride: Integer. How many frames are between the predictions (computational expansive (low update rate) vs
            high latency (high update rate))
        averaging_window_size: Integer. How many predictions are averaged. Set to 1 to disable averaging
        average_type: String. 'mean' or 'weighted'. How the predictions are averaged. Set averaging_window_size to 1 to
            disable averaging
    """
    def __init__(self, input_size=(38, 96, 96, 3), architecture='CNN2Plus1D_Light',
                 stride=38, averaging_window_size=2, average_type='mean', max_consecutive_empty=5):
        super(ClassifyVVAD, self).__init__()
        assert average_type in Average_Options, f"'{average_type}' is not in {Average_Options}"
        assert architecture in Architecture_Options, f"'{architecture}' is not in {Architecture_Options}"

        classifier = load_vvad_classifier(architecture)

        if architecture == 'LipShape':
            input_size = (38, 20, 2)
        elif architecture == 'FaceShape':
            input_size = (38, 68, 2)

        self.class_names = get_class_names('VVAD_LRS3')

        preprocess = SequentialProcessor()

        if 'Shape' in architecture:            
            preprocess.add(GetShapeFeatures(architecture=architecture)) # works on batch of face images not on batch of samples - needs to be done before buffering into a sample
            
        self.buffer_features = BufferFeatures(input_size, stride=stride, max_consecutive_empty=max_consecutive_empty)
        # We buffer the incoming face images or features
        preprocess.add(self.buffer_features)


        if 'Shape' in architecture:            
            preprocess.add(NormalizeShapeSample()) # works on batch of samples

        else:
            preprocess.add(PreprocessImage(input_size[1:3], (0.0, 0.0, 0.0)))


        self.add(PredictWithNones(classifier, preprocess))

        #  buffer predictions with stride 1 and return_incomplete_samples so so that it returns each time something comes in
        self.buffer_predictions = BufferFeatures((averaging_window_size,), stride=1, max_consecutive_empty=max_consecutive_empty, return_incomplete_samples=True)
        self.add(self.buffer_predictions)

        self.avg = AveragePredictions(weighted=True)
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
        self.avg.predictions.clear()




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

    def __init__(self, architecture='CNN2Plus1D_Light', stride=2, averaging_window_size=3,
                 average_type='weighted', offsets=[0,0], colors=[[0, 255, 0], [255, 0, 0]], min_frames=38, patience=5):
        super(ASD, self).__init__()
        self.offsets = offsets
        self.colors = colors
        self.min_frames = int(min_frames)
        self.patience = int(patience)
        self.absent_counts = []
        
        #detection
        self.copy = pr.Copy()
        self.detect = dt.HaarCascadeFrontalFace()
        self.square = SequentialProcessor()
        self.square.add(pr.SquareBoxes2D())
        self.square.add(pr.OffsetBoxes2D(offsets))
        self.clip = pr.ClipBoxes2D()
        self.crop = pr.CropBoxes2D()

        
        self.vvad_args = dict(
            stride=stride,
            averaging_window_size=averaging_window_size,
            average_type=str(average_type),
            architecture=architecture
        )
        self.classifier = pr.AddClassAndScoreToBoxes(ClassifyVVAD(**self.vvad_args))
        # TODO: Dummy predict to fully initialize in classify VVAD
        

        self.class_names = list(get_class_names('VVAD_LRS3'))

        # TODO: only draw if enabled
        self.draw = pr.DrawBoxes2D(self.class_names, self.colors, True)
        self.wrap = pr.WrapOutput(['image', 'boxes2D'])

    def call(self, image):
        image_copy = self.copy(image)
        boxes2D = self.detect(image_copy)['boxes2D']
        boxes2D = self.square(boxes2D)
        boxes2D = self.clip(image, boxes2D)
        cropped_images = self.crop(image, boxes2D)

        N = len(cropped_images)

        # TODO: add each face in the corresponding buffer
        # TODO: if we do not have enough buffers create new ones
        # TODO: get all full buffers and batch predict with self.classifier([crop], [box])

        # one classifier for all buffers
        # is pr.AddClassAndScoreToBoxes batch safe?

        while len(self.adders) < N:
            self.adders.append(pr.AddClassAndScoreToBoxes(clf))
            self.frame_counts.append(0)
            self.miss_counts.append(0)
            self.absent_counts.append(0)
            # TODO: what is the difference between absence counts and miss counts?

        # Increment counters for the first N slots (faces we actually saw this frame)
        for i in range(N):
            self.frame_counts[i] += 1
            self.miss_counts[i] = 0
            self.absent_counts[i] = 0

        # Reset counters
        for i in range(N, len(self.adders)):
            self.miss_counts[i] += 1
            self.absent_counts[i] += 1
            if self.miss_counts[i] > self.patience:
                # clear counter and clear the VVAD temporal buffer
                self.frame_counts[i] = 0
                self.classifiers[i].reset() 
                self.miss_counts[i] = 0
        # Drop dangling tail slots that have been absent long enough
        while len(self.adders) > N and self.absent_counts[-1] >= self.min_frames:
            self.adders.pop()
            self.classifiers.pop()
            self.frame_counts.pop()
            self.miss_counts.pop()
            self.absent_counts.pop()

        # Classify and update only the slots that have matured enough frames
        updated_boxes = []
        for i, (adder, crop, box) in enumerate(zip(self.adders, cropped_images, boxes2D)):
            updated = adder([crop], [box])[0] 
            if self.frame_counts[i] >= self.min_frames:
                updated_boxes.append(updated)

        boxes2D = updated_boxes
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
                                         colors=colors, min_frames=min_frames, patience=patience)
        raise DeprecationWarning("DetectVVAD is deprecated. Use ASD instead.")
if __name__ == '__main__':
    # load Processor for testing
    #test_classiffier = ClassifyVVAD(architecture='LipShape')
    # run processor for testing
    pipeline = ASD(architecture='LipShape')
    camera = Camera(0)
    player = VideoPlayer((640, 480), pipeline, camera)
    player.run()