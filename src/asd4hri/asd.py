"""Adjusted Version of ClassifyVVAD to use dlib and lib_shape or face_shape model."""

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
from paz.backend.image import show_image
from tensorflow.keras.utils import get_file






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

FaceDetector_Options = ['YuNet', 'HaarCascade', 'RetinaFace', 'InsightFace', 'YOLOv10-Face']


    

def load_vvad_classifier(architecture):
    """Load the Keras classifier model for the given architecture.

    The ``LipShape`` and ``FaceShape`` weights are downloaded to ``src/models`` on first use.

    Args:
        architecture: Name of the architecture to load. One of ``Architecture_Options``:
            'VVAD-LRS3-LSTM', 'CNN2Plus1D', 'CNN2Plus1D_Filters', 'CNN2Plus1D_Layers',
            'CNN2Plus1D_Light', 'LipShape' or 'FaceShape'.

    Returns:
        tuple: ``(model, input_shape)`` where ``model`` is the Keras model and ``input_shape`` is the
        shape of one sample expected by the model, e.g. ``(38, 96, 96, 3)`` for the image models,
        ``(38, 20, 2)`` for 'LipShape' and ``(38, 68, 2)`` for 'FaceShape'.

    Raises:
        ValueError: If ``architecture`` is not supported.
    """
    if architecture == 'VVAD-LRS3-LSTM':
        return (VVAD_LRS3_LSTM(weights='VVAD_LRS3'), (38, 96, 96, 3))
    elif architecture.startswith('CNN2Plus1D'):
        return (CNN2Plus1D(weights='VVAD_LRS3', architecture=str(architecture)), (38, 96, 96, 3))
    elif architecture == 'LipShape':
        filename = 'paz_LipShape_0.8958.keras'
        get_file(filename, URL + filename, cache_subdir=str(Path(__file__).absolute().parent.parent / "models" ))
        return (load_model(str(Path(__file__).absolute().parent.parent / "models" / 'paz_LipShape_0.8958.keras')), (38, 20, 2))
    elif architecture == 'FaceShape':
        filename = 'faceFeatureModel.keras'
        get_file(filename, URL + filename, cache_subdir=str(Path(__file__).absolute().parent.parent / "models" ))
        return (load_model(str(Path(__file__).absolute().parent.parent / "models" / 'faceFeatureModel.keras')), (38, 68, 2))
    else:
        raise ValueError(f"Unsupported architecture: {architecture}. Supported architectures are: {Architecture_Options}")


class ClassifyVVAD(SequentialProcessor):
    """Visual Voice Activity Detection pipeline for cropped face video clips.

    Classifies speaking vs. not speaking. Expects a batch of face images per call, each image
    representing a different entity (the order must be stable across calls). Images, or the shape
    features extracted from them, are buffered until a full sample for the model is collected.
    Returns a batch of (averaged) predictions, each corresponding to the entity at the same index
    of the input batch, or ``None`` where no prediction is available yet.

    Args:
        architecture: Name of the architecture to use. One of ``Architecture_Options``:
            'VVAD-LRS3-LSTM', 'CNN2Plus1D', 'CNN2Plus1D_Filters', 'CNN2Plus1D_Layers',
            'CNN2Plus1D_Light', 'LipShape' or 'FaceShape'.
        stride: Number of frames between two predictions. A low stride is computationally
            expensive (high update rate), a high stride adds latency (low update rate).
        averaging_window_size: Number of consecutive predictions that are averaged.
            Set to 1 to disable averaging.
        weighted: If True, the predictions in the averaging window are weighted by their
            position (index + 1), so newer predictions count more. If False, a plain average
            is used.
        max_consecutive_empty: Number of consecutive missing face images after which the
            buffer of an entity is cleared.
    """
    def __init__(self, architecture='CNN2Plus1D_Light',
                 stride=38, averaging_window_size=2, weighted=False, max_consecutive_empty=5):
        """Load the classifier and assemble the preprocessing, buffering and averaging steps.

        Asserts that ``architecture`` is in ``Architecture_Options``.
        """
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
            # PreprocessImage is set to not normalize the image to [0, 1] because the model is trained on images in [0, 255]
            preprocess.add(PreprocessImages(input_size[1:3], normalize=False)) # works on batch of images not on batch of samples - needs to be done before buffering into a sample 
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
        """Clear the temporal state: the feature buffers and the prediction averaging window."""
        # Clear Buffer
        self.buffer_features.reset()

        # AveragePredictions
        self.buffer_predictions.reset()




class ASD(Processor):
    """Active Speaker Detection pipeline.

    Detects faces in each RGB frame, crops them and classifies every face as speaking or not
    speaking with ``ClassifyVVAD``. Multiple consecutive frames are needed before the first
    prediction is produced; until then boxes are labelled 'No Prediction yet' with score -1.

    Example:
        ```python
        pipeline = ASD(architecture='LipShape')
        camera = Camera(0)
        player = VideoPlayer((640, 480), pipeline, camera)
        player.run()
        ```

    Args:
        architecture: Name of the architecture to use. One of ``Architecture_Options``:
            'VVAD-LRS3-LSTM', 'CNN2Plus1D', 'CNN2Plus1D_Filters', 'CNN2Plus1D_Layers',
            'CNN2Plus1D_Light', 'LipShape' or 'FaceShape'.
        stride: Number of frames between two predictions. A low stride is computationally
            expensive, a high stride adds latency.
        averaging_window_size: Number of consecutive predictions that are averaged.
            Set to 1 to disable averaging.
        decision_threshold: Score above which a face is classified as speaking.
        weighted: If True, the averaging window is weighted by the position of each prediction,
            so newer predictions count more.
        max_consecutive_empty: Number of frames a face may be missing before its buffer is
            cleared. A low number can cause short samples or missing predictions for models with
            a fixed sample length; a high number can cause glitches in samples.
        annotate_output: If True, the bounding box, score and predicted label are drawn on the
            returned image.
        detector: Face detector to use. One of ``FaceDetector_Options``. Only 'YuNet',
            'HaarCascade' and 'RetinaFace' are implemented.
        input_size: ``(width, height)`` of the input images, used by the YuNet detector.
            If None, it is set at runtime, which can slow the pipeline down slightly.
        face_tracking_approach: How faces are kept in a stable order across frames. One of
            ``FaceTracking_Options``; see ``FaceTracker``.
    """
    def __init__(self, architecture='CNN2Plus1D_Light', stride=2, averaging_window_size=3, decision_threshold=0.5,
                 weighted= True, max_consecutive_empty=2, annotate_output=False, detector='YuNet', input_size=None, face_tracking_approach='sort'):
        """Build the face detection, cropping and classification steps.

        Raises:
            NotImplementedError: If ``detector`` is 'InsightFace' or 'YOLOv10-Face'.
            ValueError: If ``detector`` is unknown.
        """
        super(ASD, self).__init__()
        self.annotate_output = annotate_output
        self.offsets = [0,0]
        self.colors = [[0, 255, 0], [255, 0, 0], [0, 0, 0]]
        self.absent_counts = []
        self.face_tracking_approach = face_tracking_approach
        if self.face_tracking_approach not in FaceTracking_Options:
            raise ValueError(f"Face tracking approach must be one of {FaceTracking_Options} but is {self.face_tracking_approach}")
        self.face_tracker = FaceTracker(approach=self.face_tracking_approach)
        #detection
        self.copy = pr.Copy()
        if detector == 'HaarCascade':
            self.detect = dt.HaarCascadeFrontalFace()
        elif detector == 'YuNet':
            self.detect = FaceDetectorYN(input_size=input_size) # TODO: parameter forwarding with kwargs?
        elif detector == 'RetinaFace':
            self.detect = FaceDetectorRetinaFace() # TODO: parameter forwarding with kwargs?
        elif detector ==  'InsightFace':
            raise NotImplementedError
        elif detector == 'YOLOv10-Face':
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown detector {detector}. Must be one of {FaceDetector_Options}")
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
        self.class_names = list(get_class_names('VVAD_LRS3'))
        corrected_class_names = [self.class_names[1], self.class_names[0]] # in PAZ the order is speaking, not-speaking but we need it the other way around.
        self.classifier = AddClassAndScoreToBoxes(ClassifyVVAD(**self.vvad_args), class_names=corrected_class_names, decision_threshold=decision_threshold)
        

        self.class_names.append('No Prediction yet')

        self.draw = pr.DrawBoxes2D(self.class_names, self.colors, True)
        self.wrap = pr.WrapOutput(['image', 'boxes2D'])

    def call(self, image):
        """Detect faces in one frame and classify each of them as speaking or not speaking.

        Args:
            image: RGB image as a numpy array of shape ``(height, width, 3)``.

        Returns:
            dict: ``{'image': image, 'boxes2D': boxes2D}``. ``boxes2D`` is a list of
            ``paz.abstract.messages.Box2D`` whose ``class_name`` and ``score`` hold the prediction.
            The image has the boxes drawn on it if ``annotate_output`` is True.
        """
        # get the face boxes and crop the faces from the image
        image_copy = self.copy(image)
        boxes2D = self.detect(image_copy)['boxes2D']
        boxes2D = self.face_tracker(boxes2D)
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
    """Deprecated version of ``ASD``. Use ``ASD`` instead."""
    def __init__(self, architecture='CNN2Plus1D_Light', stride=2, averaging_window_size=3,
                 average_type='weighted', offsets=[0,0], colors=[[0, 255, 0], [255, 0, 0]], min_frames=38, patience=5):
        """Fail on construction because the class is deprecated.

        Raises:
            DeprecationWarning: Always; use ``ASD`` instead.
        """
        super(DetectVVAD, self).__init__(architecture=architecture, stride=stride,
                                         averaging_window_size=averaging_window_size,
                                         average_type=average_type, offsets=offsets,
                                         colors=colors, max_consecutive_empty=patience)
        raise DeprecationWarning("DetectVVAD is deprecated. Use ASD instead.")
if __name__ == '__main__':
    # load Processor for testing
    #test_classiffier = ClassifyVVAD(architecture='LipShape')
    # run processor for testing
    pipeline = ASD(architecture='LipShape', annotate_output=True, detector='YuNet')
    camera = Camera(0)
    player = VideoPlayer((640, 480), pipeline, camera)
    player.run()
