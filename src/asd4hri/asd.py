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
from .timing import StageTimer, NULL_TIMER
from .av import AV_PREFIX, AVClassifier, av_architectures
from .models import StreamingFeatureLSTM, normalize_shape_frames

# end file header
__author__      = 'Adrian Auer'


Architecture_Options = ['VVAD-LRS3-LSTM', 'CNN2Plus1D', 'CNN2Plus1D_Filters', 'CNN2Plus1D_Layers',
                        'CNN2Plus1D_Light', 'LipShape', 'FaceShape',
                        'LipShapeCausal', 'FaceShapeCausal']

# Causal landmark models (#45): one prediction per frame from that frame and the frames before
Causal_Architectures = ['LipShapeCausal', 'FaceShapeCausal']
Causal_Modes = ['streaming', 'window']

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
    elif architecture in Causal_Architectures:
        filename = f'{architecture}.keras'
        get_file(filename, URL + filename, cache_subdir=str(Path(__file__).absolute().parent.parent / "models" ))
        num_landmarks = 20 if architecture.startswith('Lip') else 68
        return (load_model(str(Path(__file__).absolute().parent.parent / "models" / filename)), (None, num_landmarks, 2))
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


class ClassifyVVAD(Processor):
    """Visual Voice Activity Detection pipeline for cropped face video clips.

    Classifies speaking vs. not speaking. Expects a batch of face images per call, each image
    representing a different entity. Entities are identified either by their position in the
    batch (the order must then be stable across calls) or by the optional ``ids``. Images, or the shape
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
        shape_passes: dlib passes per face for the landmark models (``GetShapeFeatures``).
        shape_box_mode: Start box of dlib's first pass for the landmark models, one of
            ``Shape_Box_Modes`` (``GetShapeFeatures``, #105).
    """
    def __init__(self, architecture='CNN2Plus1D_Light',
                 stride=38, averaging_window_size=2, weighted=False, max_consecutive_empty=5,
                 timer=None, causal_mode='streaming', window_size=38, shape_passes=1,
                 shape_box_mode='none'):
        """Load the classifier and assemble the preprocessing, buffering and averaging steps.

        ``timer`` is an optional ``StageTimer`` that records the time of every step (#76).
        Asserts that ``architecture`` is in ``Architecture_Options``.
        """
        super(ClassifyVVAD, self).__init__()
        self.timer = timer or NULL_TIMER
        assert architecture in Architecture_Options, f"'{architecture}' is not in {Architecture_Options}"
        assert causal_mode in Causal_Modes, f"'{causal_mode}' is not in {Causal_Modes}"

        self.classifier, input_size = load_vvad_classifier(architecture)
        self.causal = architecture in Causal_Architectures
        self.causal_mode = causal_mode
        if self.causal:
            # one prediction per frame: the stride and the fixed sample length don't apply
            input_size = (window_size,) + tuple(input_size[1:])
            stride = 1
            self.stream = StreamingFeatureLSTM(self.classifier) if causal_mode == 'streaming' else None
            # compiled once for any batch size and length; eager Keras calls cost ~20 ms each
            num_landmarks = input_size[1]
            self._window_fn = tf.function(
                lambda x: self.classifier(x, training=False),
                input_signature=[tf.TensorSpec([None, None, num_landmarks, 2], tf.float32)])
            self.missed = {}
            self.max_consecutive_empty = max_consecutive_empty
        # Dummy predict to fully initialize
        self.classifier(np.zeros((1,) + tuple(input_size)))

        self.class_names = get_class_names('VVAD_LRS3')
        self.is_shape = 'Shape' in architecture

        # works on a batch of face images, not on a batch of samples, so it runs before buffering
        if self.is_shape:
            self.preprocess = GetShapeFeatures(architecture=architecture.replace('Causal', ''),
                                               passes=shape_passes, box_mode=shape_box_mode)
        else:
            # not normalized to [0, 1] because the model is trained on images in [0, 255]
            self.preprocess = PreprocessImages(input_size[1:3], normalize=False)
        # buffers the incoming face images or features into samples; a causal model in 'window'
        # mode predicts from whatever is buffered so far (1..window_size frames)
        self.buffer_features = BufferFeatures(input_size, stride=stride, max_consecutive_empty=max_consecutive_empty,
                                              return_incomplete_samples=self.causal)
        self.normalize = NormalizeShapeSample() if self.is_shape and not self.causal else None
        self.predict = PredictWithNones(model=self.classifier, postprocess=flatten_predictions)

        # Buffer predictions with stride 1 and return_incomplete_samples, so that something is
        # returned on every call. max_consecutive_empty = stride, so that a model giving no
        # prediction for stride steps does not reset the buffer.
        self.buffer_predictions = BufferFeatures((averaging_window_size,), stride=1, max_consecutive_empty=stride, return_incomplete_samples=True)
        # normalize to [0, 1] so that the output is in [0, 1] independent of averaging_window_size
        self.avg = AveragePredictions(weighted=weighted, normalize=True)

    def call(self, images, ids=None):
        """Classify a batch of face images, one per entity.

        Args:
            images: Batch of cropped RGB face images.
            ids: Optional entity ids, one per image. If given, each entity keeps its own
                temporal buffer regardless of its position in the batch.

        Returns:
            list: One (averaged) score in [0, 1] per image, or None where no prediction is
            available yet.
        """
        timer, n = self.timer, len(images)
        with timer.stage('preprocess', n):
            features = self.preprocess(images)
        if self.causal:
            return self._call_causal(features, ids)
        with timer.stage('buffer', n):
            samples = self.buffer_features(features, ids)
        n_samples = sum(sample is not None for sample in samples)
        if self.normalize is not None:
            with timer.stage('normalize', n_samples):
                samples = self.normalize(samples)
        if n_samples:
            with timer.stage('model', n_samples):
                predictions = self.predict(samples)
        else:
            predictions = None
        if predictions is None or len(predictions) == 0:
            predictions = [None] * n
        with timer.stage('average', n):
            return self.avg(self.buffer_predictions(list(predictions), ids))

    def _call_causal(self, features, ids):
        """Per-frame prediction with a causal landmark model (#45).

        'streaming' advances one LSTM state per entity by one step; 'window' runs the model on
        the buffered frames (up to ``window_size``) and takes the output of the newest frame.
        
        Args:
            features: Landmarks per face (output of ``GetShapeFeatures``).
            ids: Entity ids, or None to use the batch positions.

        Returns:
            list: One averaged score per face.
        """
        timer, n = self.timer, len(features)
        keys = list(ids) if ids is not None else list(range(n))
        frames = [normalize_shape_frames(f) for f in features]
        if self.causal_mode == 'streaming':
            with timer.stage('model', n):
                scores = list(self.stream.step(np.stack(frames), keys)) if n else []
            # forget entities that have been missing for max_consecutive_empty calls
            for key in keys:
                self.missed[key] = 0
            for key in list(self.missed):
                if key not in keys:
                    self.missed[key] += 1
                    if self.missed[key] >= self.max_consecutive_empty:
                        del self.missed[key]
                        self.stream.forget([key])
        else:
            with timer.stage('buffer', n):
                samples = self.buffer_features(frames, ids)
            scores = [None] * n
            with timer.stage('model', n):
                by_len = {}
                for k, sample in enumerate(samples):
                    by_len.setdefault(len(sample), []).append(k)
                for length, idx in by_len.items():
                    batch = np.stack([np.stack(list(samples[k])) for k in idx])
                    out = self._window_fn(tf.constant(batch, tf.float32)).numpy()[:, -1, 0]
                    for k, score in zip(idx, out):
                        scores[k] = float(score)
        with timer.stage('average', n):
            return self.avg(self.buffer_predictions([float(s) for s in scores], ids))

    def reset(self):
        """Clear the temporal state: the feature buffers and the prediction averaging window."""
        self.buffer_features.reset()
        self.buffer_predictions.reset()
        if self.is_shape:
            self.preprocess.reset()
        if self.causal and self.stream is not None:
            self.stream.reset()
            self.missed = {}




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
        profile: If True, the time of every stage is recorded in ``self.timer``
            (a ``StageTimer``); see #76.
        av_window: Frames per prediction for the audio-visual architectures ('AV:<model>').
        audio_fallback: What the audio-visual models get without audio: 'noise' or
            'visual' (visual-only back end); see ``asd4hri.av.AVClassifier``.
        causal_mode: Only for the causal architectures (``Causal_Architectures``): 'streaming'
            keeps one LSTM state per face and costs one step per frame, 'window' re-runs the
            model on the last ``window_size`` frames. Both predict every frame with zero
            look-ahead (#45).
        window_size: Frames per prediction in the causal 'window' mode.
        shape_passes: dlib passes per face for the landmark models; 2 fits loose boxes better
            (``GetShapeFeatures``, #92).
        shape_box_mode: Start box of dlib's first pass for the landmark models, one of
            ``Shape_Box_Modes`` (``GetShapeFeatures``, #105).
    """
    def __init__(self, architecture='CNN2Plus1D_Light', stride=2, averaging_window_size=3, decision_threshold=0.5,
                 weighted= True, max_consecutive_empty=2, annotate_output=False, detector='YuNet', input_size=None, face_tracking_approach='sort',
                 profile=True, causal_mode='streaming', window_size=38, av_window=25,
                 audio_fallback='noise', shape_passes=1, shape_box_mode='none'):
        """Build the face detection, cropping and classification steps.

        Raises:
            NotImplementedError: If ``detector`` is 'InsightFace' or 'YOLOv10-Face'.
            ValueError: If ``detector`` is unknown.
        """
        super(ASD, self).__init__()
        self.timer = StageTimer(enabled=profile)
        self.av = None
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
        self.detector_name = detector
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
            architecture=architecture, max_consecutive_empty=max_consecutive_empty,
            timer=self.timer, causal_mode=causal_mode, window_size=window_size,
            shape_passes=shape_passes, shape_box_mode=shape_box_mode,
        )
        self.class_names = list(get_class_names('VVAD_LRS3'))
        corrected_class_names = [self.class_names[1], self.class_names[0]] # in PAZ the order is speaking, not-speaking but we need it the other way around.
        self.decision_threshold = decision_threshold
        self.corrected_class_names = corrected_class_names
        if architecture.startswith(AV_PREFIX):
            # audio-visual model (#40, #80): crops its own faces from the frame, needs set_audio()
            self.av = AVClassifier(architecture[len(AV_PREFIX):], window=av_window,
                                   stride=stride, max_consecutive_empty=max_consecutive_empty,
                                   audio_fallback=audio_fallback, timer=self.timer)
            self.classifier = None
        else:
            self.vvad = ClassifyVVAD(**self.vvad_args)
            self.classifier = AddClassAndScoreToBoxes(self.vvad, class_names=corrected_class_names, decision_threshold=decision_threshold)
        

        self.class_names.append('No Prediction yet')

        self.draw = pr.DrawBoxes2D(self.class_names, self.colors, True)
        self.wrap = pr.WrapOutput(['image', 'boxes2D'])

    def set_audio(self, audio, fps):
        """Give the audio-visual models the audio track (int16, 16 kHz mono) and frame rate.

        Call once per video before the first frame. Ignored by the visual-only models.
        """
        if self.av is not None:
            self.av.set_audio(audio, fps)

    def call(self, image, boxes2D=None, ids=None, timestamp=None):
        """Detect faces in one frame and classify each of them as speaking or not speaking.

        Args:
            image: RGB image as a numpy array of shape ``(height, width, 3)``.
            boxes2D: Optional face boxes (``paz.abstract.messages.Box2D`` in pixels). If given,
                the face detector and tracker are skipped and these boxes are classified,
                e.g. the ground-truth boxes of a dataset (#75).
            ids: Optional entity ids, one per box in ``boxes2D``. Each id keeps its own temporal
                buffer. Without ids, faces are identified by their (tracked) position.
            timestamp: Time of the frame in seconds; the audio-visual models need it to cut the
                matching audio.

        Returns:
            dict: ``{'image': image, 'boxes2D': boxes2D}``. ``boxes2D`` is a list of
            ``paz.abstract.messages.Box2D`` whose ``class_name`` and ``score`` hold the prediction.
            The image has the boxes drawn on it if ``annotate_output`` is True.
        """
        timer = self.timer
        with timer.stage('frame'):
            source = 'gt' if boxes2D is not None else self.detector_name
            if boxes2D is None:
                # get the face boxes and crop the faces from the image
                with timer.stage('detect'):
                    image_copy = self.copy(image)
                    boxes2D = self.detect(image_copy)['boxes2D']
                with timer.stage('track', len(boxes2D)):
                    boxes2D = self.face_tracker(boxes2D)
                ids = None
            n = len(boxes2D)
            if self.av is not None:
                # TalkNet-style models crop the (unsquared) box themselves
                with timer.stage('crop', n):
                    boxes2D = self.clip(image, boxes2D)
                scores = self.av(image, [b.coordinates for b in boxes2D], ids, timestamp)
                for score, box in zip(scores, boxes2D):
                    if score is None:
                        box.score, box.class_name = -1.0, 'No Prediction yet'
                    else:
                        box.score = float(score)
                        box.class_name = self.corrected_class_names[int(score > self.decision_threshold)]
            else:
                with timer.stage('crop', n):
                    boxes2D = self.square(boxes2D)
                    boxes2D = self.clip(image, boxes2D)
                    cropped_images = self.crop(image, boxes2D)
                if self.vvad.is_shape:
                    # where the crops lie, for the start box of dlib (#105)
                    self.vvad.preprocess.set_frame([b.coordinates for b in boxes2D], source, ids)

                # call classifyVVAD for the whole batch of faces; it times its own steps
                boxes2D = self.classifier(cropped_images, boxes2D, ids)

            # only if flag is set
            if self.annotate_output:
                with timer.stage('annotate', n):
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
