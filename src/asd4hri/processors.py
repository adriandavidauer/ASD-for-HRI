"""All paz processors needed for the full pipeline."""

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
import tensorflow as tf

import cv2
import numpy as np
import onnxruntime as ort


from paz.models.classification import VVAD_LRS3_LSTM, CNN2Plus1D
from paz.datasets import get_class_names
from paz.pipelines import PreprocessImage
from paz import processors as pr
from paz.abstract import Processor, SequentialProcessor
from paz.backend.camera import VideoPlayer, Camera
import paz.pipelines.detection as dt
from paz.backend.boxes import add_class_and_score
from paz.abstract.messages import Box2D

from keras.models import load_model, Sequential
from keras.layers import Dense, Input, LSTM, TimeDistributed, BatchNormalization, Flatten
from tensorflow.keras.utils import get_file


import numpy as np
from tqdm import tqdm

# local imports
from .retinaface_onnx import cfg_re50 as cfg
from .retinaface_onnx import PriorBox, decode, py_cpu_nms

# end file header
__author__      = 'Adrian Auer'

URL = 'https://github.com/adriandavidauer/ASD-for-HRI/releases/download/models/'
FaceTracking_Options = ['sort', 'face_recognition']
class DownloadProgressBar(tqdm):
    """``tqdm`` progress bar that can be used as a ``urllib.request.urlretrieve`` report hook."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """Update the progress bar from ``urlretrieve`` report hook arguments.

        Args:
            b: Number of blocks transferred so far.
            bsize: Size of each block in bytes.
            tsize: Total size in bytes, or None if unknown.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download a file while showing a progress bar.

    Args:
        url: URL of the file to download.
        output_path: Path the file is written to.
    """
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(
            url, filename=output_path, reporthook=t.update_to)

def SHAPE_PREDICTOR_68_FACE_LANDMARKS():
    """Return the path to dlib's 68 face landmark shape predictor, downloading it if necessary.

    The compressed model is downloaded from the dlib-models repository to ``src/models``,
    decompressed, and the archive is removed.

    Returns:
        pathlib.Path: Path to ``shape_predictor_68_face_landmarks.dat``.

    Raises:
        FileNotFoundError: If the file is missing after decompression, e.g. because it was
            deleted externally.
    """
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



class BufferFeatures(Processor):
    """FIFO buffer of features (e.g. images) for models that need a time series of features.

    Handles batches dynamically with one internal buffer per batch position. The input is
    expected to be a sorted list-like object of features so that each position always belongs
    to the same entity and is matched to the same internal buffer.

    Args:
        input_size: Shape of one model sample, ``(timesteps, dim0, dim1, ...)``, e.g.
            ``(frames, height, width, channels)`` for video. Only ``timesteps`` is used as the
            buffer length.
        stride: Number of added features after which a full buffer is returned again. With an
            already full buffer and a stride of 10, the buffer is returned on every 10th call.
            Must not be larger than ``timesteps``.
        max_consecutive_empty: Number of consecutive empty (None) or missing inputs after which
            a buffer is cleared. Use None as input for an entity in the middle of the list that
            has no feature in this step; missing entities at the end can simply be left out.
        return_incomplete_samples: If True, every buffer is returned on every call, even if not
            all timesteps are collected yet. Useful for dynamic lengths with an upper bound.
    """
    def __init__(self, input_size, stride=25, max_consecutive_empty=5, return_incomplete_samples=False):
        """Set up the buffer configuration and empty internal buffers.

        Raises:
            ValueError: If ``input_size[0]`` is smaller than ``stride``.
        """
        self.buffer_size = input_size[0]
        if self.buffer_size < stride:
            raise ValueError('Buffer size must be equal or larger than stride')
        super(BufferFeatures, self).__init__()
        self.stride = stride 
        self.max_consecutive_empty = max_consecutive_empty
        self.return_incomplete_samples = return_incomplete_samples
        # Buffers: 
        self.buffers = []
        # Counters for return after stride
        self.timesteps_since_last_return = [] # I have to keep track of that for each internal buffer, because otherwise a full buffer could return even if only input for another buffer came in.
        self.missed_consecutive_timesteps = []


    def call(self, batch_of_features):
        """Add one batch of features to the internal buffers.

        Args:
            batch_of_features: Sorted list-like object of features with shape
                ``(N, dim0, dim1, ...)`` where N is the batch size. Entries may be None.

        Returns:
            list: One entry per input position: the corresponding buffer (a ``collections.deque``
            holding up to ``timesteps`` features) if it is ready to be returned, otherwise None.
        """
        batch_return = []
        i = 0
        for i, features in enumerate(batch_of_features):
            if i == len(self.buffers):
                self.buffers.append(deque(maxlen=self.buffer_size))
                self.timesteps_since_last_return.append(0)
                self.missed_consecutive_timesteps.append(0)
            if features is None:
                self.missed_consecutive_timesteps[i] += 1
            else:
                self.buffers[i].append(features)
                self.missed_consecutive_timesteps[i] = 0
            self.timesteps_since_last_return[i] += 1 
            if (self.return_incomplete_samples) or (len(self.buffers[i]) == self.buffer_size and self.timesteps_since_last_return[i] >= self.stride):
                batch_return.append(self.buffers[i])
                self.timesteps_since_last_return[i] = 0
            else:
                batch_return.append(None) # necessary to keep ordering
            # resetting
            if self.missed_consecutive_timesteps[i] >= self.max_consecutive_empty:
                self.reset(index=i)
        # update missed_consecutive_timesteps for buffers in the end of the list if input list is shorter than self.buffers
        if i < len(self.buffers)-1:
            # need to go backwards to pop
            for j in range(len(self.buffers) -1, i, -1):
                self.missed_consecutive_timesteps[j] += 1
                # remove dangeling
                if self.missed_consecutive_timesteps[j] >= self.max_consecutive_empty:
                    self.buffers.pop()
                    self.timesteps_since_last_return.pop()
                    self.missed_consecutive_timesteps.pop()
        return batch_return

    def reset(self, index=None):
        """Reset one internal buffer, or all of them.

        Args:
            index: Index of the buffer to reset. If None, all buffers are reset.
        """
        if index is None:
            for i, _ in enumerate(self.buffers):
                self.reset(index=i)
        else:
            self.buffers[index] = deque(maxlen=self.buffer_size)
            self.timesteps_since_last_return[index] = 0
            self.missed_consecutive_timesteps[index] = 0


class NormalizeShapeSample(Processor):
    """Normalize a batch of FaceShape or LipShape samples.

    Each sample is translated so that the first landmark of the first timestep is the origin,
    then scaled to [-1, 1] by its largest absolute coordinate.
    """
    def __init__(self):
        """Initialize the processor."""
        super(NormalizeShapeSample, self).__init__()

    def _getDist(self, sample):
        """Translate a sample so that the first landmark of the first timestep is the origin.

        Args:
            sample: Sample of shape ``(timesteps, num_landmarks, 2)``.

        Returns:
            numpy.ndarray: The translated sample.
        """
        # print(f'{type(sample)=}')
        base = sample[0][0]

        output_sample = np.array(sample)
        output_sample -= [base[0], base[1]]
   
        return output_sample

    def _normalize(self, arr):
        """Scale an array to [-1, 1] by dividing by its largest absolute value.

        Args:
            arr: Array to scale.

        Returns:
            numpy.ndarray: The scaled array.
        """
        arrMax = np.max(arr)
        arrMin = np.min(arr)
        absMax = np.max([np.abs(arrMax), np.abs(arrMin)])
        return arr/absMax
    
    def call(self, samples):
        """Normalize every sample in the batch.

        Args:
            samples: Batch of samples of shape ``(timesteps, num_landmarks, 2)``. Entries may be None.

        Returns:
            list: The normalized samples, with None at the positions of None inputs.
        """
        output_list = []
        for sample in samples:
            if sample is None:
                output_list.append(None)
            else:
                sample = self._getDist(sample) 
                sample = self._normalize(sample)
                output_list.append(sample)
        return output_list

class GetShapeFeatures(Processor):
    """Extract facial landmark (shape) features from a batch of cropped RGB faces with dlib.

    Args:
        architecture: 'LipShape' to return only the 20 lip landmarks (indices 48-67); any other
            value returns all 68 landmarks.
        shape_predictor_path: Path to a dlib shape predictor. If None, the 68 face landmark
            predictor is downloaded if needed and used.
    """
    def __init__(self, architecture='FaceShape', shape_predictor_path=None):
        """Load the dlib shape predictor."""
        super(GetShapeFeatures, self).__init__()
        if shape_predictor_path is None:
            shape_predictor_path = SHAPE_PREDICTOR_68_FACE_LANDMARKS()
        self.shape_predictor = dlib.shape_predictor(str(shape_predictor_path))
        self.architecture = architecture

    def call(self, face_images):
        """Convert a batch of face images to a batch of landmark coordinates.

        Args:
            face_images: Batch of cropped face images.

        Returns:
            list: One numpy array of ``(x, y)`` landmark coordinates per face, of shape ``(68, 2)``
            or ``(20, 2)`` for 'LipShape'.

        Raises:
            AssertionError: If an element of ``face_images`` is two-dimensional with a last
                dimension of 3, which indicates that a single image was passed instead of a batch.
        """
        ret_batch = []
        for image in face_images:
            if len(image.shape) == 2 and image.shape[-1] == 3:
                raise AssertionError(f"Probably not receiving a batch - could be a batch of images with the shape {image.shape} but that makes no sense.")
            shape = self.shape_predictor(image, dlib.rectangle(
                    0, 0, image.shape[1], image.shape[0]))
            if self.architecture == 'LipShape':
                # return only lip landmarks (48-67)
                ret_batch.append(np.array([(p.x, p.y) for p in shape.parts()[48:68]]))
            else:
                ret_batch.append(np.array([(p.x, p.y) for p in shape.parts()]))
        return ret_batch


def predict_with_nones(x, model, preprocess=None, postprocess=None):
    """Preprocess, predict and postprocess a batch that may contain None.

    If the batch (after preprocessing) is None, the output is None. If it contains None
    entries, the output contains None at the same positions. Postprocessing is skipped when the
    batch is None, empty, or contains only None entries.

    Args:
        x: Input batch for the model; may be None or contain None.
        model: Callable, e.g. a Keras model.
        preprocess: Callable used to preprocess ``x``.
        postprocess: Callable used to postprocess the output of the model.

    Returns:
        The (postprocessed) model output for the batch with None at the positions of None inputs,
        None if the batch is None, or an empty list for an empty batch. A ``tf.Tensor`` output
        is converted to a numpy array.
    """
    if preprocess is not None:
        x = preprocess(x)
    if x is None:
        return None # this skips post processing
    if len(x) == 0:
        return [] # this skips post processing
    elif any(sample is None for sample in x):
        # create mapping and remove Nones from the batch
        non_none_pairs = [(i, val) for i, val in enumerate(x) if val is not None and val is not [] is not None]   
        if non_none_pairs:
            indices, non_nones = zip(*non_none_pairs)     
            # apply model
            inference = model(np.array(non_nones))
            # apply mapping to recreate batch with Nones
            y = [None] * len(x)
            for idx, val in zip(indices, inference):
                y[idx] = val
        else: # if the input only contains Nones
            return x # this skips post processing
    else:
        y = model(np.array(x))
    if isinstance(y, tf.Tensor):
        y = y.numpy()
    if postprocess is not None:
        y = postprocess(y)
    return y

class PredictWithNones(Processor):
    """Preprocess, predict and postprocess batches that may contain None.

    See ``predict_with_nones``.

    Args:
        model: Callable, e.g. a Keras model.
        preprocess: Function applied to the inputs.
        postprocess: Function applied to the predictions of the model.
    """

    def __init__(self, model, preprocess=None, postprocess=None):
        """Store the model and the pre- and postprocessing functions."""
        super(PredictWithNones, self).__init__()
        self.model = model
        self.preprocess = preprocess
        self.postprocess = postprocess

    def call(self, x):
        """Run ``predict_with_nones`` on a batch.

        Args:
            x: Input batch; may be None or contain None.

        Returns:
            The prediction batch as returned by ``predict_with_nones``.
        """
        return predict_with_nones(x, self.model, self.preprocess, self.postprocess)

def flatten_predictions(batch_of_predictions):
    """Flatten a batch of single-value predictions.

    The model returns an array of shape ``(N, 1)``, but the following steps expect ``(N,)``.

    Args:
        batch_of_predictions: Batch of predictions; may contain None.

    Returns:
        numpy.ndarray of shape ``(N,)``, or a list of scalars with None kept in place if the
        batch contains None.
    """
    # TOOD: if list inlcuding nones, reshape differently
    if None in batch_of_predictions:
        return [None if pred is None else np.array(pred).reshape(-1)[0] for pred in batch_of_predictions]
    return np.array(batch_of_predictions).reshape(-1)
        

class AveragePredictions(Processor):
    """Average each list of values in a batch.

    Args:
        weighted: If True, the i-th value (1-based) is multiplied by i before averaging, so
            later (newer) values count more. If False, the plain mean is used.
        normalize: If True, the result is divided by ``(n + 1) / 2`` (the weighted mean of n
            ones), so a weighted average of values in [0, 1] stays in [0, 1].
    """

    def __init__(self, weighted=False, normalize=False):
        """Store the averaging options."""
        super(AveragePredictions, self).__init__()
        self.weighted = weighted
        self.normalize = normalize

    def call(self, batch_of_list_of_values):
        """Average every list of values in the batch.

        Args:
            batch_of_list_of_values: Batch of lists of values. Entries may be None or empty.

        Returns:
            list: One averaged value per entry, None for None or empty entries. Returns None if the
            batch is None and an empty list if the batch is empty.
        """
        if batch_of_list_of_values is None:
            return None
        if len(batch_of_list_of_values) == 0:
            return []
        else:
            # create mapping and remove Nones and empty lists from the batch
            non_none_pairs = [(i, val) for i, val in enumerate(batch_of_list_of_values) if val is not None and len(val) != 0]  
            if not non_none_pairs:
                return [None] * len(batch_of_list_of_values)
            indices, non_nones = zip(*non_none_pairs)    
            
            # Get sample lengths and dimensions
            lengths = np.array([len(x) for x in non_nones])
            max_len = lengths.max()
            # Build a padded 2D matrix filled with zeros
            padded = np.zeros((len(non_nones), max_len))
            for i, buffer in enumerate(non_nones):
                padded[i, :len(buffer)] = buffer
            if self.weighted:
                # Create index weights: [1, 2, 3] and multiply
                weights = np.arange(1, max_len+1)
                padded = padded * weights
            # Sum across rows and divide by the REAL original length of each sample
            mean = np.array(padded).sum(axis=1) / lengths

            # normalize eachto [0, 1] if desired
            if self.normalize:
                max_vals = np.array([np.sum(np.arange(1, x+1))/x for x in lengths])
                mean = mean / max_vals
            # apply mapping to recreate batch with Nones
            ret_batch = [None] * len(batch_of_list_of_values)
            for idx, val in zip(indices, mean):
                ret_batch[idx] = val
            return ret_batch
            

class AddClassAndScoreToBoxes(Processor):
    """Classify cropped images and write the class name and score into the matching boxes.

    Args:
        classifier: Callable mapping a batch of cropped images to a batch of scores (None
            where no score is available yet), e.g. ``ClassifyVVAD``.
        class_names: Two class names, ``[negative, positive]``.
        decision_threshold: Score above which the positive class is assigned.
    """
    def __init__(self, classifier, class_names, decision_threshold=0.5):
        """Store the classifier and the class names.

        Asserts that ``class_names`` contains exactly two names.
        """
        super(AddClassAndScoreToBoxes, self).__init__()
        self.classify = classifier
        self.class_name = class_names
        self.decision_threshold = decision_threshold
        assert len(self.class_name) == 2, f"AddClassAndScoreToBoxes only supports binary classes. Number of classes names must be 2 for binary classification but is {len(self.class_name)}"

    def call(self, cropped_images, boxes):
        """Classify the cropped images and update the boxes in place.

        Boxes without a score get the class name 'No Prediction yet' and a score of -1.0.

        Args:
            cropped_images: Batch of cropped images, one per box.
            boxes: List of ``paz.abstract.messages.Box2D``, in the same order as ``cropped_images``.

        Returns:
            list: The same boxes with ``class_name`` and ``score`` set.
        """
        scores = self.classify(cropped_images)
        for score, box2D in zip(scores, boxes):
            if score is None:
                class_name = 'No Prediction yet'
                score = -1.0
            else:
                class_name = self.class_name[1] if score > self.decision_threshold else self.class_name[0]
            box2D.score = score
            box2D.class_name = class_name
            # return_boxes.append(add_class_and_score({'class_name': class_name, 'scores': [score]}, box2D))
        return boxes            
class PreprocessImages(Processor):
    """Resize, and optionally normalize, a batch of RGB images.

    The batch may contain None; the output then contains None at the same positions.

    Args:
        shape: Target size as a list of two integers, ``(width, height)``.
        normalize: If True, the images are cast to float and normalized to [0, 1].
    """
    def __init__(self, shape, normalize=True):
        """Create the resize, cast and normalize processors."""
        super(PreprocessImages, self).__init__()
        self.resize = pr.ResizeImage(shape)
        self.cast = pr.CastImage(float)
        self.normalizer = pr.NormalizeImage()
        self.normalize = normalize

    def call(self, images):
        """Preprocess every image in the batch.

        Args:
            images: Batch of images; may be None or contain None.

        Returns:
            list: The preprocessed images with None kept in place, or None if ``images`` is None.
        """
        ret_batch = []
        if images is None:
            return None
        else:
            for image in images:
                if image is None:
                    ret_batch.append(None)
                else:
                    image = self.resize(image)
                    if self.normalize:
                        image = self.cast(image)
                        image = self.normalizer(image)
                    ret_batch.append(image)
        return ret_batch


class FaceDetectorYN(Processor):
    """Face detector based on OpenCV's YuNet (``cv2.FaceDetectorYN``).

    Args:
        input_size: ``(width, height)`` of the input images. If it does not match an image,
            the detector input size is updated at runtime, which can slow the pipeline down.
        class_name: Class name used when drawing boxes.
        color: RGB color for drawing boxes, e.g. ``[0, 255, 0]``.
        draw: If True, the bounding boxes are drawn on the returned image.
        model_path: Path to the ONNX model. Defaults to the 2026may YuNet model, which is
            downloaded on first use.
        conf_threshold: Boxes with a score below this value are filtered out.
        nms_threshold: Boxes with an IoU above this value are suppressed.
        topK: Number of boxes kept before non-maximum suppression.
        backendId: OpenCV DNN backend id.
        targetId: OpenCV DNN target device id.
        name: Name of the processor.
    """
    def __init__(self, input_size=[320, 320], class_name='Face', color=[0, 255, 0], draw=False, model_path=str(Path(__file__).absolute().parent.parent / "models" / 'face_detection_yunet_2026may.onnx'), conf_threshold=0.6, nms_threshold=0.3, topK=500, backendId=0, targetId=0, name='FaceDetectorYN'):
        """Create the YuNet detector, downloading the default model if needed."""
        super(FaceDetectorYN, self).__init__(name=name)
        if model_path == str(Path(__file__).absolute().parent.parent / "models" / 'face_detection_yunet_2026may.onnx'):
            filename = 'face_detection_yunet_2026may.onnx'
            get_file(filename, URL + filename, cache_subdir=str(Path(__file__).absolute().parent.parent / "models" ))
        self.detector = cv2.FaceDetectorYN.create(model=model_path,
            config="",
            input_size=input_size,
            score_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            top_k=topK,
            backend_id=backendId,
            target_id=targetId)
        self.input_size = input_size
        self.draw = draw
        self.color = color
        self.class_name = class_name
        self.drawer = pr.DrawBoxes2D([self.class_name], [self.color], True)

    def call(self, image):
        """Detect faces in an image.

        Args:
            image: Image as a numpy array of shape ``(height, width, 3)``.

        Returns:
            dict: ``{'image': image, 'boxes2D': boxes2D}`` where ``boxes2D`` is a list of
            ``paz.abstract.messages.Box2D`` in pixel coordinates with the detection score and the
            class name 'Face'. The image has the boxes drawn on it if ``draw`` is True.
        """
        # set input size if necessary
        if (image.shape[1], image.shape[0]) != self.input_size:
            self.detector.setInputSize((image.shape[1], image.shape[0]))
        retval, faces = self.detector.detect(image)
        # create boxes2D with the corresponding points and the score and the label
        boxes2D = []
        if faces is not None: 
            for face in faces:
                # fractional pixel coordinates -> round? int()? I will loose precision if I do it but drawing does not work and cropping might not work as well
                boxes2D.append(Box2D((int(face[0]), int(face[1]), int(face[0]+face[2]), int(face[1]+face[3])), face[14], 'Face'))
        if self.draw:
            image = self.drawer(image, boxes2D)
        return {'image': image, 'boxes2D': boxes2D}




class FaceDetectorRetinaFace(Processor):
    """Face detector based on a RetinaFace (ResNet-50) ONNX model.

    Runs on CUDA if available, otherwise on the CPU. Images are resized to 640x640 for inference.

    Args:
        class_name: Class name used when drawing boxes.
        color: RGB color for drawing boxes, e.g. ``[0, 255, 0]``.
        draw: If True, the bounding boxes are drawn on the returned image.
        model_path: Path to the ONNX model. Defaults to ``retinaface_resnet50.onnx``, which is
            downloaded on first use.
        conf_threshold: Boxes with a score below this value are filtered out.
        nms_threshold: Boxes with an IoU above this value are suppressed.
        pre_nms_top_k: Number of highest-scoring boxes kept before non-maximum suppression.
        topK: Number of boxes kept after non-maximum suppression.
        backendId: Unused.
        targetId: Unused.
        name: Name of the processor.
    """
    def __init__(self, class_name='Face', color=[0, 255, 0], draw=False, model_path=str(Path(__file__).absolute().parent.parent / "models" / 'retinaface_resnet50.onnx'), conf_threshold=0.6, nms_threshold=0.3, pre_nms_top_k=1000, topK=500, backendId=0, targetId=0, name='FaceDetectorRetinaFace'):
        """Create the ONNX Runtime session, downloading the default model if needed."""
        super(FaceDetectorRetinaFace, self).__init__(name=name)
        if model_path == str(Path(__file__).absolute().parent.parent / "models" / 'retinaface_resnet50.onnx'):
            filename = 'retinaface_resnet50.onnx'
            get_file(filename, URL + filename, cache_subdir=str(Path(__file__).absolute().parent.parent / "models" ))
        self.draw = draw
        self.color = color
        self.class_name = class_name
        self.drawer = pr.DrawBoxes2D([self.class_name], [self.color], True)
        # 1. Load the model session (CPU or GPU)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.retinaface = ort.InferenceSession(model_path, providers=providers)
        # self.decoder = RetinaFaceDecoder640(conf_threshold=conf_threshold, nms_threshold=nms_threshold, pre_nms_top_k=pre_nms_top_k, topK=topK)
        self.topK = topK
        self.nms_threshold = nms_threshold
        self.pre_nms_top_k = pre_nms_top_k
        self.conf_threshold = conf_threshold
        # 2. Get input and output tensor metadata
        self.input_name = self.retinaface.get_inputs()[0].name

    def call(self, image):
        """Detect faces in an image.

        Args:
            image: Image as a numpy array of shape ``(height, width, 3)``.

        Returns:
            dict: ``{'image': image, 'boxes2D': boxes2D}`` where ``boxes2D`` is a list of
            ``paz.abstract.messages.Box2D`` in pixel coordinates with the detection score and the
            class name 'Face'. The image has the boxes drawn on it if ``draw`` is True.
        """
        orig_height, orig_width, _ = image.shape

        img = cv2.resize(image, (640, 640)).astype(np.float32)
        im_height, im_width, _ = img.shape
        # scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)

        # tic = time.time()
        inputs = {"input": img}
        loc, conf, _ = self.retinaface.run(None, inputs)
        # print('net forward time: {:.4f}s'.format(time.time() - tic))

        # tic = time.time()
        priorbox = PriorBox(cfg, image_size=(im_height, im_width), format="numpy")
        priors = priorbox.forward()

        prior_data = priors

        boxes = decode(np.squeeze(loc, axis=0), prior_data, cfg['variance'])
        # rescale the boxes to original image size
        boxes = boxes * np.array([orig_width, orig_height, orig_width, orig_height])
        scores = np.squeeze(conf, axis=0)[:, 1]

        # landms = decode_landm(np.squeeze(landms.data, axis=0), prior_data, cfg['variance'])

        # scale1 = np.array([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
        #                        img.shape[3], img.shape[2], img.shape[3], img.shape[2],
        #                        img.shape[3], img.shape[2]])
        # landms = landms * scale1 / resize

        # ignore low scores
        inds = np.where(scores > self.conf_threshold)[0]
        boxes = boxes[inds]
        # landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.pre_nms_top_k]
        boxes = boxes[order]
        # landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        # landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.topK, :]
        # landms = landms[:args.keep_top_k, :]

        # dets = np.concatenate((dets, landms), axis=1)
        # print('post processing time: {:.4f}s'.format(time.time() - tic))

        boxes2D = []
        if len(dets): 
            for face in dets:
                # fractional pixel coordinates -> round? int()? I will loose precision if I do it but drawing does not work and cropping might not work as well
                boxes2D.append(Box2D((int(face[0]), int(face[1]), int(face[2]), int(face[3])), face[4], 'Face'))
        if self.draw:
            image = self.drawer(image, boxes2D)
        return {'image': image, 'boxes2D': boxes2D}

class FaceTracker(Processor):
    """Order face boxes so that each face keeps the same index across frames.

    The buffers in ``ClassifyVVAD`` identify an entity by its index in the batch, so the boxes must
    come in a stable order.

    Args:
        approach: Tracking approach. One of ``FaceTracking_Options``: 'sort' naively sorts the boxes
            by their position, 'face_recognition' (not implemented yet) would use a face
            recognition model to track the same face over time.
    """
    def __init__(self, approach='sort'):
        """Validate and store the tracking approach.

        Raises:
            ValueError: If ``approach`` is not in ``FaceTracking_Options``.
        """
        super(FaceTracker, self).__init__()
        if approach not in FaceTracking_Options:
            raise ValueError(f"Approach must be one of {FaceTracking_Options} but is {approach}")
        self.approach = approach

    def call(self, boxes2D):
        """Return the boxes in tracking order.

        Args:
            boxes2D: List of ``paz.abstract.messages.Box2D`` from the face detector.

        Returns:
            list: The same boxes, sorted by their top-left corner (x, then y) for 'sort'.

        Raises:
            NotImplementedError: If the approach is 'face_recognition'.
        """
        if self.approach == 'sort':
            # boxes2D is sorted by score descending, so the first box is the best one but I need them sorted by bounding box position for the buffer to work properly
            ret_boxes2D = sorted(boxes2D, key=lambda box: (box.coordinates[0], box.coordinates[1])) # This could go in a separate processor with different sorting technigues from this naive one to tracking.
        elif self.approach == 'face_recognition':
            raise NotImplementedError("Face recognition based tracking is not implemented yet. Please use 'sort' as approach.")
        return ret_boxes2D