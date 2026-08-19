'''
All paz processors needed for the full pipeline
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
import tensorflow as tf


from paz.models.classification import VVAD_LRS3_LSTM, CNN2Plus1D
from paz.datasets import get_class_names
from paz.pipelines import PreprocessImage
from paz import processors as pr
from paz.abstract import Processor, SequentialProcessor
from paz.backend.camera import VideoPlayer, Camera
import paz.pipelines.detection as dt
from paz.backend.boxes import add_class_and_score

from keras.models import load_model, Sequential
from keras.layers import Dense, Input, LSTM, TimeDistributed, BatchNormalization, Flatten

import numpy as np
from tqdm import tqdm

# local imports

# end file header
__author__      = 'Adrian Auer'


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



class BufferFeatures(Processor):
    """Buffers features (e.g.images) for models that need timeseries of features. The Buffer is working as a FIFO (First In First Out) buffer. 
    The Buffer handels batches dynamically with multiple buffers internally. It expects a sorted list like object of features as input to match to the internal buffers.
    The List must be sorted to assure  

    # Arguments
        input_size: Tuple of integers. Input shape to the model in following format: (timesteps, dim0, dim1, ...) 
            e.g. (frames, height, width, channels) for video
        stride: Integer, specifies after how many added features the buffer will return the all buffered features.
            In a scenario with an already full buffer and a stride of 10,
            after each 10th call the buffer will be returned.
            The stride must be smaller than the number of timesteps (the first argument of the input_size).
        max_consecutive_empty: Integer. Buffer will be cleared after this amount of consecutive empty(None) or no inputs. 
            None as input is needed if within the sorted list there is no input for an element between two others. 
            If no input is in the end, the list will just be shorter.
        return_incomplete_samples: Boolean.Flag if the samples should be returned even if not all timesteps are collected 
            - practical for dynamic length with upper bound

    # Methods
        call()
    """
    def __init__(self, input_size, stride=25, max_consecutive_empty=5, return_incomplete_samples=False):
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
        """
        # Arguments
            batch_of_features: sorted list like object of features. Must have the shape (N, dim0, dim1, ...) where N is the size of the batch.
        # Returns
            Numpy array. Batch of timeseries of features of the shape (M, timesteps, dim0, dim1, ...) where M is the number of internal Buffers that are full.
            The buffer will return a sorted list of all buffers. When the stride is not reached or the buffer is not full the element will be None.
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
        """
        reset the state of all internal buffers if not specified differently
        # Arguments:
            index: Index for the buffer that should be resetted
        """
        if index is None:
            for i, _ in enumerate(self.buffers):
                self.reset(index=i)
        else:
            self.buffers[index] = deque(maxlen=self.buffer_size)
            self.timesteps_since_last_return[index] = 0
            self.missed_consecutive_timesteps[index] = 0


class NormalizeShapeSample(Processor):
    """Processor to normalize a batch of faceShape or LipShape samples"""
    def __init__(self):
        super(NormalizeShapeSample, self).__init__()

    def _getDist(self, sample):
        """
        calcing the distance vectors for a sample

        :param sample: the sample we want the distances to be calculated
        :type sample: numpy array
        """
        # print(f'{type(sample)=}')
        base = sample[0][0]

        output_sample = np.array(sample)
        output_sample -= [base[0], base[1]]
   
        return output_sample

    def _normalize(self, arr):
        """
        Normalizes the features of the the array to [-1, 1]. 
        """
        arrMax = np.max(arr)
        arrMin = np.min(arr)
        absMax = np.max([np.abs(arrMax), np.abs(arrMin)])
        return arr/absMax
    
    def call(self, samples):
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
    """
    Processor to extract shape features from a batch of cropped RGB faces using dlib's shape predictor.

    """
    def __init__(self, architecture='FaceShape', shape_predictor_path=None):
        super(GetShapeFeatures, self).__init__()
        if shape_predictor_path is None:
            shape_predictor_path = SHAPE_PREDICTOR_68_FACE_LANDMARKS()
        self.shape_predictor = dlib.shape_predictor(str(shape_predictor_path))
        self.architecture = architecture

    def call(self, face_images):
        """
        converts a batch of face images to a batch of face_shapes
        # Argumants
            face_images: a batch of cropped images of faces.
        # Returns
            List of numpy arrays: A batch of shape features for the input faces.
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
    """Preprocess, predict and postprocess batched input.
    Batch can contain None as samples or be None itself. 
    If batch is None - output is None (no postprocessing is applied)
    If batch contains Nones - output batch will contain Nones in sam position (No postprocessing if batch only contains Nones)
    # Arguments
        x: Noneable input to model
        model: Callable i.e. Keras model.
        preprocess: Callable, used for preprocessing input x.
        postprocess: Callable, used for postprocessing output of model.

    # Note
        If model outputs a tf.Tensor is converted automatically to numpy array.
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
    """Perform input preprocessing, model prediction and output postprocessing based on batches.

    # Arguments
        model: Class with a ''predict'' method e.g. a Keras model.
        preprocess: Function applied to given inputs.
        postprocess: Function applied to outputted predictions from model.
    """

    def __init__(self, model, preprocess=None, postprocess=None):
        super(PredictWithNones, self).__init__()
        self.model = model
        self.preprocess = preprocess
        self.postprocess = postprocess

    def call(self, x):
        return predict_with_nones(x, self.model, self.preprocess, self.postprocess)

def flatten_predictions(batch_of_predictions):
    """Flattens the given predictions because the model returns an array with shape (N, 1) instead of (N,) and the postprocessing expects a list of predictions.
    # Arguments:
        batch_of_predictions: List of predictions to be flattened.
    # Returns
        List of predictions. Flattened predictions.
    """
    # TOOD: if list inlcuding nones, reshape differently
    if None in batch_of_predictions:
        return [None if pred is None else np.array(pred).reshape(-1)[0] for pred in batch_of_predictions]
    return np.array(batch_of_predictions).reshape(-1)
        

class AveragePredictions(Processor):
    """Averages the given predictions
    # Arguments
        weigthed: Boolean. 
            If True weigthed by the index+1 (otherwise the first entry will always be zero and single values will be removed) of the value.
            If False just the average over all entries
        normalize: Boolean.
            If True the outputvalue will be normalized to [0, 1]. This expects the input values to be in [0, 1] as well.
    """

    def __init__(self, weighted=False, normalize=False):
        super(AveragePredictions, self).__init__()
        self.weighted = weighted
        self.normalize = normalize

    def call(self, batch_of_list_of_values):
        """
        # Arguments:
            batch_of_list_of_values: Batch of List of values to be averaged.
        # Returns
           list of  Bool, Int or Float value. batch of Averaged value.
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
    """Adds class name and score to boxes. Expects a batch of cropped images and a batch of boxes. 

    # Arguments
        classifier: Keras model.
    """
    def __init__(self, classifier, class_names, decision_threshold=0.5):
        super(AddClassAndScoreToBoxes, self).__init__()
        self.classify = classifier
        self.class_name = class_names
        self.decision_threshold = decision_threshold
        assert len(self.class_name) == 2, f"AddClassAndScoreToBoxes only supports binary classes. Number of classes names must be 2 for binary classification but is {len(self.class_name)}"

    def __call__(self, cropped_images, boxes):
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

class PreprocessImages(SequentialProcessor):
    """Preprocess RGB images by resizing it to the given ``shape``. And cast it to the given ``dtype``. 
    Can contain Nones in the batch and will return a batch with Nones in the same position.

    # Arguments
        shape: List of two Ints.
        dtype: np.dtype. Data type to cast the image to.
    """
    def __init__(self, shape, dtype=float):
        super(PreprocessImages, self).__init__()
        self.resize = pr.ResizeImage(shape)
        self.cast = pr.CastImage(dtype)

    def __call__(self, images):
        ret_batch = []
        if images is None:
            return None
        else:
            for image in images:
                if image is None:
                    ret_batch.append(None)
                else:
                    image = self.resize(image)
                    image = self.cast(image)
                    ret_batch.append(image)
        return ret_batch

# def add_class_and_score(prediction, box):
#     """Adds class and score to box.

#     # Arguments
#         prediction: Dictionary with keys `class_name` and `scores`.
#         box: Array of shape `(num_nms_boxes, 4 + num_classes)`.
#     """
#     if box is None:
#         return None
#     if prediction is None:
#         box.class_name = 'No Prediction yet' 
#         box.score = -1.0 
#         return box
#     box.class_name = prediction['class_name']
#     box.score = np.amax(prediction['scores'])
#     return box