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

    # Methods
        call()
    """
    def __init__(self, input_size, stride=25, max_consecutive_empty=5):
        self.buffer_size = input_size[0]
        if self.buffer_size < stride:
            raise ValueError('Buffer size must be equal or larger than stride')
        super(BufferFeatures, self).__init__()
        self.stride = stride 
        self.max_consecutive_empty = max_consecutive_empty
        

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
            if len(self.buffers[i]) == self.buffer_size and self.timesteps_since_last_return[i] >= self.stride:
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


