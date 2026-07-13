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


# TODO: make Buffer Batch Ready!
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
        dtype: Numpy Datatype for the features. Default is numpy.float64

    # Methods
        call()
    """
    def __init__(self, input_size, stride=25, max_consecutive_empty=5, dtype=np.float64):
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
                self.buffers[i] = deque(maxlen=self.buffer_size)
                self.timesteps_since_last_return[i] = 0
                self.missed_consecutive_timesteps[i] = 0
        # update missed_consecutive_timesteps for buffers in the end oif the list
        if i == len(self.buffers):
            for j in range(i, len(self.buffers) -1):
                self.missed_consecutive_timesteps[j] += 1

        return batch_return