'''
Tests for the main asd processors
'''

# System imports
from pathlib import Path
import random

# 3rd party imports
import cv2
import numpy as np
import pytest

# local imports
from asd4hri.asd import *
from tests.constants import *


# end file header
__author__      = 'Adrian Auer'



def test_input_output_ClassifyVVAD():
    """
    Test for the correct input and output shapes for ClassifyVVAD pipeline
    """
    averaging_window_size = 2
    weighted = True
    max_consecutive_empty = 5
    stride = 1
    vvad = ClassifyVVAD(architecture='LipShape', averaging_window_size=averaging_window_size, weighted=weighted, max_consecutive_empty=max_consecutive_empty, stride=stride)
    for i in range(200):
        test_image = np.empty(IMAGE_INPUT_SHAPE[1:], dtype=np.uint8) # multiple different initializations giving this more variability in the input 
        ret_val = vvad([test_image]) # input a batch of images - expecting batch of predictions
        if i < IMAGE_INPUT_SHAPE[0]-1:
            assert len(ret_val) == 1  , f"ret_val should have one entry but has {len(ret_val)} entries in step {i}"
            assert ret_val[0] is None, f"ret_val should be [None] but is {ret_val[0]} in step {i}"
        else:
            assert len(ret_val) == 1  , f"ret_val should have one entry but has {len(ret_val)} entries in step {i}"
            assert ret_val[0] is not None, f"ret_val should not be None but is {ret_val[0]} in step {i}"
            assert ret_val[0] >= 0 and ret_val[0] <= 1, f"ret_val should be between 0 and 1 but is {ret_val[0]} in step {i}"

def test_input_output_batch_ClassifyVVAD():
    """
    Test for the correct input and output shapes for ClassifyVVAD pipeline
    """
    averaging_window_size = 2
    weighted = True
    max_consecutive_empty = 5
    stride = 1
    batch_lower_bound = 1
    batch_upper_bound = 10
    vvad = ClassifyVVAD(architecture='LipShape', averaging_window_size=averaging_window_size, weighted=weighted, max_consecutive_empty=max_consecutive_empty, stride=stride)
    for i in range(200):
        ret_val = vvad([test_image]*random.randint(batch_lower_bound, batch_upper_bound)) # input a batch of images - expecting batch of predictions
        if i < IMAGE_INPUT_SHAPE[0]-1:
            assert len(ret_val) <= batch_upper_bound  , f"ret_val should have less than {batch_upper_bound} entries but has {len(ret_val)} entries in step {i}"
            assert ret_val[0] is None, f"first entry of the batch should be [None] but is {ret_val[0]} in step {i}"
        else:
            assert len(ret_val) <= batch_upper_bound  , f"ret_val should have less than {batch_upper_bound} entries but has {len(ret_val)} entries in step {i}"
            assert ret_val[0] is not None, f"ret_val should not be None but is {ret_val[0]} in step {i}"
            assert ret_val[0] >= 0 and ret_val[0] <= 1, f"ret_val should be between 0 and 1 but is {ret_val[0]} in step {i}"


def test_input_output_empty_batch_ClassifyVVAD():
    """
    Test for the correct input and output shapes for ClassifyVVAD pipeline
    """
    averaging_window_size = 2
    weighted = True
    max_consecutive_empty = 5
    stride = 1
    vvad = ClassifyVVAD(architecture='LipShape', averaging_window_size=averaging_window_size, weighted=weighted, max_consecutive_empty=max_consecutive_empty, stride=stride)
    for i in range(200):
        ret_val = vvad([]) # input a batch of images - expecting batch of predictions
        if i < IMAGE_INPUT_SHAPE[0]-1:
            assert len(ret_val) == 0  , f"ret_val should have no entries but has {len(ret_val)} entries in step {i}"
            # assert ret_val[0] is None, f"first entry of the batch should be [None] but is {ret_val[0]} in step {i}"
        else:
            assert len(ret_val) == 0  , f"ret_val should have no entries but has {len(ret_val)} entries in step {i}"
            # assert ret_val[0] is not None, f"ret_val should not be None but is {ret_val[0]} in step {i}"
            # assert ret_val[0] >= 0 and ret_val[0] <= 1, f"ret_val should be between 0 and 1 but is {ret_val[0]} in step {i}"



def test_full_asd_pipeline():
    """
    Test for the correct input and output shapes for the full ASD pipeline
    """
    averaging_window_size = 2
    weighted = True
    max_consecutive_empty = 5
    stride = 1
    asd_pipeline = ASD(architecture='LipShape', averaging_window_size=averaging_window_size, weighted=weighted, max_consecutive_empty=max_consecutive_empty, stride=stride)
    # open video file and read frames as numpy arrays
    video_path = Path(__file__).parent / '001.avi'
    video = cv2.VideoCapture(str(video_path)) 
    assert video.isOpened(), "Video file could not be opened"
    i = 0 
    while True:
        ret, frame = video.read()
        if not ret:
            break
        ret_val = asd_pipeline(frame)
        i += 1
        predictions = 0
        if i < IMAGE_INPUT_SHAPE[0]-1:
            assert len(ret_val) == 2  , f"ret_val should have two entries but has {len(ret_val)} entries in step {i}"
            assert type(ret_val['image']) == np.ndarray, f"ret_val should be an array but is {type(ret_val['image'])} in step {i}"
            for box in ret_val['boxes2D']:
                assert box.class_name == 'No Prediction yet', f"box should not have a prediction yet but has {box.class_name} in step {i}"
                assert box.score == -1.0, f"box should not have a score but is {box.score} in step {i}"
        for box in ret_val['boxes2D']:
            if box.class_name != 'No Prediction yet':
                    predictions += 1
                    assert box.score >= 0.0 and box.score <= 1.0, f"box should have a valid score but is {box.score} in step {i}"
    assert predictions > 0, f"there should be at least one prediction in the video but there are {predictions} predictions"
    video.release()