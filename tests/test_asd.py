'''
Tests for the main asd processors
'''

# System imports


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
        ret_val = vvad([test_image]) # input a batch of images - expecting batch of predictions
        if i < IMAGE_INPUT_SHAPE[0]-1:
            assert len(ret_val) == 1  , f"ret_val should have one entry but has {len(ret_val)} entries in step {i}"
            assert ret_val[0] is None, f"ret_val should be [None] but is {ret_val[0]} in step {i}"
        else:
            assert len(ret_val) == 1  , f"ret_val should have one entry but has {len(ret_val)} entries in step {i}"
            assert ret_val[0] is not None, f"ret_val should not be None but is {ret_val[0]} in step {i}"
            assert ret_val[0] >= 0 and ret_val[0] <= 1, f"ret_val should be between 0 and 1 but is {ret_val[0]} in step {i}"

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
    video = cv2.VideoCapture('001.avi') #TODO: reading video does not work :DDDDD
    i = 0 
    while True:
        ret, frame = video.read()
        if not ret:
            break
        ret_val = asd_pipeline(frame)
        i += 1
        if i < IMAGE_INPUT_SHAPE[0]-1:
            assert len(ret_val) == 2  , f"ret_val should have two entries but has {len(ret_val)} entries in step {i}"
            assert ret_val[1] is None, f"ret_val should be [None] but is {ret_val[1]} in step {i}"
        else:
            # TODO: whatv is returned? the pipeline should return the image and the boundingboxes with the predictions/scores
            assert len(ret_val) == 2  , f"ret_val should have two entries but has {len(ret_val)} entries in step {i}"
            assert ret_val[1] is not None, f"ret_val should not be None but is {ret_val[1]} in step {i}"
            assert ret_val[1] >= 0 and ret_val[1] <= 1, f"ret_val should be between 0 and 1 but is {ret_val[1]} in step {i}"
    video.release()

    # for i in range(200):
    #     ret_val = asd_pipeline([test_image]) # input a batch of images - expecting batch of predictions
    #     if i < IMAGE_INPUT_SHAPE[0]-1:
    #         assert len(ret_val) == 1  , f"ret_val should have one entry but has {len(ret_val)} entries in step {i}"
    #         assert ret_val[0] is None, f"ret_val should be [None] but is {ret_val[0]} in step {i}"
    #     else:
    #         assert len(ret_val) == 1  , f"ret_val should have one entry but has {len(ret_val)} entries in step {i}"
    #         assert ret_val[0] is not None, f"ret_val should not be None but is {ret_val[0]} in step {i}"
    #         assert ret_val[0] >= 0 and ret_val[0] <= 1, f"ret_val should be between 0 and 1 but is {ret_val[0]} in step {i}"