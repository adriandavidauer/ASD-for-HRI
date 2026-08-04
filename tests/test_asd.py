'''
Tests for the main asd processors
'''

# System imports


# 3rd party imports
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
    averaging_window_size = 5
    weighted = True
    vvad = ClassifyVVAD(input_size=IMAGE_INPUT_SHAPE, architecture='LipShape', averaging_window_size=averaging_window_size, weighted=weighted)
    for i in range(1000):
        ret_val = vvad([test_image]) # input a batch of images - expecting batch of predictions
        if i < IMAGE_INPUT_SHAPE[0]:
            assert len(ret_val) == 1  , f"ret_val should have one entry but has {len(ret_val)} entries"
            assert ret_val[0] is None, f"ret_val should be [None] but is {ret_val[0]}"
        else:
            assert len(ret_val) == 1  , f"ret_val should have one entry but has {len(ret_val)} entries"
            assert ret_val[0] is not None, f"ret_val should not be None but is {ret_val[0]}"
            assert ret_val[0] >= 0 and ret_val[0] <= 1, f"ret_val should be between 0 and 1 but is {ret_val[0]}"
