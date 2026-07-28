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
    vvad = ClassifyVVAD(input_size=IMAGE_INPUT_SHAPE, architecture='LipShape')
    for i in range(1000):
        vvad([test_image])
