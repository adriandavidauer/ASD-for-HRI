'''
All paz processors needed for the full pipeline
'''

# System imports


# 3rd party imports
import numpy as np

# local imports
from asd4hri.processors import BufferFeatures

# end file header
__author__      = 'Adrian Auer'
INPUT_SHAPE=(38, 96, 96, 3)
test_data = np.empty(INPUT_SHAPE)

def test_BufferFeatures():
    buffer = BufferFeatures(input_size=INPUT_SHAPE)
