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
test_feature = np.empty(INPUT_SHAPE[1:])

def test_return_simple_BufferFeatures():
    buffer = BufferFeatures(input_size=INPUT_SHAPE, stride=1)
    for i in range(1000):
        ret_val = buffer([test_feature])
        if i < INPUT_SHAPE[0]-1:
            assert ret_val[0] is None, f"before {INPUT_SHAPE[0]} timesteps None should be returned, returned {type(ret_val[0])} in timestep {i} instead."
        else:
            ret_val = np.array(ret_val)
            assert ret_val[0].shape == INPUT_SHAPE, f"A full sample with shape {INPUT_SHAPE} should be returned, returned sample with shape {ret_val[0].shape} instead."

def test_return_batched_BufferFeatures():
    buffer = BufferFeatures(input_size=INPUT_SHAPE, stride=1)
    for i in range(1000):
        input_list = [None, None, None]
        input_list[i%3] = test_feature
        ret_val = buffer(input_list)
        if i < (INPUT_SHAPE[0]-1)*3 :
            assert ret_val == [None, None, None], f"before {INPUT_SHAPE[0]*2} timesteps [None, None, None] should be returned, returned [{type(ret_val[0])},{type(ret_val[1])},{type(ret_val[2])}]  in timestep {i} instead."
        else:
            if i%3 == 0:
                assert ret_val[0] is not None, f"In timestep {i} the first buffer should be full an returned. Returned {ret_val} in timestep {i} instead."
            if i%3 == 1:
                assert ret_val[1] is not None, f"In timestep {i} the second buffer should be full an returned. Returned {ret_val} in timestep {i} instead."       
            if i%3 == 2:
                assert ret_val[1] is not None, f"In timestep {i} the third buffer should be full an returned. Returned {ret_val} in timestep {i} instead."        

def test_reset_BufferFeatures():
    buffer = BufferFeatures(input_size=INPUT_SHAPE, stride=1)
    for i in range(1000):
        input_list = [None, None]
        if i%6 == 0:
            input_list[1] = test_feature
        input_list[0] = test_feature
        ret_val = buffer(input_list)
        assert ret_val[1] is None, f"Second buffer should always be None, because max_consecutive_empty is smaller than each step in which the buffer is filled"

# TODO: sorted list in sorted list out might run into problems when using tracking
# TODO: dangling buffers need to be removed completely otherwise list grows and grows