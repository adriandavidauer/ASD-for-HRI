'''
All paz processors needed for the full pipeline
'''

# System imports


# 3rd party imports
import numpy as np
import pytest

# local imports
from asd4hri.processors import *

# end file header
__author__      = 'Adrian Auer'
IMAGE_INPUT_SHAPE=(38, 96, 96, 3) # Example shape of an RGB image
test_image = np.empty(IMAGE_INPUT_SHAPE[1:], dtype=np.uint8) # example RGB image 
LIP_FEATURE_INPUT_SHAPE = (38, 20, 2)
test_lip_features = np.empty(LIP_FEATURE_INPUT_SHAPE)

def test_return_simple_BufferFeatures():
    """One Buffer that is filled. It should return only after the given timesteps."""
    buffer = BufferFeatures(input_size=IMAGE_INPUT_SHAPE, stride=1)
    for i in range(1000):
        ret_val = buffer([test_image])
        if i < IMAGE_INPUT_SHAPE[0]-1:
            assert ret_val[0] is None, f"before {IMAGE_INPUT_SHAPE[0]} timesteps None should be returned, returned {type(ret_val[0])} in timestep {i} instead."
        else:
            ret_val = np.array(ret_val)
            assert ret_val[0].shape == IMAGE_INPUT_SHAPE, f"A full sample with shape {IMAGE_INPUT_SHAPE} should be returned, returned sample with shape {ret_val[0].shape} instead."

def test_return_batched_BufferFeatures():
    """Multiple Buffer that are filled alternating. Each should return only after the given timesteps are filled into the buffer."""
    buffer = BufferFeatures(input_size=IMAGE_INPUT_SHAPE, stride=1)
    for i in range(1000):
        input_list = [None, None, None]
        input_list[i%3] = test_image # type: ignore
        ret_val = buffer(input_list)
        if i < (IMAGE_INPUT_SHAPE[0]-1)*3 :
            assert ret_val == [None, None, None], f"before {(IMAGE_INPUT_SHAPE[0]-1)*3} timesteps [None, None, None] should be returned, returned [{type(ret_val[0])},{type(ret_val[1])},{type(ret_val[2])}]  in timestep {i} instead."
        else:
            if i%3 == 0:
                assert ret_val[0] is not None, f"In timestep {i} the first buffer should be full an returned. Returned {ret_val} in timestep {i} instead."
            if i%3 == 1:
                assert ret_val[1] is not None, f"In timestep {i} the second buffer should be full an returned. Returned {ret_val} in timestep {i} instead."       
            if i%3 == 2:
                assert ret_val[1] is not None, f"In timestep {i} the third buffer should be full an returned. Returned {ret_val} in timestep {i} instead."        

def test_auto_reset_BufferFeatures():
    """
    Testing if the max_consecutive_empty parameter triggers acutually reseting the buffer.
    """
    buffer = BufferFeatures(input_size=IMAGE_INPUT_SHAPE, stride=1)
    for i in range(1000):
        input_list = [None, None]
        if i%6 == 0:
            input_list[1] = test_image # type: ignore
        input_list[0] = test_image # type: ignore
        ret_val = buffer(input_list)
        assert ret_val[1] is None, f"Second buffer should always be None, because max_consecutive_empty is smaller than each step in which the buffer is filled"

def test_reset_single_BufferFeatures():
    """
    Testing if resting the buffer works - filling a buffer and reset it every x steps where x is smaller timesteps - buffer should always return None.
    """
    buffer = BufferFeatures(input_size=IMAGE_INPUT_SHAPE, stride=1)
    for i in range(1000):
        ret_val = buffer([test_image])
        if i % (IMAGE_INPUT_SHAPE[0] - 2):
            buffer.reset(0)
        assert ret_val[0] is None, f"before {IMAGE_INPUT_SHAPE[0]} timesteps None should be returned, returned {type(ret_val[0])} in timestep {i} instead."


def test_reset_all_BufferFeatures():
    """
    Testing if resting the buffer works - filling a buffer and reset it every x steps where x is smaller timesteps - buffer should always return None.
    """
    buffer = BufferFeatures(input_size=IMAGE_INPUT_SHAPE, stride=1)
    for i in range(1000):
        input_list = [None, None, None]
        input_list[i%3] = test_image # type: ignore
        ret_val = buffer(input_list)
        if i % (IMAGE_INPUT_SHAPE[0] - 2):
            buffer.reset()
        assert ret_val == [None, None, None], f"before {(IMAGE_INPUT_SHAPE[0]-1)*3} timesteps [None, None, None] should be returned, returned [{type(ret_val[0])},{type(ret_val[1])},{type(ret_val[2])}]  in timestep {i} instead."


def test_GetShapeFeatures():
    """
    test if the processor returns the correct shapes
    """
    shape_features = GetShapeFeatures()
    for i in range(5):
        input_batch = [test_image]*i
        output_batch = shape_features(input_batch)
        assert len(output_batch) == len(input_batch), f"Output batch should have the same length as the input batch but they are: \ninput: {len(input_batch)}\noutput: {len(output_batch)}"

def test_no_batch_GetShapeFeatures():
    shape_features = GetShapeFeatures()
    with pytest.raises(AssertionError, match="Probably not receiving a batch"):
        output_no_batch = shape_features(test_image) # this should crash


def test_no_batch_NormalizeShapeSample():
    normalizer = NormalizeShapeSample()
    with pytest.raises(AssertionError, match="invalid index to scalar variable"):
        output = normalizer(test_lip_features)

def test_NormalizeShapeSample():
    normalizer = NormalizeShapeSample()
    batch_input = [test_lip_features]*5
    batch_output = normalizer(batch_input)
    assert batch_output.shape == np.array(batch_input).shape