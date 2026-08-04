'''
Tests for the paz processors needed for the full pipeline
'''

# System imports


# 3rd party imports
import numpy as np
import pytest

# local imports
from asd4hri.processors import *
from tests.constants import *

# end file header
__author__      = 'Adrian Auer'


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
                assert ret_val[2] is not None, f"In timestep {i} the third buffer should be full an returned. Returned {ret_val} in timestep {i} instead."        

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

def test_remove_dangeling_BufferFeatures():
    """
    Test if dangeling buffers are removed.
    """
    buffer = BufferFeatures(input_size=IMAGE_INPUT_SHAPE, stride=1, max_consecutive_empty=5)
    max_buffers = 42
    buffer([None]*max_buffers)
    assert len(buffer.buffers) == max_buffers, f"Should have {max_buffers} internal Buffers, but has {len(buffer.buffers)} instead."
    for i in range(6):
            ret_val = buffer([test_image])
    min_buffes = 1
    assert len(buffer.buffers) == min_buffes, f"Should have {min_buffes} internal Buffers, but has {len(buffer.buffers)} instead."

def test_return_incomplete_samples_BufferFeatures():
    """One Buffer that is filled. It should return every time it is filled but only grow until the size of given timesteps."""
    buffer = BufferFeatures(input_size=IMAGE_INPUT_SHAPE, stride=1, return_incomplete_samples=True)
    for i in range(1000):
        ret_val = buffer([test_image])
        if i < IMAGE_INPUT_SHAPE[0]-1:
            assert len(ret_val[0]) == i+1, f"before {IMAGE_INPUT_SHAPE[0]} timesteps a buffer of length {i+1} should be returned, returned buffer with length {len(ret_val[0])} in timestep {i} instead."
        else:
            ret_val = np.array(ret_val)
            assert ret_val[0].shape == IMAGE_INPUT_SHAPE, f"A full sample with shape {IMAGE_INPUT_SHAPE} should be returned, returned sample with shape {ret_val[0].shape} instead."
    

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
    """
    Test if processor raises an error when not feeded a batch
    """
    shape_features = GetShapeFeatures()
    with pytest.raises(AssertionError, match="Probably not receiving a batch"):
        output_no_batch = shape_features(test_image) # this should crash


def test_no_batch_NormalizeShapeSample():
    """
    Test if processor raises an error when not feeded a batch
    """
    normalizer = NormalizeShapeSample()
    with pytest.raises(IndexError, match="invalid index to scalar variable"):
        output = normalizer(test_lip_features)

def test_NormalizeShapeSample():
    """
    test if the processor returns the correct shapes
    """
    normalizer = NormalizeShapeSample()
    batch_input = [test_lip_features]*5
    batch_output = normalizer(batch_input)
    assert np.array(batch_output).shape == np.array(batch_input).shape


def test_AveragePredictions():
    """
    test if it calculates the average correctly
    """
    average = AveragePredictions(weighted=False)
    element_1 = [1,2,3]
    single_value = [4]
    input_batch = [element_1, single_value]
    excepted_output = [np.mean(element_1), np.mean(single_value)]
    output_batch = average(input_batch)
    assert np.all(np.equal(excepted_output, output_batch)), f"mean should be {excepted_output} but is {output_batch}"

def test_with_Nones_AveragePredictions():
    """
    test if it calculates the average with Nones correctly
    """
    average = AveragePredictions(weighted=False)
    element_1 = [1,2,3]
    single_value = [4]
    input_batch = [None, element_1, None, single_value, None]
    excepted_output = [None, np.mean(element_1), None, np.mean(single_value), None]
    output_batch = average(input_batch)
    assert np.all(np.equal(excepted_output, output_batch)), f"mean should be {excepted_output} but is {output_batch}"

def test_weighted_AveragePredictions():
    """
    test if it calculates the weigthed average correctly
    """
    average = AveragePredictions(weighted=True)
    element_1 = [1,2,3]
    element_2 = [3,2,1]
    single_value = [4]
    input_batch = [element_1, single_value, element_2]
    excepted_output = [np.mean(element_1 * np.arange(1, len(element_1) + 1)), np.mean(single_value), np.mean(element_2 * np.arange(1, len(element_2) + 1))]
    output_batch = average(input_batch)
    assert np.all(np.equal(excepted_output, output_batch)), f"mean should be {excepted_output} but is {output_batch}"    

def test_with_Nones_weighted_AveragePredictions():
    """
    test if it calculates the weighted average with Nones correctly
    """
    average = AveragePredictions(weighted=True)
    element_1 = [1,2,3]
    element_2 = [3,2,1]
    single_value = [4]
    input_batch = [None, element_1, None, single_value, None, element_2, None]
    excepted_output = [None, np.mean(element_1 * np.arange(1, len(element_1) + 1)), None, np.mean(single_value), None, np.mean(element_2 * np.arange(1, len(element_2) + 1)), None]
    output_batch = average(input_batch)
    assert np.all(np.equal(excepted_output, output_batch)), f"mean should be {excepted_output} but is {output_batch}"    

def test_empty_weighted_AveragePredictions():
    """
    test if it calculates the weighted average with Nones correctly
    """
    average = AveragePredictions(weighted=True)
    input_batch = []
    excepted_output = []
    output_batch = average(input_batch)
    assert np.all(np.equal(excepted_output, output_batch)), f"mean should be {excepted_output} but is {output_batch}"    

# TODO: test AveragePredictions with empty batch and with empty buffers in batch

def model(input):
        print(f"Model: {input}")
        return input
def preprocess(input):
    print(f"Preprocess: {input}")
    return input  
def postprocess(input):
    print(f"Postprocess: {input}")
    return input

def test_without_Nones_PredictWithNones():
    """
    Test if prediction works and Nones are forwarded
    """
    predictor = PredictWithNones(model=model, preprocess=preprocess, postprocess=postprocess)
    input_batch = [1, 2, 3]
    for i in range(100):
        input_batch[i % 3] = i
        output_batch = predictor(input_batch)
        assert input_batch == output_batch

def test_without_Nones_pure_np_array_PredictWithNones():
    """
    Test if prediction works and Nones are forwarded
    """
    predictor = PredictWithNones(model=model, preprocess=preprocess, postprocess=postprocess)
    input_batch = np.array([[1, 2], [2, 3], [3, 4]])
    output_batch = predictor(input_batch)
    assert np.all(np.equal(input_batch,output_batch))
        
# TODO: missing the test that fails because of elif None in x:
#       E   ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
def test_without_Nones_batchList_of_np_arrays_PredictWithNones():
    """
    Test if prediction works and Nones are forwarded
    """
    predictor = PredictWithNones(model=model, preprocess=preprocess, postprocess=postprocess)
    input_batch = [np.array([test_lip_features]*38)]
    output_batch = predictor(input_batch)
    assert np.all(np.equal(input_batch,output_batch))


def test_with_Nones_PredictWithNones():
    """
    Test if prediction works and Nones are forwarded
    """
    predictor = PredictWithNones(model=model, preprocess=preprocess, postprocess=postprocess)
    for i in range(100):
        input_batch = [None, None, None]
        input_batch[i % 3] = i
        output_batch = predictor(input_batch)
        assert input_batch == output_batch