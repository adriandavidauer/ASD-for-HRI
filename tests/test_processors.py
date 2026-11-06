"""Tests for the paz processors needed for the full pipeline."""

# System imports


# 3rd party imports
import random

import numpy as np
import pytest
from paz.backend.image import show_image


# local imports
from asd4hri.processors import *
from tests.constants import *

# end file header
__author__      = 'Adrian Auer'

def test_empty_batch_BufferFeatures():
    """One Buffer that is getting empty batches. It should return only after the given timesteps."""
    buffer = BufferFeatures(input_size=IMAGE_INPUT_SHAPE, stride=1)
    for i in range(1000):
        ret_val = buffer([])    
        assert len(ret_val) == 0, f"empty input should return empty output, returned {len(ret_val)} in timestep {i} instead."
        
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
    """Testing if the max_consecutive_empty parameter triggers acutually reseting the buffer."""
    buffer = BufferFeatures(input_size=IMAGE_INPUT_SHAPE, stride=1)
    for i in range(1000):
        input_list = [None, None]
        if i%6 == 0:
            input_list[1] = test_image # type: ignore
        input_list[0] = test_image # type: ignore
        ret_val = buffer(input_list)
        assert ret_val[1] is None, f"Second buffer should always be None, because max_consecutive_empty is smaller than each step in which the buffer is filled"

def test_reset_single_BufferFeatures():
    """Testing if resting the buffer works - filling a buffer and reset it every x steps where x is smaller timesteps - buffer should always return None."""
    buffer = BufferFeatures(input_size=IMAGE_INPUT_SHAPE, stride=1)
    for i in range(1000):
        ret_val = buffer([test_image])
        if i % (IMAGE_INPUT_SHAPE[0] - 2):
            buffer.reset(0)
        assert ret_val[0] is None, f"before {IMAGE_INPUT_SHAPE[0]} timesteps None should be returned, returned {type(ret_val[0])} in timestep {i} instead."

def test_reset_all_BufferFeatures():
    """Testing if resting the buffer works - filling a buffer and reset it every x steps where x is smaller timesteps - buffer should always return None."""
    buffer = BufferFeatures(input_size=IMAGE_INPUT_SHAPE, stride=1)
    for i in range(1000):
        input_list = [None, None, None]
        input_list[i%3] = test_image # type: ignore
        ret_val = buffer(input_list)
        if i % (IMAGE_INPUT_SHAPE[0] - 2):
            buffer.reset()
        assert ret_val == [None, None, None], f"before {(IMAGE_INPUT_SHAPE[0]-1)*3} timesteps [None, None, None] should be returned, returned [{type(ret_val[0])},{type(ret_val[1])},{type(ret_val[2])}]  in timestep {i} instead."

def test_remove_dangeling_BufferFeatures():
    """Test if dangeling buffers are removed."""
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
    """Test if the processor returns the correct shapes."""
    shape_features = GetShapeFeatures()
    for i in range(5):
        input_batch = [test_image]*i
        output_batch = shape_features(input_batch)
        assert len(output_batch) == len(input_batch), f"Output batch should have the same length as the input batch but they are: \ninput: {len(input_batch)}\noutput: {len(output_batch)}"

def test_no_batch_GetShapeFeatures():
    """Test if processor raises an error when not feeded a batch."""
    shape_features = GetShapeFeatures()
    with pytest.raises(AssertionError, match="Probably not receiving a batch"):
        output_no_batch = shape_features(test_image) # this should crash


def test_no_batch_NormalizeShapeSample():
    """Test if processor raises an error when not feeded a batch."""
    normalizer = NormalizeShapeSample()
    with pytest.raises(IndexError, match="invalid index to scalar variable"):
        output = normalizer(test_lip_features)

def test_NormalizeShapeSample():
    """Test if the processor returns the correct shapes."""
    normalizer = NormalizeShapeSample()
    batch_input = [test_lip_features]*5
    batch_output = normalizer(batch_input)
    assert np.array(batch_output).shape == np.array(batch_input).shape


def test_AveragePredictions():
    """Test if it calculates the average correctly."""
    average = AveragePredictions(weighted=False)
    element_1 = [1,2,3]
    single_value = [4]
    input_batch = [element_1, single_value]
    excepted_output = [np.mean(element_1), np.mean(single_value)]
    output_batch = average(input_batch)
    assert np.all(np.equal(excepted_output, output_batch)), f"mean should be {excepted_output} but is {output_batch}"

def test_with_Nones_AveragePredictions():
    """Test if it calculates the average with Nones correctly."""
    average = AveragePredictions(weighted=False)
    element_1 = [1,2,3]
    single_value = [4]
    input_batch = [None, element_1, None, single_value, None]
    excepted_output = [None, np.mean(element_1), None, np.mean(single_value), None]
    output_batch = average(input_batch)
    assert np.all(np.equal(excepted_output, output_batch)), f"mean should be {excepted_output} but is {output_batch}"

def test_weighted_AveragePredictions():
    """Test if it calculates the weigthed average correctly."""
    average = AveragePredictions(weighted=True)
    element_1 = [0,1,2,3]
    element_2 = [3,2,1,0]
    single_value = [4]
    input_batch = [element_1, single_value, element_2]
    excepted_output = [np.mean(element_1 * np.arange(1, len(element_1) + 1)), np.mean(single_value), np.mean(element_2 * np.arange(1, len(element_2) + 1))]
    output_batch = average(input_batch)
    assert np.all(np.equal(excepted_output, output_batch)), f"mean should be {excepted_output} but is {output_batch}"    

def test_with_Nones_weighted_AveragePredictions():
    """Test if it calculates the weighted average with Nones correctly."""
    average = AveragePredictions(weighted=True)
    element_1 = [1,2,3]
    element_2 = [3,2,1]
    single_value = [4]
    input_batch = [None, element_1, None, single_value, None, element_2, None]
    excepted_output = [None, np.mean(element_1 * np.arange(1, len(element_1) + 1)), None, np.mean(single_value), None, np.mean(element_2 * np.arange(1, len(element_2) + 1)), None]
    output_batch = average(input_batch)
    assert np.all(np.equal(excepted_output, output_batch)), f"mean should be {excepted_output} but is {output_batch}"    

def test_empty_batch_AveragePredictions():
    """Test if it calculates the (weighted) average with an empty batch correctly."""
    # weighted
    average = AveragePredictions(weighted=True)
    input_batch = []
    excepted_output = []
    output_batch = average(input_batch)
    assert np.all(np.equal(excepted_output, output_batch)), f"mean should be {excepted_output} but is {output_batch}"
    # unweighted
    average = AveragePredictions(weighted=False)
    output_batch = average(input_batch)
    assert np.all(np.equal(excepted_output, output_batch)), f"mean should be {excepted_output} but is {output_batch}"

def test_empty_buffer_in_batch_AveragePredictions():
    """Test if it calculates the (weighted) average with an empty buffer in batch correctly."""
    # weighted
    average = AveragePredictions(weighted=True)
    element_1 = [1,2,3]
    element_2 = [3,2,1]
    empty_value = []
    input_batch = [None, element_1, None, empty_value, None, element_2, None]
    excepted_output = [None, np.mean(element_1 * np.arange(1, len(element_1) + 1)), None, None, None, np.mean(element_2 * np.arange(1, len(element_2) + 1)), None]
    output_batch = average(input_batch)
    assert np.all(np.equal(excepted_output, output_batch)), f"mean should be {excepted_output} but is {output_batch}"    
    # unweighted
    average = AveragePredictions(weighted=False)
    output_batch = average(input_batch)
    excepted_output = [None, np.mean(element_1), None, None, None, np.mean(element_2), None]
    assert np.all(np.equal(excepted_output, output_batch)), f"mean should be {excepted_output} but is {output_batch}"

def test_only_empty_buffer_in_batch_AveragePredictions():
    """Test if it calculates the (weighted) average with only an empty buffer in batch correctly."""
    # weighted
    average = AveragePredictions(weighted=True)
    input_batch = [[]]
    excepted_output = [None]
    output_batch = average(input_batch)
    assert np.all(np.equal(excepted_output, output_batch)), f"mean should be {excepted_output} but is {output_batch}"    
    # unweighted
    average = AveragePredictions(weighted=False)
    output_batch = average(input_batch)
    assert np.all(np.equal(excepted_output, output_batch)), f"mean should be {excepted_output} but is {output_batch}"

def test_normalize_AveragePredictions():
    """Test if it calculates the (weighted) average with normalization correctly."""
    # weighted
    average = AveragePredictions(weighted=True, normalize=True)
    element_1 = [1,1,1]
    element_2 = [0.0,0.0,0.0]
    single_value = [0.1]
    element_3 = [0.1,0.2,0.3]
    max_for_element_3 = np.sum(np.arange(1, len(element_3) + 1))
    input_batch = [element_1, single_value, element_2, element_3]
    excepted_output = [1.0, 0.1, 0.0, np.mean(element_3 * np.arange(1, len(element_3) + 1))/(max_for_element_3/len(element_3))]
    output_batch = average(input_batch)
    assert np.all(np.equal(excepted_output, output_batch)), f"mean should be {excepted_output} but is {output_batch}"    

def model(input):
        print(f"Model: {input}")
        return input
def preprocess(input):
    print(f"Preprocess: {input}")
    return input  
def postprocess(input):
    print(f"Postprocess: {input}")
    return input

def test_empty_batch_PredictWithNones():
    """Test if prediction works with an empty batch."""
    predictor = PredictWithNones(model=model, preprocess=preprocess, postprocess=postprocess)
    input_batch = []
    for i in range(100):
        output_batch = predictor(input_batch)
        assert np.all(np.equal(input_batch, output_batch)), f"predictions should be {input_batch} but is {output_batch}"
 

def test_without_Nones_PredictWithNones():
    """Test if prediction works and Nones are forwarded."""
    predictor = PredictWithNones(model=model, preprocess=preprocess, postprocess=postprocess)
    input_batch = [1, 2, 3]
    for i in range(100):
        input_batch[i % 3] = i
        output_batch = predictor(input_batch)
        assert np.all(np.equal(input_batch, output_batch)), f"predictions should be {input_batch} but is {output_batch}"
        
def test_without_Nones_pure_np_array_PredictWithNones():
    """Test if prediction works and Nones are forwarded."""
    predictor = PredictWithNones(model=model, preprocess=preprocess, postprocess=postprocess)
    input_batch = np.array([[1, 2], [2, 3], [3, 4]])
    output_batch = predictor(input_batch)
    assert np.all(np.equal(input_batch,output_batch))
        
def test_without_Nones_batchList_of_np_arrays_PredictWithNones():
    """Test if prediction works and Nones are forwarded."""
    predictor = PredictWithNones(model=model, preprocess=preprocess, postprocess=postprocess)
    input_batch = [np.array([test_lip_features]*38)]
    output_batch = predictor(input_batch)
    assert np.all(np.equal(np.array(input_batch),np.array(output_batch))), f"predictions should be {input_batch} but is {output_batch}"


def test_with_Nones_PredictWithNones():
    """Test if prediction works and Nones are forwarded."""
    predictor = PredictWithNones(model=model, preprocess=preprocess, postprocess=postprocess)
    for i in range(100):
        input_batch = [None, None, None]
        input_batch[i % 3] = i
        output_batch = predictor(input_batch)
        assert input_batch == output_batch

def test_flatten_predictions():
    """Test if flattening predictions works."""
    input_batch = np.array([[1]]) # [np.array([[1], [2], [3]]), np.array([[4], [5], [6]])]
    output_batch = flatten_predictions(input_batch)
    expected_output = np.array([1]) # [np.array([1, 2, 3]), np.array([4, 5, 6])]
    assert np.all(np.equal(output_batch, expected_output)), f"flattened predictions should be {expected_output} but is {output_batch}"

def test_flatten_predictions_with_Nones():
    """Test if flattening predictions works with Nones."""
    input_batch = [np.array([[1]]), None, np.array([[3]])]
    output_batch = flatten_predictions(input_batch)
    expected_output = np.array([1, None, 3]) # [np.array([1, 2, 3]), np.array([4, 5, 6])]
    assert np.all(np.equal(output_batch, expected_output)), f"flattened predictions should be {expected_output} but is {output_batch}"


def test_FaceDetectorYN_output():
    """Test if the output with a test image is correct."""
    image = cv2.imread(str(Path(__file__).parent / 'researcher_faces.jpeg'))
    orig_img = image.copy()
    h, w, _ = image.shape
    detector = FaceDetectorYN(input_size=(w, h), draw=False)
    output = detector(image)
    assert np.all(np.equal(output['image'], orig_img)), 'image forwarding does not work correctly' 
    assert type(output['boxes2D']) == list
    expected_boxes2D = [Box2D((376, 49, 424, 112), 0.9374542236328125, 'Face'), Box2D((189, 38, 231, 95), 0.9337112307548523, 'Face'), Box2D((87, 70, 126, 130), 0.9286843538284302, 'Face'), Box2D((461, 67, 509, 126), 0.9188944101333618, 'Face'), Box2D((289, 33, 338, 93), 0.9140876531600952, 'Face')]
    for exp_box, box in zip(expected_boxes2D, output['boxes2D']):
        assert exp_box.coordinates == box.coordinates
        assert exp_box.score == box.score
        assert exp_box.class_name == box.class_name

def test_FaceDetectorYN_output_no_inpt_size():
    """Test if the output with a test image is correct even if no input_size is given (automatic setting during runtime)."""
    image = cv2.imread(str(Path(__file__).parent / 'researcher_faces.jpeg'))
    orig_img = image.copy()
    h, w, _ = image.shape
    detector = FaceDetectorYN()
    output = detector(image)
    assert np.all(np.equal(output['image'], orig_img)), 'image forwarding does not work correctly' 
    assert type(output['boxes2D']) == list
    expected_boxes2D = [Box2D((376, 49, 424, 112), 0.9374542236328125, 'Face'), Box2D((189, 38, 231, 95), 0.9337112307548523, 'Face'), Box2D((87, 70, 126, 130), 0.9286843538284302, 'Face'), Box2D((461, 67, 509, 126), 0.9188944101333618, 'Face'), Box2D((289, 33, 338, 93), 0.9140876531600952, 'Face')]
    for exp_box, box in zip(expected_boxes2D, output['boxes2D']):
        assert exp_box.coordinates == box.coordinates
        assert exp_box.score == box.score
        assert exp_box.class_name == box.class_name


def test_PreprocessImages():
    """Test if the processor returns the correct output shape and range for values."""
    n = random.randint(1,10) # number of images
    w = random.randint(0, 1024)
    h = random.randint(0, 1024)
    preprocess_image = PreprocessImages((w, h), normalize=True)
    output_images = preprocess_image([test_image] * n)
    for output_image in output_images:
        assert output_image.shape == (h,w,3), "should be resized to given shape"
        assert output_image.dtype == float, "should be casted to float"
        assert np.all((output_image.min() >= 0) and (output_image.max() <= 1))

def test_PreprocessImages_no_normalization():
    """Test if the processor returns the correct output shape and range for values."""
    n = random.randint(1,10) # number of images
    w = random.randint(0, 1024)
    h = random.randint(0, 1024)
    preprocess_image = PreprocessImages((w, h), normalize=False)
    output_images = preprocess_image([test_image] * n)
    for output_image in output_images:
        assert output_image.shape == (h,w,3), "should be resized to given shape"
        assert output_image.dtype == np.uint8, "should be uint8"
        assert np.all((output_image.min() >= 0) and (output_image.max() <= 255))

def test_PreprocessImages_with_Nones():
    """Test if the processor returns the correct output shape and range for values."""
    n = random.randint(1,10) # number of images
    w = random.randint(0, 1024)
    h = random.randint(0, 1024)
    preprocess_image = PreprocessImages((w, h), normalize=True)
    input_images = [None, test_image, None] * n
    output_images = preprocess_image(input_images)
    for input_image, output_image in zip(input_images, output_images):
        if input_image is None:
            assert output_image is None, "None should be forwarded but is not"
        else:
            assert output_image.shape == (h,w,3), "should be resized to given shape"
            assert output_image.dtype == float, "should be casted to float"
            assert np.all((output_image.min() >= 0) and (output_image.max() <= 1))


def test_FaceDetectorRetinaFace_output():
    """Test if the output with a test image is correct."""
    image = cv2.imread(str(Path(__file__).parent / 'researcher_faces.jpeg'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_img = image.copy()

    h, w, _ = image.shape
    detector = FaceDetectorRetinaFace(draw=False)
    output = detector(image)
    # show_image(output['image'], 'test', True)
    assert np.all(np.equal(output['image'], orig_img)), 'image forwarding does not work correctly' 
    assert type(output['boxes2D']) == list
    expected_boxes2D = [Box2D((184, 39, 234, 92), 0.987154483795166, 'Face'), 
                        Box2D((460, 68, 508, 125), 0.9813412427902222, 'Face'), 
                        Box2D((85, 72, 135, 132), 0.9796932935714722, 'Face'), 
                        Box2D((373, 50, 424, 110), 0.9790538549423218, 'Face'), 
                        Box2D((286, 39, 339, 92), 0.9758303165435791, 'Face')]
    for exp_box, box in zip(expected_boxes2D, output['boxes2D']):
        assert exp_box.coordinates == box.coordinates
        assert exp_box.score == box.score
        assert exp_box.class_name == box.class_name


def test_stride_BufferFeatures():
    """A full buffer is returned first after `timesteps` inputs and then after every `stride` inputs."""
    buffer = BufferFeatures(input_size=(4,), stride=3)
    returned = {}
    for step in range(1, 11):
        ret_val = buffer([step])
        if ret_val[0] is not None:
            returned[step] = list(ret_val[0])
    assert returned == {4: [1, 2, 3, 4], 7: [4, 5, 6, 7], 10: [7, 8, 9, 10]}


def test_stride_larger_than_buffer_BufferFeatures():
    """A stride larger than the number of timesteps is rejected."""
    with pytest.raises(ValueError, match='Buffer size must be equal or larger than stride'):
        BufferFeatures(input_size=(4,), stride=5)


def test_max_consecutive_empty_clears_buffer_BufferFeatures():
    """An entity missing for `max_consecutive_empty` steps starts over; shorter gaps keep the buffer."""
    buffer = BufferFeatures(input_size=(3,), stride=1, max_consecutive_empty=2)
    buffer([1])
    buffer([2])
    buffer([None])  # one gap is tolerated
    assert buffer([3]) == [buffer.buffers[0]] and list(buffer.buffers[0]) == [1, 2, 3]
    buffer([None])
    buffer([None])  # second consecutive gap clears the buffer
    assert list(buffer.buffers[0]) == []
    assert buffer([4]) == [None]


def test_entities_appear_and_disappear_BufferFeatures():
    """Trailing entities that disappear keep their buffer for a short gap and are removed after a long one."""
    buffer = BufferFeatures(input_size=(2,), stride=1, max_consecutive_empty=2)
    buffer(['a1', 'b1'])
    buffer(['a2'])  # b missing once: kept
    assert len(buffer.buffers) == 2
    assert buffer(['a3', 'b2'])[1] is not None and list(buffer.buffers[1]) == ['b1', 'b2']
    buffer(['a4'])
    buffer(['a5'])  # b missing twice: removed
    assert len(buffer.buffers) == 1
    assert buffer(['a6', 'c1']) == [buffer.buffers[0], None]  # new entity starts empty
    assert list(buffer.buffers[1]) == ['c1']


def test_empty_batch_ages_all_buffers_BufferFeatures():
    """Frames without any entity count as missing for every buffer, including the first one."""
    buffer = BufferFeatures(input_size=(2,), stride=1, max_consecutive_empty=2)
    buffer(['a1'])
    buffer([])
    buffer([])
    assert len(buffer.buffers) == 0
    assert buffer(['a2']) == [None]  # no stale features from before the gap


def test_missing_middle_entity_keeps_later_buffer_BufferFeatures():
    """A dropped entity that is not the last one is cleared without removing the buffers after it."""
    buffer = BufferFeatures(input_size=(3,), stride=1, max_consecutive_empty=2)
    buffer(['a1', None, 'c1'])  # b missed once, c present
    buffer(['a2'])  # b missed twice, c once
    assert len(buffer.buffers) == 3
    assert list(buffer.buffers[1]) == []
    buffer(['a3', 'b1', 'c2'])
    assert list(buffer.buffers[2]) == ['c1', 'c2']


def test_None_batch_AveragePredictions():
    """A None batch is forwarded as None."""
    assert AveragePredictions()(None) is None


def test_different_lengths_weighted_AveragePredictions():
    """Shorter buffers in a batch are not distorted by the zero padding of longer ones."""
    average = AveragePredictions(weighted=True, normalize=True)
    output_batch = average([[0.6], None, [0.0, 0.5, 1.0]])
    expected = [0.6, None, np.average([0.0, 0.5, 1.0], weights=[1, 2, 3])]
    assert output_batch[1] is None
    np.testing.assert_allclose([output_batch[0], output_batch[2]], [expected[0], expected[2]])


def test_normalized_weighted_is_weighted_average_AveragePredictions():
    """Normalized weighted averaging equals the average weighted by the positions 1..n."""
    values = [0.2, 0.9, 0.4, 0.7]
    output_batch = AveragePredictions(weighted=True, normalize=True)([values])
    np.testing.assert_allclose(output_batch, [np.average(values, weights=[1, 2, 3, 4])])


class RecordingModel:
    """Fake model that records its inputs and returns a ``tf.Tensor`` of doubled inputs."""

    def __init__(self):
        """Start without recorded calls."""
        self.calls = []

    def __call__(self, x):
        """Record the batch and predict ``2 * x``.

        Args:
            x: Batch as a numpy array.

        Returns:
            tf.Tensor: The doubled batch.
        """
        self.calls.append(x)
        return tf.constant(2 * x)


def test_model_sees_only_non_None_PredictWithNones():
    """Only non-None samples reach the model and predictions are mapped back to their positions."""
    model = RecordingModel()
    output_batch = PredictWithNones(model=model)([None, 1.5, None, 3.0])
    assert len(model.calls) == 1
    np.testing.assert_array_equal(model.calls[0], [1.5, 3.0])
    assert output_batch[0] is None and output_batch[2] is None
    assert output_batch[1] == 3.0 and output_batch[3] == 6.0


def test_tensor_output_and_postprocess_PredictWithNones():
    """A ``tf.Tensor`` prediction is converted to numpy before postprocessing."""
    seen = []
    def postprocess(y):
        """Record the type of the predictions.

        Args:
            y: Predictions of the model.

        Returns:
            The unchanged predictions.
        """
        seen.append(type(y))
        return y
    output_batch = PredictWithNones(model=RecordingModel(), postprocess=postprocess)(np.array([[1.0], [2.0]]))
    assert seen == [np.ndarray]
    np.testing.assert_array_equal(output_batch, [[2.0], [4.0]])


def test_skip_model_and_postprocess_PredictWithNones():
    """None batches and batches of only None skip the model and the postprocessing."""
    model = RecordingModel()
    def postprocess(y):
        """Fail if called.

        Args:
            y: Predictions of the model.

        Raises:
            AssertionError: Always.
        """
        raise AssertionError('postprocess must not be called')
    predictor = PredictWithNones(model=model, preprocess=lambda x: x, postprocess=postprocess)
    assert predictor(None) is None
    assert predictor([None, None]) == [None, None]
    assert model.calls == []


def test_model_output_flatten_predictions():
    """A model output of shape (N, 1) is flattened to (N,)."""
    output_batch = flatten_predictions(np.array([[0.1], [0.7], [0.4]]))
    assert output_batch.shape == (3,)
    np.testing.assert_allclose(output_batch, [0.1, 0.7, 0.4])


def test_sort_FaceTracker():
    """Boxes are ordered by their left edge, then by their top edge, independent of the score order."""
    boxes2D = [Box2D((50, 10, 60, 20), 0.9, 'Face'),
               Box2D((5, 40, 15, 50), 0.8, 'Face'),
               Box2D((5, 30, 15, 40), 0.7, 'Face')]
    tracked = FaceTracker(approach='sort')(boxes2D)
    assert [box.coordinates for box in tracked] == [(5, 30, 15, 40), (5, 40, 15, 50), (50, 10, 60, 20)]
    assert FaceTracker()([]) == []


def test_invalid_approach_FaceTracker():
    """An unknown tracking approach is rejected."""
    with pytest.raises(ValueError, match='Approach must be one of'):
        FaceTracker(approach='kalman')


def test_face_recognition_FaceTracker():
    """Face recognition based tracking is not implemented yet."""
    with pytest.raises(NotImplementedError):
        FaceTracker(approach='face_recognition')([])


def test_AddClassAndScoreToBoxes():
    """Scores above the threshold get the positive class, missing scores get 'No Prediction yet'."""
    boxes2D = [Box2D((0, 0, 10, 10), 0.9, 'Face') for _ in range(3)]
    add_class = AddClassAndScoreToBoxes(lambda images: [None, 0.7, 0.3], ['not speaking', 'speaking'],
                                        decision_threshold=0.5)
    boxes2D = add_class([test_image] * 3, boxes2D)
    assert [(box.class_name, box.score) for box in boxes2D] == [
        ('No Prediction yet', -1.0), ('speaking', 0.7), ('not speaking', 0.3)]


def test_non_binary_AddClassAndScoreToBoxes():
    """Only two class names are supported."""
    with pytest.raises(AssertionError, match='only supports binary classes'):
        AddClassAndScoreToBoxes(lambda images: [], ['a', 'b', 'c'])
