"""Tests for the main asd processors."""

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
    """Test for the correct input and output shapes for ClassifyVVAD pipeline."""
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
    """Test for the correct input and output shapes for ClassifyVVAD pipeline."""
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
    """Test for the correct input and output shapes for ClassifyVVAD pipeline."""
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
    """Test for the correct input and output shapes for the full ASD pipeline."""
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
        ret_val = asd_pipeline(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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


def test_invalid_architecture_load_vvad_classifier():
    """An unknown architecture is rejected before anything is downloaded."""
    with pytest.raises(ValueError, match='Unsupported architecture'):
        load_vvad_classifier('ResNet')


def test_invalid_architecture_ClassifyVVAD():
    """ClassifyVVAD only accepts the architectures in Architecture_Options."""
    with pytest.raises(AssertionError, match='is not in'):
        ClassifyVVAD(architecture='ResNet')


def test_invalid_face_tracking_approach_ASD():
    """ASD rejects unknown face tracking approaches."""
    with pytest.raises(ValueError, match='Face tracking approach must be one of'):
        ASD(face_tracking_approach='kalman')


@pytest.mark.parametrize('detector, error', [('InsightFace', NotImplementedError),
                                             ('YOLOv10-Face', NotImplementedError),
                                             ('MTCNN', ValueError)])
def test_invalid_detector_ASD(detector, error):
    """ASD rejects detectors that are unknown or not implemented yet."""
    with pytest.raises(error):
        ASD(detector=detector)


def test_reset_ClassifyVVAD():
    """After a reset, a full sample has to be buffered again before the next prediction."""
    vvad = ClassifyVVAD(architecture='LipShape', stride=1)
    for _ in range(IMAGE_INPUT_SHAPE[0]):
        ret_val = vvad([test_image])
    assert ret_val[0] is not None
    vvad.reset()
    assert vvad([test_image]) == [None]

def test_ClassifyVVAD_ids_keep_entities_apart():
    """With ids, swapping the batch order gives each entity the same score as without swapping."""
    rng = np.random.default_rng(0)
    faces = {k: rng.integers(0, 255, (60, 60, 3), dtype=np.uint8) for k in 'ab'}
    plain = ClassifyVVAD(architecture='LipShape', stride=1, averaging_window_size=1)
    keyed = ClassifyVVAD(architecture='LipShape', stride=1, averaging_window_size=1)
    for i in range(IMAGE_INPUT_SHAPE[0]):
        out_plain = plain([faces['a'], faces['b']])
        order = ['a', 'b'] if i % 2 else ['b', 'a']
        out_keyed = dict(zip(order, keyed([faces[k] for k in order], ids=order)))
    assert out_plain[0] is not None
    assert out_keyed['a'] == pytest.approx(out_plain[0])
    assert out_keyed['b'] == pytest.approx(out_plain[1])


def test_ASD_with_given_boxes_skips_detector():
    """Given boxes and ids are classified directly; the detector is not called."""
    asd = ASD(architecture='LipShape', stride=1, averaging_window_size=1)
    asd.detect = None                         # would fail if called
    image = cv2.cvtColor(cv2.imread(str(Path(__file__).parent / 'researcher_faces.jpeg')),
                         cv2.COLOR_BGR2RGB)
    for _ in range(IMAGE_INPUT_SHAPE[0]):
        boxes = [Box2D((10, 10, 110, 110), 1.0, 'Face')]
        out = asd(image, boxes, ['e1'])
    assert len(out['boxes2D']) == 1
    assert 0.0 <= out['boxes2D'][0].score <= 1.0

@pytest.mark.parametrize('detector_class', [FaceDetectorYN, FaceDetectorRetinaFace])
def test_face_detectors_take_rgb(detector_class):
    """Detectors take RGB (the ASD convention) and still find the faces (#77).

    The test image is read with OpenCV (BGR) and converted to RGB like ``run_asd_on_video``.
    Feeding the detector the BGR image instead must not give the identical result, which shows
    that the channel order reaches the model.
    """
    bgr = cv2.imread(str(Path(__file__).parent / 'researcher_faces.jpeg'))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    detector = detector_class()
    boxes_rgb = detector(rgb)['boxes2D']
    boxes_bgr = detector(bgr)['boxes2D']
    assert len(boxes_rgb) > 0
    scores_rgb = sorted(float(b.score) for b in boxes_rgb)
    scores_bgr = sorted(float(b.score) for b in boxes_bgr)
    assert scores_rgb != scores_bgr


@pytest.mark.parametrize('architecture', ['LipShapeCausal', 'FaceShapeCausal',
                                          'LipShapeCausal_ava', 'FaceShapeCausal_ava'])
@pytest.mark.parametrize('causal_mode', ['streaming', 'window'])
def test_causal_architectures_predict_from_the_first_frame(architecture, causal_mode):
    """Causal models give a score for every face from its very first frame (#45)."""
    vvad = ClassifyVVAD(architecture=architecture, averaging_window_size=1,
                        causal_mode=causal_mode, window_size=10)
    rng = np.random.default_rng(0)
    for i in range(15):
        ids = ['a', 'b'] if i < 10 else ['b']
        faces = [rng.integers(0, 255, (80, 80, 3), dtype=np.uint8) for _ in ids]
        out = vvad(faces, ids=ids)
        assert all(s is not None and 0.0 <= s <= 1.0 for s in out)
    assert len(out) == 1


def test_causal_streaming_equals_window_while_window_not_full():
    """Until the window is full, both modes see the same frames and give the same score."""
    rng = np.random.default_rng(1)
    faces = [rng.integers(0, 255, (80, 80, 3), dtype=np.uint8) for _ in range(6)]
    stream = ClassifyVVAD(architecture='LipShapeCausal', averaging_window_size=1)
    window = ClassifyVVAD(architecture='LipShapeCausal', averaging_window_size=1,
                          causal_mode='window', window_size=38)
    for face in faces:
        s, w = stream([face], ids=['a']), window([face], ids=['a'])
    assert s[0] == pytest.approx(w[0], abs=1e-4)


@pytest.mark.parametrize('architecture, shape, points', [
    ('LipShapeCausal_ava', 'LipShape', 20), ('FaceShapeCausal_ava', 'FaceShape', 68)])
def test_ava_models_get_the_landmarks_they_were_trained_on(architecture, shape, points):
    """The fine-tuned models (#92) take the lip or face points like the models they come from."""
    vvad = ClassifyVVAD(architecture=architecture, averaging_window_size=1)
    assert vvad.preprocess.architecture == shape
    assert vvad.classifier.input_shape[-2] == points


def test_every_release_model_is_pinned_by_hash():
    """Every file the code downloads from the models release has a SHA-256 (#107)."""
    from asd4hri.processors import MODEL_SHA256
    for architecture in Causal_Architectures:
        assert f'{architecture}.keras' in MODEL_SHA256
    assert all(len(h) == 64 for h in MODEL_SHA256.values())
