"""Tests for the RetinaFace prior boxes, box decoding and NMS."""

# 3rd party imports
import numpy as np

# local imports
from asd4hri.retinaface_onnx import PriorBox, cfg_re50, decode, py_cpu_nms

# end file header
__author__      = 'Adrian Auer'


def test_PriorBox_count_and_first_priors():
    """Feature maps of a 32x64 input yield 2 priors per cell, normalized by height and width."""
    priors = PriorBox(cfg_re50, image_size=(32, 64)).forward()
    # feature maps for steps 8, 16, 32: 4x8, 2x4, 1x2 cells with 2 min_sizes each
    assert priors.shape == ((4 * 8 + 2 * 4 + 1 * 2) * 2, 4)
    # first cell of the stride-8 map: center (4, 4) px, sizes 16 and 32 px
    np.testing.assert_allclose(priors[:3], [[4 / 64, 4 / 32, 16 / 64, 16 / 32],
                                            [4 / 64, 4 / 32, 32 / 64, 32 / 32],
                                            [12 / 64, 4 / 32, 16 / 64, 16 / 32]])
    # last prior: only cell of the stride-32 map in the second column, size 512 px
    np.testing.assert_allclose(priors[-1], [48 / 64, 16 / 32, 512 / 64, 512 / 32])


def test_PriorBox_clip():
    """With clip enabled all prior values are clipped to [0, 1]."""
    priors = PriorBox(dict(cfg_re50, clip=True), image_size=(32, 64)).forward()
    assert priors.min() >= 0 and priors.max() <= 1
    np.testing.assert_allclose(priors[-1], [48 / 64, 16 / 32, 1, 1])


def test_decode_zero_offsets_returns_prior_corners():
    """Zero offsets decode to the corners of the prior boxes."""
    priors = np.array([[0.5, 0.5, 0.2, 0.4], [0.1, 0.2, 0.1, 0.1]])
    boxes = decode(np.zeros((2, 4)), priors, [0.1, 0.2])
    np.testing.assert_allclose(boxes, [[0.4, 0.3, 0.6, 0.7], [0.05, 0.15, 0.15, 0.25]])


def test_decode_offsets():
    """Center offsets scale with variance and prior size, size offsets are exponential."""
    priors = np.array([[0.5, 0.5, 0.2, 0.4]])
    loc = np.array([[1.0, -1.0, np.log(2) / 0.2, 0.0]])
    boxes = decode(loc, priors, [0.1, 0.2])
    # center (0.52, 0.46), size (0.4, 0.4)
    np.testing.assert_allclose(boxes, [[0.32, 0.26, 0.72, 0.66]])


def test_py_cpu_nms():
    """Overlapping lower-scoring boxes are suppressed above the IoU threshold only."""
    dets = np.array([
        [1, 1, 10, 10, 0.8],    # IoU with box 2 is 81 / 119 ~ 0.68
        [20, 20, 29, 29, 0.7],  # no overlap
        [0, 0, 9, 9, 0.9],
    ])
    assert py_cpu_nms(dets, 0.5) == [2, 1]
    assert py_cpu_nms(dets, 0.7) == [2, 0, 1]


def test_py_cpu_nms_empty():
    """No detections give no kept indices."""
    assert py_cpu_nms(np.zeros((0, 5)), 0.5) == []
