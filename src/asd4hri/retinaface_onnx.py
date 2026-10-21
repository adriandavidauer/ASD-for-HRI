"""Code from https://github.com/kingardor/retinaface-onnx."""

# System imports
from math import ceil

from itertools import product as product

# 3rd party imports
import numpy as np


# local imports

# end file header
__author__      = 'Adrian Auer'
cfg_re50 = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}

class PriorBox(object):
    """Generate the RetinaFace anchor (prior) boxes for a given input size.

    From https://github.com/kingardor/retinaface-onnx/blob/main/layers/functions/prior_box.py.
    """
    def __init__(self, cfg, format:str="tensor", image_size=None, phase='train'):
        """Compute the feature map sizes for the given configuration and image size.

        Args:
            cfg: RetinaFace configuration, e.g. ``cfg_re50``. Uses the keys 'min_sizes', 'steps'
                and 'clip'.
            format: Kept for compatibility with the original code; the output is always numpy.
            image_size: ``(height, width)`` of the network input.
            phase: Unused; kept for compatibility with the original code.
        """
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"
        self.__format = format

    def forward(self):
        """Generate the prior boxes.

        Returns:
            numpy.ndarray: Array of shape ``(num_priors, 4)`` with priors in center-offset form
            ``(cx, cy, w, h)``, normalized to the image size.
        """
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        # if self.__format == "tensor":
        #     output = torch.Tensor(anchors).view(-1, 4)
        # elif self.__format == "numpy":
        output = np.array(anchors).reshape(-1, 4)
        # else:
        #     print(TypeError(("ERROR: INVALID TYPE OF FORMAT")))

        if self.clip:
            # if self.__format == "tensor":
            #     output.clamp_(max=1, min=0)
            # else:
            output = np.clip(output, 0, 1)

        return output


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode box predictions using the priors.

    Undoes the encoding applied for offset regression at training time.

    Args:
        loc: Location predictions of shape ``(num_priors, 4)``.
        priors: Prior boxes in center-offset form, shape ``(num_priors, 4)``.
        variances: Variances of the prior boxes, e.g. ``[0.1, 0.2]``.

    Returns:
        numpy.ndarray: Decoded boxes of shape ``(num_priors, 4)`` as
        ``(x_min, y_min, x_max, y_max)`` in normalized coordinates.
    """
    boxes = None
    # if isinstance(loc, torch.Tensor) and isinstance(priors, torch.Tensor) :
    #     boxes = torch.cat((
    #         priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
    #         priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)

    # elif isinstance(loc, np.ndarray) and isinstance(priors, np.ndarray):
    boxes = np.concatenate((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), axis=1)

    # else:
    #     print(type(loc), type(priors))
    #     print(TypeError("ERROR: INVALID TYPE OF BOUNDING BOX"))

    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def py_cpu_nms(dets, thresh):
    """Apply non-maximum suppression (pure numpy baseline).

    Args:
        dets: Detections of shape ``(N, 5)`` as ``(x_min, y_min, x_max, y_max, score)``.
        thresh: IoU threshold above which the lower-scoring box is suppressed.

    Returns:
        list: Indices into ``dets`` of the kept detections, in descending score order.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep