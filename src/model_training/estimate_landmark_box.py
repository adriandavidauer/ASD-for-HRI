"""Estimate where dlib's landmarks lie in the face crop, per box source (#105).

The result is ``LANDMARK_BOX_IN_CROP`` in ``asd4hri.processors``: the median tight box of the
2-pass landmarks as fractions ``(x0, y0, x1, y1)`` of the square crop. ``GetShapeFeatures``
with ``box_mode='fixed'`` starts dlib there instead of in the whole crop.

* GT boxes: from the ``.npz`` files of ``extract_ava_landmarks --passes 2``.
* A detector: runs the detector, the pipeline's square/clip/crop and 2-pass dlib on frames
  sampled evenly from some videos.

Usage (from the repo root, in the image)::

    python -m src.model_training.estimate_landmark_box gt /ft/ava_train_landmarks_p2
    python -m src.model_training.estimate_landmark_box YuNet /data/videos/val/*.mp4
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

MIN_SIZE = 40      # smaller crops give coarse landmarks


def summarize(fractions):
    """Median and spread of the landmark boxes.

    Args:
        fractions: ``(N, 4)`` landmark boxes as fractions of the crop.

    Returns:
        dict: ``box`` (median ``x0, y0, x1, y1``), ``std``, ``n``.
    """
    fractions = np.asarray(fractions, np.float64)
    return {'box': [round(float(v), 3) for v in np.median(fractions, 0)],
            'std': [round(float(v), 3) for v in fractions.std(0)], 'n': len(fractions)}


def from_extracted(files, min_size=MIN_SIZE):
    """Landmark boxes of the extracted (GT-box) faces.

    Args:
        files: ``.npz`` files of ``extract_ava_landmarks``.
        min_size: Smallest crop side to use.

    Returns:
        np.ndarray: ``(N, 4)`` landmark boxes as fractions of the square crop.
    """
    out = []
    for path in files:
        d = np.load(path)
        keep = d['size'] >= min_size
        points = d['landmarks'][keep].astype(np.float64)
        size = d['size'][keep].astype(np.float64)[:, None]
        out.append(np.concatenate([points.min(1), points.max(1)], 1) / size)
    return np.concatenate(out) if out else np.zeros((0, 4))


def from_detector(videos, detector='YuNet', frames_per_video=100, min_size=MIN_SIZE):
    """Landmark boxes of the faces a detector finds, cropped as in the pipeline.

    Args:
        videos: Video files.
        detector: One of ``FaceDetector_Options``.
        frames_per_video: Frames sampled evenly from each video.
        min_size: Smallest crop side to use.

    Returns:
        np.ndarray: ``(N, 4)`` landmark boxes as fractions of the crop width/height.
    """
    import cv2

    from asd4hri.asd import ASD
    from asd4hri.processors import GetShapeFeatures
    asd = ASD(architecture='LipShapeCausal', detector=detector)
    shape = GetShapeFeatures('FaceShape', passes=2)
    out = []
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for index in np.linspace(0, max(total - 1, 0), frames_per_video).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                continue
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = asd.clip(image, asd.square(asd.detect(image.copy())['boxes2D']))
            crops = [c for c in asd.crop(image, boxes)
                     if c is not None and min(c.shape[:2]) >= min_size]
            for crop, points in zip(crops, shape(crops) if crops else []):
                h, w = crop.shape[:2]
                out.append(np.r_[points.min(0) / (w, h), points.max(0) / (w, h)])
        cap.release()
    return np.array(out).reshape(-1, 4)


def main():
    """Print the median landmark box of one box source as JSON."""
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('source', help="'gt' or a detector name (e.g. YuNet)")
    p.add_argument('inputs', nargs='+', help='landmark folder (gt) or video files (detector)')
    p.add_argument('--frames_per_video', type=int, default=100)
    args = p.parse_args()
    if args.source == 'gt':
        files = sorted(f for f in Path(args.inputs[0]).glob('*.npz') if '.part' not in f.name)
        fractions = from_extracted(files)
    else:
        fractions = from_detector(args.inputs, args.source, args.frames_per_video)
    json.dump({'source': args.source, **summarize(fractions)}, sys.stdout)
    print()


if __name__ == '__main__':
    main()
