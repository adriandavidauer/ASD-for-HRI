"""Extract dlib landmarks of the AVA-ActiveSpeaker GT face tracks for fine-tuning (#92).

The faces are cut exactly as the pipeline does with ``--boxes gt``: every video frame gets the
GT boxes of the nearest annotated timestamp (within half a frame), each box is made square,
clipped to the image and cropped, and dlib finds the 68 landmarks in the crop. The lip model
uses points 48-67 of the same landmarks, so one extraction serves LipShape and FaceShape.

One ``<video_id>.npz`` per video, with one row per (video frame, face):

* ``entity``: entity id (str), ``timestamp``: GT timestamp (s), ``frame``: video frame index
* ``landmarks``: ``(N, 68, 2)`` int16 pixel coordinates in the crop
* ``label``: 1 for SPEAKING_AUDIBLE (the AVA positive class), else 0
* ``size``: side length of the square crop in pixels
* ``fps``: native frame rate of the video

Usage (from the repo root, in the image)::

    python -m src.model_training.extract_ava_landmarks /Data/data/ava_train out/ --workers 8

The default of 2 dlib passes is what the released ``*_ava`` weights were trained on. In the
pipeline they run with the default of one pass: only on AVA's loose GT boxes does a second pass
(or ``--shape_start_box ava_gt``) help, see #105.
"""

import argparse
import csv
import logging
import multiprocessing
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger('extract_ava_landmarks')
POSITIVE = 'SPEAKING_AUDIBLE'


def read_annotations(csv_path):
    """Read one per-video AVA-ActiveSpeaker CSV.

    Args:
        csv_path: ``<video_id>-activespeaker.csv``.

    Returns:
        tuple: ``(groundtruth, labels)``: the list of ``(timestamp, entity_id, (x1, y1, x2,
        y2))`` normalised boxes that ``_GroundTruthFrames`` takes, and a dict
        ``(timestamp, entity_id) -> 0/1``.
    """
    groundtruth, labels = [], {}
    with open(csv_path, newline='') as f:
        for row in csv.reader(f):
            if len(row) < 8 or row[0] == 'video_id':
                continue
            ts, entity = float(row[1]), row[7]
            groundtruth.append((ts, entity, tuple(float(v) for v in row[2:6])))
            labels[(ts, entity)] = int(row[6] == POSITIVE)
    return groundtruth, labels


class FaceLandmarks:
    """The crop and landmark steps of the pipeline for GT boxes (``ASD`` + ``GetShapeFeatures``)."""

    def __init__(self, passes=1):
        """Build the square, clip and crop processors and load dlib.

        Args:
            passes: dlib passes per face (``GetShapeFeatures``).
        """
        import paz.processors as pr
        from paz.abstract import SequentialProcessor

        from asd4hri.processors import GetShapeFeatures
        self.square = SequentialProcessor([pr.SquareBoxes2D(), pr.OffsetBoxes2D([0, 0])])
        self.clip = pr.ClipBoxes2D()
        self.crop = pr.CropBoxes2D()
        self.landmarks = GetShapeFeatures(architecture='FaceShape', passes=passes)

    def __call__(self, image, boxes):
        """Landmarks of every box of one RGB frame.

        Args:
            image: RGB frame.
            boxes: ``paz`` Box2D list in pixels.

        Returns:
            tuple: ``(landmarks, sizes, keep)``: one ``(68, 2)`` array and crop size per kept
            box, and the indices of the kept boxes (empty crops are dropped).
        """
        boxes = self.clip(image, self.square(boxes))
        crops = self.crop(image, boxes)
        keep = [i for i, c in enumerate(crops) if c is not None and c.size and min(c.shape[:2]) > 1]
        if not keep:
            return [], [], []
        points = self.landmarks([crops[i] for i in keep])
        sizes = [max(crops[i].shape[:2]) for i in keep]
        return points, sizes, keep


def extract_video(video_path, csv_path, out_path, passes=1):
    """Extract the landmarks of all GT faces of one video and write them to ``out_path``.

    Args:
        video_path: The video.
        csv_path: Its annotation CSV.
        out_path: The ``.npz`` to write.
        passes: dlib passes per face (``GetShapeFeatures``).

    Returns:
        int: Number of extracted faces.

    Raises:
        RuntimeError: If the video cannot be read.
    """
    import cv2

    from src.experiments.run_asd_on_video import _GroundTruthFrames
    groundtruth, labels = read_annotations(csv_path)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or not fps:
        raise RuntimeError(f'cannot read {video_path}')
    index = _GroundTruthFrames(groundtruth, fps)
    first, last = index.timestamps[0], index.timestamps[-1]
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(first * fps) - 2))
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    face = FaceLandmarks(passes)
    rows = {k: [] for k in ('entity', 'timestamp', 'frame', 'landmarks', 'label', 'size')}
    try:
        while frame_idx / fps <= last + 0.5 / fps:
            ok, frame = cap.read()
            if not ok:
                break
            gt_ts, gt_frame = index.at(frame_idx / fps)
            if gt_frame:
                height, width = frame.shape[:2]
                boxes, ids = index.boxes2D(gt_frame, width, height)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                points, sizes, keep = face(image, boxes)
                for p, s, i in zip(points, sizes, keep):
                    rows['entity'].append(ids[i])
                    rows['timestamp'].append(gt_ts)
                    rows['frame'].append(frame_idx)
                    rows['landmarks'].append(np.asarray(p, dtype=np.int16))
                    rows['label'].append(labels[(gt_ts, ids[i])])
                    rows['size'].append(s)
            frame_idx += 1
    finally:
        cap.release()
    n = len(rows['entity'])
    np.savez_compressed(
        out_path, entity=np.array(rows['entity'], dtype=str),
        timestamp=np.array(rows['timestamp'], dtype=np.float64),
        frame=np.array(rows['frame'], dtype=np.int32),
        landmarks=np.stack(rows['landmarks']) if n else np.zeros((0, 68, 2), np.int16),
        label=np.array(rows['label'], dtype=np.uint8),
        size=np.array(rows['size'], dtype=np.int16), fps=np.float64(fps))
    return n


def _job(args):
    """Worker: extract one video unless its output exists.

    Args:
        args: ``(video_path, csv_path, out_path, passes)``.

    Returns:
        tuple: ``(video name, faces or error message)``.
    """
    video_path, csv_path, out_path, passes = args
    if out_path.exists():
        return video_path.name, 'exists'
    tmp = out_path.with_name(out_path.stem + '.part.npz')
    try:
        n = extract_video(video_path, csv_path, tmp, passes)
    except Exception as exc:          # one broken video must not stop the others
        return video_path.name, f'FAILED {exc}'
    tmp.rename(out_path)
    return video_path.name, n


def jobs(ava_dir, out_dir):
    """One ``(video, csv, output)`` triple per annotated video that is present.

    Args:
        ava_dir: Folder with ``videos/`` and ``annotations/<id>-activespeaker.csv``.
        out_dir: Output folder.

    Returns:
        list: The triples, sorted by video id.
    """
    ava_dir, out_dir = Path(ava_dir), Path(out_dir)
    videos = {p.stem: p for p in (ava_dir / 'videos').iterdir() if '.part' not in p.name}
    out = []
    for csv_path in sorted((ava_dir / 'annotations').glob('*-activespeaker.csv')):
        vid = csv_path.name[:-len('-activespeaker.csv')]
        if vid in videos:
            out.append((videos[vid], csv_path, out_dir / f'{vid}.npz'))
    return out


def main():
    """Extract all videos in parallel."""
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('ava_dir', help='folder with videos/ and annotations/')
    p.add_argument('out_dir')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--passes', type=int, default=2,
                   help='dlib passes per face (2: refit on the landmark box, see #92)')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    todo = [job + (args.passes,) for job in jobs(args.ava_dir, args.out_dir)]
    LOGGER.info('%d videos', len(todo))
    with multiprocessing.get_context('spawn').Pool(args.workers) as pool:
        for i, (name, result) in enumerate(pool.imap_unordered(_job, todo), 1):
            LOGGER.info('%d/%d %s: %s', i, len(todo), name, result)


if __name__ == '__main__':
    main()
