r"""Estimate where dlib's own face detector puts its box in the face crop, per box source (#132).

dlib's 68-point shape predictor was trained on the boxes of dlib's frontal face detector, so
its single pass works best when it starts there. This script measures, on train frames, where
that box lies inside the crop the pipeline cuts for the landmark models (square, clip, crop as
in ``ASD.call``), and the median goes into ``SHAPE_START_BOXES`` in ``asd4hri.processors``
(``--shape_start_box adjusted``).

Box sources (train splits only):

* ``ava``: AVA-ActiveSpeaker train GT boxes (``<root>/videos``, ``<root>/annotations``).
* ``wasd``: WASD train GT boxes; the sub-clips are cut from the source videos into a scratch
  folder exactly as ``WasdDataset.cut_clips`` does.
* ``easycom``: EasyCom train sessions, boxes converted with ``easycom_dataset.convert_file``
  and frames cropped to the camera image (``frame_crop``).
* ``unitalk``: UniTalk train GT boxes (``<root>/videos/train``, ``<root>/csv/train_orig.csv``,
  as written by ``download_uni_talk``); videos still downloading are skipped.
* ``YuNet``: the pipeline's detector, run on the same frames of all datasets.

Every sampled frame is stored with the crop regions of the GT boxes and of YuNet and with all
of dlib's detections (``detector.run(image, 1)``: one upsampling, threshold 0), so the
matching rule can be changed without running dlib again (``summarize``).

Matching rule (``match``): dlib's boxes are much tighter than loose GT crops (on AVA train a
matched dlib box has ~0.6 of the crop's side, median IoU 0.38), so IoU >= 0.5 alone keeps only
~1 in 10 correct AVA pairs. A pair is therefore a candidate if its IoU is >= ``MIN_IOU`` (tight
boxes, where dlib's box often sticks out at the chin) or if at least ``MIN_INSIDE`` of the dlib
box lies inside the crop and its side is at least ``MIN_SIDE`` of the crop side (so a small
background face in a large crop does not count). Candidates are matched one-to-one in order of
decreasing IoU. Crops smaller than ``MIN_SIZE`` are skipped, as in ``estimate_landmark_box``.
On the train samples the medians do not change for ``MIN_SIDE`` 0.2-0.4: 99 % of the dlib
boxes inside a crop have more than 0.45 of its side.

Usage (from the repo root, in the image)::

    python -m src.model_training.estimate_detector_box collect ava /data/ava_train out/ava.jsonl
    python -m src.model_training.estimate_detector_box collect wasd /data/wasd out/wasd.jsonl \\
        --clip_dir out/wasd_clips
    python -m src.model_training.estimate_detector_box collect easycom /data/easycom \\
        out/easycom.jsonl
    python -m src.model_training.estimate_detector_box collect unitalk /data out/unitalk.jsonl
    python -m src.model_training.estimate_detector_box summarize out/*.jsonl
    python -m src.model_training.estimate_detector_box proxy /data/ava out/proxy.json \\
        --groundtruth /data/ava/annotations/ava_val_groundtruth.csv
"""

import argparse
import csv
import glob
import json
import logging
import multiprocessing
import os
import random
import sys
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger('estimate_detector_box')
DATASETS = ('ava', 'wasd', 'easycom', 'unitalk')
MIN_SIZE = 40          # smaller crops give coarse landmarks (as in estimate_landmark_box)
MIN_INSIDE = 0.8       # share of the dlib box that must lie inside the crop
MIN_SIDE = 0.3         # dlib box side relative to the crop side
MIN_IOU = 0.5          # tight boxes (EasyCom, YuNet): a dlib box may stick out of the crop
UPSAMPLE = 1           # dlib detector upsampling (finds faces from ~40 px)


# ── data ──────────────────────────────────────────────────────────────────────

def read_groundtruth(csv_path, video_id=None):
    """Read AVA-format GT rows, grouped by video.

    Args:
        csv_path: CSV with AVA columns (with or without header).
        video_id: If given, the video id of all rows (per-video AVA files), else column 0.

    Returns:
        dict: ``video_id -> [(timestamp, entity_id, (x1, y1, x2, y2)), ...]`` normalised.
    """
    out = {}
    with open(csv_path, newline='') as f:
        for row in csv.reader(f):
            if len(row) < 8 or row[0] == 'video_id':
                continue
            out.setdefault(video_id or row[0], []).append(
                (float(row[1]), row[7], tuple(float(v) for v in row[2:6])))
    return out


def ava_videos(root, groundtruth_csv=None):
    """AVA videos and their GT boxes.

    Args:
        root: Folder with ``videos/`` and, without ``groundtruth_csv``, ``annotations/
            <id>-activespeaker.csv``.
        groundtruth_csv: One CSV with the GT of all videos (e.g. AVA val), or None.

    Returns:
        list[tuple]: ``(video_id, path, frame_crop, groundtruth)`` per video present.
    """
    root = Path(root)
    videos = {p.stem: p for p in (root / 'videos').iterdir() if '.part' not in p.name}
    if groundtruth_csv:
        gt = read_groundtruth(groundtruth_csv)
    else:
        gt = {}
        for path in sorted((root / 'annotations').glob('*-activespeaker.csv')):
            vid = path.name[:-len('-activespeaker.csv')]
            gt.update(read_groundtruth(path, vid))
    return [(vid, str(videos[vid]), None, rows) for vid, rows in sorted(gt.items())
            if vid in videos]


def unitalk_videos(root, n_videos=None, seed=0):
    """UniTalk train videos and their GT boxes (AVA format).

    Args:
        root: Folder with ``videos/train/<id>.mp4`` and ``csv/train_orig.csv``; only read.
        n_videos: Number of videos to sample, or None for all present.
        seed: Seed of the sample.

    Returns:
        list[tuple]: ``(video_id, path, frame_crop, groundtruth)`` per video present.
    """
    root = Path(root)
    gt = read_groundtruth(root / 'csv' / 'train_orig.csv')
    # yt-dlp's partial and not yet merged files have another stem (``<id>.f137``, ``.part``)
    videos = {p.stem: p for p in (root / 'videos' / 'train').glob('*.mp4')}
    ids = sorted(vid for vid in gt if vid in videos)
    if n_videos is not None:
        ids = sorted(random.Random(seed).sample(ids, min(n_videos, len(ids))))
    return [(vid, str(videos[vid]), None, gt[vid]) for vid in ids]


def wasd_videos(root, clip_dir, n_clips, seed=0):
    """A random sample of WASD train sub-clips, cut into ``clip_dir``.

    Args:
        root: WASD folder (``csv/train_orig.csv``, ``WASD_videos/``); only read.
        clip_dir: Writable folder for the cut sub-clips.
        n_clips: Number of sub-clips to sample.
        seed: Seed of the sample.

    Returns:
        list[tuple]: ``(video_id, path, frame_crop, groundtruth)`` per cut sub-clip.
    """
    from src.experiments.wasd_dataset import WasdDataset
    ds = WasdDataset(root, split='train', download=False)
    ids = sorted(random.Random(seed).sample(ds.video_ids, min(n_clips, len(ds.video_ids))))
    ds.clip_dir, ds.video_ids = str(clip_dir), ids
    ds.cut_clips()
    gt = read_groundtruth(ds.groundtruth_csv)
    return [(vid, ds.clip_path(vid), None, gt[vid]) for vid in ids
            if os.path.isfile(ds.clip_path(vid))]


def easycom_videos(root, n_files=None, seed=0):
    """EasyCom train files and their GT boxes, converted as in ``EasyComDataset``.

    Args:
        root: EasyCom folder with ``Main/``; only read.
        n_files: Number of one-minute files to sample, or None for all.
        seed: Seed of the sample.

    Returns:
        list[tuple]: ``(video_id, path, frame_crop, groundtruth)`` per file.
    """
    from src.experiments.easycom_dataset import (SPLITS, EasyComDataset, _load_json,
                                                 convert_file)
    main = os.path.join(root, 'Main')
    files = []
    for session in SPLITS['train']:
        for path in sorted(glob.glob(os.path.join(main, 'Video_Compressed', f'Session_{session}',
                                                  '*.mp4'))):
            stem = Path(path).stem
            boxes = os.path.join(main, 'Face_Bounding_Boxes', f'Session_{session}', f'{stem}.json')
            if os.path.isfile(boxes):
                files.append((f'Session_{session}_{stem}', path, boxes))
    if n_files is not None and n_files < len(files):
        files = sorted(random.Random(seed).sample(files, n_files))
    out = []
    for vid, path, boxes in files:
        crop = EasyComDataset.frame_crop(path)
        rows = convert_file(vid, _load_json(boxes), [], x_offset=crop[0])
        out.append((vid, path, crop, [(r[1], r[7], tuple(r[2:6])) for r in rows]))
    return out


def sample_timestamps(groundtruth, k):
    """``k`` annotated timestamps spread evenly over a video.

    Args:
        groundtruth: ``(timestamp, entity_id, box)`` rows.
        k: Number of timestamps.

    Returns:
        list[float]: Sorted timestamps.
    """
    stamps = sorted({float(ts) for ts, _e, _b in groundtruth})
    if len(stamps) <= k:
        return stamps
    return [stamps[int(i)] for i in np.linspace(0, len(stamps) - 1, k).round()]


# ── geometry ──────────────────────────────────────────────────────────────────

def match(regions, detections, min_inside=MIN_INSIDE, min_side=MIN_SIDE, min_iou=MIN_IOU):
    """Match dlib's detections to crop regions one-to-one.

    A pair is a candidate if its IoU is at least ``min_iou`` or the detection lies mostly inside
    the region (``min_inside``) and is not much smaller than it (``min_side``); candidates are
    taken in order of decreasing IoU.

    Args:
        regions: Crop regions ``(x0, y0, x1, y1)`` in pixels.
        detections: dlib boxes ``(left, top, right, bottom, ...)`` in pixels.
        min_inside: Smallest share of a detection's area inside the region.
        min_side: Smallest detection side (mean of width and height) relative to the region
            side (mean of width and height).
        min_iou: IoU from which a pair is a candidate anyway (tight boxes).

    Returns:
        list[tuple]: ``(region index, detection index)`` pairs.
    """
    candidates = []
    for i, (x0, y0, x1, y1) in enumerate(regions):
        side = ((x1 - x0) + (y1 - y0)) / 2
        for j, det in enumerate(detections):
            l, t, r, b = det[:4]
            area = max(r - l, 0) * max(b - t, 0)
            inter = max(min(x1, r) - max(x0, l), 0) * max(min(y1, b) - max(y0, t), 0)
            if not area or not inter:
                continue
            iou = inter / (area + (x1 - x0) * (y1 - y0) - inter)
            inside = inter / area >= min_inside and ((r - l) + (b - t)) / 2 >= min_side * side
            if iou >= min_iou or inside:
                candidates.append((iou, i, j))
    pairs, used_i, used_j = [], set(), set()
    for _iou, i, j in sorted(candidates, reverse=True):
        if i not in used_i and j not in used_j:
            pairs.append((i, j))
            used_i.add(i)
            used_j.add(j)
    return sorted(pairs)


def as_fractions(region, detection):
    """A dlib box as fractions of the crop of a region.

    ``GetShapeFeatures`` turns the fractions back into ``dlib.rectangle(round(x0 * w), ...)``
    with ``w, h`` the crop size, so the fractions are relative to the crop's top-left corner and
    size, and ``right``/``bottom`` stay dlib's inclusive coordinates.

    Args:
        region: Crop region ``(x0, y0, x1, y1)`` in pixels.
        detection: dlib box ``(left, top, right, bottom, ...)`` in pixels.

    Returns:
        np.ndarray: ``(x0, y0, x1, y1)`` fractions of the crop.
    """
    x0, y0, x1, y1 = region
    w, h = x1 - x0, y1 - y0
    l, t, r, b = detection[:4]
    return np.array([(l - x0) / w, (t - y0) / h, (r - x0) / w, (b - y0) / h])


def summarize(fractions):
    """Median, spread and number of the matched boxes.

    Args:
        fractions: ``(N, 4)`` boxes as fractions of the crop.

    Returns:
        dict: ``box`` (median ``x0, y0, x1, y1``), ``iqr`` (25th and 75th percentiles per
        value), ``std`` and ``n``.
    """
    fractions = np.asarray(fractions, np.float64).reshape(-1, 4)
    if not len(fractions):
        return {'box': None, 'iqr': None, 'std': None, 'n': 0}
    q25, q50, q75 = np.percentile(fractions, [25, 50, 75], axis=0)
    return {'box': [round(float(v), 3) for v in q50],
            'iqr': [[round(float(a), 3), round(float(b), 3)] for a, b in zip(q25, q75)],
            'std': [round(float(v), 3) for v in fractions.std(0)], 'n': len(fractions)}


def matched_fractions(records, source, min_inside=MIN_INSIDE, min_side=MIN_SIDE,
                      min_size=MIN_SIZE):
    """The matched dlib boxes of one box source as fractions of its crops.

    Args:
        records: Frame records of ``collect`` (dicts with ``gt``, ``yunet`` and ``dlib``).
        source: 'gt' or 'yunet'.
        min_inside: See ``match``.
        min_side: See ``match``.
        min_size: Smallest crop side.

    Returns:
        np.ndarray: ``(N, 4)`` fractions.
    """
    out = []
    for rec in records:
        regions = [r for r in rec[source] if min(r[2] - r[0], r[3] - r[1]) >= min_size]
        for i, j in match(regions, rec['dlib'], min_inside, min_side):
            out.append(as_fractions(regions[i], rec['dlib'][j]))
    return np.array(out).reshape(-1, 4)


# ── frames ────────────────────────────────────────────────────────────────────

class _Worker:
    """The pipeline's square/clip/crop steps, YuNet and dlib's detector, built once per process."""

    def __init__(self, yunet=True, start_boxes=()):
        """Build the processors.

        Args:
            yunet: Also run the YuNet detector.
            start_boxes: Keys of ``SHAPE_START_BOXES`` for the landmark proxy (``proxy``).
        """
        import dlib
        import paz.processors as pr
        from paz.abstract import SequentialProcessor

        from asd4hri.processors import FaceDetectorYN, GetShapeFeatures
        # as in ASD.__init__: square (+ zero offset), clip, crop
        self.square = SequentialProcessor([pr.SquareBoxes2D(), pr.OffsetBoxes2D([0, 0])])
        self.clip = pr.ClipBoxes2D()
        self.crop = pr.CropBoxes2D()
        self.dlib = dlib.get_frontal_face_detector()
        self.yunet = FaceDetectorYN(input_size=None) if yunet else None
        self.start_boxes = list(start_boxes)
        self.shapes = {key: GetShapeFeatures('FaceShape', start_box=key) for key in start_boxes}
        if start_boxes:
            self.shape = GetShapeFeatures('FaceShape')
            self.shape2 = GetShapeFeatures('FaceShape', passes=2)

    def regions(self, image, boxes):
        """Crop regions of boxes, as ``ASD.call`` cuts them for the landmark models.

        Args:
            image: RGB frame.
            boxes: ``paz`` Box2D list in pixels.

        Returns:
            list[tuple]: ``(x0, y0, x1, y1)`` per box.
        """
        return [tuple(int(v) for v in b.coordinates) for b in self.clip(image, self.square(boxes))]

    def frame(self, image, gt_frame):
        """Crop regions and dlib detections of one frame.

        Args:
            image: RGB frame (already cropped to ``frame_crop``).
            gt_frame: ``[(entity_id, normalised box), ...]`` of the frame.

        Returns:
            dict: ``gt`` and ``yunet`` regions, ``dlib`` boxes ``(l, t, r, b, score)``, and
            ``proxy`` (per GT region, landmark errors, see ``proxy_errors``) if start boxes
            are set.
        """
        from src.experiments.run_asd_on_video import _GroundTruthFrames
        height, width = image.shape[:2]
        boxes, _ids = _GroundTruthFrames.boxes2D(gt_frame, width, height)
        rec = {'size': [height, width], 'gt': self.regions(image, boxes)}
        if self.yunet is not None:
            rec['yunet'] = self.regions(image, self.yunet(image.copy())['boxes2D'])
        rects, scores, _kind = self.dlib.run(image, UPSAMPLE, 0.0)
        rec['dlib'] = [(r.left(), r.top(), r.right(), r.bottom(), round(float(s), 3))
                       for r, s in zip(rects, scores)]
        if self.start_boxes:
            rec['proxy'] = self.proxy_errors(image, rec)
        return rec

    def proxy_errors(self, image, rec):
        """Landmark errors of 1-pass dlib from each start box, per GT crop.

        References: the 2-pass landmarks from the whole crop (``p2``) and, for crops with a
        matched dlib detection, the 1-pass landmarks started from that detection (``det``).
        The error is the mean point distance divided by the outer-eye-corner distance of the
        reference (NME).

        Args:
            image: RGB frame.
            rec: Frame record with ``gt`` and ``dlib``.

        Returns:
            list[dict]: Per kept GT crop, ``size`` and ``{reference: {start box: NME}}``.
        """
        import dlib

        from paz.abstract.messages import Box2D
        regions = [r for r in rec['gt'] if min(r[2] - r[0], r[3] - r[1]) >= MIN_SIZE]
        pairs = dict(match(regions, rec['dlib']))
        crops = self.crop(image, [Box2D(r, 1.0, 'Face') for r in regions])
        out = []
        for i, (region, crop) in enumerate(zip(regions, crops)):
            refs = {'p2': self.shape2([crop])[0]}
            if i in pairs:
                l, t, r, b = as_fractions(region, rec['dlib'][pairs[i]])
                h, w = crop.shape[:2]
                rect = dlib.rectangle(round(l * w), round(t * h), round(r * w), round(b * h))
                parts = self.shape.shape_predictor(crop, rect).parts()
                refs['det'] = np.array([(p.x, p.y) for p in parts])
            starts = {key: self.shapes[key]([crop])[0] for key in self.start_boxes}
            errors = {}
            for name, ref in refs.items():
                eyes = max(np.linalg.norm(ref[36] - ref[45]), 1.0)
                errors[name] = {key: round(float(np.linalg.norm(p - ref, axis=1).mean() / eyes), 4)
                                for key, p in starts.items()}
            out.append({'size': max(crop.shape[:2]), 'errors': errors})
        return out


_WORKER = None


def _init(yunet, start_boxes):
    """Pool initializer: build the worker of this process.

    Args:
        yunet: See ``_Worker``.
        start_boxes: See ``_Worker``.
    """
    global _WORKER
    _WORKER = _Worker(yunet, start_boxes)


def process_video(job):
    """Records of the sampled frames of one video.

    Frame ``i`` of the video has the timestamp ``i / fps``, as in ``run_asd_on_video``, so a GT
    timestamp ``ts`` is read from frame ``round(ts * fps)``.

    Args:
        job: ``(dataset, video_id, path, frame_crop, groundtruth, timestamps)``.

    Returns:
        list[dict]: One record per frame read; empty if the video cannot be read.
    """
    import cv2

    from src.experiments.run_asd_on_video import _GroundTruthFrames
    dataset, vid, path, frame_crop, groundtruth, timestamps = job
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or not fps:
        LOGGER.warning('cannot read %s', path)
        return []
    index = _GroundTruthFrames(groundtruth, fps)
    out = []
    try:
        for ts in timestamps:
            frame_idx = int(round(ts * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            gt_ts, gt_frame = index.at(frame_idx / fps)
            if not ok or not gt_frame:
                continue
            if frame_crop is not None:
                x0, y0, x1, y1 = frame_crop
                frame = frame[y0:y1, x0:x1]
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.append({'dataset': dataset, 'video': vid, 'ts': gt_ts,
                        **_WORKER.frame(image, gt_frame)})
    finally:
        cap.release()
    return out


def run(jobs, out_path, workers, yunet=True, start_boxes=()):
    """Process the jobs in parallel and write one JSON record per frame.

    Args:
        jobs: Arguments of ``process_video``.
        out_path: The ``.jsonl`` file to write.
        workers: Number of processes.
        yunet: See ``_Worker``.
        start_boxes: See ``_Worker``.

    Returns:
        list[dict]: All records.
    """
    records = []
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f, multiprocessing.Pool(
            workers, initializer=_init, initargs=(yunet, tuple(start_boxes))) as pool:
        for n, recs in enumerate(pool.imap_unordered(process_video, jobs), 1):
            for rec in recs:
                f.write(json.dumps(rec) + '\n')
            f.flush()
            records.extend(recs)
            LOGGER.info('%d/%d videos, %d frames', n, len(jobs), len(records))
    return records


def load_records(paths):
    """Read the records of ``collect``.

    Args:
        paths: ``.jsonl`` files.

    Returns:
        list[dict]: The records.
    """
    records = []
    for path in paths:
        with open(path) as f:
            records.extend(json.loads(line) for line in f if line.strip())
    return records


# ── commands ──────────────────────────────────────────────────────────────────

def _videos(args):
    """Videos of the ``collect`` command.

    Args:
        args: Parsed arguments.

    Returns:
        list[tuple]: ``(video_id, path, frame_crop, groundtruth)``.

    Raises:
        ValueError: If the dataset is unknown.
    """
    if args.dataset == 'ava':
        return ava_videos(args.root)
    if args.dataset == 'wasd':
        return wasd_videos(args.root, args.clip_dir or Path(args.out).with_suffix('') / 'clips',
                           args.videos or 150, args.seed)
    if args.dataset == 'easycom':
        return easycom_videos(args.root, args.videos, args.seed)
    if args.dataset == 'unitalk':
        return unitalk_videos(args.root, args.videos, args.seed)
    raise ValueError(f'dataset must be one of {DATASETS}')


def report(records):
    """Matched boxes per box source and dataset.

    Args:
        records: Frame records of ``collect``.

    Returns:
        dict: ``{source: {dataset or 'all': summarize(...)}}`` for 'gt' (per dataset) and
        'yunet' (per dataset and all together).
    """
    out = {'gt': {}, 'yunet': {}}
    datasets = sorted({r['dataset'] for r in records})
    for ds in datasets:
        recs = [r for r in records if r['dataset'] == ds]
        out['gt'][ds] = summarize(matched_fractions(recs, 'gt'))
        if all('yunet' in r for r in recs):
            out['yunet'][ds] = summarize(matched_fractions(recs, 'yunet'))
    if out['yunet']:
        out['yunet']['all'] = summarize(matched_fractions(
            [r for r in records if 'yunet' in r], 'yunet'))
    return out


def proxy_report(records):
    """Median and mean landmark error per reference and start box.

    Args:
        records: Frame records of ``proxy``.

    Returns:
        dict: ``{reference: {start box: {'median', 'mean', 'n'}}}``.
    """
    errors = {}
    for rec in records:
        for face in rec.get('proxy', []):
            for ref, by_box in face['errors'].items():
                for box, err in by_box.items():
                    errors.setdefault(ref, {}).setdefault(box, []).append(err)
    return {ref: {box: {'median': round(float(np.median(v)), 4),
                        'mean': round(float(np.mean(v)), 4), 'n': len(v)}
                  for box, v in by_box.items()} for ref, by_box in errors.items()}


def main():
    """Run one command: ``collect``, ``summarize`` or ``proxy``; print JSON."""
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('collect', help='sample train frames, write crop regions + dlib boxes')
    c.add_argument('dataset', choices=DATASETS)
    c.add_argument('root', help='dataset folder (read only)')
    c.add_argument('out', help='.jsonl to write')
    c.add_argument('--frames_per_video', type=int, default=8)
    c.add_argument('--videos', type=int, default=None,
                   help='WASD sub-clips (default 150) / EasyCom files / UniTalk videos (default all) '
                        'to sample')
    c.add_argument('--clip_dir', default=None, help='writable folder for the WASD sub-clips')
    c.add_argument('--no_yunet', action='store_true', help='skip the YuNet box source')
    c.add_argument('--seed', type=int, default=0)
    c.add_argument('--workers', type=int, default=4)
    s = sub.add_parser('summarize', help='match and print the median boxes')
    s.add_argument('records', nargs='+')
    x = sub.add_parser('proxy', help='landmark error of 1-pass dlib per start box (validation)')
    x.add_argument('root', help='AVA folder with videos/')
    x.add_argument('out', help='.jsonl to write')
    x.add_argument('--groundtruth', required=True, help='AVA GT CSV, e.g. the val split')
    x.add_argument('--start_boxes', nargs='+', default=['none', 'ava_gt', 'ava'])
    x.add_argument('--videos', type=int, default=None, help='videos to use (default all)')
    x.add_argument('--frames_per_video', type=int, default=4)
    x.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if args.command == 'summarize':
        result = report(load_records(args.records))
    elif args.command == 'collect':
        jobs = [(args.dataset, vid, path, crop, gt, sample_timestamps(gt, args.frames_per_video))
                for vid, path, crop, gt in _videos(args)]
        result = report(run(jobs, args.out, args.workers, yunet=not args.no_yunet))
    else:
        videos = ava_videos(args.root, args.groundtruth)[:args.videos]
        jobs = [('ava', vid, path, crop, gt, sample_timestamps(gt, args.frames_per_video))
                for vid, path, crop, gt in videos]
        result = proxy_report(run(jobs, args.out, args.workers, yunet=False,
                                  start_boxes=args.start_boxes))
    json.dump(result, sys.stdout, indent=1)
    print()


if __name__ == '__main__':
    main()
