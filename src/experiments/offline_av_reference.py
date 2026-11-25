r"""Offline reference for the audio-visual models: score each whole face track at once.

The pipeline is live: a face is scored from at most its last ``av_window`` frames, for its newest
frame, without look-ahead (``asd4hri.av.AVClassifier``). The papers evaluate offline: the model
sees the whole face track, past and future, and scores every frame of it. This script does the
latter with exactly the pipeline's own pieces (GT boxes of each frame, ``face_crop``,
``mfcc_window``, ``AVModel``, the same scoring), so the difference to a ``--boxes gt`` benchmark
run is only live vs. offline.

A track is an entity's run of GT boxes, cut where the entity is missing for more than
``max_gap`` frames (the pipeline drops a face's buffer after 5). Tracks longer than
``max_frames`` are scored in chunks of that length.

Usage (in the image, from ``/app``)::

    python -m src.experiments.offline_av_reference --data_dir /app/data --model LR-ASD \
        --out benchmark/offline_ava_AV-LR-ASD_gt
"""

import argparse
import csv
import logging
import os
from pathlib import Path

import cv2
import numpy as np

from asd4hri.av import AVModel, face_crop, load_audio, mfcc_window

from .run_asd_on_video import _PREDICTION_FIELDS, _GroundTruthFrames, append_aggregate_time
from .run_evaluation import _DEFAULT_GROUNDTRUTH
from .run_full_pipeline import annotated_range, load_groundtruth_boxes
from .stats import run_stats

LOGGER = logging.getLogger('offline_av')


def collect_tracks(video_path, gt_rows, max_gap=5):
    """Read the annotated span of a video and cut the GT faces into tracks.

    Args:
        video_path: The video.
        gt_rows: ``(timestamp, entity_id, box)`` rows of the video (normalised boxes).
        max_gap: Frames an entity may be missing within one track.

    Returns:
        tuple: ``(tracks, fps, width, height, frames)``. ``tracks`` is a list of dicts with
        ``entity``, ``crops`` (T, 112, 112), ``times`` (frame timestamps), ``gt_times`` and
        ``boxes`` (normalised), one entry per frame of the track.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    index = _GroundTruthFrames(gt_rows, fps)
    start_s, end_s = annotated_range(gt_rows)
    frame_idx = max(0, int(start_s * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    open_tracks, done, width, height, frames = {}, [], 0, 0, 0
    while frame_idx / fps <= end_s:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        height, width = frame.shape[:2]
        timestamp = frame_idx / fps
        gt_ts, gt_frame = index.at(timestamp)
        if gt_frame:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, ids = index.boxes2D(gt_frame, width, height)
            for (entity, norm_box), box in zip(gt_frame, boxes):
                track = open_tracks.get(entity)
                if track is None or frame_idx - track['last'] > max_gap + 1:
                    if track is not None:
                        done.append(track)
                    track = open_tracks[entity] = {'entity': entity, 'crops': [], 'times': [],
                                                   'gt_times': [], 'boxes': [], 'frames': []}
                track['crops'].append(face_crop(rgb, box.coordinates))
                track['times'].append(timestamp)
                track['gt_times'].append(gt_ts)
                track['boxes'].append(norm_box)
                track['frames'].append(frame_idx)
                track['last'] = frame_idx
        frame_idx += 1
    cap.release()
    done += open_tracks.values()
    for t in done:
        t['crops'] = np.stack(t['crops'])
    return done, fps, width, height, frames


def score_tracks(model, tracks, audio, fps, max_frames=1500):
    """Score every frame of every track from the whole track.

    Args:
        model: ``AVModel``.
        tracks: From ``collect_tracks``.
        audio: int16 samples at 16 kHz.
        fps: Video frame rate.
        max_frames: Longest chunk scored at once.

    Returns:
        list: ``(frame_idx, gt_timestamp, entity, box, score)`` rows.
    """
    rows = []
    for t in tracks:
        n = len(t['crops'])
        for a in range(0, n, max_frames):
            b = min(n, a + max_frames)
            mfcc = mfcc_window(audio, t['times'][a], b - a, fps)
            scores = model.scores(t['crops'][None, a:b], mfcc[None])[0]
            rows += [(t['frames'][i], t['gt_times'][i], t['entity'], t['boxes'][i],
                      float(s)) for i, s in zip(range(a, b), scores)]
    return rows


def main(argv=None):
    """Command line entry point.

    Raises:
        SystemExit: For a dataset other than AVA.
    """
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--data_dir', default='data')
    p.add_argument('--dataset', default='ava', choices=list(_DEFAULT_GROUNDTRUTH))
    p.add_argument('--split', default='val')
    p.add_argument('--model', default='LR-ASD')
    p.add_argument('--out', required=True, help='run folder, relative to --data_dir')
    p.add_argument('--videos', nargs='*', help='only these video ids')
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    data_dir = Path(args.data_dir)
    out = data_dir / args.out
    pred_dir = out / 'predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_csv = data_dir / _DEFAULT_GROUNDTRUTH[args.dataset].format(split=args.split)
    all_gt = load_groundtruth_boxes(gt_csv)
    if args.dataset != 'ava':
        raise SystemExit('only AVA so far: its videos live in <data_dir>/ava/videos')
    video_dir = data_dir / 'ava' / 'videos'
    model = AVModel(args.model)
    for name in sorted(os.listdir(video_dir)):
        vid = os.path.splitext(name)[0]
        if vid not in all_gt or (args.videos and vid not in args.videos):
            continue
        csv_path = pred_dir / f'{vid}.csv'
        if csv_path.is_file():
            continue
        tracks, fps, width, height, frames = collect_tracks(video_dir / name, all_gt[vid])
        rows = score_tracks(model, tracks, load_audio(video_dir / name), fps)
        rows.sort(key=lambda r: (r[0], r[2]))
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(_PREDICTION_FIELDS)
            for frame_idx, ts, entity, box, score in rows:
                label = 'speaking' if score > 0.5 else 'not-speaking'
                w.writerow([frame_idx, repr(ts), label, *(repr(float(v)) for v in box),
                            f'{score:.6f}', entity])
        append_aggregate_time(str(pred_dir / 'aggregate_time.csv'), vid, 0.0, frames, 0.0,
                              width, height)
        LOGGER.info('%s: %d tracks, %d scores', vid, len(tracks), len(rows))
    run_stats(str(pred_dir), str(gt_csv), str(out / 'stats'),
              log_dir=str(data_dir / 'logs_stats'),
              run_info={'dataset': args.dataset, 'split': args.split,
                        'architecture': f'AV:{args.model}', 'boxes': 'gt',
                        'causal_mode': 'offline (whole track)'})


if __name__ == '__main__':
    main()
