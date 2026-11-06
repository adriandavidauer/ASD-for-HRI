"""WASD dataset adapter (#81): source videos, sub-clips, ground truth and categories.

WASD (Roxo et al., https://github.com/Tiago-Roxo/WASD) annotates 30-s sub-clips of YouTube
videos in AVA format; ``video_id`` is ``<youtube id>_<start s>-<end s>`` and the timestamps
start at 0 in every sub-clip. The sub-clips are cut from the source videos exactly as WASD's
own ``prepare_setup.py`` does (moviepy's ``ffmpeg_extract_subclip``: input seek, stream
copy), so the frames line up with the annotations.

Layout under ``root`` (e.g. ``data/wasd``)::

    csv/val_orig.csv            ground truth (AVA columns + label_id, instance_id)
    csv/wasd_metadata.csv       category per sub-clip
    WASD_videos/<id>.mp4, .txt  source videos and their sub-clip time ranges
    clips/<video_id>.mp4        the cut sub-clips that the pipeline runs on
"""

import csv
import logging
import os
import subprocess
import zipfile
from collections import Counter

LOGGER = logging.getLogger('wasd')

CATEGORIES = ['OC', 'SI', 'FO', 'HVN', 'SS']   # metadata Category 0..4
CATEGORY_NAMES = {
    'OC': 'Optimal Conditions', 'SI': 'Speech Impairment', 'FO': 'Face Occlusion',
    'HVN': 'Human Voice Noise', 'SS': 'Surveillance Settings',
}
_CSV_GDRIVE_ID = '1calMd83IIzYnvETY3bee7juRACMBuuyR'          # from WASD's prepare_setup.py
_VIDEOS_GDRIVE_ID = '1F6pYUNz1u23Q-PvPHpxzm4RUhgFCfVYL'       # from WASD's prepare_setup.py
_METADATA_URL = 'https://raw.githubusercontent.com/Tiago-Roxo/WASD/main/eval/wasd_metadata.csv'


class WasdDataset:
    """Prepare WASD for the evaluation pipeline.

    Args:
        root: Dataset directory.
        split: 'val' (default) or 'train'.
        download: If True, missing CSVs / videos are downloaded with gdown (Google Drive links
            of the official repo) and missing sub-clips are cut with ffmpeg.
    """

    def __init__(self, root, split='val', download=True):
        """Locate (and if allowed, fetch) the files of the split."""
        self.root = root
        self.split = split
        self.csv_dir = os.path.join(root, 'csv')
        self.source_dir = os.path.join(root, 'WASD_videos')
        self.clip_dir = os.path.join(root, 'clips')
        self.groundtruth_csv = os.path.join(self.csv_dir, f'{split}_orig.csv')
        self.metadata_csv = os.path.join(self.csv_dir, 'wasd_metadata.csv')
        if download:
            self._download()
        self.video_ids = self._read_video_ids()
        if download:
            self.cut_clips()

    # ── download ──────────────────────────────────────────────────────────────
    def _gdown(self, file_id, target):
        LOGGER.info('downloading %s -> %s', file_id, target)
        subprocess.run(['gdown', '--id', file_id, '-O', target], check=True)

    def _download(self):
        os.makedirs(self.csv_dir, exist_ok=True)
        if not os.path.isfile(self.groundtruth_csv):
            archive = os.path.join(self.root, 'WASD_csv.zip')
            self._gdown(_CSV_GDRIVE_ID, archive)
            with zipfile.ZipFile(archive) as z:
                z.extractall(self.root)
            os.remove(archive)
        if not os.path.isfile(self.metadata_csv):
            import urllib.request
            urllib.request.urlretrieve(_METADATA_URL, self.metadata_csv)
        if not os.path.isdir(self.source_dir):
            archive = os.path.join(self.root, 'WASD_videos.zip')
            if not os.path.isfile(archive):
                self._gdown(_VIDEOS_GDRIVE_ID, archive)
            with zipfile.ZipFile(archive) as z:
                z.extractall(self.source_dir)       # the archive has no top-level folder
            os.remove(archive)

    # ── clips ─────────────────────────────────────────────────────────────────
    def _read_video_ids(self):
        with open(self.groundtruth_csv, newline='') as f:
            return sorted({row['video_id'] for row in csv.DictReader(f)})

    def clip_path(self, video_id):
        """Path of the sub-clip video for a WASD ``video_id``."""
        return os.path.join(self.clip_dir, f'{video_id}.mp4')

    @staticmethod
    def parse_video_id(video_id):
        """Split ``<youtube id>_<start>-<end>`` into ``(youtube id, start s, end s)``."""
        source, times = video_id.rsplit('_', 1)
        start, end = times.split('-')
        return source, int(start), int(end)

    def cut_clips(self):
        """Cut every missing sub-clip of the split from its source video.

        Same ffmpeg call as moviepy's ``ffmpeg_extract_subclip`` used by WASD (input seek,
        stream copy), so the clips match the ones the annotations were made on.
        """
        os.makedirs(self.clip_dir, exist_ok=True)
        for video_id in self.video_ids:
            target = self.clip_path(video_id)
            if os.path.isfile(target):
                continue
            source, start, end = self.parse_video_id(video_id)
            src = os.path.join(self.source_dir, f'{source}.mp4')
            if not os.path.isfile(src):
                LOGGER.warning('source video missing for %s: %s', video_id, src)
                continue
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', f'{start:.2f}',
                            '-i', src, '-t', f'{end - start:.2f}', '-map', '0',
                            '-vcodec', 'copy', '-acodec', 'copy', target], check=True)

    def video_list(self):
        """``[(video_id, clip path, None)]`` in the format of ``load_video_list``."""
        return [(vid, self.clip_path(vid), None) for vid in self.video_ids]

    # ── categories ────────────────────────────────────────────────────────────
    def categories(self):
        """Map every ``video_id`` of the split to its category code (OC, SI, FO, HVN, SS).

        Uses the numeric ``Category`` column of ``wasd_metadata.csv``. A few val sub-clips are
        missing there; they get the most common category of the other sub-clips of the same
        source video.
        """
        with open(self.metadata_csv, newline='') as f:
            meta = {row['Video']: CATEGORIES[int(row['Category'])] for row in csv.DictReader(f)}
        by_source = {}
        for clip, cat in meta.items():
            by_source.setdefault(self.parse_video_id(clip)[0], Counter())[cat] += 1
        out = {}
        for vid in self.video_ids:
            if vid in meta:
                out[vid] = meta[vid]
            else:
                counts = by_source.get(self.parse_video_id(vid)[0])
                out[vid] = counts.most_common(1)[0][0] if counts else ''
        return out

    def write_categories_csv(self, path):
        """Write ``video_id,category`` for ``stats.py --categories_csv``."""
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['video_id', 'category'])
            for vid, cat in sorted(self.categories().items()):
                writer.writerow([vid, cat])
