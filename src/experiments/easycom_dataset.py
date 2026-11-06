r"""EasyCom dataset adapter (#82): AR-glasses conversations, converted to AVA-style ASD labels.

EasyCom (Donley et al. 2021, https://github.com/facebookresearch/EasyComDataset, CC-BY-NC-4.0)
has no official ASD protocol, so this module defines one:

* ``video_id`` = ``Session_<s>_<file>`` (one-minute files, 20 fps, frames numbered 1..1200),
  ``entity_id`` = ``<video_id>:<participant id>``.
* Boxes: the face boxes of ``Main/Face_Bounding_Boxes`` are in pixels of the full video frame,
  which has a participant banner on the left; the banner width (video width - 1920) is
  subtracted and the boxes are normalised by the 1920x1080 camera image.
  ``frame_timestamp`` = (frame - 1) / 20.
* Label: SPEAKING_AUDIBLE if the frame lies inside one of the participant's
  ``Speech_Transcriptions`` segments [Start_Frame, End_Frame], else NOT_SPEAKING. The camera
  wearer has no face box and does not appear.
* Splits from the EasyCom README: train 1,2,3,5-9; val 4,12; test 10,11; plus 'all'.
* The videos have a participant-photo banner on the left (width 2123-2203 px), so the pipeline
  crops every frame to the rightmost 1920 px (``frame_crop``), the image the boxes refer to.

Download (only the needed folders, ~36 GB, streamed so the 70 GB archive is never stored)::

    for i in $(seq -w 1 33); do curl -sSfL \\
      https://github.com/facebookresearch/EasyComDataset/releases/download/v1.0.0/EasyComDataset.tar.gz.0$i
    done | tar -xz --wildcards '*Main/Video_Compressed/*' '*Main/Face_Bounding_Boxes/*' \\
                               '*Main/Speech_Transcriptions/*'
"""

import csv
import glob
import json
import os
import subprocess

SPLITS = {
    'train': [1, 2, 3, 5, 6, 7, 8, 9],
    'val': [4, 12],
    'test': [10, 11],
    'all': list(range(1, 13)),
}
FPS = 20.0
WIDTH, HEIGHT = 1920, 1080
_COLUMNS = ['video_id', 'frame_timestamp', 'entity_box_x1', 'entity_box_y1', 'entity_box_x2',
            'entity_box_y2', 'label', 'entity_id']


def _load_json(path):
    """Read an EasyCom JSON file; a few transcription files are Latin-1, not UTF-8.

    Args:
        path: The JSON file.

    Returns:
        The parsed content.
    """
    raw = open(path, 'rb').read()
    try:
        return json.loads(raw.decode('utf-8'))
    except UnicodeDecodeError:
        return json.loads(raw.decode('latin-1'))


def convert_file(video_id, boxes_json, speech_json, x_offset=0):
    """AVA-style GT rows for one one-minute file.

    Args:
        video_id: Id written into the rows.
        boxes_json: Parsed ``Face_Bounding_Boxes`` file (list of frames).
        speech_json: Parsed ``Speech_Transcriptions`` file (list of segments), or [].
        x_offset: Width of the video's banner in pixels. The face boxes are in pixels of the
            full video frame (banner included), so it is subtracted before normalising.

    Returns:
        list[list]: Rows in ``_COLUMNS`` order.
    """
    speaking = {}
    for seg in speech_json:
        speaking.setdefault(int(seg['Participant_ID']), []).append(
            (int(seg['Start_Frame']), int(seg['End_Frame'])))
    rows = []
    for frame in boxes_json:
        number = int(frame['Frame_Number'])
        ts = round((number - 1) / FPS, 3)
        for p in frame.get('Participants', []):
            pid = int(p['Participant_ID'])
            x1 = max(0.0, (p['x1'] - x_offset) / WIDTH)
            x2 = min(1.0, (p['x2'] - x_offset) / WIDTH)
            y1, y2 = max(0.0, p['y1'] / HEIGHT), min(1.0, p['y2'] / HEIGHT)
            if x2 <= x1 or y2 <= y1:
                continue
            talking = any(s <= number <= e for s, e in speaking.get(pid, ()))
            rows.append([video_id, ts, round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6),
                         'SPEAKING_AUDIBLE' if talking else 'NOT_SPEAKING', f'{video_id}:{pid}'])
    return rows


class EasyComDataset:
    """Video list, ground truth and frame crops for one EasyCom split.

    Args:
        root: Directory that contains ``Main/`` (e.g. ``data/easycom``).
        split: 'train', 'val', 'test' (default) or 'all'.

    Raises:
        ValueError: If ``split`` is unknown.
    """

    def __init__(self, root, split='test'):
        """Index the videos of the split and write the GT CSV if missing.

        Raises:
            ValueError: If ``split`` is unknown.
        """
        if split not in SPLITS:
            raise ValueError(f'split must be one of {list(SPLITS)}, not {split}')
        self.root, self.split = root, split
        self.main = os.path.join(root, 'Main')
        self.groundtruth_csv = os.path.join(root, 'csv', f'{split}_orig.csv')
        self.videos = {}
        for session in SPLITS[split]:
            pattern = os.path.join(self.main, 'Video_Compressed', f'Session_{session}', '*.mp4')
            for path in sorted(glob.glob(pattern)):
                stem = os.path.splitext(os.path.basename(path))[0]
                self.videos[f'Session_{session}_{stem}'] = (session, stem, path)
        if not os.path.isfile(self.groundtruth_csv):
            self.write_groundtruth()

    def write_groundtruth(self):
        """Convert the split's face boxes + transcriptions into one AVA-format CSV."""
        os.makedirs(os.path.dirname(self.groundtruth_csv), exist_ok=True)
        tmp = self.groundtruth_csv + '.tmp'
        with open(tmp, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(_COLUMNS)
            for video_id, (session, stem, path) in self.videos.items():
                boxes = os.path.join(self.main, 'Face_Bounding_Boxes', f'Session_{session}',
                                     f'{stem}.json')
                speech = os.path.join(self.main, 'Speech_Transcriptions', f'Session_{session}',
                                      f'{stem}.json')
                if not os.path.isfile(boxes):
                    continue
                writer.writerows(convert_file(video_id, _load_json(boxes),
                                              _load_json(speech) if os.path.isfile(speech)
                                              else [], x_offset=self.frame_crop(path)[0]))
        os.replace(tmp, self.groundtruth_csv)

    def video_list(self):
        """List the split's videos in the format of ``load_video_list``.

        Returns:
            list[tuple]: ``(video_id, path, None)``.
        """
        return [(vid, path, None) for vid, (_s, _f, path) in self.videos.items()]

    @staticmethod
    def frame_crop(video_path):
        """Region of the camera image inside a video frame (the banner removed).

        Args:
            video_path: The video.

        Returns:
            tuple: ``(x0, y0, x1, y1)`` in pixels.
        """
        out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                              '-show_entries', 'stream=width,height', '-of', 'csv=p=0',
                              str(video_path)], capture_output=True, text=True, check=True)
        width, height = (int(v) for v in out.stdout.strip().split(',')[:2])
        return width - WIDTH, 0, width, min(height, HEIGHT)
