"""Tests for the EasyCom conversion (#82)."""

import csv
import json

import pytest

from src.experiments.easycom_dataset import EasyComDataset, convert_file


def test_convert_file_labels_frames_inside_speech_segments():
    boxes = [{'Frame_Number': n, 'Participants': [
        {'Participant_ID': 3, 'x1': 960, 'y1': 540, 'x2': 1152, 'y2': 756},
        {'Participant_ID': 4, 'x1': 0, 'y1': 0, 'x2': 96, 'y2': 108}]} for n in (1, 2, 3, 21)]
    speech = [{'Start_Frame': 2, 'End_Frame': 3, 'Participant_ID': 3, 'Transcription': 'hi',
               'Target_of_Speech': ['Group']}]
    rows = convert_file('Session_9_00-00-000', boxes, speech)
    assert len(rows) == 8
    by = {(r[1], r[7]): r for r in rows}
    # frame 1 -> t=0, frame 21 -> t=1.0 s at 20 fps
    assert (0.0, 'Session_9_00-00-000:3') in by
    assert by[(1.0, 'Session_9_00-00-000:3')][6] == 'NOT_SPEAKING'
    assert by[(0.05, 'Session_9_00-00-000:3')][6] == 'SPEAKING_AUDIBLE'
    assert by[(0.1, 'Session_9_00-00-000:3')][6] == 'SPEAKING_AUDIBLE'
    assert by[(0.05, 'Session_9_00-00-000:4')][6] == 'NOT_SPEAKING'
    # normalised by the 1920x1080 camera image
    assert by[(0.0, 'Session_9_00-00-000:3')][2:6] == [0.5, 0.5, 0.6, 0.7]


def test_dataset_writes_groundtruth_for_the_split(tmp_path):
    main = tmp_path / 'Main'
    for sub in ('Video_Compressed', 'Face_Bounding_Boxes', 'Speech_Transcriptions'):
        (main / sub / 'Session_10').mkdir(parents=True)
    (main / 'Video_Compressed' / 'Session_10' / '00-00-000.mp4').write_bytes(b'')
    (main / 'Face_Bounding_Boxes' / 'Session_10' / '00-00-000.json').write_text(json.dumps(
        [{'Frame_Number': 1, 'Participants': [
            {'Participant_ID': 1, 'x1': 10, 'y1': 10, 'x2': 50, 'y2': 50}]}]))
    # Latin-1 transcription, as in some real EasyCom files
    (main / 'Speech_Transcriptions' / 'Session_10' / '00-00-000.json').write_bytes(json.dumps(
        [{'Start_Frame': 1, 'End_Frame': 5, 'Participant_ID': 1, 'Transcription': 'café',
          'Target_of_Speech': ['Group']}], ensure_ascii=False).encode('latin-1'))
    ds = EasyComDataset(str(tmp_path), split='test')
    assert [v for v, _p, _u in ds.video_list()] == ['Session_10_00-00-000']
    with open(ds.groundtruth_csv) as f:
        rows = list(csv.DictReader(f))
    assert rows[0]['label'] == 'SPEAKING_AUDIBLE'
    with pytest.raises(ValueError):
        EasyComDataset(str(tmp_path), split='dev')
