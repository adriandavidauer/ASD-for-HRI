"""Tests for the AVA landmark extraction used for fine-tuning (#92)."""

from pathlib import Path

import cv2
import numpy as np

from asd4hri.asd import ASD
from asd4hri.processors import GetShapeFeatures
from src.experiments.run_asd_on_video import _GroundTruthFrames
from src.model_training.extract_ava_landmarks import extract_video, read_annotations

VIDEO = Path(__file__).parent / '001.avi'


def _write_csv(path, fps, frames=12):
    """AVA-style annotations for two faces on the first ``frames`` frames of the test video.

    Args:
        path: CSV to write.
        fps: Frame rate of the video.
        frames: Number of annotated frames.
    """
    with open(path, 'w') as f:
        for i in range(frames):
            ts = round(i / fps, 3)
            f.write(f'001,{ts},0.1,0.1,0.5,0.6,SPEAKING_AUDIBLE,001:a\n')
            label = 'NOT_SPEAKING' if i % 2 else 'SPEAKING_NOT_AUDIBLE'
            f.write(f'001,{ts},0.45,0.2,0.9,0.95,{label},001:b\n')


def test_extracted_landmarks_are_those_of_the_pipeline(tmp_path, monkeypatch):
    """The landmarks and labels equal what ASD computes for the same GT boxes."""
    fps = cv2.VideoCapture(str(VIDEO)).get(cv2.CAP_PROP_FPS)
    csv_path = tmp_path / '001-activespeaker.csv'
    _write_csv(csv_path, fps)
    n = extract_video(VIDEO, csv_path, tmp_path / 'out.npz')
    data = np.load(tmp_path / 'out.npz')
    assert n == len(data['entity']) == 24

    # the same frames through the pipeline, recording what GetShapeFeatures returns
    seen = []
    original = GetShapeFeatures.call
    def record(self, images):
        """Call GetShapeFeatures and keep its output.

        Returns:
            list: The landmarks.
        """
        out = original(self, images)
        seen.extend(out)
        return out
    monkeypatch.setattr(GetShapeFeatures, 'call', record)
    asd = ASD(architecture='FaceShapeCausal')
    groundtruth, labels = read_annotations(csv_path)
    index = _GroundTruthFrames(groundtruth, fps)
    cap = cv2.VideoCapture(str(VIDEO))
    for i in range(12):
        _, frame = cap.read()
        _, gt_frame = index.at(i / fps)
        boxes, ids = index.boxes2D(gt_frame, frame.shape[1], frame.shape[0])
        asd(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), boxes, ids, timestamp=i / fps)
    np.testing.assert_array_equal(data['landmarks'], np.stack(seen))
    # only SPEAKING_AUDIBLE is positive
    assert data['label'].tolist() == [1, 0] * 12
    assert data['entity'].tolist() == ['001:a', '001:b'] * 12
