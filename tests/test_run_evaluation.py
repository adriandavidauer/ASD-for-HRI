"""Tests for the output folder handling in experiments/run_evaluation.py."""

from pathlib import Path

import pytest

from experiments.run_evaluation import relative_to_data_dir


@pytest.mark.parametrize('path, expected', [
    ('predictions_x', 'predictions_x'),
    ('/app/data/predictions_x', 'predictions_x'),
    ('/app/data/runs/predictions_x/', 'runs/predictions_x'),
])
def test_folder_made_relative_to_data_dir(path, expected):
    """Relative folders are kept; absolute ones inside data_dir become relative.

    Args:
        path: Folder as passed on the command line.
        expected: Folder relative to the data dir.
    """
    assert relative_to_data_dir('/app/data', path) == Path(expected)


@pytest.mark.parametrize('path', ['/elsewhere/predictions_x', '/app/data/../predictions_x'])
def test_folder_outside_data_dir_fails(path):
    """An absolute folder outside data_dir is rejected.

    Args:
        path: Folder outside the data dir.
    """
    with pytest.raises(ValueError, match='not inside --data_dir'):
        relative_to_data_dir('/app/data', path)
