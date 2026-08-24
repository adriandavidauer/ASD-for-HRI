import logging
import os
from datetime import datetime

LOGGER = logging.getLogger(__name__)

CONTAINMENT_THRESHOLD = 0.5  # accept a match when the smaller box is ≥50% covered

_LABEL_MAP = {
    'SPEAKING_AUDIBLE': 'speaking',
    'NOT_SPEAKING':     'not-speaking',
}

_PredBox = namedtuple('_PredBox', ['coordinates', 'class_name'])


def setup_logging(log_name="unitalk", verbose=False, path=None):
    if path is None:
        path = '/app/data/logs'
    os.makedirs(path, exist_ok=True)
    path = f'{path}/{log_name}_{datetime.now():%Y%m%d_%H%M%S}.log'

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    fh = logging.FileHandler(path, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO if verbose else logging.WARNING)
    ch.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    root.addHandler(ch)

    LOGGER.info('logging initialised file=%s', path)
    return path
