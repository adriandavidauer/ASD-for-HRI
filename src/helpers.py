import logging
import os
from datetime import datetime

LOGGER = logging.getLogger(__name__)


def setup_logging(log_name='unitalk', verbose=False, log_dir='/app/data/logs'):
    """Attach a debug file handler and a console handler to the root logger."""
    os.makedirs(log_dir, exist_ok=True)
    path = f'{log_dir}/{log_name}_{datetime.now():%Y%m%d_%H%M%S}.log'

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
