"""Per-stage timing of the ASD pipeline (#76)."""

import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np


class StageTimer:
    """Collect wall-clock times of named pipeline stages.

    Each ``with timer.stage(name, items):`` block adds one call of ``name`` together with the
    number of items (e.g. faces) it processed. The overhead is about one microsecond per call.
    TensorFlow/PyTorch/ONNX Runtime return numpy arrays on the CPU, so a GPU model's time
    includes the device synchronisation.

    Args:
        enabled: If False, ``stage`` does nothing and nothing is recorded.
    """

    def __init__(self, enabled=True):
        """Start with no records."""
        self.enabled = enabled
        self.reset()

    def reset(self):
        """Forget all recorded calls."""
        self.seconds = defaultdict(list)
        self.items = defaultdict(list)

    @contextmanager
    def stage(self, name, items=None):
        """Time the enclosed block as one call of stage ``name``.

        Args:
            name: Stage name, e.g. 'detect' or 'model'.
            items: Number of items processed in this call (e.g. faces), or None.

        Yields:
            None
        """
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name].append(time.perf_counter() - start)
            self.items[name].append(items)

    def summary(self):
        """Aggregate the recorded calls per stage.

        Returns:
            list[dict]: One dict per stage, in the order the stages were first seen, with
            ``stage``, ``calls``, ``total_s``, ``mean_ms``, ``median_ms``, ``p95_ms``,
            ``max_ms``, ``items`` (sum, or None if never given) and ``ms_per_item``.
        """
        rows = []
        for name, secs in self.seconds.items():
            arr = np.asarray(secs) * 1000.0
            counts = [n for n in self.items[name] if n is not None]
            items = int(sum(counts)) if counts else None
            rows.append({
                'stage': name,
                'calls': len(arr),
                'total_s': float(arr.sum() / 1000.0),
                'mean_ms': float(arr.mean()),
                'median_ms': float(np.median(arr)),
                'p95_ms': float(np.percentile(arr, 95)),
                'max_ms': float(arr.max()),
                'items': items,
                'ms_per_item': float(arr.sum() / items) if items else None,
            })
        return rows


NULL_TIMER = StageTimer(enabled=False)
