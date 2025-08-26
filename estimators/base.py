class EstimatorBase:
    """Base class for all frequency estimators."""
    def __init__(self, fs, frame_len, **kwargs):
        self.fs = fs
        self.frame_len = frame_len
        self.params = kwargs

    def reset(self):
        """Reset internal state."""
        pass

    def update(self, x):
        """Update with a chunk of samples."""
        raise NotImplementedError

    def report(self):
        """Return current frequency estimate and diagnostics."""
        return {"f": None, "rocof": None, "theta": None, "latency": None}
