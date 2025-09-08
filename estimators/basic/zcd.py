import numpy as np
from estimators.base import EstimatorBase

class ZeroCrossing(EstimatorBase):
    """
    Online Zero-Crossing frequency estimator.

    Works by detecting negative→positive zero crossings.
    Each crossing gives a period estimate -> frequency = 1 / period.
    """

    def __init__(self, fs, **kwargs):
        super().__init__(fs, frame_len=1, **kwargs)  # no frame_len needed
        self.prev_sample = None
        self.prev_cross_idx = None
        self.last_freq = None
        self.sample_idx = 0

    def update(self, x):
        """
        Feed one sample at a time (online).
        Returns current frequency estimate (Hz).
        """
        # Ensure scalar
        x = float(np.squeeze(x))
        self.sample_idx += 1

        # First sample
        if self.prev_sample is None:
            self.prev_sample = x
            return None

        # Check for zero crossing (negative → positive)
        if self.prev_sample < 0 and x >= 0:
            if self.prev_cross_idx is not None:
                period_samples = self.sample_idx - self.prev_cross_idx
                if period_samples > 0:
                    self.last_freq = self.fs / period_samples
            self.prev_cross_idx = self.sample_idx

        self.prev_sample = x
        return self.last_freq
