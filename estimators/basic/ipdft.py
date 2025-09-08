import numpy as np
from estimators.base import EstimatorBase

class IpDFT(EstimatorBase):
    """
    Interpolated DFT (IpDFT) frequency estimator with a sliding buffer.
    Works online: one sample at a time, maintains internal buffer.
    """

    def __init__(self, fs, frame_len=50, **kwargs):
        super().__init__(fs, frame_len, **kwargs)
        self.buffer = np.zeros(frame_len, dtype=float)
        self.ptr = 0
        self.filled = False
        self.last_freq = None

    def update(self, sample):
        """
        Feed one sample at a time, return current frequency estimate (Hz).
        Returns None until the buffer is full.
        """
        self.buffer[self.ptr] = sample
        self.ptr = (self.ptr + 1) % self.frame_len

        if not self.filled and self.ptr == 0:
            self.filled = True

        if not self.filled:
            return None  # not enough data yet

        # Perform FFT
        X = np.fft.fft(self.buffer, n=self.frame_len)
        mag = np.abs(X[: self.frame_len // 2])
        k = np.argmax(mag)  # peak bin

        # Neighbor interpolation
        if 1 <= k < len(mag) - 1:
            delta = 0.5 * (mag[k - 1] - mag[k + 1]) / (mag[k - 1] - 2 * mag[k] + mag[k + 1])
        else:
            delta = 0.0

        freq = (k + delta) * self.fs / self.frame_len
        self.last_freq = freq
        return freq
