import numpy as np
from estimators.base import EstimatorBase

class IpDFT(EstimatorBase):
    """Interpolated DFT estimator (simplified)."""

    def update(self, x):
        X = np.fft.fft(x, n=self.frame_len)
        mag = np.abs(X[: self.frame_len // 2])
        k = np.argmax(mag)
        if k == 0 or k == len(mag) - 1:
            return k * self.fs / self.frame_len
        # quadratic interpolation
        alpha = mag[k - 1]
        beta = mag[k]
        gamma = mag[k + 1]
        p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
        return (k + p) * self.fs / self.frame_len
