import numpy as np

def make_clean(f0=60.0, df=0.0, duration=5.0, fs=5000):
    """Generate a clean sinusoid with optional frequency ramp."""
    t = np.arange(0, duration, 1/fs)
    f = f0 + df * t / duration
    theta = 2 * np.pi * np.cumsum(f) / fs
    signal = np.sin(theta)
    return signal, f
