import numpy as np

def frequency_step(f0=60.0, f_step=59.5, t_step=2.0, t_back=4.0, duration=6.0, fs=5000):
    """
    Generate a sinusoidal waveform with a step change in frequency.

    Parameters
    ----------
    f0 : float
        Nominal frequency (Hz).
    f_step : float
        Frequency during disturbance (Hz).
    t_step : float
        Time (s) when frequency steps from f0 to f_step.
    t_back : float
        Time (s) when frequency returns to f0.
    duration : float
        Total duration (s).
    fs : float
        Sampling frequency (Hz).

    Returns
    -------
    signal : np.ndarray
        Generated sinusoidal signal.
    f : np.ndarray
        Instantaneous frequency array (Hz).
    """
    t = np.arange(0, duration, 1/fs)

    # Piecewise frequency profile
    f = np.full_like(t, f0, dtype=float)
    f[(t >= t_step) & (t < t_back)] = f_step  # apply disturbance

    # Phase accumulation
    theta = 2 * np.pi * np.cumsum(f) / fs
    signal = np.sin(theta)

    return signal, f

