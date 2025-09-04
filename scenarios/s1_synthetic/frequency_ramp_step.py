import numpy as np

def frequency_ramp_step(f0=60.0, f_step=59.5, t_step=2.0, t_back=4.0,
                        duration=6.0, fs=5000, rocof=1.0):
    """
    Generate a sinusoidal waveform with a smooth frequency disturbance:
    ramp from f0 to f_step at a given RoCoF, hold, then ramp back.

    Parameters
    ----------
    f0 : float
        Nominal frequency (Hz).
    f_step : float
        Disturbance frequency (Hz).
    t_step : float
        Time (s) when ramp to f_step begins.
    t_back : float
        Time (s) when ramp back to f0 begins.
    duration : float
        Total duration (s).
    fs : float
        Sampling frequency (Hz).
    rocof : float
        Rate of change of frequency in Hz/s.

    Returns
    -------
    signal : np.ndarray
        Generated sinusoidal signal.
    f : np.ndarray
        Instantaneous frequency array (Hz).
    """
    t = np.arange(0, duration, 1/fs)
    f = np.full_like(t, f0, dtype=float)

    # Time needed to complete ramp
    ramp_time = abs(f_step - f0) / rocof

    for i, ti in enumerate(t):
        if ti < t_step:
            # Before disturbance
            f[i] = f0
        elif t_step <= ti < t_step + ramp_time:
            # Ramp from f0 to f_step
            f[i] = f0 + np.sign(f_step - f0) * rocof * (ti - t_step)
        elif t_step + ramp_time <= ti < t_back:
            # Hold at f_step
            f[i] = f_step
        elif t_back <= ti < t_back + ramp_time:
            # Ramp back to f0
            f[i] = f_step - np.sign(f_step - f0) * rocof * (ti - t_back)
        else:
            # After returning
            f[i] = f0

    # Accumulate phase
    theta = 2 * np.pi * np.cumsum(f) / fs
    signal = np.sin(theta)

    return signal, f
