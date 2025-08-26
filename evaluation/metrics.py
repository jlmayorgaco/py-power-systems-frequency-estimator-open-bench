import numpy as np

def frequency_error(f_hat, f_true):
    """RMSE of frequency estimate vs ground truth."""
    f_hat = np.array(f_hat)
    f_true = np.array(f_true[:len(f_hat)])
    return np.sqrt(np.mean((f_hat - f_true)**2))
