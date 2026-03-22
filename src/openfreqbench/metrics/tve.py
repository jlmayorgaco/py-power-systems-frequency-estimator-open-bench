import numpy as np

def compute_tve(
    a_est: np.ndarray, p_est: np.ndarray, a_true: np.ndarray, p_true: np.ndarray
) -> np.ndarray:
    \"\"\"
    Compute the Total Vector Error (TVE) percentage elementwise
    given amplitude (a) and phase (p) of the estimated and true phasors.
    
    IEEE C37.118-compliant formula:
    TVE_n = sqrt( (Xr_est - Xr_true)^2 + (Xi_est - Xi_true)^2 ) / sqrt(Xr_true^2 + Xi_true^2) * 100
    \"\"\"
    ae = np.asarray(a_est, dtype=float)
    pe = np.asarray(p_est, dtype=float)
    at = np.asarray(a_true, dtype=float)
    pt = np.asarray(p_true, dtype=float)
    
    # Check boundaries
    n = min(ae.size, pe.size, at.size, pt.size)
    if n <= 0:
        return np.array([])
        
    ae, pe, at, pt = ae[:n], pe[:n], at[:n], pt[:n]
    
    # Compute Re and Im directly
    xr_est, xi_est = ae * np.cos(pe), ae * np.sin(pe)
    xr_true, xi_true = at * np.cos(pt), at * np.sin(pt)
    
    err_sq = (xr_est - xr_true)**2 + (xi_est - xi_true)**2
    true_sq = xr_true**2 + xi_true**2
    
    # Avoid zero division
    valid_sq = np.clip(true_sq, 1e-12, None)
    
    tve_pct = np.sqrt(err_sq / valid_sq) * 100.0
    return tve_pct

def compute_tve_complex(v_est: np.ndarray, v_true: np.ndarray) -> np.ndarray:
    \"\"\"
    Compute the Total Vector Error (TVE) on complex phasor arrays directly.
    \"\"\"
    ve = np.asarray(v_est, dtype=complex)
    vt = np.asarray(v_true, dtype=complex)
    n = min(ve.size, vt.size)
    if n <= 0:
        return np.array([])
    ve, vt = ve[:n], vt[:n]
    
    err_mag = np.abs(ve - vt)
    true_mag = np.clip(np.abs(vt), 1e-12, None)
    
    return (err_mag / true_mag) * 100.0
