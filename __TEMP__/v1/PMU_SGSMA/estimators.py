#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from collections import deque
import math

from ekf2 import EKF2  # Asegúrate de tener ekf2.py en el mismo directorio

# Intento opcional de importar PyTorch para el estimador PI-GRU
try:
    import torch
except ImportError:  # pragma: no cover - entorno sin torch
    torch = None

# =============================================================
# 0. CONFIGURACIÓN GLOBAL (SOLO NÚMEROS, SIN PLOTTING)
# =============================================================
SEED = 42
np.random.seed(SEED)

FS_PHYSICS = 1000000.0  # 1 MHz (Physics/Ground Truth)
FS_DSP = 10000.0  # 10 kHz (IED/Relay Sampling Rate)
RATIO = int(FS_PHYSICS / FS_DSP)
DT_DSP = 1.0 / FS_DSP
F_NOM = 60.0

# Umbrales de error para métricas
SETTLING_THRESHOLD = 0.2  # Hz
TRIP_THRESHOLD = 0.5  # Hz


# =============================================================
# 1. SCENARIO GENERATOR (With IEEE + IBR Multi-Event)
# =============================================================
def get_test_signals():
    # Escenarios cortos (1.5 s)
    t = np.arange(0, 1.5, 1.0 / FS_PHYSICS)
    n = len(t)
    signals = {}

    # --- A: MAGNITUDE STEP (IEEE C37.118.1) ---
    f_a = np.ones(n) * 60.0
    amp_a = np.ones(n)
    amp_a[t > 0.5] = 1.1  # +10% Step
    phi_a = 2 * np.pi * 60.0 * t
    v_a = amp_a * np.sin(phi_a) + np.random.normal(0, 0.001, n)

    meta_a = {
        "description": "Sudden Voltage Magnitude Step",
        "parameters": "Step +10% magnitude at t=0.5s",
        "harmonics": "None",
        "noise": "Gaussian White Noise (sigma=0.001 pu)",
        "dynamics": "Steady State Frequency (60Hz), Amplitude Transient",
    }
    signals["IEEE_Mag_Step"] = (t, v_a, f_a, meta_a)

    # --- B: FREQUENCY RAMP (High ROCOF) ---
    f_b = np.ones(n) * 60.0
    mask_ramp = (t > 0.3) & (t < 1.0)
    # 5 Hz/s Ramp (Aggressive for IBR)
    f_b[mask_ramp] = 60.0 + 5.0 * (t[mask_ramp] - 0.3)
    f_b[t >= 1.0] = f_b[t < 1.0][-1]
    phi_b = np.cumsum(2 * np.pi * f_b * (1.0 / FS_PHYSICS))
    v_b = np.sin(phi_b) + np.random.normal(0, 0.001, n)

    meta_b = {
        "description": "High ROCOF Frequency Ramp",
        "parameters": "Ramp +5 Hz/s starting at t=0.3s, ending t=1.0s",
        "harmonics": "None",
        "noise": "Gaussian White Noise (sigma=0.001 pu)",
        "dynamics": "Linear Frequency Change (Quadratic Phase)",
    }
    signals["IEEE_Freq_Ramp"] = (t, v_b, f_b, meta_b)

    # --- C: MODULATION (AM/FM) ---
    fm = 2.0
    kx = 0.1
    f_c = np.ones(n) * 60.0
    # AM Modulation
    mod_signal = 1.0 + kx * np.cos(2 * np.pi * fm * t)
    v_c = mod_signal * np.sin(2 * np.pi * 60.0 * t) + np.random.normal(0, 0.001, n)

    meta_c = {
        "description": "Amplitude Modulation (AM/LFO)",
        "parameters": "AM Depth 10% (kx=0.1), Mod Freq 2 Hz",
        "harmonics": "None",
        "noise": "Gaussian White Noise (sigma=0.001 pu)",
        "dynamics": "Steady Frequency with Inter-harmonic interference",
    }
    signals["IEEE_Modulation"] = (t, v_c, f_c, meta_c)

    # --- D: NIGHTMARE (Phase Jump + Harmonics) ---
    f_d = np.ones(n) * 60.0
    phase_accum = np.zeros(n)
    curr_phi = 0.0
    for i in range(n):
        if i > 0:
            # Phase Jump at 0.7s (Simulates Islanding/Switching)
            if 0.6999 < t[i] < 0.7001:
                curr_phi += np.pi / 3.0  # 60 deg jump
            curr_phi += 2 * np.pi * 60.0 * (1.0 / FS_PHYSICS)
        phase_accum[i] = curr_phi

    v_d = np.sin(phase_accum)
    v_d += 0.05 * np.sin(5 * phase_accum)  # 5th Harmonic
    v_d += 0.02 * np.sin(2 * np.pi * 32.5 * t)  # Inter-harmonic
    v_d += np.random.normal(0, 0.005, n)  # Higher Noise floor

    meta_d = {
        "description": "IBR Islanding Scenario (True Nightmare)",
        "parameters": "Instantaneous Phase Jump +60 deg at t=0.7s",
        "harmonics": "5th Harmonic (5%), Inter-harmonic 32.5Hz (2%)",
        "noise": "High Gaussian Noise (sigma=0.005 pu)",
        "dynamics": "Phase Discontinuity (Dirac delta frequency)",
    }
    signals["IBR_Nightmare"] = (t, v_d, f_d, meta_d)

    # --- E: IBR MULTI EVENT (Super Escenario 5s) ---
    t_e = np.arange(0, 5.0, 1.0 / FS_PHYSICS)
    n_e = len(t_e)

    # Frecuencia base
    f_e = np.ones(n_e) * 60.0

    # Segmentos de tiempo
    seg1 = t_e < 1.0
    seg2 = (t_e >= 1.0) & (t_e < 1.2)
    seg3 = (t_e >= 1.2) & (t_e < 2.5)
    seg4 = (t_e >= 2.5) & (t_e < 3.5)
    seg5 = t_e >= 3.5

    # 3) ROCOF negativo rápido: 1.2s -> 2.5s (de 60 a 52 Hz) ~ -6 Hz/s
    f_e[seg3] = 60.0 - 6.0 * (t_e[seg3] - 1.2)

    # 4) Ring-down de 2º orden (3 Hz, amortiguado) en 2.5–3.5s
    tau = t_e[seg4] - 2.5
    A_osc = 2.0  # amplitud en Hz
    sigma = 3.0  # amortiguamiento
    f_osc = 3.0  # frecuencia de la oscilación en Hz
    f_e[seg4] = 60.0 + A_osc * np.exp(-sigma * tau) * np.sin(2.0 * np.pi * f_osc * tau)
    # seg1, seg2, seg5 ya están en 60 Hz

    # Construir fase con integración de frecuencia + saltos de fase
    phi_e = np.zeros(n_e)
    curr_phi = 0.0
    for i in range(1, n_e):
        # Saltos de fase:
        if 0.999999 < t_e[i] < 1.000001:  # ~1.0 s
            curr_phi += np.deg2rad(40.0)  # +40°
        if 2.499999 < t_e[i] < 2.500001:  # ~2.5 s
            curr_phi += np.deg2rad(80.0)  # +80°

        curr_phi += 2.0 * np.pi * f_e[i] * (1.0 / FS_PHYSICS)
        phi_e[i] = curr_phi

    # Amplitud con pequeña dinámica
    amp_e = np.ones(n_e)
    # Pequeña caída de amplitud durante el ROCOF
    amp_e[seg3] = 1.0 - 0.15 * (t_e[seg3] - 1.2) / (2.5 - 1.2)  # cae ~15%
    # Overshoot de amplitud y recuperación en el ring-down
    amp_e[seg4] = 0.85 + 0.2 * np.exp(-sigma * tau) * np.cos(2.0 * np.pi * f_osc * tau)
    # Post-fault un poco más “sucia”
    amp_e[seg5] = 1.05

    # Señal base + armónicos
    v_base = amp_e * np.sin(phi_e)
    # 5ª y 7ª armónicas tipo IBR
    v_base += 0.05 * np.sin(5 * phi_e)  # 5%
    v_base += 0.03 * np.sin(7 * phi_e)  # 3%

    # Ruido gaussiano + impulsivo (no gaussiano)
    noise_gauss = np.random.normal(0, 0.003, n_e)
    impulse_mask = np.random.rand(n_e) < 5e-4  # impulsos raros
    noise_imp = impulse_mask * np.random.normal(0, 0.05, n_e)
    v_e = v_base + noise_gauss + noise_imp

    meta_e = {
        "description": "IBR Multi-Event Super Scenario",
        "parameters": (
            "5 s: (0-1s) IBR steady w/ 5th & 7th; "
            "(1.0s) +40° phase jump; "
            "(1.2-2.5s) ROCOF -6 Hz/s; "
            "(2.5s) +80° phase jump; "
            "(2.5-3.5s) 2nd-order ring-down in freq; "
            "(3.5-5s) harmonic-rich steady-state w/ impulsive noise."
        ),
        "harmonics": "5th (5%), 7th (3%), impulsive noise spikes",
        "noise": "Gaussian (σ=0.003) + rare impulsive spikes (σ=0.05, p≈5e-4)",
        "dynamics": "Composite: phase jumps, fast ROCOF, ring-down, non-Gaussian disturbances.",
    }

    signals["IBR_MultiEvent"] = (t_e, v_e, f_e, meta_e)

    return signals


# =============================================================
# 2. SIGNAL PROCESSING HELPERS (Filters & Normalizers)
# =============================================================
class IIR_Bandpass:
    """2nd Order IIR Bandpass Filter (Butterworth).
    Center: 60Hz, Bandwidth: 40Hz (Wide enough for ramps).
    """

    def __init__(self):
        w0 = 2 * np.pi * 60.0 / FS_DSP
        bw = 2 * np.pi * 40.0 / FS_DSP
        R = 1.0 - (bw / 2.0)
        self.a1 = 2.0 * R * np.cos(w0)
        self.a2 = -(R * R)
        self.b0 = (1.0 - self.a2) / 2.0 * 0.5
        self.b2 = -self.b0
        self.x = deque([0.0, 0.0], maxlen=2)
        self.y = deque([0.0, 0.0], maxlen=2)

    def step(self, v_in):
        v_out = (
            self.b0 * v_in
            + self.b2 * self.x[1]
            + self.a1 * self.y[0]
            + self.a2 * self.y[1]
        )
        self.x.appendleft(v_in)
        self.y.appendleft(v_out)
        return v_out


class FastRMS_Normalizer:
    """Fast Automatic Gain Control (AGC) using sliding RMS.
    Decouples amplitude dynamics from frequency estimation.
    """

    def __init__(self):
        # Window: 1/2 cycle for fast response vs ripple trade-off
        self.win_len = int(FS_DSP / 60.0 / 2.0)
        self.buf = deque(maxlen=self.win_len)

    def step(self, val):
        sq_val = val * val
        self.buf.append(sq_val)
        rms = np.sqrt(np.mean(self.buf))
        if rms < 0.1:
            rms = 0.1  # Safety floor
        return val / (rms * 1.41421356)  # Normalize peak to ~1.0


# =============================================================
# 3. ESTIMATION ALGORITHMS (BENCHMARK SUITE)
# =============================================================


# --- 3.1 BASELINE: IpDFT (Interpolated DFT) ---
class TunableIpDFT:
    def __init__(self, cycles):
        self.sz = int((FS_DSP / 60.0) * cycles)
        self.buf = deque(maxlen=self.sz)
        self.win = np.hanning(self.sz)
        self.res = FS_DSP / self.sz
        self.name = f"IpDFT_{cycles}cyc"

    def step(self, z):
        self.buf.append(z)
        if len(self.buf) < self.sz:
            return 60.0
        sp = np.abs(np.fft.rfft(np.array(self.buf) * self.win))
        k = np.argmax(sp)
        if k == 0 or k == len(sp) - 1:
            return k * self.res
        den = sp[k - 1] + 2 * sp[k] + sp[k + 1]
        delta = 0 if den == 0 else 2.0 * (sp[k + 1] - sp[k - 1]) / den
        return (k + delta) * self.res


# --- 3.2 INDUSTRY STANDARD: SRF-PLL (with MAF) ---
class StandardPLL:
    def __init__(self, kp, ki):
        self.kp = kp
        self.ki = ki
        self.integrator = 0.0
        self.theta = 0.0
        self.name = "SRF-PLL"
        self.maf_win = int(FS_DSP / 60.0)  # 1-cycle Moving Average Filter
        self.buf = deque(maxlen=self.maf_win)

    def step(self, z):
        # Phase Detector (Park-based simplification)
        pd_out = z * np.cos(self.theta)
        # PI Controller
        self.integrator += self.ki * pd_out * DT_DSP
        w_dev = self.kp * pd_out + self.integrator
        w_raw = 2 * np.pi * 60.0 + w_dev
        # Oscillator
        self.theta += w_raw * DT_DSP
        self.theta %= 2 * np.pi
        # Output Filter (MAF on instantaneous frequency)
        self.buf.append(w_raw)
        if len(self.buf) < self.maf_win:
            return w_raw / (2 * np.pi)
        else:
            return np.mean(self.buf) / (2 * np.pi)


# --- 3.3 PROPOSED: EKF (Extended Kalman Filter) ---
class ClassicEKF:
    def __init__(self, q_param, r_param):
        # State: [Phase, Freq(rad/s), Amplitude]
        self.x = np.array([0.0, 2 * np.pi * 60.0, 1.0])
        self.P = np.eye(3) * 1.0
        self.Q = np.diag([1e-6, q_param, 1e-4])
        self.R = np.array([[r_param]])
        self.I = np.eye(3)
        self.init = False
        self.name = "EKF"

    def step(self, z):
        if not self.init:
            if abs(z) < 0.99:
                self.x[0] = np.arcsin(z)
                self.init = True
            return 60.0
        # Predict
        self.x[0] += self.x[1] * DT_DSP
        F = np.eye(3)
        F[0, 1] = DT_DSP
        self.P = F @ self.P @ F.T + self.Q
        # Update
        theta, amp = self.x[0], self.x[2]
        y_pred = amp * np.sin(theta)
        inn = z - y_pred
        H = np.array([[amp * np.cos(theta), 0.0, np.sin(theta)]])
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T * (1.0 / (S[0, 0] + 1e-12))
        self.x += (K * inn).flatten()
        self.P = (self.I - K @ H) @ self.P
        # Constraints
        self.x[1] = np.clip(self.x[1], 2 * np.pi * 40.0, 2 * np.pi * 80.0)
        self.x[0] %= 2 * np.pi
        self.x[2] = max(0.1, self.x[2])
        return self.x[1] / (2 * np.pi)


# --- 3.4 SOTA 1: SOGI-FLL (Robust Discretization) ---
class SOGI_FLL:
    def __init__(self, k_gain, gamma):
        self.k = k_gain
        self.gamma = gamma
        self.w = 2 * np.pi * 60.0
        self.v_alpha = 0.0
        self.v_beta = 0.0
        self.name = "SOGI-FLL"

    def step(self, z):
        # Improved Euler (Heun's Method approximation for stability)
        e = z - self.v_alpha
        v_alpha_pred = self.v_alpha + DT_DSP * (
            self.w * e * self.k - self.w * self.v_beta
        )
        v_beta_pred = self.v_beta + DT_DSP * (self.w * self.v_alpha)

        e_pred = z - v_alpha_pred
        dot_alpha_avg = 0.5 * (
            (self.w * e * self.k - self.w * self.v_beta)
            + (self.w * e_pred * self.k - self.w * v_beta_pred)
        )
        dot_beta_avg = 0.5 * ((self.w * self.v_alpha) + (self.w * v_alpha_pred))

        self.v_alpha += DT_DSP * dot_alpha_avg
        self.v_beta += DT_DSP * dot_beta_avg

        # FLL Adaptation
        mag_sq = self.v_alpha**2 + self.v_beta**2
        if mag_sq < 0.01:
            mag_sq = 0.01

        w_dot = -self.gamma * (z - self.v_alpha) * self.v_beta * self.w / mag_sq
        self.w += w_dot * DT_DSP
        self.w = np.clip(self.w, 2 * np.pi * 40.0, 2 * np.pi * 80.0)
        return self.w / (2 * np.pi)


# =============================================================
# Helper: AR(2) -> frecuencia (con dt configurable)
# =============================================================
def _ar2_to_freq(theta0, dt=DT_DSP):
    """
    Convierte el primer coeficiente AR(2) (a1) en frecuencia [Hz]
    usando el mapeo estándar:
        a1 = 2 cos(w * dt)
    donde dt es el paso de muestreo efectivo.

    Incluye clamps suaves para evitar NaN y valores fuera de rango físico.
    """
    # Clamp de estabilidad
    a1 = float(np.clip(theta0, -1.9999, 1.9999))

    # Relación a1 = 2 cos(w dt)
    val = a1 / 2.0
    val = float(np.clip(val, -0.9999, 0.9999))

    try:
        # w [rad/s]
        w = math.acos(val) / dt
    except ValueError:
        # Por seguridad numérica
        return 60.0

    f_inst = w / (2.0 * math.pi)

    if math.isnan(f_inst) or not np.isfinite(f_inst):
        return 60.0

    # rango físico razonable para aplicaciones de red
    return float(np.clip(f_inst, 40.0, 80.0))


# --- 3.5 SOTA 2: RLS (decimado, normalizado, sin VFF) ---
# --- 3.5 SOTA 2: RLS (decimado, normalizado, sin VFF) ---
class RLS_Estimator:
    """
    RLS AR(2) para estimación de frecuencia.

    - Front-end (IIR + AGC) corre a FS_DSP (=10 kHz).
    - Núcleo RLS trabaja sobre una señal decimada:
         FS_eff = FS_DSP / decim
      Esto mejora la condición numérica del mapeo a1 -> f
      y, con la inicialización correcta, arranca ya en ~60 Hz.
    """

    def __init__(self, lam, win_smooth, decim=20):
        """
        lam        : factor de olvido fijo (0.98–0.999 recomendado)
        win_smooth : ventana de suavizado sobre frecuencia (en muestras RLS)
        decim      : factor de decimación interna (entero ≥1).
                      FS_eff = FS_DSP / decim > 2*F_MAX (≈ 160 Hz).
        """
        self.lam = float(lam)

        # Decimación e instante de muestreo efectivo
        self.decim = int(decim)
        if self.decim < 1:
            self.decim = 1
        self.DT_eff = self.decim * DT_DSP

        # Coeficientes AR(2) iniciales consistentes con 60 Hz
        # y(n) = 2 cos(w0 dt) y(n-1) - y(n-2)
        w0 = 2.0 * math.pi * 60.0
        a1_60 = 2.0 * math.cos(w0 * self.DT_eff)
        self.theta = np.array([a1_60, -1.0], dtype=float)

        # Covarianza inicial moderada (no tan agresiva como 100·I)
        self.P = np.eye(2) * 10.0

        self.y_buf = deque([0.0, 0.0], maxlen=2)
        self.name = "RLS"

        self._cnt = 0

        self.smooth_win = int(win_smooth)
        self.f_buf = deque(maxlen=self.smooth_win)

        # Front-end
        self.bp = IIR_Bandpass()
        self.norm = FastRMS_Normalizer()

        self._last_f = 60.0

    def step(self, z_raw):
        # Front-end continuo a FS_DSP
        z_bp = self.bp.step(z_raw)
        z = self.norm.step(z_bp)

        # Decimación interna: solo actualizamos RLS cada 'decim' muestras
        self._cnt += 1
        if self._cnt < self.decim:
            return self._last_f
        self._cnt = 0

        # A partir de aquí estamos en tiempo efectivo FS_eff
        if len(self.y_buf) < 2:
            self.y_buf.append(z)
            self._last_f = 60.0
            return self._last_f

        # Vector de entrada AR(2): [y(n-1), y(n-2)]
        phi_raw = np.array([self.y_buf[1], self.y_buf[0]], dtype=float)
        # Normalización para estabilidad numérica (no cambia la solución)
        norm_phi = np.linalg.norm(phi_raw) + 1e-9
        phi = phi_raw / norm_phi
        d = z / norm_phi

        # Predicción y error
        y_pred = float(self.theta @ phi)
        e = d - y_pred

        # RLS estándar
        Pphi = self.P @ phi
        denom = self.lam + float(phi @ Pphi)
        if denom <= 0.0:
            denom = 1e-9
        K = Pphi / denom
        self.theta = self.theta + K * e
        self.P = (self.P - np.outer(K, Pphi)) / self.lam

        # Simetrizar P para estabilidad numérica
        self.P = 0.5 * (self.P + self.P.T)

        self.y_buf.append(z)

        # AR(2) -> frecuencia con dt efectivo
        f_inst = _ar2_to_freq(self.theta[0], dt=self.DT_eff)

        self.f_buf.append(f_inst)
        self._last_f = f_inst

        if len(self.f_buf) < self.smooth_win:
            return 60.0
        return float(np.mean(self.f_buf))


# --- 3.6 SOTA 3: TEAGER (DESA-2 + Normalization + robust clamp) ---
class Teager_Estimator:
    def __init__(self, smooth_win):
        self.buf = deque(maxlen=5)
        self.win = int(smooth_win)
        self.f_buf = deque(maxlen=self.win)
        self.name = "Teager"
        self.bp = IIR_Bandpass()
        self.norm = FastRMS_Normalizer()

    def step(self, z_raw):
        z_bp = self.bp.step(z_raw)
        z = self.norm.step(z_bp)
        self.buf.append(z)
        if len(self.buf) < 5:
            return 60.0

        # DESA-2 Algorithm
        x_n = self.buf[2]
        x_nm1 = self.buf[1]
        x_np1 = self.buf[3]
        x_nm2 = self.buf[0]
        x_np2 = self.buf[4]

        # Teager Operators
        psi_x = x_n**2 - x_nm1 * x_np1
        y_n = x_np1 - x_nm1
        y_nm1 = x_n - x_nm2
        y_np1 = x_np2 - x_n
        psi_y = y_n**2 - y_nm1 * y_np1

        # Umbral algo más conservador para SNR baja
        if psi_x <= 1e-4:
            f = self.f_buf[-1] if len(self.f_buf) > 0 else 60.0
        else:
            val = 1.0 - psi_y / (2.0 * psi_x)
            if abs(val) > 1.0:
                val = math.copysign(1.0, val)
            w = 0.5 * math.acos(val)  # DESA-2 factor
            f = (w / DT_DSP) / (2 * math.pi)

        if f > 80 or f < 40 or math.isnan(f):
            f = 60.0

        # Clamp ligero para evitar saltos aislados absurdos
        if len(self.f_buf) > 0:
            prev = self.f_buf[-1]
            if abs(f - prev) > 5.0:
                f = prev

        self.f_buf.append(f)
        if len(self.f_buf) < self.win:
            return 60.0
        return float(np.mean(self.f_buf))


# --- 3.7 SOTA 4: TFT (K=2 Quadratic Model) ---
class TFT_Estimator:
    def __init__(self, win_cycles):
        self.N = int((FS_DSP / 60.0) * win_cycles)
        self.buf = deque(maxlen=self.N)
        self.t_vec = np.arange(self.N) * DT_DSP
        self.t_vec = self.t_vec - np.mean(self.t_vec)
        self.name = "TFT"
        w = 2 * np.pi * 60.0

        # Basis Matrix (K=2)
        self.H = np.zeros((self.N, 6))
        self.H[:, 0] = np.cos(w * self.t_vec)
        self.H[:, 1] = np.sin(w * self.t_vec)
        self.H[:, 2] = self.t_vec * np.cos(w * self.t_vec)
        self.H[:, 3] = self.t_vec * np.sin(w * self.t_vec)
        self.H[:, 4] = (self.t_vec**2) * np.cos(w * self.t_vec)
        self.H[:, 5] = (self.t_vec**2) * np.sin(w * self.t_vec)
        self.H_pinv = np.linalg.pinv(self.H)

    def step(self, z):
        self.buf.append(z)
        if len(self.buf) < self.N:
            return 60.0
        y = np.array(self.buf)
        coeffs = self.H_pinv @ y
        a0, b0, a1, b1 = coeffs[0:4]

        num = b0 * a1 - a0 * b1
        den = a0**2 + b0**2
        if den < 1e-6:
            return 60.0
        df = (1.0 / (2 * np.pi)) * (num / den)
        return 60.0 + df


# --- 3.8 SOTA+: RLS con Variable Forgetting Factor (VFF-RLS) ---
class RLS_VFF_Estimator:
    """
    RLS AR(2) con factor de olvido variable tipo VFF.

    Opera sobre una señal:
      - Filtrada (IIR_Bandpass)
      - Normalizada (FastRMS)
      - Decimada por 'decim'
    """

    def __init__(
        self,
        lam_min=0.98,
        lam_max=0.9995,
        Ka=3.0,
        Kb=None,
        win_smooth=40,
        decim=20,
        alpha=None,
    ):
        """
        lam_min   : cota inferior de lambda_k
        lam_max   : cota superior (<= 1.0)
        Ka, Kb    : parámetros de las ventanas exponenciales
                    (si Kb es None, se toma Kb = Ka)
        win_smooth: tamaño de ventana de suavizado sobre frecuencia (en muestras RLS)
        decim     : factor de decimación interna
        alpha     : alias opcional para Ka (compatibilidad hacia atrás)
        """
        # Alias alpha -> Ka (para compatibilidad con código previo)
        if alpha is not None:
            Ka = alpha

        self.lam_min = float(lam_min)
        self.lam_max = float(min(lam_max, 1.0))
        self.Ka = float(Ka)
        self.Kb = float(Ka if Kb is None else Kb)

        # Decimación y dt efectivo
        self.decim = int(decim) if int(decim) > 0 else 1
        self._cnt = 0
        self.DT_eff = self.decim * DT_DSP

        # Coeficientes AR(2) iniciales consistentes con 60 Hz
        w0 = 2.0 * math.pi * 60.0
        a1_60 = 2.0 * math.cos(w0 * self.DT_eff)
        self.theta = np.array([a1_60, -1.0], dtype=float)

        # Matriz de correlación inversa inicial: P(0) = delta^-1 I
        # delta algo más grande para que no sea tan agresivo
        delta = 0.1  # antes 0.01 -> 100·I (muy brusco)
        self.P = np.eye(2) / delta

        self.y_buf = deque([0.0, 0.0], maxlen=2)
        self.name = "RLS-VFF"

        # Suavizado sobre frecuencia estimada
        self.smooth_win = int(win_smooth)
        self.f_buf = deque(maxlen=self.smooth_win)

        # Front-end
        self.bp = IIR_Bandpass()
        self.norm = FastRMS_Normalizer()

        # Variables para VFF
        self.lambda_k = 1.0
        self.sigma_e = 1.0
        self.sigma_q = 1.0
        self.sigma_v = 1.0

        # Coeficientes de ventana exponencial (filter_len = 2 en AR(2))
        filter_len = 2.0
        self.alpha = 1.0 - 1.0 / (self.Ka * filter_len)
        self.beta = 1.0 - 1.0 / (self.Kb * filter_len)

        # Último valor de frecuencia, para rellenar en muestras decimadas
        self._last_f = 60.0

    def _update_lambda(self):
        """
        Actualiza lambda_k según las potencias estimadas.
        """
        se = max(self.sigma_e, 1e-12)
        sv = max(self.sigma_v, 1e-12)

        gamma = math.sqrt(se) / math.sqrt(sv)

        if 1.0 < gamma <= 2.0:
            lam = 1.0
        else:
            sq = max(self.sigma_q, 1e-12)
            num = math.sqrt(sq) * math.sqrt(sv)
            den = abs(math.sqrt(se) - math.sqrt(sv)) + 1e-8
            lam = min(num / den, 1.0)

        lam = float(np.clip(lam, self.lam_min, self.lam_max))
        # Extra seguridad: no bajar de 0.97 para evitar explosiones
        lam = max(lam, 0.97)

        self.lambda_k = lam
        return lam

    def step(self, z_raw):
        # Front-end a FS_DSP
        z_bp = self.bp.step(z_raw)
        z = self.norm.step(z_bp)

        # Decimación: sólo actualizamos RLS una de cada 'decim' muestras
        self._cnt += 1
        if self._cnt < self.decim:
            return self._last_f
        self._cnt = 0

        # Llenado inicial del buffer AR(2)
        if len(self.y_buf) < 2:
            self.y_buf.append(z)
            self._last_f = 60.0
            return self._last_f

        # Vector de entrada u = [y(k-1), y(k-2)] (normalizado)
        u = np.array([self.y_buf[1], self.y_buf[0]], dtype=float)
        norm_u = np.linalg.norm(u) + 1e-9
        u_n = u / norm_u
        d = z / norm_u

        # Kalman gain con lambda_k actual
        lam = self.lambda_k
        Pu = self.P @ u_n
        q = float(u_n @ Pu)  # u^T P u (para sigma_q)

        den = 1.0 + lam**-1 * q
        if den <= 0.0:
            den = 1e-9
        kalman = (lam**-1 * Pu) / den

        # Error a priori
        y_hat = float(self.theta @ u_n)
        e = d - y_hat

        # Actualización de coeficientes AR(2)
        self.theta = self.theta + kalman * e

        # Actualización de P (RLS clásico con lambda)
        self.P = lam**-1 * self.P - lam**-1 * (np.outer(kalman, u_n) @ self.P)

        # Mantenimiento numérico: simetrizar P suavemente
        self.P = 0.5 * (self.P + self.P.T)

        # Actualización de potencias
        self.sigma_e = self.alpha * self.sigma_e + (1.0 - self.alpha) * (e * e)
        self.sigma_q = self.alpha * self.sigma_q + (1.0 - self.alpha) * (q * q)
        self.sigma_v = self.beta * self.sigma_v + (1.0 - self.beta) * (e * e)

        # Nuevo lambda_k para la próxima iteración
        self._update_lambda()

        # Actualizar buffer de salida (para el siguiente paso)
        self.y_buf.append(z)

        # AR(2) -> frecuencia con dt efectivo (por decimación)
        f_inst = _ar2_to_freq(self.theta[0], dt=self.DT_eff)

        # Guardar para muestras decimadas que no actualizan RLS
        self.f_buf.append(f_inst)
        self._last_f = f_inst

        if len(self.f_buf) < self.smooth_win:
            return 60.0

        return float(np.mean(self.f_buf))


# --- 3.9 SOTA+: Unscented Kalman Filter (UKF) ---
class UKF_Estimator:
    """
    UKF para el mismo modelo de estado que el EKF:
        x = [theta, omega, A]
        z = A * sin(theta)
    Evita Jacobianos explícitos y es robusto a no linealidades fuertes.
    """

    def __init__(self, q_param, r_param, smooth_win=10, alpha=0.3, beta=2.0, kappa=0.0):
        self.name = "UKF"
        self.n = 3
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lmbda = self.alpha**2 * (self.n + self.kappa) - self.n

        self.Wm = np.zeros(2 * self.n + 1)
        self.Wc = np.zeros(2 * self.n + 1)
        self.Wm[0] = self.lmbda / (self.n + self.lmbda)
        self.Wc[0] = self.Wm[0] + (1 - self.alpha**2 + self.beta)
        self.Wm[1:] = 1.0 / (2 * (self.n + self.lmbda))
        self.Wc[1:] = self.Wm[1:]

        self.x = np.array([0.0, 2 * np.pi * 60.0, 1.0])
        self.P = np.diag([1e-2, (2 * np.pi) ** 2, 1e-2])

        self.Q = np.diag([1e-6, q_param, 1e-4])
        self.R = np.array([[r_param]])

        self.init = False
        self.smooth_win = int(smooth_win)
        self.f_buf = deque(maxlen=self.smooth_win)

    def _sigma_points(self, x, P):
        n = self.n
        sigma = np.zeros((2 * n + 1, n))
        sigma[0] = x
        c = n + self.lmbda
        try:
            A = np.linalg.cholesky(c * P)
        except np.linalg.LinAlgError:
            # Regulariza si P no es SPD
            A = np.linalg.cholesky(c * (P + 1e-9 * np.eye(n)))

        for i in range(n):
            sigma[i + 1] = x + A[:, i]
            sigma[i + 1 + n] = x - A[:, i]
        return sigma

    def _f(self, x):
        # Dinámica del estado
        theta, omega, A = x
        theta_new = theta + omega * DT_DSP
        return np.array([theta_new, omega, A])

    def _h(self, x):
        theta, omega, A = x
        return np.array([A * math.sin(theta)])

    def step(self, z):
        if not self.init:
            if abs(z) < 0.99:
                self.x[0] = math.asin(z)
                self.init = True
            return 60.0

        # 1) Sigma points
        sigma = self._sigma_points(self.x, self.P)

        # 2) Predicción
        sigma_f = np.array([self._f(s) for s in sigma])
        x_pred = np.sum(self.Wm[:, None] * sigma_f, axis=0)

        P_pred = self.Q.copy()
        for i in range(2 * self.n + 1):
            dx = (sigma_f[i] - x_pred).reshape(-1, 1)
            P_pred += self.Wc[i] * (dx @ dx.T)

        # 3) Predicción de medición
        sigma_h = np.array([self._h(s) for s in sigma_f])
        z_pred = np.sum(self.Wm[:, None] * sigma_h, axis=0)

        S = self.R.copy()
        Pxz = np.zeros((self.n, 1))
        for i in range(2 * self.n + 1):
            dz = (sigma_h[i] - z_pred).reshape(-1, 1)
            dx = (sigma_f[i] - x_pred).reshape(-1, 1)
            S += self.Wc[i] * (dz @ dz.T)
            Pxz += self.Wc[i] * (dx @ dz.T)

        # 4) Actualización
        K = Pxz @ np.linalg.inv(S + 1e-12 * np.eye(1))
        inn = np.array([[z]]) - z_pred.reshape(-1, 1)
        self.x = x_pred + (K @ inn).flatten()
        self.P = P_pred - K @ S @ K.T

        # Constraints
        self.x[1] = float(np.clip(self.x[1], 2 * np.pi * 40.0, 2 * np.pi * 80.0))
        self.x[0] = float(self.x[0] % (2 * np.pi))
        self.x[2] = max(0.1, float(self.x[2]))

        f_inst = self.x[1] / (2 * np.pi)
        if f_inst > 80 or f_inst < 40 or np.isnan(f_inst):
            f_inst = 60.0

        self.f_buf.append(f_inst)
        if len(self.f_buf) < self.smooth_win:
            return 60.0
        return float(np.mean(self.f_buf))


# --- 3.10 SOTA+: Koopman / RK-DPMU simplificado ---
class Koopman_RKDPmu:
    """
    Estimador tipo Koopman / RK-DPMU:
    - Ventana deslizante de estados embebidos en 2D
      s_k = [x_k, x_{k-1}]^T
    - Ajuste de operador lineal K mediante mínimos cuadrados:
      S1 ≈ K S0
    - Frecuencia = arg(eig_max(K)) / (2π * DT_DSP)
    """

    def __init__(self, window_samples=200, smooth_win=20):
        self.N = int(window_samples)
        self.buf = deque(maxlen=self.N + 2)  # necesitamos al menos N+2 puntos
        self.name = "Koopman-RKDPmu"
        self.smooth_win = int(smooth_win)
        self.f_buf = deque(maxlen=self.smooth_win)
        self.bp = IIR_Bandpass()
        self.norm = FastRMS_Normalizer()

    def step(self, z_raw):
        z_bp = self.bp.step(z_raw)
        z = self.norm.step(z_bp)

        self.buf.append(z)
        if len(self.buf) < (self.N + 2):
            return 60.0

        x = np.array(self.buf)
        # Construimos estados embebidos 2D
        x0 = x[:-2]
        x1 = x[1:-1]
        x2 = x[2:]

        S0 = np.vstack((x1, x0))  # shape (2, L-2)
        S1 = np.vstack((x2, x1))  # shape (2, L-2)

        # Koopman K ~ S1 S0^+
        S0_pinv = np.linalg.pinv(S0 + 1e-9)
        K = S1 @ S0_pinv  # 2x2

        eigvals, _ = np.linalg.eig(K)
        # Tomamos el modo con mayor módulo (dominante)
        idx = np.argmax(np.abs(eigvals))
        lam = eigvals[idx]
        angle = np.angle(lam)

        # Frecuencia instantánea (rad/s -> Hz)
        w = angle / DT_DSP
        f_inst = w / (2 * np.pi)

        if f_inst > 80 or f_inst < 40 or np.isnan(f_inst):
            f_inst = 60.0

        # Suavizado
        self.f_buf.append(f_inst)
        if len(self.f_buf) < self.smooth_win:
            return 60.0
        return float(np.mean(self.f_buf))


# --- 3.11 ML: Physics-Informed GRU Frequency Estimator (PI-GRU) ---
class PIGRU_FreqEstimator:
    """
    Physics-Informed GRU Frequency Estimator (PI-GRU).

    - Entrada: ventana deslizante de muestras de tensión ya pre-filtradas
      (bandpass 60 Hz) y normalizadas (AGC RMS).
    - Modelo: red GRU ligera entrenada offline (PyTorch).
      Debe recibir un tensor de forma (batch=1, T, 1) y devolver
      un escalar o un vector 1D con frecuencia instantánea en Hz.
    - Latencia estructural ≈ window_len / FS_DSP.
    """

    def __init__(
        self,
        model=None,
        window_len_samples=40,
        smooth_win=10,
        device="cpu",
        name="PI-GRU",
    ):
        """
        model: instancia de torch.nn.Module ya entrenada (o ScriptModule),
               con forward(x: [1,T,1]) -> [1] o [1,T_out].
        window_len_samples: tamaño de ventana en muestras a 10 kHz.
        smooth_win: tamaño de ventana de suavizado sobre la salida.
        device: 'cpu' o 'cuda' según dónde esté el modelo.
        """
        self.name = name
        self.window_len = int(window_len_samples)
        self.buf = deque(maxlen=self.window_len)
        self.smooth_win = int(smooth_win)
        self.f_buf = deque(maxlen=self.smooth_win)
        self.bp = IIR_Bandpass()
        self.norm = FastRMS_Normalizer()

        self._torch_available = torch is not None
        self._warned = False

        if self._torch_available and (model is not None):
            self.model = model.to(device)
            self.model.eval()
            self.device = device
            for p in self.model.parameters():
                p.requires_grad_(False)
        else:
            self.model = None
            self.device = "cpu"

    def _fallback(self):
        """Salida de emergencia cuando no hay modelo/torch."""
        if not self._warned:
            print(
                "[PI-GRU WARNING] PyTorch o el modelo no están disponibles. "
                "El estimador devuelve 60 Hz como dummy baseline."
            )
            self._warned = True
        return 60.0

    def step(self, z_raw):
        # Si no hay soporte ML, devolvemos baseline seguro
        if (not self._torch_available) or (self.model is None):
            return self._fallback()

        # Front-end físico: bandpass + AGC
        z_bp = self.bp.step(z_raw)
        z = self.norm.step(z_bp)

        self.buf.append(z)
        if len(self.buf) < self.window_len:
            return 60.0

        # Preparar tensor para el modelo: shape (1, T, 1)
        x_np = np.array(self.buf, dtype=np.float32).reshape(1, -1, 1)
        x_t = torch.from_numpy(x_np).to(self.device)

        with torch.no_grad():
            y = self.model(x_t)

        if y.ndim == 0:
            f_inst = float(y.cpu().item())
        else:
            y_np = y.detach().cpu().numpy().flatten()
            f_inst = float(y_np[-1])

        if np.isnan(f_inst) or f_inst < 40.0 or f_inst > 80.0:
            f_inst = 60.0

        if len(self.f_buf) > 0:
            prev = self.f_buf[-1]
            if abs(f_inst - prev) > 5.0:
                f_inst = prev

        self.f_buf.append(f_inst)
        if len(self.f_buf) < self.smooth_win:
            return 60.0
        return float(np.mean(self.f_buf))


# =============================================================
# 4. METRICS (SIN PLOTTING)
# =============================================================
def _max_contiguous_time(mask_bool, dt):
    """Longest continuous time where mask_bool is True."""
    max_len = 0
    current = 0
    for val in mask_bool:
        if val:
            current += 1
            if current > max_len:
                max_len = current
        else:
            current = 0
    return max_len * dt


def calculate_metrics(est_trace, true_trace, exec_time, structural_samples=None):
    """
    Compute accuracy + risk + complexity metrics.
    exec_time: total CPU time in seconds for the full trace.
    structural_samples: effective window / memory of the algorithm (for latency).
    """
    start_idx = int(0.15 * FS_DSP)  # discard initial transient of estimators
    e = est_trace[start_idx:] - true_trace[start_idx:]
    abs_e = np.abs(e)
    n_eff = len(abs_e) if len(abs_e) > 0 else len(est_trace)

    rmse = float(np.sqrt(np.mean(abs_e**2))) if n_eff > 0 else 0.0
    max_peak = float(np.max(abs_e)) if n_eff > 0 else 0.0
    settling_time = float(np.sum(abs_e > SETTLING_THRESHOLD) * (1.0 / FS_DSP))
    energy = float(np.sum(abs_e**2) * (1.0 / FS_DSP))

    trip_mask = abs_e > TRIP_THRESHOLD
    total_trip_time = float(np.sum(trip_mask) * (1.0 / FS_DSP))
    max_cont_trip_time = float(_max_contiguous_time(trip_mask, 1.0 / FS_DSP))

    total_time = float(exec_time)
    n_total = len(est_trace) if len(est_trace) > 0 else 1
    t_per_sample = total_time / n_total
    t_per_sample_us = float(t_per_sample * 1e6)

    if structural_samples is not None and structural_samples > 0:
        latency_s = structural_samples / FS_DSP
    else:
        latency_s = 1.0 / FS_DSP
    latency_ms = float(latency_s * 1000.0)

    return {
        "RMSE": rmse,
        "MAX_PEAK": max_peak,
        "SETTLING": settling_time,
        "ENERGY": energy,
        "TRIP_TIME_0p5": total_trip_time,
        "MAX_CONTIGUOUS_0p5": max_cont_trip_time,
        "COMPLEXITY": total_time,
        "TIME_PER_SAMPLE_US": t_per_sample_us,
        "STRUCTURAL_LATENCY_MS": latency_ms,
    }


# =============================================================
# 5. MASSIVE HYPERPARAMETER TUNING HELPERS
# =============================================================
def tune_ipdft(v, f, c_vals):
    best = {"RMSE": 1e9}
    for c in c_vals:
        algo = TunableIpDFT(c)
        tr = np.array([algo.step(x) for x in v])
        m = calculate_metrics(tr, f, 0.0, structural_samples=algo.sz)
        if m["RMSE"] < best["RMSE"]:
            best = {"RMSE": m["RMSE"], "p": f"{c} cycles"}
    return best["p"]


def tune_pll(v, f, kp_vals, ki_vals, sc_name=None):
    best = {"RMSE": 1e9}
    for kp in kp_vals:
        for ki in ki_vals:
            algo = StandardPLL(kp, ki)
            tr = np.array([algo.step(x) for x in v])
            m = calculate_metrics(tr, f, 0.0, structural_samples=algo.maf_win)
            if m["RMSE"] < best["RMSE"]:
                best = {"RMSE": m["RMSE"], "p": f"Kp{kp},Ki{ki}", "v": (kp, ki)}
    return best["p"], best.get("v", (10, 50))


def tune_ekf(v, f, q_vals, r_vals, sc_name=None):
    best = {"RMSE": 1e9}
    for q in q_vals:
        for r in r_vals:
            algo = ClassicEKF(q, r)
            tr = np.array([algo.step(x) for x in v])
            m = calculate_metrics(tr, f, 0.0, structural_samples=1)
            if m["RMSE"] < best["RMSE"]:
                best = {"RMSE": m["RMSE"], "p": f"Q{q},R{r}", "v": (q, r)}
    return best["p"], best.get("v", (0.1, 1.0))


def tune_ekf2(v, f, sc_name=None):
    best = {"RMSE": 1e9, "params": None}

    for params in EKF2.tuning_grid():
        algo = EKF2(**params)
        tr = np.array([algo.step(x) for x in v])
        m = calculate_metrics(tr, f, 0.0, structural_samples=1)

        if m["RMSE"] < best["RMSE"]:
            best["RMSE"] = m["RMSE"]
            best["params"] = params

    if best["params"] is None:
        default_params = {"q_param": 0.1, "r_param": 1.0, "inn_ref": 0.5}
        return EKF2.describe_params(default_params), default_params

    p_str = EKF2.describe_params(best["params"])
    return p_str, best["params"]


def tune_sogi(v, f, k_vals, g_vals):
    best = {"RMSE": 1e9}
    for k in k_vals:
        for g in g_vals:
            algo = SOGI_FLL(k, g)
            tr = np.array([algo.step(x) for x in v])
            m = calculate_metrics(tr, f, 0.0, structural_samples=int(FS_DSP / 60.0))
            if m["RMSE"] < best["RMSE"]:
                best = {"RMSE": m["RMSE"], "p": f"k{k},g{g}", "v": (k, g)}
    return best["p"], best.get("v", (1.0, 50))


def tune_rls(v, f, lam_vals, win_vals, sc_name=None):
    best = {"RMSE": 1e9}
    for l in lam_vals:
        for w in win_vals:
            algo = RLS_Estimator(lam=l, win_smooth=w, decim=20)
            tr = np.array([algo.step(x) for x in v])
            m = calculate_metrics(
                tr, f, 0.0, structural_samples=algo.smooth_win * algo.decim
            )
            if m["RMSE"] < best["RMSE"]:
                best = {"RMSE": m["RMSE"], "p": f"Lam{l},Win{w}", "v": (l, w)}
    return best["p"], best.get("v", (0.995, 80))


def tune_teager(v, f, win_vals):
    best = {"RMSE": 1e9}
    for w in win_vals:
        algo = Teager_Estimator(w)
        tr = np.array([algo.step(x) for x in v])
        m = calculate_metrics(tr, f, 0.0, structural_samples=max(5, algo.win))
        if m["RMSE"] < best["RMSE"]:
            best = {"RMSE": m["RMSE"], "p": f"Win{w}", "v": w}
    return best["p"], best.get("v", 10)


def tune_tft(v, f, win_vals):
    best = {"RMSE": 1e9}
    for w in win_vals:
        algo = TFT_Estimator(w)
        tr = np.array([algo.step(x) for x in v])
        m = calculate_metrics(tr, f, 0.0, structural_samples=algo.N)
        if m["RMSE"] < best["RMSE"]:
            best = {"RMSE": m["RMSE"], "p": f"Cycles{w}", "v": w}
    return best["p"], best.get("v", 2)


def tune_vff_rls(v, f, lam_min_vals, alpha_vals, sc_name=None):
    """
    Tuning para RLS_VFF_Estimator:
      - lam_min: mínimo factor de olvido
      - alpha : aquí se usa como Ka (y Kb=Ka) de las ventanas exponenciales
    lam_max se fija alto (0.9995) para buen tracking en steady state.
    """
    best = {"RMSE": 1e9}
    for lam_min in lam_min_vals:
        for Ka in alpha_vals:
            algo = RLS_VFF_Estimator(
                lam_min=lam_min, lam_max=0.9995, Ka=Ka, Kb=None, win_smooth=40, decim=20
            )
            tr = np.array([algo.step(x) for x in v])
            m = calculate_metrics(
                tr, f, 0.0, structural_samples=algo.smooth_win * algo.decim
            )
            if m["RMSE"] < best["RMSE"]:
                best = {
                    "RMSE": m["RMSE"],
                    "p": f"lamMin{lam_min},Ka{Ka}",
                    "v": (lam_min, Ka),
                }
    return best["p"], best.get("v", (0.985, 3.0))


def tune_ukf(v, f, q_vals, r_vals, sc_name=None):
    best = {"RMSE": 1e9}
    for q in q_vals:
        for r in r_vals:
            algo = UKF_Estimator(q_param=q, r_param=r, smooth_win=10)
            tr = np.array([algo.step(x) for x in v])
            m = calculate_metrics(tr, f, 0.0, structural_samples=1)
            if m["RMSE"] < best["RMSE"]:
                best = {"RMSE": m["RMSE"], "p": f"Q{q},R{r}", "v": (q, r)}
    return best["p"], best.get("v", (1.0, 0.01))


def tune_koopman(v, f, win_vals, sc_name=None):
    best = {"RMSE": 1e9}
    for w in win_vals:
        algo = Koopman_RKDPmu(window_samples=w, smooth_win=w)
        tr = np.array([algo.step(x) for x in v])
        m = calculate_metrics(tr, f, 0.0, structural_samples=w)
        if m["RMSE"] < best["RMSE"]:
            best = {"RMSE": m["RMSE"], "p": f"Win{w}", "v": w}
    return best["p"], best.get("v", 200)


# --- 3.X: Linear Kalman Filter (LKF-AR2 Hybrid) ---
class LKF_Estimator:
    """
    Linear Kalman Filter + AR(2) hybrid frequency estimator.

    - LKF filtra la señal (estado x = [y(k), y(k-1)]).
    - Cada cierto número de muestras (decim) hace un pequeño ajuste AR(2)
      para obtener a1 y a2, y luego usa _ar2_to_freq(a1, dt) para f_inst.
    """

    def __init__(self, q_val=1e-4, r_val=1e-2, smooth_win=20, decim=10):
        """
        q_val      : varianza de ruido de proceso (Q)
        r_val      : varianza de ruido de medición (R)
        smooth_win : ventana de suavizado sobre frecuencia (en muestras LKF)
        decim      : factor de decimación para el ajuste AR(2)
        """
        self.name = "LKF"

        # Estado LKF: x = [y(k), y(k-1)]
        self.x = np.zeros(2)

        # Matrices LKF (modelo simple de retardo)
        # x(k|k-1) = [y(k-1); y(k-2)]
        self.A = np.array(
            [[1.0, 0.0], [1.0, 0.0]]
        )  # se usa sólo para la forma del filtro
        self.C = np.array([[1.0, 0.0]])

        self.P = np.eye(2) * 1.0
        self.Q = np.eye(2) * float(q_val)
        self.R = np.array([[float(r_val)]])

        # Para AR(2)
        self.decim = int(decim) if decim >= 1 else 1
        self._cnt = 0
        self.buf = deque(maxlen=3)  # y[n], y[n-1], y[n-2]

        # Inicialización AR(2) consistente con 60 Hz
        w0 = 2.0 * math.pi * 60.0
        a1_60 = 2.0 * math.cos(w0 * DT_DSP * self.decim)
        self.theta = np.array([a1_60, -1.0], dtype=float)

        # Suavizado de frecuencia
        self.smooth_win = int(smooth_win)
        self.f_buf = deque(maxlen=self.smooth_win)

        # Front-end
        self.bp = IIR_Bandpass()
        self.norm = FastRMS_Normalizer()

        self._last_f = 60.0

    def step(self, z_raw):
        # --- Pre-filtrado 60Hz y AGC ---
        z_bp = self.bp.step(z_raw)
        z = self.norm.step(z_bp)

        # --- PREDICCIÓN LKF ---
        # Modelo: x_pred = [y_prev; y_prev_prev]
        x_pred = np.array([self.x[0], self.x[1]])
        P_pred = self.P + self.Q

        # --- UPDATE LKF ---
        y_pred = x_pred[0]
        inn = z - y_pred
        S = P_pred[0, 0] + self.R[0, 0]
        if S <= 0.0:
            S = 1e-9
        K = P_pred[:, 0] / S
        self.x = x_pred + K * inn
        self.P = (np.eye(2) - np.outer(K, self.C)) @ P_pred

        # Señal filtrada
        y_filt = float(self.x[0])

        # --- AR(2) cada 'decim' muestras ---
        self.buf.append(y_filt)
        self._cnt += 1
        if len(self.buf) < 3:
            return 60.0
        if self._cnt < self.decim:
            return self._last_f
        self._cnt = 0

        # Datos AR(2): y[n] = a1 y[n-1] + a2 y[n-2]
        y0, y1, y2 = self.buf[-1], self.buf[-2], self.buf[-3]
        phi = np.array([y1, y2])
        denom = (phi @ phi) + 1e-6
        a1_est = (y0 * y1) / denom
        # Modelo linealizado: a2 ~ -1 para oscilador subamortiguado
        a2_est = -1.0

        self.theta = np.array([a1_est, a2_est], dtype=float)

        # AR(2) -> frecuencia usando el dt efectivo (decimado)
        f_inst = _ar2_to_freq(self.theta[0], dt=DT_DSP * self.decim)

        if math.isnan(f_inst) or f_inst < 40.0 or f_inst > 80.0:
            f_inst = self._last_f

        self.f_buf.append(f_inst)
        self._last_f = f_inst

        if len(self.f_buf) < self.smooth_win:
            return 60.0

        return float(np.mean(self.f_buf))


def tune_lkf(v, f, q_vals, r_vals, sc_name=None):
    """
    Grid search para el LKF_Estimator, análogo a tune_ukf():

      - q_vals: lista de posibles Q (ruido de proceso)
      - r_vals: lista de posibles R (ruido de medición)

    Devuelve:
      p_str: string con la combinación ganadora
      (q_opt, r_opt): tupla de valores óptimos
    """
    best = {"RMSE": 1e9}

    for q in q_vals:
        for r in r_vals:
            algo = LKF_Estimator(q_val=q, r_val=r, smooth_win=20, decim=10)
            tr = np.array([algo.step(x) for x in v])

            # Latencia estructural ≈ smooth_win * decim
            structural_samples = algo.smooth_win * algo.decim

            m = calculate_metrics(tr, f, 0.0, structural_samples=structural_samples)

            if m["RMSE"] < best["RMSE"]:
                best = {"RMSE": m["RMSE"], "p": f"Q{q},R{r}", "v": (q, r)}

    # Fallback razonable por si algo raro pasa
    return best["p"], best.get("v", (1e-4, 1e-2))
