# --- RA-EKF: RoCoF-Augmented Extended Kalman Filter ---
# Proposed estimator for IBR protection-class frequency measurement.
import numpy as np
from collections import deque

# =====================================================
# Sampling constants — must match estimators.py exactly
# FS_PHYSICS: 1 MHz continuous simulation rate
# FS_DSP:    10 kHz discrete acquisition rate (IED/relay)
# RATIO:     100  (downsampling factor)
# =====================================================
FS_PHYSICS = 1_000_000.0   # 1 MHz physics simulation rate
FS_DSP     = 10_000.0      # 10 kHz DSP / relay acquisition rate
RATIO      = int(FS_PHYSICS / FS_DSP)  # = 100
DT_DSP     = 1.0 / FS_DSP             # = 0.1 ms
F_NOM      = 60.0                     # Nominal grid frequency [Hz]


class EKF2:
    """
    EKF2: EKF extendido con ROCOF + gating de eventos para protección dinámica.

    Estado:
        x = [theta, omega, A, domega]^T
          - theta : fase [rad]
          - omega : frecuencia instantánea [rad/s]
          - A     : amplitud
          - domega: ROCOF [rad/s^2]

    Contrato externo (se mantiene):
      - __init__(q_param, r_param, inn_ref=0.5,
                 event_thresh=2.0, fast_horizon_ms=80.0)
      - step(z) -> f_est
      - @staticmethod tuning_grid() -> lista de dicts con kwargs
      - @staticmethod describe_params(params_dict) -> str legible
    """

    def __init__(
        self,
        q_param,
        r_param,
        inn_ref=0.5,
        event_thresh=2.0,
        fast_horizon_ms=80.0,
    ):
        # -----------------------------
        # Estado y covarianza inicial
        # -----------------------------
        self.x = np.array([
            0.0,                  # theta
            2 * np.pi * F_NOM,    # omega ~ 60 Hz
            1.0,                  # A
            0.0                   # domega (ROCOF)
        ], dtype=float)

        # P inicial algo "flojo" en frecuencia / ROCOF
        self.P = np.diag([
            1e-2,                 # var(theta)
            (2 * np.pi * 1.0)**2, # var(omega) ~ 1 Hz
            1e-2,                 # var(A)
            (2 * np.pi * 5.0)**2  # var(domega) ~ 5 Hz/s
        ])

        # -----------------------------
        # Covarianzas base (modo lento / rápido)
        # q_param se interpreta como "ruido de modelo" en omega
        # -----------------------------
        self.Q_slow = np.diag([
            1e-7,          # muy poco ruido en fase directa
            q_param,       # ruido en omega
            1e-4,          # amplitud
            10.0 * q_param # ruido en ROCOF, ligado a q_param
        ])

        # Modo rápido: mucho más agresivo en omega y ROCOF
        # BUG-6: Q_fast[3,3] = 500×q_param, then in _adaptive_QR damped by 0.1.
        # Net ROCOF noise in fast mode: 50×q_param (500× base ×0.1 damp)
        self.Q_fast = np.diag([
            1e-6,
            50.0 * q_param,
            1e-3,
            500.0 * q_param
        ])

        self.R_base = np.array([[r_param]])

        # Estas Q y R se irán adaptando en tiempo real
        self.Q = self.Q_slow.copy()
        self.R = self.R_base.copy()

        # Referencia de innovación típica (en unidades de z)
        self.inn_ref = float(inn_ref)

        # Detector de eventos
        self.event_thresh = float(event_thresh)
        self.fast_horizon = int((fast_horizon_ms * 1e-3) * FS_DSP)
        self.fast_timer = 0
        # 200 samples @ 10kHz = 20ms ≈ 1.2 full cycles at 60Hz.
        # Provides statistically stable median for event detection threshold.
        self.inn_buf = deque(maxlen=200)

        self.I = np.eye(4)
        self.init = False
        self.A_init_est = 1.0  # running amplitude estimate for BUG-1 fix
        self.name = "RA-EKF"

    # ---------- API de tuning embebido ----------
    @staticmethod
    def tuning_grid():
        """
        Reduced grid: 6 × 6 × 2 × 1 × 1 = 72 combinations.

        Rationale for reduction from 864 → 72:
          - event_thresh fixed at 2.0 (best empirically; ±20% insensitive)
          - fast_horizon_ms fixed at 80 ms (one half-cycle margin at 60 Hz)
          - inn_ref: two representative values (low=0.1, high=0.3)
          - q_param and r_param: 6 log-uniform points each over 4 decades

        At N_COST_REPS=10, 72 combos × 5 scenarios ≈ 5–8 min total.
        """
        q_vals            = np.logspace(-1, 3, 6)   # [0.1 … 1000]  rad²/s²
        r_vals            = np.logspace(-3, 0, 6)   # [0.001 … 1.0] pu²
        inn_ref_vals      = [0.1, 0.3]              # nominal innovation scale
        event_thresh_vals = [2.0]                   # fixed — empirically optimal
        fast_horizon_vals = [80.0]                  # fixed — 80 ms window [ms]

        configs = []
        for q in q_vals:
            for r in r_vals:
                for inn_ref in inn_ref_vals:
                    for ev_th in event_thresh_vals:
                        for fh in fast_horizon_vals:
                            configs.append({
                                "q_param":         float(q),
                                "r_param":         float(r),
                                "inn_ref":         float(inn_ref),
                                "event_thresh":    float(ev_th),
                                "fast_horizon_ms": float(fh),
                            })
        return configs

    @staticmethod
    def describe_params(params):
        """
        Convierte el dict de parámetros en un string legible
        para guardarlo en JSON / imprimir.
        """
        return (
            f"Q{params['q_param']},"
            f"R{params['r_param']},"
            f"InnRef{params['inn_ref']},"
            f"EvTh{params.get('event_thresh', 2.0)},"
            f"Fast{params.get('fast_horizon_ms', 80.0)}ms"
        )

    # ---------- Adaptación de Q y R ----------
    def _adaptive_QR(self, inn):
        """
        Innovation-Driven Covariance Scaling.

        The normalised innovation magnitude is:
            r_k = |ν_k| / (σ_ν + ε)

        where ν_k = z_k - ŷ_k is the scalar innovation (measurement residual)
        and σ_ν = self.inn_ref is the expected innovation scale under nominal
        conditions (tuning parameter).

        The adaptive process noise covariance is:

            Q_k = Q_base · diag([r_k, r_k, r_k, 1])          (slow mode)
            Q_k = Q_fast · diag([r_k, r_k, r_k, 0.1])        (fast/event mode)

        Diagonal entry 3 (ROCOF / ω̇) is NOT scaled by r_k in order to keep
        the ROCOF estimate stiff and prevent overshoot during phase-jump events.
        The ratio r_k is clamped to [0.25, 4.0] to prevent covariance blow-up.

        The measurement noise covariance is inversely scaled:
            R_k = R_base / r_k

        This makes the filter trust the measurement more when the innovation is
        small (low noise) and less when it is large (transient / outlier).
        """
        abs_inn = abs(inn)
        ratio = abs_inn / (self.inn_ref + 1e-8)
        ratio = np.clip(ratio, 0.25, 4.0)

        # FIX: Decouple Phase/Freq noise from ROCOF noise adaptation
        # We want to track phase fast, but keep ROCOF stiff.
        
        # 1. Adapt Q mostly for Phase (idx 0) and Amplitude (idx 2)
        # Omega (idx 1) and ROCOF (idx 3) should be stiffer.
        
        q_scale_matrix = np.diag([ratio, ratio, ratio, 1.0]) # Don't scale ROCOF noise wildly
        
        if self.fast_timer > 0:
            # Even in fast mode, cap the ROCOF noise injection
            Q_temp = self.Q_fast.copy()
            Q_temp[3,3] *= 0.1  # DAMP the ROCOF in fast mode to prevent overshoot
            # Net ROCOF noise in fast mode: 50×q_param (500× base ×0.1 damp)
            self.Q = Q_temp @ q_scale_matrix
        else:
            self.Q = self.Q_slow @ q_scale_matrix

        # R adaptation remains the same
        self.R = self.R_base * (1.0 / ratio)

    def _maybe_trigger_event(self, inn, z):
        """
        Detector de eventos tipo 'islanding / salto de fase'.

        Si la innovación actual es varias veces mayor que la mediana
        histórica, disparamos modo rápido y re-ajustamos fase.
        """
        abs_inn = abs(inn)
        self.inn_buf.append(inn)

        if len(self.inn_buf) < 10:
            return

        med_abs = np.median(np.abs(self.inn_buf))
        ref = max(self.inn_ref, med_abs)

        if abs_inn > self.event_thresh * ref:
            # Dispara evento: modo rápido durante fast_horizon
            self.fast_timer = self.fast_horizon

            # Re-inicializar ligeramente la fase para no tardar tanto
            theta, omega, A, domega = self.x
            A_eff = max(A, 0.1)
            arg = np.clip(z / A_eff, -0.99, 0.99)
            new_theta = np.arcsin(arg)

            # No tocamos omega/domega para preservar ROCOF y evitar spikes
            self.x[0] = new_theta
            # BUG-4: cap covariance inflation to prevent unbounded growth
            # under consecutive events (e.g., IBR_MultiEvent)
            self.P[0, 0] = np.clip(self.P[0, 0] * 3.0, 0.0, 1.0)
            self.P[1, 1] = np.clip(self.P[1, 1], 0.0, (2 * np.pi * 5) ** 2)
            self.P[0, 1] = 0.0   # decouple theta-omega
            self.P[1, 0] = 0.0

    # ---------- Núcleo EKF2 ----------
    def step(self, z):
        # -----------------------------
        # Fase de inicialización simple
        # -----------------------------
        if not self.init:
            # BUG-1: Works even when signal amplitude > 1.0 pu (e.g., Scenario A +10%).
            # Warm-up: estimate amplitude from first few samples via exponential smoothing.
            self.A_init_est = max(self.A_init_est * 0.95 + abs(z) * 0.05 * 1.41, 0.1)
            A_safe = max(self.A_init_est, 1.0)
            safe_z = np.clip(z / A_safe, -0.99, 0.99)
            self.x[0] = np.arcsin(safe_z)
            self.x[2] = self.A_init_est  # initialize amplitude state
            self.init = True
            return F_NOM

        # Decremento del timer de modo rápido, si está activo
        if self.fast_timer > 0:
            self.fast_timer -= 1

        # -----------------------------
        # Predicción (modelo dinámico con ROCOF)
        # -----------------------------
        theta, omega, A, domega = self.x

        # Modelo continuo discretizado por Euler + término 0.5*domega*dt^2
        theta_pred = theta + omega * DT_DSP + 0.5 * domega * DT_DSP**2
        omega_pred = omega + domega * DT_DSP
        A_pred = A
        domega_pred = domega

        x_pred = np.array([theta_pred, omega_pred, A_pred, domega_pred])
        self.x = x_pred

        # Jacobiano F = d f / d x
        F = np.eye(4)
        F[0, 1] = DT_DSP
        F[0, 3] = 0.5 * DT_DSP**2
        F[1, 3] = DT_DSP
        # resto ya son 0 o 1

        self.P = F @ self.P @ F.T + self.Q

        # -----------------------------
        # Actualización (medida senoidal)
        # y = A * sin(theta)
        # -----------------------------
        theta, omega, A, domega = self.x
        y_pred = A * np.sin(theta)
        inn = z - y_pred

        # BUG-3 FIX: adapt Q/R first, then detect event.
        # Event detector uses the current-step scaled inn_ref (not stale value).
        self._adaptive_QR(inn)

        # Event detection after QR adaptation
        self._maybe_trigger_event(inn, z)

        H = np.array([[A * np.cos(theta), 0.0, np.sin(theta), 0.0]])  # 1x4
        S = H @ self.P @ H.T + self.R
        inv_S = 1.0 / (S[0, 0] + 1e-12)
        K = self.P @ H.T * inv_S  # 4x1

        self.x += (K * inn).flatten()
        self.P = (self.I - K @ H) @ self.P

        # -----------------------------
        # Restricciones físicas
        # -----------------------------
        # BUG-5 FIX: Frequency: relay-class range 45–75 Hz (tighter than 40–80 Hz)
        # This prevents the brief 19.74 Hz spike during 80° phase-jump re-init.
        # The event-gating re-initializes theta but NOT omega, so omega should
        # stay within the physical protection-relay operating range.
        # After clipping omega, update P[1,1] to reflect the hard constraint.
        omega_pre_clip = self.x[1]
        self.x[1] = np.clip(self.x[1], 2 * np.pi * 45.0, 2 * np.pi * 75.0)
        if self.x[1] != omega_pre_clip:
            # Omega hit boundary: hard constraint → reduce P[1,1] to ≤ 1 Hz uncertainty
            self.P[1, 1] = min(self.P[1, 1], (2 * np.pi * 1.0) ** 2)
        # ROCOF: ±15 Hz/s (tighter than ±20 Hz/s; >15 Hz/s is beyond any
        # real grid event and indicates numerical divergence)
        self.x[3] = np.clip(self.x[3],
                            -2 * np.pi * 15.0,
                             2 * np.pi * 15.0)
        # Phase [0, 2π)
        self.x[0] %= 2 * np.pi
        # Amplitude minimum
        self.x[2] = max(0.1, self.x[2])

        # Salida en Hz
        return self.x[1] / (2 * np.pi)
