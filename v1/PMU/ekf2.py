# --- 4.3b PROPOSED: EKF2 (self-contained tuning) ---
import numpy as np
FS_PHYSICS = 1000000.0   # 100 kHz (Physics/Ground Truth)
FS_DSP = 10000.0        # 10 kHz (IED/Relay Sampling Rate)
RATIO = int(FS_PHYSICS / FS_DSP)
DT_DSP = 1.0 / FS_DSP
F_NOM = 60.0

from collections import deque

# =====================================================
# Frecuencias de referencia (manténlas coherentes con el resto del código)
# =====================================================
FS_PHYSICS = 100000.0   # 100 kHz (Physics/Ground Truth)
FS_DSP = 10000.0        # 10 kHz (IED/Relay Sampling Rate)
RATIO = int(FS_PHYSICS / FS_DSP)
DT_DSP = 1.0 / FS_DSP
F_NOM = 60.0


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
        self.inn_buf = deque(maxlen=25)  # hist. corta de innovaciones

        self.I = np.eye(4)
        self.init = False
        self.name = "RA-EKF"

    # ---------- API de tuning embebido ----------
    @staticmethod
    def tuning_grid():
        """
        Devuelve una lista de diccionarios de parámetros para barrer.
        Cada dict se pasa luego como EKF2(**params).

        Se barre:
          - q_param: ruido de modelo en frecuencia
          - r_param: ruido de medida
          - inn_ref: escala típica de innovación
          - event_thresh: cuántas veces inn_ref dispara "evento"
          - fast_horizon_ms: duración del modo rápido
        """
        q_vals = np.logspace(-2, 4, 6)         # menos puntos que antes
        r_vals = np.logspace(-4, 1, 6)
        inn_ref_vals = [0.05, 0.1, 0.2, 0.5]
        event_thresh_vals = [1.5, 2.0, 3.0]
        fast_horizon_vals = [40.0, 80.0]       # en ms

        configs = []
        for q in q_vals:
            for r in r_vals:
                for inn_ref in inn_ref_vals:
                    for ev_th in event_thresh_vals:
                        for fh in fast_horizon_vals:
                            configs.append({
                                "q_param": q,
                                "r_param": r,
                                "inn_ref": inn_ref,
                                "event_thresh": ev_th,
                                "fast_horizon_ms": fh,
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
            Q_temp[3,3] *= 0.1 # DAMP the ROCOF in fast mode to prevent overshoot
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

            # No tocamos omega/domega para no romper ROCOF
            self.x[0] = new_theta
            # A y P se mantienen, pero podemos "inflar" la covarianza
            self.P *= 2.0  # más incertidumbre tras el evento

    # ---------- Núcleo EKF2 ----------
    def step(self, z):
        # -----------------------------
        # Fase de inicialización simple
        # -----------------------------
        if not self.init:
            if abs(z) < 0.99:
                self.x[0] = np.arcsin(z)
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

        # Primero detectamos evento
        self._maybe_trigger_event(inn, z)

        # Después adaptamos Q y R según innovación y modo
        self._adaptive_QR(inn)

        H = np.array([[A * np.cos(theta), 0.0, np.sin(theta), 0.0]])  # 1x4
        S = H @ self.P @ H.T + self.R
        inv_S = 1.0 / (S[0, 0] + 1e-12)
        K = self.P @ H.T * inv_S  # 4x1

        self.x += (K * inn).flatten()
        self.P = (self.I - K @ H) @ self.P

        # -----------------------------
        # Restricciones físicas
        # -----------------------------
        # Frecuencia en 40–80 Hz
        self.x[1] = np.clip(self.x[1], 2 * np.pi * 40.0, 2 * np.pi * 80.0)
        # ROCOF razonable, p.ej. ±20 Hz/s
        self.x[3] = np.clip(self.x[3],
                            -2 * np.pi * 20.0,
                            2 * np.pi * 20.0)
        # Fase [0, 2π)
        self.x[0] %= 2 * np.pi
        # Amplitud mínima
        self.x[2] = max(0.1, self.x[2])

        # Salida en Hz
        return self.x[1] / (2 * np.pi)
