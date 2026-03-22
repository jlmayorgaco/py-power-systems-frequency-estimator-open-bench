import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuración
# ============================================================
CSV_FREQ = "freq_matlab_exacta.csv"
CSV_VOLTAGE = "chamorro_case7_3ph.csv"
OUTPUT_DIR = "figures_noise_rls_fit_scaled"

PROFILE_NOISE_PCT = 0.05   # residual perfilado: 5% del pico fundamental
GAUSS_HF_PCT = 0.01        # ruido gaussiano HF: sigma = 1% del pico fundamental
RNG_SEED = 42


# ============================================================
# Utilidades
# ============================================================
def first_order_alpha(fc_hz: float, fs: float) -> float:
    ts = 1.0 / fs
    return (2.0 * np.pi * fc_hz * ts) / (1.0 + 2.0 * np.pi * fc_hz * ts)


def moving_average_same(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    y = np.convolve(xpad, kernel, mode="same")
    return y[pad:-pad]


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def center_to_nominal_prefault(t: np.ndarray, f: np.ndarray, f_nom: float = 50.0) -> np.ndarray:
    mask = t < 0.95
    if np.any(mask):
        return f - (np.mean(f[mask]) - f_nom)
    return f.copy()


def phase_from_frequency(t: np.ndarray, f: np.ndarray, theta0: float = 0.0) -> np.ndarray:
    theta = np.zeros_like(t)
    theta[0] = theta0
    dt = np.diff(t)

    for k in range(1, len(t)):
        omega_avg = 2.0 * np.pi * 0.5 * (f[k - 1] + f[k])
        theta[k] = theta[k - 1] + omega_avg * dt[k - 1]

    return theta


def scale_signal_to_peak_fraction(x: np.ndarray, ref_peak: float, frac: float) -> np.ndarray:
    """
    Escala x para que su valor máximo absoluto sea frac * ref_peak.
    """
    x0 = x - np.mean(x)
    xmax = np.max(np.abs(x0))
    if xmax < 1e-12:
        return np.zeros_like(x0)
    return x0 * ((frac * ref_peak) / xmax)


def make_highfreq_gaussian_noise(n: int, sigma: float, seed: int = 42) -> np.ndarray:
    """
    Ruido gaussiano con contenido más de alta frecuencia.
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 1.0, n)

    hf = np.empty(n)
    hf[0] = 0.0
    hf[1:] = np.diff(w)

    std_hf = np.std(hf)
    if std_hf < 1e-12:
        return np.zeros_like(hf)

    hf = hf / std_hf
    hf = sigma * hf
    return hf


# ============================================================
# RLS para ajustar:
# v(k) ≈ a(k)*sin(w0 t) + b(k)*cos(w0 t) + c(k)
# ============================================================
def adaptive_rls_50hz_fit(
    t: np.ndarray,
    v: np.ndarray,
    f0: float = 50.0,
    lam: float = 0.9995,
    delta: float = 1e5,
):
    w0 = 2.0 * np.pi * f0
    n = len(t)

    theta_hat = np.zeros(3)   # [a, b, c]
    P = delta * np.eye(3)

    a_hist = np.zeros(n)
    b_hist = np.zeros(n)
    c_hist = np.zeros(n)
    v_fit = np.zeros(n)
    err_hist = np.zeros(n)
    amp_hist = np.zeros(n)
    phi_hist = np.zeros(n)

    for k in range(n):
        phi_vec = np.array([
            np.sin(w0 * t[k]),
            np.cos(w0 * t[k]),
            1.0,
        ])

        y_hat = phi_vec @ theta_hat
        err = v[k] - y_hat

        denom = lam + phi_vec.T @ P @ phi_vec
        K = (P @ phi_vec) / denom

        theta_hat = theta_hat + K * err
        P = (P - np.outer(K, phi_vec) @ P) / lam

        a_k, b_k, c_k = theta_hat
        amp_k = np.sqrt(a_k**2 + b_k**2)
        phi_k = np.arctan2(b_k, a_k)

        a_hist[k] = a_k
        b_hist[k] = b_k
        c_hist[k] = c_k
        v_fit[k] = a_k * np.sin(w0 * t[k]) + b_k * np.cos(w0 * t[k]) + c_k
        err_hist[k] = v[k] - v_fit[k]
        amp_hist[k] = amp_k
        phi_hist[k] = phi_k

    phi_hist = np.unwrap(phi_hist)

    return {
        "a": a_hist,
        "b": b_hist,
        "c": c_hist,
        "amp": amp_hist,
        "phi": phi_hist,
        "v_fit": v_fit,
        "residual": err_hist,
    }


# ============================================================
# PLL monofásico SOGI
# ============================================================
class SOGIPLL1P:
    def __init__(
        self,
        fs: float,
        f_nom: float = 50.0,
        kp: float = 40.0,
        ki: float = 700.0,
        k_sogi: float = np.sqrt(2.0),
        fc_out: float = 9.0,
        max_rocof_hz_s: float = 300.0,
        theta0: float = 0.0,
    ):
        self.fs = fs
        self.ts = 1.0 / fs
        self.f_nom = f_nom
        self.w_nom = 2.0 * np.pi * f_nom

        self.kp = kp
        self.ki = ki
        self.k_sogi = k_sogi

        self.theta = theta0
        self.integral = 0.0

        self.v_alpha = 0.0
        self.v_beta = 0.0

        self.w_est = self.w_nom
        self.dw_max = 2.0 * np.pi * max_rocof_hz_s * self.ts

        self.alpha_out = first_order_alpha(fc_out, fs)
        self.f_out = f_nom

    def step(self, v: float) -> float:
        err_sogi = v - self.v_alpha
        self.v_alpha += self.ts * (self.k_sogi * self.w_est * err_sogi - self.w_est * self.v_beta)
        self.v_beta += self.ts * (self.w_est * self.v_alpha)

        c = np.cos(self.theta)
        s = np.sin(self.theta)

        vd = self.v_alpha * c + self.v_beta * s
        vq = -self.v_alpha * s + self.v_beta * c

        vmag = max(np.sqrt(vd * vd + vq * vq), 1e-9)
        err = np.clip(vq / vmag, -2.0, 2.0)

        self.integral += self.ki * err * self.ts
        delta_w = self.kp * err + self.integral

        w_target = self.w_nom + delta_w
        self.w_est = np.clip(w_target, self.w_est - self.dw_max, self.w_est + self.dw_max)

        self.theta = (self.theta + self.w_est * self.ts) % (2.0 * np.pi)

        f_raw = self.w_est / (2.0 * np.pi)
        self.f_out += self.alpha_out * (f_raw - self.f_out)
        return self.f_out


def warm_up_1ph(pll: SOGIPLL1P, v: np.ndarray, n_pf: int, repeats: int = 8):
    for _ in range(repeats):
        for sample in v[:n_pf]:
            pll.step(sample)


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("1. Cargando frecuencia real...")
    df_f = pd.read_csv(CSV_FREQ)
    t_f = df_f["t"].to_numpy(dtype=float)
    f_real = df_f["f"].to_numpy(dtype=float)

    print("2. Cargando voltaje real...")
    df_v = pd.read_csv(CSV_VOLTAGE)
    t_v = df_v["Tiempo"].to_numpy(dtype=float)
    v_real = df_v["Va"].to_numpy(dtype=float)

    if len(t_v) != len(t_f) or not np.allclose(t_v, t_f, atol=1e-12):
        print("   Interpolando voltaje a la malla temporal de frecuencia...")
        v_real = np.interp(t_f, t_v, v_real)

    t = t_f
    fs = 1.0 / np.median(np.diff(t))
    print(f"   fs = {fs:.3f} Hz")

    print("3. Ajustando senoide 50 Hz con RLS...")
    fit = adaptive_rls_50hz_fit(
        t=t,
        v=v_real,
        f0=50.0,
        lam=0.9995,
        delta=1e5,
    )

    v_fit_50 = fit["v_fit"]
    residual_raw = fit["residual"]
    amp_hat = fit["amp"]
    phi_hat = fit["phi"]
    c_hat = fit["c"]

    pref_mask = t < 0.95
    A_ref = float(np.median(amp_hat[pref_mask]))
    phi_ref = float(np.median(phi_hat[pref_mask]))
    dc_ref = float(np.median(c_hat[pref_mask]))

    print(f"   A_ref   = {A_ref:.6f} V")
    print(f"   phi_ref = {phi_ref:.6f} rad")
    print(f"   dc_ref  = {dc_ref:.6f} V")
    print(f"   residual raw std     = {np.std(residual_raw):.6f} V")
    print(f"   residual raw max abs = {np.max(np.abs(residual_raw)):.6f} V")

    print("4. Construyendo voltaje sintético...")
    theta_real = phase_from_frequency(t, f_real, theta0=0.0)

    # Fundamental limpia guiada por la frecuencia real
    v_synthetic_clean = A_ref * np.sin(theta_real + phi_ref) + dc_ref

    # Residual perfilado escalado a ±5% del pico fundamental
    residual_profile = scale_signal_to_peak_fraction(
        residual_raw,
        ref_peak=A_ref,
        frac=PROFILE_NOISE_PCT,
    )

    # Ruido gaussiano HF con sigma = 1% del pico fundamental
    gauss_hf = make_highfreq_gaussian_noise(
        n=len(t),
        sigma=GAUSS_HF_PCT * A_ref,
        seed=RNG_SEED,
    )

    # Señal final
    v_synthetic = v_synthetic_clean + residual_profile + gauss_hf

    print(f"   residual_profile max abs = {np.max(np.abs(residual_profile)):.6f} V")
    print(f"   gauss_hf std            = {np.std(gauss_hf):.6f} V")
    print(f"   v_synthetic max abs     = {np.max(np.abs(v_synthetic)):.6f} V")

    print("5. Corriendo PLL sobre el voltaje sintético...")
    v_base = max(A_ref, 1e-9)
    v_syn_pu = (v_synthetic - np.mean(v_synthetic[pref_mask])) / v_base

    pll = SOGIPLL1P(
        fs=fs,
        f_nom=50.0,
        kp=40.0,
        ki=700.0,
        fc_out=9.0,
        max_rocof_hz_s=300.0,
        theta0=0.0,
    )

    n_pf = int(np.sum(pref_mask))
    warm_up_1ph(pll, v_syn_pu, n_pf=n_pf, repeats=8)

    f_pll = np.zeros_like(t)
    for k in range(len(t)):
        f_pll[k] = pll.step(v_syn_pu[k])

    f_pll = center_to_nominal_prefault(t, f_pll, 50.0)

    win = int(max(5, round(0.010 * fs)))
    if win % 2 == 0:
        win += 1
    f_pll_s = moving_average_same(f_pll, win)

    pll_rmse = rmse(f_pll_s, f_real)
    print(f"   RMSE PLL vs f_real = {pll_rmse:.6f} Hz")

    print("6. Guardando PNGs...")
    plt.rcParams.update({"font.size": 11, "font.family": "serif"})

    # 1) Frecuencia real
    fig1, ax1 = plt.subplots(figsize=(10, 3.5))
    ax1.plot(t, f_real, color="tab:red", linewidth=1.8, label="Frecuencia real")
    ax1.axhline(50.0, color="k", linestyle="--", alpha=0.5, label="Nominal")
    ax1.set_title("Frecuencia real vs tiempo")
    ax1.set_xlabel("Tiempo [s]")
    ax1.set_ylabel("Frecuencia [Hz]")
    ax1.grid(True, linestyle=":", alpha=0.7)
    ax1.legend(loc="upper right")
    fig1.savefig(os.path.join(OUTPUT_DIR, "01_Frecuencia_Real_vs_t.png"), dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # 2) Voltaje real
    fig2, ax2 = plt.subplots(figsize=(10, 3.5))
    ax2.plot(t, v_real, color="tab:blue", linewidth=0.8, label="Voltaje real")
    ax2.set_title("Voltaje real fase A")
    ax2.set_xlabel("Tiempo [s]")
    ax2.set_ylabel("Voltaje [V]")
    ax2.grid(True, linestyle=":", alpha=0.7)
    ax2.legend(loc="upper right")
    fig2.savefig(os.path.join(OUTPUT_DIR, "02_Voltaje_Real.png"), dpi=300, bbox_inches="tight")
    plt.close(fig2)

    # 3) Voltaje real vs senoide 50 Hz ajustada RLS
    fig3, ax3 = plt.subplots(figsize=(10, 3.5))
    ax3.plot(t, v_real, linewidth=0.8, alpha=0.65, label="Voltaje real")
    ax3.plot(t, v_fit_50, linewidth=1.4, label="Senoide 50 Hz ajustada RLS")
    ax3.set_title("Voltaje real vs senoide 50 Hz ajustada")
    ax3.set_xlabel("Tiempo [s]")
    ax3.set_ylabel("Voltaje [V]")
    ax3.grid(True, linestyle=":", alpha=0.7)
    ax3.legend(loc="upper right")
    fig3.savefig(os.path.join(OUTPUT_DIR, "03_Voltaje_Real_vs_50Hz_RLS.png"), dpi=300, bbox_inches="tight")
    plt.close(fig3)

    # 4) Perturbaciones añadidas
    fig4, ax4 = plt.subplots(figsize=(10, 3.5))
    ax4.plot(t, residual_profile, color="tab:purple", linewidth=0.9, label="Residual perfilado 5%")
    ax4.plot(t, gauss_hf, color="tab:gray", linewidth=0.7, alpha=0.8, label="Gauss HF 1%")
    ax4.axhline(0.0, color="k", linestyle="--", alpha=0.4)
    ax4.set_title("Perturbaciones añadidas al voltaje sintético")
    ax4.set_xlabel("Tiempo [s]")
    ax4.set_ylabel("Voltaje [V]")
    ax4.grid(True, linestyle=":", alpha=0.7)
    ax4.legend(loc="upper right")
    fig4.savefig(os.path.join(OUTPUT_DIR, "04_Perturbaciones_5pct_1pct.png"), dpi=300, bbox_inches="tight")
    plt.close(fig4)

    # 5) Voltaje sintético
    fig5, ax5 = plt.subplots(figsize=(10, 3.5))
    ax5.plot(t, v_synthetic_clean, color="tab:green", linewidth=1.1, alpha=0.85, label="Fundamental limpia")
    ax5.plot(t, v_synthetic, color="tab:olive", linewidth=0.9, alpha=0.95, label="Voltaje sintético final")
    ax5.set_title("Voltaje sintético final")
    ax5.set_xlabel("Tiempo [s]")
    ax5.set_ylabel("Voltaje [V]")
    ax5.grid(True, linestyle=":", alpha=0.7)
    ax5.legend(loc="upper right")
    fig5.savefig(os.path.join(OUTPUT_DIR, "05_Voltaje_Sintetico.png"), dpi=300, bbox_inches="tight")
    plt.close(fig5)

    # 6) Frecuencia estimada por PLL
    fig6, ax6 = plt.subplots(figsize=(10, 3.5))
    ax6.plot(t, f_real, color="tab:red", linewidth=1.8, label="Frecuencia real")
    ax6.plot(t, f_pll_s, color="tab:orange", linewidth=1.4, label="PLL estimada")
    ax6.axhline(50.0, color="k", linestyle="--", alpha=0.5, label="Nominal")
    ax6.text(
        0.02,
        0.05,
        f"RMSE = {pll_rmse:.4f} Hz",
        transform=ax6.transAxes,
        bbox=dict(facecolor="white", alpha=0.85),
    )
    ax6.set_title("Frecuencia estimada por PLL sobre voltaje sintético")
    ax6.set_xlabel("Tiempo [s]")
    ax6.set_ylabel("Frecuencia [Hz]")
    ax6.grid(True, linestyle=":", alpha=0.7)
    ax6.legend(loc="upper right")
    fig6.savefig(os.path.join(OUTPUT_DIR, "06_PLL_Frecuencia_Estimada.png"), dpi=300, bbox_inches="tight")
    plt.close(fig6)

    # 7) Amplitud estimada por RLS
    fig7, ax7 = plt.subplots(figsize=(10, 3.3))
    ax7.plot(t, amp_hat, linewidth=1.2, label="Amplitud estimada RLS")
    ax7.axhline(A_ref, color="k", linestyle="--", alpha=0.5, label="A_ref")
    ax7.set_title("Amplitud estimada de la fundamental 50 Hz")
    ax7.set_xlabel("Tiempo [s]")
    ax7.set_ylabel("Amplitud [V]")
    ax7.grid(True, linestyle=":", alpha=0.7)
    ax7.legend(loc="upper right")
    fig7.savefig(os.path.join(OUTPUT_DIR, "07_Amplitud_RLS.png"), dpi=300, bbox_inches="tight")
    plt.close(fig7)

    # 8) Residual crudo
    fig8, ax8 = plt.subplots(figsize=(10, 3.3))
    ax8.plot(t, residual_raw, linewidth=0.9, label="Residual crudo")
    ax8.axhline(0.0, color="k", linestyle="--", alpha=0.4)
    ax8.set_title("Residual crudo = v_real - ajuste RLS")
    ax8.set_xlabel("Tiempo [s]")
    ax8.set_ylabel("Voltaje [V]")
    ax8.grid(True, linestyle=":", alpha=0.7)
    ax8.legend(loc="upper right")
    fig8.savefig(os.path.join(OUTPUT_DIR, "08_Residual_Crudo.png"), dpi=300, bbox_inches="tight")
    plt.close(fig8)

    # CSV de salida
    df_out = pd.DataFrame(
        {
            "t": t,
            "f_real": f_real,
            "v_real": v_real,
            "v_fit_50_rls": v_fit_50,
            "residual_raw": residual_raw,
            "residual_profile_5pct": residual_profile,
            "gauss_hf_1pct": gauss_hf,
            "amp_hat": amp_hat,
            "phi_hat": phi_hat,
            "c_hat": c_hat,
            "v_synthetic_clean": v_synthetic_clean,
            "v_synthetic": v_synthetic,
            "f_pll": f_pll,
            "f_pll_s": f_pll_s,
        }
    )
    df_out.to_csv(os.path.join(OUTPUT_DIR, "signals_and_pll_results.csv"), index=False)

    print("Listo.")
    print(f"Resultados guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()