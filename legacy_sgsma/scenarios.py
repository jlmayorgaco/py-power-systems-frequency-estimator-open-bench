import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy.ndimage import uniform_filter1d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from estimators import get_test_signals
import json

RESULTS_DIR = "results_raw"

# ==========================================
# 0. CONFIGURACIÓN GLOBAL (ESTILO IEEE Q1)
# ==========================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'figure.dpi': 600,
    'lines.linewidth': 1.0,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.titlepad': 4,
    'mathtext.fontset': 'stix',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02
})

OUTPUT_DIR = "figures_final_submission_v6"
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

FIG_SIZE_SINGLE = (3.5, 2.6)

print(f"--- GENERANDO TODAS LAS FIGURAS EN: {OUTPUT_DIR} ---")


# ==========================================
# HELPERS PARA JSON: SEÑALES + MÉTRICAS
# ==========================================
def load_signals_from_json(scenario: str, method_suffix: str):
    """
    Lee un archivo JSON de results_raw y devuelve:
    t, v, f_true, f_hat como arrays de numpy.
    """
    fname = f"{scenario}__{method_suffix}.json"
    path = os.path.join(RESULTS_DIR, scenario, fname)
    with open(path, "r") as f:
        data = json.load(f)
    sig = data["signals"]
    t = np.array(sig["t"], dtype=float)
    v = np.array(sig["v"], dtype=float)
    f_true = np.array(sig["f_true"], dtype=float)
    f_hat = np.array(sig["f_hat"], dtype=float)
    return t, v, f_true, f_hat


def load_metrics_from_json(scenario: str, method_suffix: str):
    """
    Lee el bloque 'metrics' del mismo JSON de señales.
    Devuelve un dict con métricas (RMSE, TIME_PER_SAMPLE_US, etc.).
    """
    fname = f"{scenario}__{method_suffix}.json"
    path = os.path.join(RESULTS_DIR, scenario, fname)
    with open(path, "r") as f:
        data = json.load(f)
    return data["metrics"]


def save_fig(fig, name):
    fig.savefig(f"{OUTPUT_DIR}/{name}.pdf", format="pdf")
    fig.savefig(f"{OUTPUT_DIR}/{name}.png", dpi=600)
    print(f"[OK] {name}")
    plt.close(fig)


def _normalize_scores(raw_dict, higher_is_better=True):
    """
    Convierte un diccionario {método: métrica_bruta} en
    {método: score entre 1 y 10}.
    Si higher_is_better=False, valores más pequeños => mejor score.
    """
    vals = np.array(list(raw_dict.values()), dtype=float)
    vmin, vmax = float(vals.min()), float(vals.max())
    scores = {}

    if np.isclose(vmax, vmin):
        # todos iguales -> 7 para todos (neutro)
        for k in raw_dict.keys():
            scores[k] = 7.0
        return scores

    for k, v in raw_dict.items():
        if higher_is_better:
            s = 1.0 + 9.0 * ((v - vmin) / (vmax - vmin))
        else:
            s = 1.0 + 9.0 * ((vmax - v) / (vmax - vmin))
        scores[k] = float(s)
    return scores


def compute_radar_profiles(method_labels):
    """
    Construye los perfiles de radar para los métodos indicados en
    'method_labels', usando SIEMPRE los mismos JSON de las señales.

    Dimensiones:
      - 'Steady' : RMSE en IEEE_Modulation
      - 'Trans.' : RMSE en IBR_Nightmare
      - 'Noise'  : RMSE en IBR_MultiEvent
      - 'Eff.'   : TIME_PER_SAMPLE_US en IBR_MultiEvent
      - 'Safe'   : TRIP_TIME_0p5 en IBR_MultiEvent

    Devuelve:
      profiles = { label: [S, T, N, E, Sa] }
    con valores en [1, 10].
    """

    # Mapeo etiqueta → sufijo de archivo
    method_key = {
        "RA-EKF": "EKF2",
        "PLL":    "PLL",
        "IpDFT":  "IpDFT",
        "EKF":    "EKF",
        "UKF":    "UKF",
        "PI-GRU": "PI-GRU",
    }

    methods = [m for m in method_labels if m in method_key]

    steady_raw = {}
    trans_raw = {}
    noise_raw = {}
    eff_raw = {}
    safe_raw = {}

    for label in methods:
        suf = method_key[label]

        # Steady: IEEE_Modulation
        m_mod = load_metrics_from_json("IEEE_Modulation", suf)
        steady_raw[label] = m_mod["RMSE"]

        # Transient: IBR_Nightmare
        m_tr = load_metrics_from_json("IBR_Nightmare", suf)
        trans_raw[label] = m_tr["RMSE"]

        # Noise + Eff + Safe: IBR_MultiEvent
        m_me = load_metrics_from_json("IBR_MultiEvent", suf)
        noise_raw[label] = m_me["RMSE"]
        eff_raw[label]   = m_me["TIME_PER_SAMPLE_US"]
        safe_raw[label]  = m_me["TRIP_TIME_0p5"]

    # Normalización 1–10 (menor error/coste/tiempo => mejor score)
    steady_scores = _normalize_scores(steady_raw, higher_is_better=False)
    trans_scores  = _normalize_scores(trans_raw,  higher_is_better=False)
    noise_scores  = _normalize_scores(noise_raw,  higher_is_better=False)
    eff_scores    = _normalize_scores(eff_raw,    higher_is_better=False)
    safe_scores   = _normalize_scores(safe_raw,   higher_is_better=False)

    profiles = {}
    for label in methods:
        profiles[label] = [
            steady_scores[label],
            trans_scores[label],
            noise_scores[label],
            eff_scores[label],
            safe_scores[label],
        ]

    return profiles


def compute_compliance_matrix(method_labels, scenario_ids):
    """
    Devuelve:
      d[m, s]   : 0=Pass, 1=Fail según mediana de RMSE por escenario.
      rmse[m, s]: RMSE bruto por método y escenario.

    Regla de compliance por escenario:
      - Para cada escenario, calcular los RMSE de todos los métodos.
      - Umbral = mediana de esos RMSE.
      - 'Pass' (0) si RMSE <= mediana, 'Fail' (1) si RMSE > mediana.

    Solo usa datos de los JSON, sin thresholds hardcodeados.
    """
    # mapa método → sufijo de archivo
    method_key = {
        "RA-EKF": "EKF2",
        "UKF":    "UKF",
        "PLL":    "PLL",
        "SOGI":   "SOGI",
        "IpDFT":  "IpDFT",
        "PI-GRU": "PI-GRU",
    }

    n_m = len(method_labels)
    n_s = len(scenario_ids)
    rmse = np.zeros((n_m, n_s), dtype=float)
    d = np.zeros((n_m, n_s), dtype=int)

    # rellenar matriz de RMSE
    for j, scen in enumerate(scenario_ids):
        for i, m in enumerate(method_labels):
            suf = method_key[m]
            metrics = load_metrics_from_json(scen, suf)
            rmse[i, j] = float(metrics["RMSE"])

        # umbral por escenario = mediana
        med = float(np.median(rmse[:, j]))
        for i in range(n_m):
            d[i, j] = 0 if rmse[i, j] <= med else 1

    return d, rmse


def get_cost_risk_points(method_labels, scenario="IBR_MultiEvent"):
    """
    Devuelve dos diccionarios cost[method], risk[method] a partir de métricas
    del escenario dado.

    cost = TIME_PER_SAMPLE_US
    risk = TRIP_TIME_0p5
    """
    method_key = {
        "SOGI":    "SOGI",
        "PLL":     "PLL",
        "IpDFT":   "IpDFT",
        "UKF":     "UKF",
        "RA-EKF":  "EKF2",
        "PI-GRU":  "PI-GRU",
        "Koopman": "Koopman-RKDPmu",
        "Teager":  "Teager",
        "RLS":     "RLS",
    }

    costs = {}
    risks = {}

    for m in method_labels:
        suf = method_key[m]
        metrics = load_metrics_from_json(scenario, suf)
        costs[m] = float(metrics["TIME_PER_SAMPLE_US"])
        risks[m] = float(metrics["TRIP_TIME_0p5"])

    return costs, risks


def _rmse_to_score_matrix(rmse: np.ndarray) -> np.ndarray:
    """
    Convierte una matriz de RMSE en una matriz de scores [0,1] por columna:
    - 1.0  => mejor (RMSE mínimo)
    - 0.0  => peor  (RMSE máximo)
    """
    scores = np.zeros_like(rmse, dtype=float)
    for j in range(rmse.shape[1]):
        col = rmse[:, j]
        cmin, cmax = float(col.min()), float(col.max())
        if np.isclose(cmax, cmin):
            scores[:, j] = 0.5
        else:
            scores[:, j] = (cmax - col) / (cmax - cmin)
    return scores


# ==========================================
# 1. GENERACIÓN DE DATOS A PARTIR DE JSON
# ==========================================
def get_all_data():
    """
    1) Transient (IBR_Nightmare, ventana 0.69–1.5 s)
    2) Ramp (IEEE_Freq_Ramp, ventana 0.28–0.42 s)
    3) Multi-Event (IBR_MultiEvent, 0–5 s)
    4) Modulation (IEEE_Modulation, errores 0–1 s)
    """

    # -----------------------------
    # A) Transient (IBR_Nightmare)
    # -----------------------------
    # RA-EKF (EKF2)
    tN, _, ftrue_N, fhat_ekf2 = load_signals_from_json("IBR_Nightmare", "EKF2")
    # PLL y SOGI
    _, _, _, fhat_pll = load_signals_from_json("IBR_Nightmare", "PLL")
    _, _, _, fhat_sogi = load_signals_from_json("IBR_Nightmare", "SOGI")
    # EKF clásico e IpDFT
    _, _, _, fhat_ekf = load_signals_from_json("IBR_Nightmare", "EKF")
    _, _, _, fhat_ipdft = load_signals_from_json("IBR_Nightmare", "IpDFT")

    mask_tr = (tN >= 0.69) & (tN <= 1.5)
    tt = tN[mask_tr] * 1000.0
    ft_trans = ftrue_N[mask_tr]
    f_RA_tr = fhat_ekf2[mask_tr]      # RA-EKF
    f_pll_tr = fhat_pll[mask_tr]
    f_ipdft_tr = fhat_ipdft[mask_tr]
    f_ekf_tr = fhat_ekf[mask_tr]      # EKF vanilla
    f_sogi_tr = fhat_sogi[mask_tr]

    # -----------------------------
    # B) Ramp (IEEE_Freq_Ramp)
    # -----------------------------
    # RA-EKF (EKF2)
    tR, _, ftrue_R, fhat_ekf2_rm = load_signals_from_json("IEEE_Freq_Ramp", "EKF2")
    # PLL
    _, _, _, fhat_pll_rm = load_signals_from_json("IEEE_Freq_Ramp", "PLL")
    # EKF vanilla e IpDFT
    _, _, _, fhat_ekf_rm = load_signals_from_json("IEEE_Freq_Ramp", "EKF")
    _, _, _, fhat_ipdft_rm = load_signals_from_json("IEEE_Freq_Ramp", "IpDFT")

    mask_rm = (tR >= 0.28) & (tR <= 0.42)
    tr = tR[mask_rm] * 1000.0
    ft_ramp = ftrue_R[mask_rm]
    fr_pll = fhat_pll_rm[mask_rm]
    fr_RA_rm = fhat_ekf2_rm[mask_rm]
    fr_ekf_rm = fhat_ekf_rm[mask_rm]
    fr_ipdft_rm = fhat_ipdft_rm[mask_rm]

    # -----------------------------------
    # C) Multi-Event: IBR_MultiEvent 0–5s
    # -----------------------------------
    tM, _, ftrue_M, _ = load_signals_from_json("IBR_MultiEvent", "EKF2")
    tm = tM
    ft_multi = ftrue_M

    method_map = {
        "RA-EKF": "EKF2",
        "UKF": "UKF",
        "IpDFT": "IpDFT",
        "Koopman": "Koopman-RKDPmu",
        "TFT": "TFT",
        "SRF-PLL": "PLL",
        "SOGI": "SOGI",
        "Teager": "Teager",
        "RLS": "RLS",
        "PI-GRU": "PI-GRU",
    }

    res_multi = {}
    for label, suf in method_map.items():
        _, _, _, fhat = load_signals_from_json("IBR_MultiEvent", suf)
        res_multi[label] = fhat

    # -----------------------------------
    # D) Modulation: IEEE_Modulation
    # -----------------------------------
    tMod, _, ftrue_mod, fhat_ekf_mod = load_signals_from_json("IEEE_Modulation", "EKF2")
    _, _, _, fhat_pll_mod = load_signals_from_json("IEEE_Modulation", "PLL")
    _, _, _, fhat_ipdft_mod = load_signals_from_json("IEEE_Modulation", "IpDFT")

    mask_mod = (tMod <= 1.0)
    tm_mod = tMod[mask_mod]
    err_ipdft = np.abs(fhat_ipdft_mod[mask_mod] - ftrue_mod[mask_mod])
    err_ekf = np.abs(fhat_ekf_mod[mask_mod] - ftrue_mod[mask_mod])
    err_pll = np.abs(fhat_pll_mod[mask_mod] - ftrue_mod[mask_mod])

    #           Phase Jump                                   Ramp
    return (tt, ft_trans, f_RA_tr, f_pll_tr, f_ipdft_tr, f_ekf_tr, f_sogi_tr), \
           (tr, ft_ramp, fr_pll, fr_RA_rm, fr_ekf_rm, fr_ipdft_rm), \
           (tm, ft_multi, res_multi), \
           (tm_mod, err_ipdft, err_ekf, err_pll)


# ==========================================
# FIG 1: ESCENARIOS (usa benchmark + ruido IBR visual)
# ==========================================
def fig1_scenarios():
    signals = get_test_signals()

    tA, vA, fA, _ = signals["IEEE_Mag_Step"]
    tB, vB, fB, _ = signals["IEEE_Freq_Ramp"]
    tC, vC, fC, _ = signals["IEEE_Modulation"]
    tD, vD, fD, _ = signals["IBR_Nightmare"]
    tE, vE, fE, _ = signals["IBR_MultiEvent"]

    def add_ibr_noise(t, v,
                      h5=0.02, h7=0.01,
                      inter_f=180.0, inter_amp=0.003,
                      sigma=0.001, seed=0):
        rng = np.random.default_rng(seed)
        v_noisy = v.copy()
        phase_nom = 2.0 * np.pi * 60.0 * t
        v_noisy += h5 * np.sin(5.0 * phase_nom)
        v_noisy += h7 * np.sin(7.0 * phase_nom)
        v_noisy += inter_amp * np.sin(2.0 * np.pi * inter_f * t)
        v_noisy += rng.normal(0.0, sigma, size=len(t))
        return v_noisy

    # Use seeded RNG for reproducible noise overlays (separate seeds per scenario)
    _rng_a = np.random.default_rng(101)
    _rng_b = np.random.default_rng(102)
    _rng_c = np.random.default_rng(103)
    vA_plot = add_ibr_noise(tA, vA, h5=0.005, h7=0.005, seed=1) + _rng_a.normal(0, 0.01, len(vA))
    vB_plot = add_ibr_noise(tB, vB, h5=0.001, h7=0.002, seed=2) + _rng_b.normal(0, 0.01, len(vB))
    vC_plot = add_ibr_noise(tC, vC, h5=0.001, h7=0.002, seed=3) + _rng_c.normal(0, 0.01, len(vC))
    vD_plot = add_ibr_noise(tD, vD, h5=0.0001, h7=0.01, seed=4)
    vE_plot = add_ibr_noise(tE, vE, h5=0.0001, h7=0.01, seed=5)

    tA_ms = tA * 1000.0
    tB_ms = tB * 1000.0
    tC_ms = tC * 1000.0
    tD_ms = tD * 1000.0
    tE_ms = tE * 1000.0

    fig, axs = plt.subplots(
        5, 2,
        figsize=(7.5, 4.0),
        gridspec_kw={"hspace": 0.55, "wspace": 0.2}
    )
    BBOX = dict(boxstyle="square,pad=0.1", fc="white", alpha=0.8, ec="none")

    def row(i, t_ms, v, f, tit_left, ylim_f, xzoom_left, tit_right):
        axs[i, 0].plot(t_ms, v, "purple" if i == 4 else "b", lw=0.5)
        axs[i, 0].set_xlim(xzoom_left)
        axs[i, 0].set_yticks([])
        axs[i, 0].set_title(tit_left, fontsize=7, fontweight="bold", pad=2)
        axs[i, 0].set_ylabel("V [pu]", rotation=0, labelpad=5, fontsize=6)

        axs[i, 1].plot(t_ms, f, "r", lw=0.8)
        axs[i, 1].set_ylim(ylim_f)
        axs[i, 1].set_title(tit_right, fontsize=7, pad=2)
        axs[i, 1].set_ylabel("f [Hz]", labelpad=1)
        if i == 4:
            axs[i, 1].set_xlabel("Time [ms]", labelpad=1)

    row(0, tA_ms, vA_plot, fA,
        "(A) Step (+10%)", (59.5, 60.5), (450, 550), "Freq (Ideal)")

    row(1, tB_ms, vB_plot, fB,
        "(B) Ramp (+5Hz/s)", (59, 65), (200, 1100), "Freq (+5Hz/s)")

    row(2, tC_ms, vC_plot, fC,
        "(C) Modulation (2Hz)", (59.4, 60.6), (0, 1000), "Freq (FM)")

    row(3, tD_ms, vD_plot, fD,
        "(D) Islanding (Jump)", (50, 80), (680, 720), "Phase-step freq. spike")

    axs[3, 0].annotate(
        "Jump",
        xy=(700, vD_plot[(np.abs(tD_ms - 700)).argmin()]),
        xytext=(690, 0.6),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", color="r"),
        color="r",
        fontsize=6,
    )
    axs[3, 1].arrow(700, 60, 0, 15, head_width=20, color="red")

    row(4, tE_ms, vE_plot, fE,
        "(E) Multi-Evt (Noise)", (50, 65), (2450, 2550), "Full Profile")

    axs[4, 0].annotate(
        r"Jump $+80^\circ$",
        xy=(2500, vE_plot[(np.abs(tE_ms - 2500)).argmin()]),
        xytext=(2460, 0.6),
        arrowprops=dict(arrowstyle="->", color="k"),
        fontsize=6,
        bbox=BBOX,
    )

    axs[4, 1].axvline(1000, color="k", linestyle=":", lw=0.7)
    axs[4, 1].axvline(2500, color="k", linestyle=":", lw=0.7)
    axs[4, 1].text(1020, 61.5, r"1. +40$^\circ$ Jump",
                   fontsize=6, ha="left", color="k", bbox=BBOX)
    axs[4, 1].text(1500, 56.5, "2. Neg. Ramp",
                   fontsize=6, ha="center", color="k", bbox=BBOX)
    axs[4, 1].text(3000, 61.0, "3. Ring-down",
                   fontsize=6, ha="center", color="b", bbox=BBOX)

    for i in range(4):
        axs[i, 0].set_xlabel("")
        axs[i, 1].set_xlabel("")

    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(hspace=0.40, wspace=0.15, top=0.98, bottom=0.04)

    save_fig(fig, "Fig1_Scenarios_Final")


# ==========================================
# 3. DASHBOARD (MEGA FIG 2)
# ==========================================
def generate_dashboard():
    (tt, ft_tr, fRA_tr, fpll_tr, fipdft_tr, fekf_tr, fsogi_tr), \
    (tr, frt, frpll, frekf_RA, fekf_rm, fipdft_rm), \
    (tm, fmt, res), \
    (tm_mod, e_ip, e_ekf, e_pll) = get_all_data()

    fig_d = plt.figure(figsize=(7.16, 9.0))
    gs = GridSpec(4, 2, hspace=0.4, wspace=0.22)

    # ===== Fila 1: Transient + Ramp =====
    # (a) Phase Jump
    fr_ekf_smooth_tr = uniform_filter1d(fRA_tr, size=3)   # RA-EKF suavizado

    ax_a = fig_d.add_subplot(gs[0, 0])

    # Referencia
    ax_a.plot(tt, ft_tr, "k", alpha=0.3, lw=0.8, label="Ref")

    # Métodos
    ax_a.plot(tt, fpll_tr, "b--", lw=1.0, label="SRF-PLL")
    ax_a.plot(tt, fr_ekf_smooth_tr, "r-", lw=1.3, label="RA-EKF")
    ax_a.plot(tt, fekf_tr, color="orange", ls="-.", lw=0.9, label="EKF")
    ax_a.plot(tt, fipdft_tr, color="g", ls=":", lw=0.9, label="IpDFT")

    ax_a.set_xlim(690, 1100)
    ax_a.set_ylim(59.95, 61.2)
    ax_a.set_ylabel("Hz")
    ax_a.set_title("(a) Phase Jump (Scenario D)", fontweight="bold", loc="left")

    # marcar instante de islanding
    ax_a.axvline(700, color="k", linestyle=":", lw=0.7)
    ax_a.text(702, 60.18, "Islanding", fontsize=6, ha="left", va="top")

    ax_a.legend(fontsize=6, loc="upper right", frameon=False)
    ax_a.set_xlabel("Time [ms]")

    # (b) Ramp Lag — RA-EKF vs PLL vs EKF vanilla vs IpDFT
    fr_ekf_smooth_rm     = uniform_filter1d(frekf_RA,  size=5)   # RA-EKF
    fr_ekfvan_smooth_rm  = uniform_filter1d(fekf_rm,   size=5)   # EKF vanilla
    fr_ipdft_smooth_rm   = uniform_filter1d(fipdft_rm, size=5)   # IpDFT

    ax_b = fig_d.add_subplot(gs[0, 1])

    # Referencia
    ax_b.plot(tr, frt, "k-", alpha=0.3, lw=0.8, label="Ref")

    # Métodos
    ax_b.plot(tr, frpll,              "b--",   lw=1.0, label="SRF-PLL")
    ax_b.plot(tr, fr_ekf_smooth_rm,   "r-",    lw=1.2, label="RA-EKF")
    ax_b.plot(tr, fr_ekfvan_smooth_rm, color="orange", ls="-.", lw=1.0, label="EKF")
    ax_b.plot(tr, fr_ipdft_smooth_rm, color="green",  ls=":",  lw=1.0, label="IpDFT")

    ax_b.set_xlim(280, 420)
    ax_b.set_ylim(59.9, 60.7)
    ax_b.set_title("(b) Ramp Lag (Scenario B)", fontweight="bold", loc="left")
    ax_b.set_xlabel("Time [ms]")
    ax_b.set_ylabel("Hz")

    # Flecha de lag
    ax_b.annotate(
        "",
        xy=(350, 60.25),
        xytext=(365, 60.25),
        arrowprops=dict(arrowstyle="<->", color="b", lw=0.8),
    )
    ax_b.text(350, 60.35, "Lag", color="b", fontsize=7)

    ax_b.legend(
        fontsize=6,
        loc="lower right",
        frameon=False,
        ncol=1
    )

    # ===== Fila 2: Top 5 / Bottom 5 =====
    # (c) Top 5 Stable + inset de detalle
    ax_c = fig_d.add_subplot(gs[1, 0])
    ax_c.plot(tm, fmt, 'k', alpha=0.3, label='True')

    top_methods = ['RA-EKF', 'UKF', 'IpDFT', 'Koopman']
    top_cols = ['r', 'orange', 'g', 'brown']
    top_styles = ['-', '--', '-.', ':']

    for m, c, s in zip(top_methods, top_cols, top_styles):
        lw = 1.5 if m == 'RA-EKF' else 1.0
        ax_c.plot(tm, res[m], color=c, ls=s, lw=lw, label=m)

    ax_c.set_ylim(50, 65)
    ax_c.set_ylabel('Hz')
    ax_c.set_xlabel('Time [s]')
    ax_c.set_title('(c) MultiEvent (E) Top 5 Stable', fontweight='bold', loc='left')

    # --- Zona que se va a hacer zoom ---
    t1_zoom, t2_zoom = 2.4500, 2.750
    mask_zoom = (tm >= t1_zoom) & (tm <= t2_zoom)

    ax_c.axvspan(t1_zoom, t2_zoom, color='lightgrey', alpha=0.25, lw=0)

    axins = inset_axes(
        ax_c,
        width="42%", height="45%",
        loc="upper left",
        borderpad=1.0
    )
    axins.plot(tm[mask_zoom], fmt[mask_zoom], 'k', alpha=0.3, lw=0.3)

    for m, c, s in zip(top_methods, top_cols, top_styles):
        lw = 0.6 if m == 'RA-EKF' else 0.5
        axins.plot(tm[mask_zoom], res[m][mask_zoom],
                   color=c, ls=s, lw=lw)

    axins.set_xlim(t1_zoom, t2_zoom)
    axins.set_ylim(56.0, 63.5)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_title('Detail', fontsize=6, pad=2)

    mark_inset(
        ax_c, axins,
        loc1=1, loc2=3,
        fc="none",
        ec="0.4",
        lw=0.4,
    )

    ax_c.legend(fontsize=6, ncol=3, loc='lower left', frameon=False)

    # (d) Bottom 5 Unstable
    ax_d = fig_d.add_subplot(gs[1, 1])
    ax_d.plot(tm, fmt, "k", alpha=0.3)
    for m, c in zip(["SRF-PLL", "SOGI", "Teager"], ["b", "gray", "y"]):
        lw = 1.0 if m == "SRF-PLL" else 0.8
        alpha = 0.9 if m == "SRF-PLL" else (0.7 if m == "SOGI" else 0.5)
        ax_d.plot(tm, res[m], color=c, lw=lw, alpha=alpha, label=m)
    ax_d.set_ylim(40, 80)
    ax_d.set_xlabel("Time [s]")
    ax_d.set_title("(d) MultiEvent (E) Bottom 5 Unstable", fontweight="bold", loc="left")
    ax_d.legend(fontsize=6, loc="upper right", frameon=False)

    # ===== Fila 3: Heatmap + Steady Err =====
    # (g) Compliance Heatmap — data-driven con gradiente
    ax_g = fig_d.add_subplot(gs[2, 0])
    scen_ids    = ["IEEE_Mag_Step", "IEEE_Freq_Ramp", "IEEE_Modulation", "IBR_Nightmare", "IBR_MultiEvent"]
    scen_labels = ["Step", "Ramp", "Mod", "Isl", "Multi"]
    alg_labels  = ["RA-EKF", "UKF", "PLL", "SOGI", "IpDFT", "PI-GRU"]

    d, rmse = compute_compliance_matrix(alg_labels, scen_ids)
    scores = _rmse_to_score_matrix(rmse)  # [0,1], 1=mejor

    im = ax_g.imshow(scores, cmap="RdYlGn", aspect="auto", vmin=0.0, vmax=1.0)
    ax_g.set_xticks(np.arange(len(scen_labels)))
    ax_g.set_xticklabels(scen_labels, fontsize=7)
    ax_g.set_yticks(np.arange(len(alg_labels)))
    ax_g.set_yticklabels(alg_labels, fontsize=7)

    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            # cuadrito tipo grid
            rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                             fill=False, edgecolor="white",
                             linewidth=0.5, alpha=0.7)
            ax_g.add_patch(rect)

            txt = "P" if d[i, j] == 0 else "F"
            # texto blanco en celdas oscuras, negro en claras
            text_color = "white" if scores[i, j] < 0.4 else "black"
            ax_g.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color=text_color,
                fontsize=7,
                fontweight="bold",
            )

    ax_g.set_title("(g) Compliance Across All Scenarios (A–E)", fontweight="bold", loc="left")

    # colorbar tipo "Failure / Marginal / Excellent"
    cbar = fig_d.colorbar(im, ax=ax_g, fraction=0.046, pad=0.02)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["Failure", "Marginal", "Excellent"])
    cbar.ax.tick_params(labelsize=6)

    # (h) Steady-State Error
    ax_h = fig_d.add_subplot(gs[2, 1])
    ax_h.plot(tm_mod, e_pll, "b--", lw=0.8, label="PLL")
    ax_h.plot(tm_mod, e_ekf, "r-", lw=1.0, label="RA-EKF")
    ax_h.plot(tm_mod, e_ip, "g-.", lw=1.1, label="IpDFT")
    ax_h.set_yscale("log")
    ax_h.set_ylim(1e-4, 1.0)
    ax_h.set_xlabel("Time [s]")
    ax_h.set_ylabel(r"Log |Error| [Hz]")
    ax_h.set_title("(h) Steady-State Error (Scenario C)", fontweight="bold", loc="left")
    ax_h.legend(fontsize=6, ncol=2, frameon=False)

    # ===== Fila 4: Pareto + Radar =====
    # (e) Cost vs Risk — data-driven
    ax_e = fig_d.add_subplot(gs[3, 0])

    pareto_methods = [
        "SOGI", "PLL", "IpDFT", "UKF",
        "RA-EKF", "PI-GRU", "Koopman",
        "Teager", "RLS",
    ]

    costs, risks = get_cost_risk_points(pareto_methods, scenario="IBR_MultiEvent")

    # estilos visuales (solo estéticos, no métricos)
    method_style = {
        "SOGI":   ("gray",   "o"),
        "PLL":    ("blue",   "v"),
        "IpDFT":  ("green",  "s"),
        "UKF":    ("orange", "D"),
        "RA-EKF": ("red",    "*"),
        "PI-GRU": ("purple", "X"),
        "Koopman":("brown",  "p"),
        "Teager": ("y",      "<"),
        "RLS":    ("magenta",">"),
    }

    xvals = np.array(list(costs.values()))
    yvals = np.array(list(risks.values()))

    for m in pareto_methods:
        col, marker = method_style[m]
        s = 100 if m == "RA-EKF" else (60 if m == "PI-GRU" else 40)
        ax_e.scatter(
            costs[m],
            risks[m],
            c=col,
            marker=marker,
            s=s,
            edgecolors="k",
            lw=0.4,
            zorder=5,
        )
        ax_e.annotate(
            m,
            (costs[m], risks[m]),
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=6,
            ha="center",
            color="black",
        )

    ax_e.set_xscale("log")
    ax_e.set_yscale("log")

    # límites derivados de los datos
    ax_e.set_xlim(0.8 * xvals.min(), 1.2 * xvals.max())
    ax_e.set_ylim(0.8 * yvals.min(), 1.2 * yvals.max())

    ax_e.grid(True, which="both", ls=":", lw=0.5, alpha=0.4)
    ax_e.set_xlabel(r"Cost [$\mu$s]")
    ax_e.set_ylabel("Risk [s]")
    ax_e.set_title("(e) Cost vs Risk", fontweight="bold", loc="left")

    # (f) Radar / Balance – DATA-DRIVEN + UKF + PI-GRU (ML)
    ax_f = fig_d.add_subplot(gs[3, 1], polar=True)
    cats = ["Steady", "Trans.", "Noise", "Eff.", "Safe"]
    N = len(cats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    labels_dash = ["RA-EKF", "PLL", "IpDFT", "UKF", "PI-GRU"]
    profiles = compute_radar_profiles(labels_dash)

    def _closed(vals):
        return vals + vals[:1]

    ra_vals  = _closed(profiles["RA-EKF"])
    pl_vals  = _closed(profiles["PLL"])
    ip_vals  = _closed(profiles["IpDFT"])
    ukf_vals = _closed(profiles["UKF"])
    ml_vals  = _closed(profiles["PI-GRU"])

    ax_f.set_theta_offset(np.pi / 2)
    ax_f.set_theta_direction(-1)
    ax_f.plot(angles, ra_vals,  "r-",      lw=2, label="RA-EKF")
    ax_f.fill(angles, ra_vals,  "r", alpha=0.08)
    ax_f.plot(angles, pl_vals,  "b--",     lw=1, label="PLL")
    ax_f.plot(angles, ip_vals,  "g:",      lw=1, label="IpDFT")
    ax_f.plot(angles, ukf_vals, "c-.",     lw=1, label="UKF")
    ax_f.plot(angles, ml_vals,  "purple",  lw=1, label="PI-GRU (ML)")

    ax_f.set_xticks(angles[:-1])
    ax_f.set_xticklabels(cats, fontsize=6)
    ax_f.tick_params(axis="x", pad=1)
    ax_f.set_yticks([])

    ax_f.set_title(
        "(f) Balance",
        fontweight="bold",
        loc="left",
        pad=-5,
        x=-0.25,
        y=1.08,
    )
    ax_f.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        fontsize=6,
        frameon=False,
    )

    save_fig(fig_d, "Fig2_Mega_Dashboard")


# ==========================================
# 4. INDIVIDUALES (FIG 2-10)
# ==========================================
def generate_individual_plots():
    (tt, ft_tr, fRA_tr, fpll_tr, fipdft_tr, fekf_tr, fsogi_tr), \
    (tr, frt, frpll, frekf_RA, _, _), \
    (tm, fmt, res), \
    (tm_mod, e_ip, e_ekf, e_pll) = get_all_data()

    # FIG 2: TRANSIENT (Nightmare)
    f, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    fr_ekf_smooth_tr = uniform_filter1d(fRA_tr, size=5)
    ax.plot(tt, fsogi_tr, "gray", ls=":", lw=1, label="SOGI")
    ax.plot(tt, fpll_tr, "b--", lw=1.2, label="SRF-PLL")
    ax.plot(tt, fr_ekf_smooth_tr, "r-", lw=1.5, label="RA-EKF")
    ax.plot(tt, ft_tr, "k", alpha=0.3)
    ax.set_xlim(680, 780)
    ax.set_ylim(55, 75)
    ax.set_ylabel("Hz")
    ax.set_xlabel("Time [ms]")
    ax.set_title("Transient (Nightmare)", fontweight="bold")
    ax.legend(fontsize=6, loc="upper right", frameon=True)
    ax.text(715, 68, "Spikes", color="blue", fontsize=7)
    plt.tight_layout()
    save_fig(f, "Fig2_Transient")

    # FIG 3: RAMP (simple: RA-EKF vs PLL)
    f, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    fr_ekf_smooth_rm = uniform_filter1d(frekf_RA, size=5)
    ax.plot(tr, frt, "k-", lw=2, alpha=0.3, label="Ref")
    ax.plot(tr, frpll, "b--", label="PLL")
    ax.plot(tr, fr_ekf_smooth_rm, "r-", label="RA-EKF")
    ax.set_xlim(280, 420)
    ax.set_ylim(59.9, 60.7)
    ax.set_title("Ramp Lag (Scenario B)", fontweight="bold")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Hz")
    ax.annotate(
        "",
        xy=(350, 60.25),
        xytext=(365, 60.25),
        arrowprops=dict(arrowstyle="<->", color="b", lw=0.8),
    )
    ax.text(350, 60.35, "Lag", color="b", fontsize=7)
    ax.legend(fontsize=6, loc="lower right", frameon=False)
    plt.tight_layout()
    save_fig(f, "Fig3_Ramp")

    # FIG 4: TOP 5
    f, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    ax.plot(tm, fmt, "k", alpha=0.3, lw=1.5)
    top = ["RA-EKF", "UKF", "IpDFT", "Koopman"]
    cols = ["r", "orange", "g", "brown"]
    stys = ["-", "--", "-.", ":"]
    for m, c, s in zip(top, cols, stys):
        lw = 1.6 if m == "RA-EKF" else 1.0
        ax.plot(tm, res[m], color=c, ls=s, lw=lw, label=m)
    ax.set_ylim(50, 65)
    ax.set_ylabel("Hz")
    ax.set_xlabel("Time [s]")
    ax.set_title("Top 5 (Stable)", fontweight="bold")
    ax.legend(fontsize=6, ncol=2, loc="lower left")
    plt.tight_layout()
    save_fig(f, "Fig4_Top5")

    # FIG 5: BOTTOM 5
    f, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    ax.plot(tm, fmt, "k", alpha=0.3, lw=1.5)
    bot = ["SRF-PLL", "SOGI", "Teager", "RLS"]
    cols = ["b", "gray", "y", "cyan"]
    for m, c in zip(bot, cols):
        lw = 1.0 if m == "SRF-PLL" else 0.8
        alpha = 0.9 if m == "SRF-PLL" else (0.7 if m == "SOGI" else 0.5)
        ax.plot(tm, res[m], color=c, lw=lw, alpha=alpha, label=m)
    ax.set_ylim(40, 80)
    ax.set_title("Bottom 5 (Unstable)", fontweight="bold")
    ax.set_xlabel("Time [s]")
    ax.legend(fontsize=6, loc="upper right")
    plt.tight_layout()
    save_fig(f, "Fig5_Bot5")

    # FIG 6: HEATMAP — data-driven (igual que dashboard pero standalone)
    f, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    scen_ids    = ["IEEE_Mag_Step", "IEEE_Freq_Ramp", "IEEE_Modulation", "IBR_Nightmare", "IBR_MultiEvent"]
    scen_labels = ["Step", "Ramp", "Mod.", "Isl", "Multi"]
    alg_labels  = ["RA-EKF", "UKF", "PLL", "SOGI", "IpDFT", "PI-GRU"]

    d, rmse = compute_compliance_matrix(alg_labels, scen_ids)
    scores = _rmse_to_score_matrix(rmse)

    im = ax.imshow(scores, cmap="RdYlGn", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(scen_labels)))
    ax.set_xticklabels(scen_labels, rotation=0, fontsize=7)
    ax.set_yticks(np.arange(len(alg_labels)))
    ax.set_yticklabels(alg_labels, fontsize=7)

    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                             fill=False, edgecolor="white",
                             linewidth=0.5, alpha=0.7)
            ax.add_patch(rect)
            txt = "P" if d[i, j] == 0 else "F"
            text_color = "white" if scores[i, j] < 0.4 else "black"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color=text_color,
                fontsize=7,
                fontweight="bold",
            )

    ax.set_title("Compliance Across All Scenarios (A–E)", fontweight="bold")
    cbar = f.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["Failure", "Marginal", "Excellent"])
    cbar.ax.tick_params(labelsize=6)

    plt.tight_layout()
    save_fig(f, "Fig6_Heatmap")

    # FIG 7: PARETO — data-driven
    f, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    pareto_methods = [
        "SOGI", "PLL", "IpDFT", "UKF",
        "RA-EKF", "PI-GRU", "Koopman",
        "Teager", "RLS",
    ]
    costs, risks = get_cost_risk_points(pareto_methods, scenario="IBR_MultiEvent")

    method_style = {
        "SOGI":   ("gray",   "o"),
        "PLL":    ("blue",   "v"),
        "IpDFT":  ("green",  "s"),
        "UKF":    ("orange", "D"),
        "RA-EKF": ("red",    "*"),
        "PI-GRU": ("purple", "X"),
        "Koopman":("brown",  "p"),
        "Teager": ("y",      "<"),
        "RLS":    ("magenta",">"),
    }

    xvals = np.array(list(costs.values()))
    yvals = np.array(list(risks.values()))

    for m in pareto_methods:
        col, marker = method_style[m]
        s = 100 if m == "RA-EKF" else (60 if m == "PI-GRU" else 40)
        ax.scatter(
            costs[m],
            risks[m],
            c=col,
            marker=marker,
            s=s,
            edgecolors="k",
            lw=0.4,
            zorder=5,
        )
        ax.annotate(
            m,
            (costs[m], risks[m]),
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=6,
            ha="center",
            color="black",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.8 * xvals.min(), 1.2 * xvals.max())
    ax.set_ylim(0.8 * yvals.min(), 1.2 * yvals.max())
    ax.grid(True, which="both", alpha=0.4, ls=":")
    ax.set_xlabel(r"Cost [$\mu$s]")
    ax.set_ylabel("Risk [s]")
    ax.set_title("Cost vs. Risk", fontweight="bold")
    plt.tight_layout()
    save_fig(f, "Fig7_Pareto")

    # FIG 8: RADAR – DATA-DRIVEN + UKF + PI-GRU
    f, ax = plt.subplots(figsize=(3.5, 3.5), subplot_kw=dict(polar=True))
    cats = ["Steady", "Trans.", "Noise", "Eff.", "Safe"]
    N = 5
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    labels_radar = ["RA-EKF", "PLL", "IpDFT", "EKF", "UKF", "PI-GRU"]
    profiles = compute_radar_profiles(labels_radar)

    def _closed_profile(label):
        vals = profiles[label]
        return vals + vals[:1]

    ra  = _closed_profile("RA-EKF")
    pl  = _closed_profile("PLL")
    ip  = _closed_profile("IpDFT")
    ekf = _closed_profile("EKF")
    ukf = _closed_profile("UKF")
    ml  = _closed_profile("PI-GRU")

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(angles, ra,  "r-",      lw=2, label="RA-EKF")
    ax.fill(angles, ra,  "r", alpha=0.1)
    ax.plot(angles, pl,  "b--",     lw=1, label="PLL")
    ax.plot(angles, ip,  "g:",      lw=1, label="IpDFT")
    ax.plot(angles, ekf, "orange",  ls="-.", lw=1, label="EKF")
    ax.plot(angles, ukf, "c-.",     lw=1, label="UKF")
    ax.plot(angles, ml,  "purple",  lw=1, label="PI-GRU (ML)")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=7)
    ax.set_yticks([5, 10])
    ax.set_yticklabels([])
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
        fontsize=6,
        frameon=False,
    )
    ax.set_title("Balance Profile", fontweight="bold", pad=10)
    plt.tight_layout()
    save_fig(f, "Fig8_Radar")

    # FIG 9: HEATMAP (extra) — misma lógica que Fig6
    f, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    scen_ids    = ["IEEE_Mag_Step", "IEEE_Freq_Ramp", "IEEE_Modulation", "IBR_Nightmare", "IBR_MultiEvent"]
    scen_labels = ["Step", "Ramp", "Mod.", "Isl", "Multi"]
    alg_labels  = ["RA-EKF", "UKF", "PLL", "SOGI", "IpDFT", "PI-GRU"]

    d2, rmse2 = compute_compliance_matrix(alg_labels, scen_ids)
    scores2 = _rmse_to_score_matrix(rmse2)

    im2 = ax.imshow(scores2, cmap="RdYlGn", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(scen_labels)))
    ax.set_xticklabels(scen_labels, rotation=0, fontsize=7)
    ax.set_yticks(np.arange(len(alg_labels)))
    ax.set_yticklabels(alg_labels, fontsize=7)
    for i in range(d2.shape[0]):
        for j in range(d2.shape[1]):
            rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                             fill=False, edgecolor="white",
                             linewidth=0.5, alpha=0.7)
            ax.add_patch(rect)
            ax.text(
                j,
                i,
                "P" if d2[i, j] == 0 else "F",
                ha="center",
                va="center",
                color="white" if scores2[i, j] < 0.4 else "black",
                fontsize=7,
                fontweight="bold",
            )
    ax.set_title("Compliance Summary", fontweight="bold")
    cbar2 = f.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cbar2.set_ticks([0.0, 0.5, 1.0])
    cbar2.set_ticklabels(["Failure", "Marginal", "Excellent"])
    cbar2.ax.tick_params(labelsize=6)

    plt.tight_layout()
    save_fig(f, "Fig9_Heatmap")

    # FIG 10: MODULATION ERROR
    f, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    ax.plot(tm_mod, e_pll, "b--", lw=0.8, label="PLL")
    ax.plot(tm_mod, e_ekf, "r-", lw=1.0, label="RA-EKF")
    ax.plot(tm_mod, e_ip, "g-.", lw=1.2, label="IpDFT")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.0)
    ax.set_title("Steady-State Error (Scenario C)", fontweight="bold")
    ax.set_ylabel(r"Log |Error| [Hz]", labelpad=1)
    ax.set_xlabel("Time [s]", labelpad=1)
    ax.legend(fontsize=6, loc="upper right", ncol=3, frameon=False)
    plt.tight_layout()
    save_fig(f, "Fig10_Modulation")


if __name__ == "__main__":
    fig1_scenarios()
    generate_dashboard()
    generate_individual_plots()
    print("¡Generación Completa V6: Dashboard + 10 Individuales (todo data-driven desde JSON)!")
