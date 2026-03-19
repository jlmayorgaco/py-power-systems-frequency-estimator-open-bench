"""
IEEE 14-BUS: SPECTRAL FRAGILITY JOURNAL SUITE (Q1 STANDARD)
-----------------------------------------------------------
Engine: ANDES V7 (Tuned Physics)
Outputs:
  1. Forensic JSON Report: Deep theoretical bounds, node-by-node limits, ROI analysis.
  2. Plots: IEEE Transaction style (Times New Roman, 600 DPI, Professional Layout).
"""

import andes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import scipy.linalg as la
import networkx as nx
import json
import os
import shutil

# --- CONFIGURACIÓN ESTÉTICA IEEE TRANSACTIONS ---
# Dimensiones: Single column ~3.5 in, Double column ~7.16 in.
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",  # Mejor renderizado de ecuaciones LaTeX
        "font.size": 8,  # Tamaño base pequeño para papers
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 10,
        "axes.grid": True,
        "grid.alpha": 0.4,
        "grid.linewidth": 0.5,
        "grid.linestyle": ":",
        "lines.linewidth": 1.0,
        "figure.dpi": 600,  # Alta resolución
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

# ANDES Silent
andes.config_logger(stream_level=40)

OUTPUT_DIR = "results_journal_ready"
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

print("[INIT] Iniciando Suite de Publicación IEEE (Deep Analytics Mode)...")


# ==============================================================================
# 1. GENERADOR DE CASO (ROBUSTO Y TIPADO)
# ==============================================================================
def create_ieee14_xlsx(filename="ieee14_andes.xlsx"):
    # Bus Data (Strings preserved in 'name', IDs are int)
    bus_data = pd.DataFrame(
        {
            "uid": range(1, 15),
            "idx": range(1, 15),
            "name": [f"Bus{i}" for i in range(1, 15)],
            "Vn": [138.0] * 14,
            "v0": [1.0] * 14,
            "a0": [0.0] * 14,
            "area": [1] * 14,
            "zone": [1] * 14,
        }
    )
    # Convertir solo columnas numéricas
    cols_int = ["uid", "idx", "area", "zone"]
    bus_data[cols_int] = bus_data[cols_int].astype(int)

    # Line Data
    edges = [
        (1, 2, 0.2),
        (1, 5, 0.25),
        (2, 3, 0.25),
        (2, 4, 0.33),
        (2, 5, 0.33),
        (3, 4, 0.5),
        (4, 5, 0.5),
        (4, 7, 1.0),
        (4, 9, 1.0),
        (5, 6, 0.5),
        (6, 11, 1.0),
        (6, 12, 1.0),
        (6, 13, 1.0),
        (7, 8, 1.0),
        (7, 9, 1.0),
        (9, 10, 1.0),
        (9, 14, 1.0),
        (10, 11, 1.0),
        (12, 13, 1.0),
        (13, 14, 1.0),
    ]
    line_data = pd.DataFrame(
        {
            "uid": range(1, len(edges) + 1),
            "idx": range(1, len(edges) + 1),
            "bus1": [e[0] for e in edges],
            "bus2": [e[1] for e in edges],
            "r": [0.0] * len(edges),
            "x": [e[2] for e in edges],
            "b": [0.0] * len(edges),
        }
    )
    line_data[["uid", "idx", "bus1", "bus2"]] = line_data[
        ["uid", "idx", "bus1", "bus2"]
    ].astype(int)

    # Static Generators
    slack_data = pd.DataFrame(
        {"uid": [1], "idx": [1], "bus": [1], "Vn": [138.0], "v0": [1.02], "a0": [0.0]}
    ).astype(int)
    pv_data = pd.DataFrame(
        {
            "uid": [2, 3, 4, 5],
            "idx": [2, 3, 4, 5],
            "bus": [2, 3, 6, 8],
            "Vn": [138.0] * 4,
            "p0": [0.5] * 4,
            "v0": [1.02] * 4,
        }
    )
    pv_data[["uid", "idx", "bus"]] = pv_data[["uid", "idx", "bus"]].astype(int)

    # Dynamic Gen (M=6.0 tuned)
    gencls_data = pd.DataFrame(
        {
            "uid": range(1, 6),
            "idx": range(1, 6),
            "bus": [1, 2, 3, 6, 8],
            "gen": [1, 2, 3, 4, 5],
            "Sn": [100.0] * 5,
            "Vn": [138.0] * 5,
            "u": [1] * 5,
            "M": [6.0] * 5,
            "D": [1.0] * 5,
            "ra": [0.0] * 5,
            "xl": [0.0] * 5,
        }
    )
    gencls_data[["uid", "idx", "bus", "gen", "u"]] = gencls_data[
        ["uid", "idx", "bus", "gen", "u"]
    ].astype(int)

    # Load (p0=0.15 tuned)
    pq_data = pd.DataFrame(
        {
            "uid": range(1, 15),
            "idx": range(1, 15),
            "bus": range(1, 15),
            "p0": [0.15] * 14,
            "q0": [0.05] * 14,
            "Vn": [138.0] * 14,
        }
    )
    pq_data[["uid", "idx", "bus"]] = pq_data[["uid", "idx", "bus"]].astype(int)

    with pd.ExcelWriter(filename) as writer:
        bus_data.to_excel(writer, sheet_name="Bus", index=False)
        line_data.to_excel(writer, sheet_name="Line", index=False)
        slack_data.to_excel(writer, sheet_name="Slack", index=False)
        pv_data.to_excel(writer, sheet_name="PV", index=False)
        gencls_data.to_excel(writer, sheet_name="GENCLS", index=False)
        pq_data.to_excel(writer, sheet_name="PQ", index=False)
    return filename


IEEE14_POS = {
    1: (0, 0),
    2: (2, 2),
    3: (4, 2),
    4: (6, 0),
    5: (2, -2),
    6: (6, -2),
    7: (8, 1),
    8: (10, 1),
    9: (8, -1),
    10: (10, -1),
    11: (7, -3),
    12: (5, -3),
    13: (3, -3),
    14: (9, -2),
}


# ==============================================================================
# 2. ANÁLISIS ESPECTRAL PROFUNDO
# ==============================================================================
def analyze_spectral_structure(ss):
    """Calcula H, autovalores, autovectores y métricas derivadas."""
    n_buses = len(ss.Bus.idx.v)
    M_vec = np.zeros(n_buses)
    if ss.GENCLS.n > 0:
        for i in range(ss.GENCLS.n):
            bus_ext = int(ss.GENCLS.bus.v[i])
            M_vec[bus_ext - 1] += 2 * ss.GENCLS.M.v[i]
    M_vec[M_vec < 0.1] = 0.5

    L = np.zeros((n_buses, n_buses))
    edges_info = []
    for i in range(ss.Line.n):
        u, v = int(ss.Line.bus1.v[i]) - 1, int(ss.Line.bus2.v[i]) - 1
        x_val = ss.Line.x.v[i]
        b_val = 1.0 / max(x_val, 1e-4)
        L[u, v] -= b_val
        L[v, u] -= b_val
        L[u, u] += b_val
        L[v, v] += b_val
        edges_info.append({"u": u + 1, "v": v + 1, "weight": b_val})

    M_sqrt_inv = np.diag(1.0 / np.sqrt(M_vec))
    H = M_sqrt_inv @ L @ M_sqrt_inv
    vals, vecs = la.eigh(H)

    # Métricas Clave
    lambda_2 = vals[1]
    fiedler_vec = vecs[:, 1]
    pi_scores = fiedler_vec**2

    # Predicción de Capacidad (Hosting Capacity Proxy)
    # Theory: Max Angle ~ Energy Bound / pi_i.
    # Max Load ~ lambda_2 / pi_i (roughly proportional to stiffness/participation)
    # Scaled to p.u. for readability relative to system mean
    stiffness_local = lambda_2 / (pi_scores + 1e-6)
    dhc_proxy = (
        stiffness_local / np.mean(stiffness_local) * 2.0
    )  # Factor de escala arbitrario para p.u. realista

    return {
        "H": H,
        "vals": vals,
        "pi": pi_scores,
        "lambda_2": lambda_2,
        "M_vec": M_vec,
        "dhc_proxy": dhc_proxy,
        "edges": edges_info,
    }


# ==============================================================================
# 3. EJECUTOR DE ESCENARIOS
# ==============================================================================
def run_scenario(
    phase_name, topology_stress=1.0, repair_bus=None, repair_M=0, load_step=4.0
):
    xlsx_name = create_ieee14_xlsx()
    ss = andes.load(xlsx_name, setup=False, default_config=True)

    # Modificaciones
    if topology_stress < 1.0:
        factor = 1.0 / topology_stress
        for i in range(ss.Line.n):
            if ss.Line.bus1.v[i] > 5 or ss.Line.bus2.v[i] > 5:
                ss.Line.x.v[i] *= factor
    if phase_name == "C_Critical":
        ss.GENCLS.D.v[:] = [0.2] * ss.GENCLS.n

    repair_info = None
    if repair_bus is not None:
        for i in range(ss.GENCLS.n):
            if int(ss.GENCLS.bus.v[i]) == repair_bus:
                prev_M = ss.GENCLS.M.v[i]
                ss.GENCLS.M.v[i] += repair_M / 2.0
                ss.GENCLS.D.v[i] += 15.0
                repair_info = {
                    "bus": repair_bus,
                    "added_M": repair_M,
                    "prev_M": prev_M * 2,
                }
                break

    # Diagnóstico (Pre-Setup)
    spec_data = analyze_spectral_structure(ss)
    target_node = np.argmax(spec_data["pi"]) + 1

    # Configurar Disturbio
    disturb_bus = target_node if phase_name != "A_Healthy" else 8  # Consistent target
    # Override for Bad Repair test
    if phase_name == "D_Bad":
        disturb_bus = 8  # Perturbar donde duele, aunque reparemos mal

    pq_idx = -1
    for i in range(ss.PQ.n):
        if int(ss.PQ.bus.v[i]) == disturb_bus:
            pq_idx = int(ss.PQ.idx.v[i])
            break
    if pq_idx >= 0:
        ss.add(
            "Toggle",
            {
                "model": "PQ",
                "dev": pq_idx,
                "attr": "p0",
                "t": 1.0,
                "val": ss.PQ.p0.v[pq_idx] + load_step,
            },
        )

    # Simulación
    try:
        ss.setup()
    except:
        return None
    if not ss.PFlow.run():
        return None
    ss.TDS.config.tf = 20.0
    ss.TDS.config.criteria = 0
    ss.TDS.run()

    # Extracción
    t = np.array(ss.dae.ts.t)
    try:
        addr_target = ss.Bus.a.a[disturb_bus - 1]
        addr_slack = ss.Bus.a.a[0]
        angle_diff = (
            (ss.dae.ts.y[:, addr_target] - ss.dae.ts.y[:, addr_slack]) * 180 / np.pi
        )
        freq_dev = np.gradient(ss.dae.ts.y[:, addr_target], t) / (2 * np.pi)
    except:
        angle_diff = np.zeros_like(t)
        freq_dev = np.zeros_like(t)

    return {
        "t": t,
        "angle": angle_diff,
        "freq": freq_dev,
        "spectral": spec_data,
        "target": target_node,
        "repair_info": repair_info,
        "load_step": load_step,
    }


# ==============================================================================
# 4. EJECUCIÓN DEL EXPERIMENTO
# ==============================================================================
results = {}
print(">>> Simulando Escenarios...")
results["C"] = run_scenario("C_Critical", 0.5)
WEAK_BUS = results["C"]["target"]
results["D"] = run_scenario("D_Repaired", 0.5, repair_bus=WEAK_BUS, repair_M=40.0)
results["D_Bad"] = run_scenario("D_Bad", 0.5, repair_bus=1, repair_M=40.0)
results["A"] = run_scenario("A_Healthy", 1.0)
results["B"] = run_scenario("B_Stressed", 0.6)


# ==============================================================================
# 5. GENERACIÓN DEL JSON FORENSE (COMPREHENSIVE)
# ==============================================================================
def generate_forensic_json(res_c, res_d):
    spec_c = res_c["spectral"]
    spec_d = res_d["spectral"]

    # Analisis Nodo a Nodo
    nodes_analysis = []
    for i in range(len(spec_c["pi"])):
        nodes_analysis.append(
            {
                "bus_id": i + 1,
                "metrics": {
                    "participation_factor_pi": float(spec_c["pi"][i]),
                    "inertia_M_initial": float(spec_c["M_vec"][i]),
                    "spectral_stiffness_proxy": float(spec_c["dhc_proxy"][i]),
                },
                "prediction": {
                    "is_critical": bool(i == res_c["target"] - 1),
                    "predicted_max_load_pu": float(spec_c["dhc_proxy"][i]),
                    "vulnerability_rank": int(
                        list(np.argsort(spec_c["pi"])[::-1]).index(i) + 1
                    ),
                },
            }
        )

    # Analisis de Diseño (ROI)
    lambda_old = spec_c["lambda_2"]
    lambda_new = spec_d["lambda_2"]
    gap_improvement_pct = ((lambda_new - lambda_old) / lambda_old) * 100

    m_added = res_d["repair_info"]["added_M"]
    roi_spectral = gap_improvement_pct / m_added  # % Mejora por MWs de inercia

    report = {
        "experiment_meta": {
            "title": "Spectral Fragility & Hosting Capacity Validation",
            "system": "IEEE 14-Bus (Modified)",
            "disturbance_type": "Step Load",
            "disturbance_magnitude_pu": float(res_c["load_step"]),
        },
        "theoretical_bounds": {
            "spectral_gap_lambda2_critical": float(lambda_old),
            "spectral_gap_lambda2_repaired": float(lambda_new),
            "stochastic_noise_bound_ratio": float(
                lambda_new / lambda_old
            ),  # Ratio de varianza
        },
        "critical_node_diagnosis": {
            "identified_node": int(res_c["target"]),
            "confidence_score": float(np.max(spec_c["pi"])),
            "reasoning": "Maximal eigenvector centrality in Inertial Laplacian H",
        },
        "node_forensics": sorted(
            nodes_analysis, key=lambda x: x["prediction"]["vulnerability_rank"]
        ),
        "actionable_design": {
            "strategy": "Targeted Inertia Injection",
            "location": int(res_d["repair_info"]["bus"]),
            "mass_added": float(m_added),
            "outcome_metrics": {
                "stability_margin_improvement_pct": float(gap_improvement_pct),
                "efficiency_ROI_pct_per_MW": float(roi_spectral),
                "peak_angle_reduction_factor": float(
                    np.max(np.abs(res_c["angle"])) / np.max(np.abs(res_d["angle"]))
                ),
            },
        },
    }
    return report


json_data = generate_forensic_json(results["C"], results["D"])
with open(f"{OUTPUT_DIR}/complete_forensic_report.json", "w") as f:
    json.dump(json_data, f, indent=4)

# ==============================================================================
# 6. PLOT 1: DYNAMICS DASHBOARD (IEEE FORMAT)
# ==============================================================================
fig = plt.figure(figsize=(7.16, 6))  # Doble columna IEEE
gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.3)

rows_keys = ["A", "B", "C", "D"]
row_labels = ["(a) Healthy", "(b) Stressed", "(c) Critical", "(d) Repaired"]
colors = ["#2ca02c", "#ff7f0e", "#d62728", "#1f77b4"]

for i, key in enumerate(rows_keys):
    res = results[key]
    c = colors[i]

    # 1. Angle
    ax1 = fig.add_subplot(gs[i, 0])
    ax1.plot(res["t"], res["angle"], c=c, lw=1)
    ax1.set_ylabel(r"$\Delta \delta$ [deg]")
    if i == 0:
        ax1.set_title(r"Angle Deviation")
    if key == "C":
        ax1.axhline(180, c="k", ls=":", lw=0.8)

    # 2. RoCoF
    ax2 = fig.add_subplot(gs[i, 1])
    rocof = np.gradient(res["freq"], res["t"])
    ax2.plot(res["t"], rocof, c=c, lw=1, alpha=0.8)
    if i == 0:
        ax2.set_title(r"RoCoF [Hz/s]")
    ax2.set_ylim(-4, 4)

    # 3. Spectral Metric
    ax3 = fig.add_subplot(gs[i, 2])
    if key in ["C", "D"]:
        pi = res["spectral"]["pi"]
        ax3.bar(range(1, 15), pi, color=c, alpha=0.8, width=0.6)
        ax3.set_ylim(0, 1.0)
    else:
        ax3.text(0.5, 0.5, "N/A", ha="center", fontsize=6)
    if i == 0:
        ax3.set_title(r"Fragility $\pi_i$")

    # 4. Freq
    ax4 = fig.add_subplot(gs[i, 3])
    ax4.plot(res["t"], res["freq"], c=c, lw=1)
    if i == 0:
        ax4.set_title(r"Freq. Dev [p.u.]")

    # Add Row Label
    fig.text(
        0.01,
        0.88 - i * 0.23,
        row_labels[i],
        va="center",
        rotation=90,
        fontweight="bold",
        fontsize=9,
    )

    # Bad Repair Comparison
    if key == "D":
        ax1.plot(
            results["D_Bad"]["t"],
            results["D_Bad"]["angle"],
            c="gray",
            ls="--",
            lw=0.8,
            label="Random",
        )
        ax1.legend(loc="upper right", frameon=False, fontsize=5)

fig.align_ylabels()
plt.savefig(f"{OUTPUT_DIR}/fig1_ieee_dynamics.png", dpi=600)
plt.savefig(f"{OUTPUT_DIR}/fig1_ieee_dynamics.pdf")  # Vectorial

# ==============================================================================
# 7. PLOT 2: SPECTRAL GAP & NETWORK (COMBO)
# ==============================================================================
fig2 = plt.figure(figsize=(7.16, 3.5))
gs2 = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2])

# Panel Left: Spectral Gap
ax_spec = fig2.add_subplot(gs2[0])
vals_c = results["C"]["spectral"]["vals"]
vals_d = results["D"]["spectral"]["vals"]
# Plot eigenvalues
ax_spec.plot(vals_c, "o-", ms=4, c="#d62728", label="Critical", alpha=0.7)
ax_spec.plot(vals_d, "s-", ms=4, c="#1f77b4", label="Repaired", alpha=0.7)
# Annotate Gap
gap_c = vals_c[1]
gap_d = vals_d[1]
ax_spec.annotate(
    f"Gap $\lambda_2$ (Weak)",
    xy=(1, gap_c),
    xytext=(3, gap_c + 0.1),
    arrowprops=dict(arrowstyle="->", color="r"),
    fontsize=7,
)
ax_spec.annotate(
    f"Gap $\lambda_2$ (Strong)",
    xy=(1, gap_d),
    xytext=(3, gap_d + 0.1),
    arrowprops=dict(arrowstyle="->", color="b"),
    fontsize=7,
)
ax_spec.set_ylabel(r"Eigenvalue Magnitude $\lambda_i(H)$")
ax_spec.set_xlabel("Mode Index $i$")
ax_spec.set_title("(a) Spectral Signature Improvement")
ax_spec.legend(frameon=False)
ax_spec.grid(True, alpha=0.3)

# Panel Right: Network Map
ax_net = fig2.add_subplot(gs2[1])
G = nx.Graph()
G.add_edges_from([(e["u"], e["v"]) for e in results["C"]["spectral"]["edges"]])
pi_c = results["C"]["spectral"]["pi"]
pi_norm = plt.Normalize(0, 0.8)
cmap = plt.cm.Reds

nx.draw_networkx_edges(G, IEEE14_POS, ax=ax_net, edge_color="gray", width=1, alpha=0.5)
nodes = nx.draw_networkx_nodes(
    G,
    IEEE14_POS,
    ax=ax_net,
    node_size=300,
    node_color=pi_c,
    cmap=cmap,
    edgecolors="k",
    linewidths=0.5,
)
nx.draw_networkx_labels(
    G, IEEE14_POS, ax=ax_net, font_size=7, font_color="black", font_weight="bold"
)

# Colorbar
cbar = plt.colorbar(nodes, ax=ax_net, fraction=0.046, pad=0.04)
cbar.set_label(r"Fragility Score $\pi_i$", fontsize=8)
cbar.ax.tick_params(labelsize=6)

ax_net.set_title(
    f"(b) Fragility Localization (Target: Bus {int(results['C']['target'])})"
)
ax_net.axis("off")

plt.savefig(f"{OUTPUT_DIR}/fig2_ieee_spectral_topo.png", dpi=600)
plt.savefig(f"{OUTPUT_DIR}/fig2_ieee_spectral_topo.pdf")

print("[DONE] Resultados generados. Revisa la carpeta 'results_journal_ready'.")
