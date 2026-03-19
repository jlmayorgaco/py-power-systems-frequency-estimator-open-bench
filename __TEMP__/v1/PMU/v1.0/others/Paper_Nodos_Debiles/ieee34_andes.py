"""
IEEE 34-NODE: ADVANCED SPECTRAL SUITE (V2.0)
--------------------------------------------
Objective: Scalability test with deep forensic reporting.
System: IEEE 34-Node Test Feeder (Balanced Equivalent).
Outputs:
  - Comprehensive JSON (Nodes, Eigenvalues, Dynamics).
  - High-Res Plots (Duel, Metrics, Heatmaps).
"""

import andes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import networkx as nx
import json
import os
import shutil

# --- CONFIGURACIÓN ESTÉTICA IEEE ---
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
    }
)

andes.config_logger(stream_level=40)
OUTPUT_DIR = "results_ieee34_suite_v2"
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

print("[INIT] Generando Suite IEEE 34 V2...")

# ==============================================================================
# 1. TOPOLOGÍA IEEE 34
# ==============================================================================
# Mapeo de IDs originales a internos
ORIGINAL_IDS = [
    800,
    802,
    806,
    808,
    810,
    812,
    814,
    816,
    818,
    820,
    822,
    824,
    826,
    828,
    830,
    832,
    834,
    836,
    838,
    840,
    842,
    844,
    846,
    848,
    850,
    852,
    854,
    856,
    858,
    860,
    862,
    864,
    888,
    890,
]
NODE_MAP = {oid: i + 1 for i, oid in enumerate(ORIGINAL_IDS)}
REV_MAP = {v: k for k, v in NODE_MAP.items()}
N_BUSES = len(NODE_MAP)


def create_ieee34_xlsx(filename="ieee34_andes.xlsx"):
    bus_data = pd.DataFrame(
        {
            "uid": list(NODE_MAP.values()),
            "idx": list(NODE_MAP.values()),
            "name": [str(k) for k in NODE_MAP.keys()],
            "Vn": [138.0] * N_BUSES,
            "v0": [1.0] * N_BUSES,
            "a0": [0.0] * N_BUSES,
            "area": [1] * N_BUSES,
            "zone": [1] * N_BUSES,
        }
    ).astype(int)

    # Conectividad
    connections = [
        (800, 802, 0.05),
        (802, 806, 0.05),
        (806, 808, 0.08),
        (808, 810, 0.05),
        (808, 812, 0.08),
        (812, 814, 0.08),
        (814, 850, 0.05),
        (850, 816, 0.05),
        (816, 818, 0.05),
        (816, 824, 0.08),
        (818, 820, 0.15),
        (820, 822, 0.15),
        (824, 826, 0.05),
        (824, 828, 0.05),
        (828, 830, 0.15),
        (830, 854, 0.05),
        (854, 856, 0.05),
        (854, 852, 0.08),
        (852, 832, 0.08),
        (832, 858, 0.05),
        (832, 888, 0.08),
        (888, 890, 0.15),
        (858, 864, 0.05),
        (858, 834, 0.08),
        (834, 860, 0.05),
        (834, 842, 0.05),
        (860, 836, 0.08),
        (836, 840, 0.05),
        (836, 862, 0.05),
        (862, 838, 0.05),
        (842, 844, 0.05),
        (844, 846, 0.05),
        (846, 848, 0.05),
    ]

    line_rows = []
    for i, (u, v, x) in enumerate(connections):
        line_rows.append(
            {
                "uid": i + 1,
                "idx": i + 1,
                "u": NODE_MAP[u],
                "v": NODE_MAP[v],
                "bus1": NODE_MAP[u],
                "bus2": NODE_MAP[v],
                "r": x * 0.1,
                "x": x,
                "b": 0.001,
            }
        )
    line_data = pd.DataFrame(line_rows)

    slack_data = pd.DataFrame(
        {
            "uid": [1],
            "idx": [1],
            "bus": [NODE_MAP[800]],
            "Vn": [138.0],
            "v0": [1.05],
            "a0": [0.0],
        }
    )

    gen_rows = []
    for i in range(1, N_BUSES + 1):
        is_slack = i == NODE_MAP[800]
        gen_rows.append(
            {
                "uid": i,
                "idx": i,
                "bus": i,
                "gen": i,
                "Sn": 100.0,
                "Vn": 138.0,
                "u": 1,
                "M": 100.0 if is_slack else 2.0,
                "D": 5.0 if is_slack else 0.1,
                "ra": 0.0,
                "xl": 0.0,
            }
        )
    gencls_data = pd.DataFrame(gen_rows)

    pv_rows = []
    for i in range(1, N_BUSES + 1):
        if i == NODE_MAP[800]:
            continue
        pv_rows.append(
            {
                "uid": i,
                "idx": i,
                "bus": i,
                "Vn": 138.0,
                "p0": 0.0,
                "v0": 1.02,
                "qmax": 9999,
                "qmin": -9999,
            }
        )
    pv_data = pd.DataFrame(pv_rows)

    pq_rows = []
    for i in range(1, N_BUSES + 1):
        pq_rows.append(
            {"uid": i, "idx": i, "bus": i, "p0": 0.02, "q0": 0.005, "Vn": 138.0}
        )
    pq_data = pd.DataFrame(pq_rows)

    with pd.ExcelWriter(filename) as writer:
        bus_data.to_excel(writer, "Bus", index=False)
        line_data.to_excel(writer, "Line", index=False)
        slack_data.to_excel(writer, "Slack", index=False)
        pv_data.to_excel(writer, "PV", index=False)
        gencls_data.to_excel(writer, "GENCLS", index=False)
        pq_data.to_excel(writer, "PQ", index=False)
    return filename


# ==============================================================================
# 2. MOTOR DE ANÁLISIS
# ==============================================================================
def analyze_system(ss):
    # 1. PVR Calculation (Mockup if dynamic run)
    try:
        v = ss.dae.y[ss.Bus.v.a]
        pvr = v / 1.05
        pvr[NODE_MAP[800] - 1] = 999
        weak_pvr = np.argmin(pvr) + 1
    except:
        pvr = np.zeros(N_BUSES)
        weak_pvr = 1

    # 2. Spectral Analysis
    M = np.zeros(N_BUSES)
    for i in range(ss.GENCLS.n):
        idx = int(ss.GENCLS.bus.v[i]) - 1
        M[idx] += 2 * ss.GENCLS.M.v[i]

    L = np.zeros((N_BUSES, N_BUSES))
    edges = []
    for i in range(ss.Line.n):
        if hasattr(ss.Line, "bus1"):
            u, v = int(ss.Line.bus1.v[i]) - 1, int(ss.Line.bus2.v[i]) - 1
        else:
            u, v = int(ss.Line.u.v[i]) - 1, int(ss.Line.v.v[i]) - 1
        b = 1.0 / ss.Line.x.v[i]
        L[u, v] -= b
        L[v, u] -= b
        L[u, u] += b
        L[v, v] += b
        edges.append((u, v))

    vals, vecs = la.eigh(np.diag(1 / np.sqrt(M)) @ L @ np.diag(1 / np.sqrt(M)))
    pi = vecs[:, 1] ** 2
    pi[NODE_MAP[800] - 1] = -1  # Ignore Slack
    weak_spec = np.argmax(pi) + 1

    return {
        "weak_pvr": weak_pvr,
        "weak_spec": weak_spec,
        "pvr_vals": pvr,
        "pi_vals": pi,
        "vals": vals,
        "edges": edges,
        "M": M,
        "volatility": np.sum(1.0 / vals[1:]),
    }


# Baseline Analysis
create_ieee34_xlsx()
ss_base = andes.load("ieee34_andes.xlsx", setup=False, default_config=True)
ss_base.setup()
ss_base.PFlow.run()
METRICS_BASE = analyze_system(ss_base)
WEAK_PVR = METRICS_BASE["weak_pvr"]
WEAK_SPEC = METRICS_BASE["weak_spec"]
print(f"Targets IEEE 34: Voltage={REV_MAP[WEAK_PVR]}, Spectral={REV_MAP[WEAK_SPEC]}")


# ==============================================================================
# 3. EXPERIMENTO DINÁMICO
# ==============================================================================
def run_dynamic(mode):
    xlsx = create_ieee34_xlsx()
    ss = andes.load(xlsx, setup=False, default_config=True)

    # REPAIR
    target = None
    if mode == "Fix_Voltage":
        target = WEAK_PVR
    elif mode == "Fix_Spectral":
        target = WEAK_SPEC

    if target:
        for i in range(ss.GENCLS.n):
            if int(ss.GENCLS.bus.v[i]) == target:
                ss.GENCLS.M.v[i] += 200.0
                ss.GENCLS.D.v[i] += 50.0
                break

    # ANALYZE STATE (For JSON)
    spec_state = analyze_system(ss)

    # DISTURB
    dist_node = WEAK_SPEC
    pq_uid, pq_idx = -1, -1
    for i in range(ss.PQ.n):
        if int(ss.PQ.bus.v[i]) == dist_node:
            pq_uid = int(ss.PQ.idx.v[i])
            pq_idx = i
            break

    if pq_uid != -1:
        base = ss.PQ.p0.v[pq_idx]
        ss.add(
            "Toggle",
            {"model": "PQ", "dev": pq_uid, "attr": "p0", "t": 1.0, "val": base + 5.0},
        )

    ss.setup()
    if not ss.PFlow.run():
        return None
    ss.TDS.config.tf = 15.0
    ss.TDS.run()

    t = np.array(ss.dae.ts.t)
    angle = (
        (
            ss.dae.ts.y[:, ss.Bus.a.a[dist_node - 1]]
            - ss.dae.ts.y[:, ss.Bus.a.a[NODE_MAP[800] - 1]]
        )
        * 180
        / np.pi
    )

    return {
        "t": t,
        "angle": angle,
        "max_angle": np.max(np.abs(angle)),
        "spec": spec_state,
    }


print(">>> Simulando Dinámica IEEE 34...")
res_crit = run_dynamic("Critical")
res_spec = run_dynamic("Fix_Spectral")

# ==============================================================================
# 4. JSON FORENSE COMPLETO
# ==============================================================================
json_full = {
    "meta": {
        "experiment": "IEEE 34 Scalability Test",
        "description": "Validation of Spectral Fragility on Larger Topology",
    },
    "metrics_summary": {
        "voltage_target": int(REV_MAP[WEAK_PVR]),
        "spectral_target": int(REV_MAP[WEAK_SPEC]),
        "spectral_gap_improvement_pct": float(
            (res_crit["spec"]["vals"][1] - res_spec["spec"]["vals"][1])
            / res_crit["spec"]["vals"][1]
            * 100
        ),
    },
    "dynamics": {
        "max_angle_critical": float(res_crit["max_angle"]),
        "max_angle_repaired": float(res_spec["max_angle"]),
        "improvement_pct": float(
            (res_crit["max_angle"] - res_spec["max_angle"])
            / res_crit["max_angle"]
            * 100
        ),
    },
    "spectral_forensics": {
        "eigenvalues_base": [
            float(x) for x in res_crit["spec"]["vals"][:15]
        ],  # First 15 modes
        "eigenvalues_repaired": [float(x) for x in res_spec["spec"]["vals"][:15]],
        "volatility_base": float(res_crit["spec"]["volatility"]),
        "volatility_repaired": float(res_spec["spec"]["volatility"]),
    },
    "nodal_data": [],
}

# Populate Nodal Data
for i in range(N_BUSES):
    nid = i + 1
    real_id = int(REV_MAP[nid])
    json_full["nodal_data"].append(
        {
            "node_id": nid,
            "name": str(real_id),
            "pvr_score": float(METRICS_BASE["pvr_vals"][i]),
            "pi_score": float(METRICS_BASE["pi_vals"][i]),
            "inertia_base": float(METRICS_BASE["M"][i]),
            "is_weak_voltage": bool(nid == WEAK_PVR),
            "is_weak_spectral": bool(nid == WEAK_SPEC),
        }
    )

with open(f"{OUTPUT_DIR}/ieee34_advanced_report.json", "w") as f:
    json.dump(json_full, f, indent=4)

# ==============================================================================
# 5. PLOTS
# ==============================================================================
# FIG 1: DUEL
plt.figure(figsize=(8, 4))
plt.plot(res_crit["t"], res_crit["angle"], "r-", label="Critical")
plt.plot(res_spec["t"], res_spec["angle"], "b-", label="Repaired (Spectral)")
plt.title(f"IEEE 34 Stability Response")
plt.xlabel("Time [s]")
plt.ylabel("Angle [deg]")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f"{OUTPUT_DIR}/fig1_duel.png")

# FIG 2: METRICS
fig, ax = plt.subplots(figsize=(10, 4))
idx = range(N_BUSES)
pvr_n = 1.1 - METRICS_BASE["pvr_vals"]
pvr_n = (pvr_n - pvr_n.min()) / (pvr_n.max() - pvr_n.min())
pi_n = METRICS_BASE["pi_vals"]
pi_n = (pi_n - pi_n.min()) / (pi_n.max() - pi_n.min())
ax.bar([i - 0.2 for i in idx], pvr_n, 0.4, color="gray", label="Voltage")
ax.bar([i + 0.2 for i in idx], pi_n, 0.4, color="red", label="Spectral")
ax.set_xticks(idx)
ax.set_xticklabels([str(REV_MAP[i + 1]) for i in idx], rotation=90, fontsize=6)
ax.set_xlim(-1, N_BUSES)
ax.legend()
ax.set_title("Node Identification")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig2_metrics.png")

# FIG 3: TOPOLOGY
fig, ax = plt.subplots(figsize=(8, 8))
G = nx.Graph()
G.add_edges_from(METRICS_BASE["edges"])
pos = nx.kamada_kawai_layout(G)
nx.draw(
    G,
    pos,
    ax=ax,
    node_color=METRICS_BASE["pi_vals"],
    cmap="Reds",
    node_size=300,
    edge_color="#555555",
)
labels = {
    i: str(REV_MAP[i + 1]) for i in range(N_BUSES) if METRICS_BASE["pi_vals"][i] > 0.05
}
nx.draw_networkx_labels(G, pos, ax=ax, labels=labels, font_size=8, font_weight="bold")
ax.set_title("IEEE 34 Spectral Fragility Map")
plt.savefig(f"{OUTPUT_DIR}/fig3_topology.png")

print(f"[DONE] Resultados IEEE 34 en: {OUTPUT_DIR}")
