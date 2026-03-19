"""
IEEE 14-BUS: SPECTRAL THEORY VALIDATION SUITE
Outputs:
  1. Story Plot (Map + Time)
  2. NEW: 4x4 Matrix Dashboard (The "Scanner" view)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.linalg as la
from scipy.integrate import odeint
import json
import os
import shutil

# ==============================================================================
# 0. SETUP
# ==============================================================================
OUTPUT_DIR = "results_ieee14_dashboard"
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)
print(f"[INIT] Output directory '{OUTPUT_DIR}' ready.")

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
        "lines.linewidth": 1.2,
        "mathtext.fontset": "cm",
        "figure.dpi": 200,
    }
)


# ==============================================================================
# 1. PHYSICS ENGINE (LTI)
# ==============================================================================
class PowerSystemLTI:
    def __init__(self, n_nodes, adj_matrix, M_vals, D_vals):
        self.n = n_nodes
        self.M_inv = np.diag(1.0 / M_vals)
        self.D = np.diag(D_vals)

        # Laplacian
        self.L = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            self.L[i, i] = np.sum(adj_matrix[i, :])
            for j in range(n_nodes):
                if i != j:
                    self.L[i, j] = -adj_matrix[i, j]

        # Inertial Laplacian H
        M_sqrt_inv = np.diag(1.0 / np.sqrt(M_vals))
        self.H = M_sqrt_inv @ self.L @ M_sqrt_inv

        # State Matrix A
        Z = np.zeros((self.n, self.n))
        I = np.eye(self.n)
        self.A = np.block([[Z, I], [-self.M_inv @ self.L, -self.M_inv @ self.D]])

    def get_spectral_metric(self):
        # Return full pi vector (Localization Score)
        vals, vecs = la.eigh(self.H)
        u_2 = vecs[:, 1]
        return u_2**2

    def simulate(self, t, fault_node, p_step=1.0):
        x0 = np.zeros(2 * self.n)
        P_dist = np.zeros(2 * self.n)
        P_dist[self.n + fault_node] = p_step * self.M_inv[fault_node, fault_node]

        def model(x, t_val):
            dx = self.A @ x
            if t_val > 1.0:
                dx += P_dist
            return dx

        return odeint(model, x0, t)


# ==============================================================================
# 2. IEEE 14 CONFIG
# ==============================================================================
def get_ieee14_params(topology_stress=1.0, inertia_level="high", damping_level="high"):
    n_nodes = 14
    edges = [
        (0, 1, 5.0),
        (0, 4, 4.0),
        (1, 2, 4.0),
        (1, 3, 3.0),
        (1, 4, 3.0),
        (2, 3, 2.0),
        (3, 4, 2.0),
        (3, 6, 1.0),
        (3, 8, 1.0),
        (4, 5, 2.0),
        (5, 10, 1.0),
        (5, 11, 1.0),
        (5, 12, 1.0),
        (6, 7, 1.0),
        (6, 8, 1.0),
        (8, 9, 1.0),
        (8, 13, 1.0),
        (9, 10, 1.0),
        (11, 12, 1.0),
        (12, 13, 1.0),
    ]

    adj = np.zeros((n_nodes, n_nodes))
    for u, v, w in edges:
        weight = w
        if u > 5 or v > 5:
            weight *= topology_stress
        adj[u, v] = adj[v, u] = weight

    # Generators: 0, 1, 2, 5, 7
    gen_buses = [0, 1, 2, 5, 7]
    M_base = 0.5 if inertia_level == "low" else 1.0
    D_base = 0.05 if damping_level == "low" else 0.2

    M = np.ones(n_nodes) * M_base
    D = np.ones(n_nodes) * D_base
    for g in gen_buses:
        M[g] = 10.0
        D[g] = 2.0

    return adj, M, D


# ==============================================================================
# 3. SCENARIO EXECUTION
# ==============================================================================
t_sim = np.linspace(0, 40, 2000)
dt = t_sim[1] - t_sim[0]

# --- A. HEALTHY ---
a_adj, a_M, a_D = get_ieee14_params(1.0, "high", "high")
sys_A = PowerSystemLTI(14, a_adj, a_M, a_D)

# --- B. STRESSED ---
b_adj, b_M, b_D = get_ieee14_params(0.4, "high", "high")
sys_B = PowerSystemLTI(14, b_adj, b_M, b_D)

# --- C. CRITICAL ---
c_adj, c_M, c_D = get_ieee14_params(0.25, "low", "low")
c_D[9:] = 0.01
sys_C = PowerSystemLTI(14, c_adj, c_M, c_D)

# Diagnosis
pi_scores = sys_C.get_spectral_metric()
target_node = np.argmax(pi_scores)
print(f"[INFO] Target Node: {target_node+1}")

# --- D. REPAIRED ---
d_M = c_M.copy()
d_M[target_node] += 15.0
d_D = c_D.copy()
d_D[target_node] += 10.0
sys_D = PowerSystemLTI(14, c_adj, d_M, d_D)

# SIMULATE
dist_node = target_node - 1 if target_node > 0 else 1
idx_freq = 14 + target_node
idx_angle = target_node

systems = [sys_A, sys_B, sys_C, sys_D]
labels = ["A. Healthy", "B. Stressed", "C. Critical", "D. Repaired"]
colors = ["green", "orange", "red", "blue"]

results = []
for sys in systems:
    sol = sys.simulate(t_sim, dist_node)

    # 1. Angle Stress (Proxy for Voltage Stress)
    angle = sol[:, idx_angle]
    angle = angle - angle[0]  # Deviation

    # 2. RoCoF
    freq = sol[:, idx_freq]
    rocof = np.gradient(freq, dt)

    # 3. Metric (Pi Scores)
    pi = sys.get_spectral_metric()

    # 4. Freq
    # (Already have freq)

    results.append({"angle": angle, "rocof": rocof, "pi": pi, "freq": freq})

# ==============================================================================
# 4. PLOT GENERATION: THE 4x4 DASHBOARD
# ==============================================================================
fig, axs = plt.subplots(4, 4, figsize=(16, 10), constrained_layout=True)

cols = [
    "Angle Stress (deg)\n(Proxy for V-Stability)",
    "Standard RoCoF\n(Hz/s)",
    r"Geometric Metric $\pi_i$" "\n(Spectral Localization)",
    "Frequency Deviation\n(p.u.)",
]

for i, ax in enumerate(axs[0]):
    ax.set_title(cols[i], fontweight="bold", fontsize=11)

for row in range(4):
    res = results[row]
    lbl = labels[row]
    col_c = colors[row]

    # Col 1: Angle Stress (Time)
    axs[row, 0].plot(t_sim, res["angle"] * 57.29, color=col_c)  # Rad to Deg
    axs[row, 0].set_ylabel(lbl, fontweight="bold", fontsize=10)
    axs[row, 0].set_ylim(-5, 90) if row > 1 else axs[row, 0].set_ylim(-5, 20)
    axs[row, 0].grid(True, alpha=0.3)

    # Col 2: RoCoF (Time)
    axs[row, 1].plot(t_sim, res["rocof"], color=col_c, alpha=0.8)
    axs[row, 1].set_ylim(-0.5, 0.5)

    # Col 3: Geometric Metric (Bar Chart)
    # Highlight the target node
    bars = axs[row, 2].bar(range(1, 15), res["pi"], color="gray", alpha=0.5)
    # Color the max bar
    idx_max = np.argmax(res["pi"])
    bars[idx_max].set_color(col_c)
    bars[idx_max].set_alpha(1.0)
    axs[row, 2].set_ylim(0, 1.0)
    axs[row, 2].text(
        idx_max + 1,
        res["pi"][idx_max] + 0.05,
        f"Bus {idx_max+1}",
        ha="center",
        fontsize=8,
        fontweight="bold",
        color=col_c,
    )

    # Col 4: Frequency (Time)
    axs[row, 3].plot(t_sim, res["freq"], color=col_c, lw=1.5)
    axs[row, 3].set_ylim(-0.15, 0.20)
    # Mark Peak
    pk = np.max(np.abs(res["freq"]))
    axs[row, 3].text(
        25, 0.12, f"Peak: {pk:.3f}", color=col_c, fontsize=9, fontweight="bold"
    )

# Labels x-axis only on bottom
for ax in axs[3, :]:
    if ax == axs[3, 2]:
        ax.set_xlabel("Bus Index")
    else:
        ax.set_xlabel("Time [s]")

fig.suptitle(
    "IEEE 14-Bus Spectral Validation Dashboard: From Fragility to Repair",
    fontsize=16,
    y=1.02,
)
out_path = os.path.join(OUTPUT_DIR, "ieee14_dashboard_4x4.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"[DONE] Dashboard saved to: {out_path}")
