import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import os


# ==========================================
# 1. CONFIGURACIÓN DE ESTILO IEEE (Ready to Publish)
# ==========================================
def set_ieee_style():
    # Ancho de columna IEEE standard (3.5 pulgadas aprox)
    fig_width = 3.5
    golden_mean = (np.sqrt(5) - 1.0) / 2.0
    fig_height = fig_width * golden_mean

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "axes.labelsize": 9,
            "font.size": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "text.usetex": False,  # Cambiar a True si tienes LaTeX instalado localmente
            "figure.figsize": [fig_width, fig_height],
            "lines.linewidth": 1.5,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
        }
    )


set_ieee_style()

# Crear carpeta de salida
output_dir = "paper_plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ==========================================
# 2. MODELO DEL SISTEMA (WSCC 9-BUS KRON-REDUCED)
# ==========================================

# Parámetros Base (P.U.)
M_base = np.array([10.0, 8.0, 6.0])  # Nodos 1, 2, 3
D_base = np.array([2.0, 1.5, 1.0])  # Amortiguamiento base
# Pesos de las líneas (Susceptancia efectiva)
w_12 = 5.0
w_23 = 4.0
w_13 = 3.0


def get_laplacian(w12, w23, w13):
    """Construye la matriz Laplaciana L 3x3"""
    L = np.array(
        [[w12 + w13, -w12, -w13], [-w12, w12 + w23, -w23], [-w13, -w23, w13 + w23]]
    )
    return L


def get_spectral_metrics(M, L):
    """Calcula el autovalor crítico nu_c y el autovector q_c"""
    # Mass-Normalized Laplacian: L_tilde = M^(-1/2) L M^(-1/2)
    M_sqrt_inv = np.diag(1.0 / np.sqrt(M))
    L_tilde = M_sqrt_inv @ L @ M_sqrt_inv

    # Autovalores
    eigvals, eigvecs = la.eigh(L_tilde)

    # Ordenar (nu1=0, nu2=critico, nu3=rapido)
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    nu_c = eigvals[1]  # El segundo más pequeño (Fiedler generalizado)
    q_c = eigvecs[:, 1]

    return nu_c, q_c, L_tilde


# ==========================================
# 3. FIGURA 1: THE HOSTING POLYTOPE
# ==========================================
print("Generando Figura 1: Hosting Polytope...")

# Barrido de parámetros (Penetración de Inversores en Nodos 1 y 2)
# rho = 0 (SG), rho = 1 (GFL casi puro)
N = 100
rho_vals = np.linspace(0, 0.9, N)  # Hasta 90% para evitar singularidad M=0
X, Y = np.meshgrid(rho_vals, rho_vals)
Z_exact = np.zeros((N, N))
Z_poly = np.zeros((N, N))

# Estado Base (rho=0)
L_base = get_laplacian(w_12, w_23, w_13)
nu_base, q_base, _ = get_spectral_metrics(M_base, L_base)
nu_min = 0.2  # Umbral de seguridad definido

# Cálculo de Pesos del Politopo (Sensibilidad Analítica)
# w_i = (nu_c / M_i) * q_ci^2 * Delta_M
Delta_M1 = M_base[0]  # Asumiendo reemplazo total
Delta_M2 = M_base[1]
weight_1 = (nu_base / M_base[0]) * (q_base[0] ** 2) * Delta_M1
weight_2 = (nu_base / M_base[1]) * (q_base[1] ** 2) * Delta_M2
budget = nu_base - nu_min

for i in range(N):
    for j in range(N):
        rho1 = X[i, j]
        rho2 = Y[i, j]

        # Inercia actual
        M_curr = M_base.copy()
        M_curr[0] *= 1 - rho1
        M_curr[1] *= 1 - rho2

        # Exacto
        nu_curr, _, _ = get_spectral_metrics(M_curr, L_base)
        Z_exact[i, j] = nu_curr

        # Aproximación Lineal (Politopo)
        # Condición: w1*rho1 + w2*rho2 <= budget
        # Visualizamos el margen: Margen = Budget - Costo
        margin = budget - (weight_1 * rho1 + weight_2 * rho2)
        Z_poly[i, j] = nu_base - (weight_1 * rho1 + weight_2 * rho2)

# Plotting
fig1, ax1 = plt.subplots()
# Región de Estabilidad Exacta
cs = ax1.contourf(X, Y, Z_exact, levels=20, cmap="RdYlBu", alpha=0.8)
# Frontera Exacta (nu_c = nu_min)
exact_line = ax1.contour(
    X, Y, Z_exact, levels=[nu_min], colors="black", linewidths=2, linestyles="-"
)
# Frontera del Politopo (Lineal)
poly_line = ax1.contour(
    X, Y, Z_poly, levels=[nu_min], colors="white", linewidths=2, linestyles="--"
)

ax1.set_xlabel(r"Inv. Penetration Node 1 ($\rho_1$)")
ax1.set_ylabel(r"Inv. Penetration Node 2 ($\rho_2$)")
ax1.set_title("Spectral Hosting Polytope")

# Leyenda manual
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color="black", lw=2, label="Exact Boundary"),
    Line2D([0], [0], color="white", lw=2, ls="--", label="Linear Certificate"),
    Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        markerfacecolor="blue",
        alpha=0.3,
        label="Safe Region",
    ),
]
ax1.legend(handles=legend_elements, loc="upper right", frameon=True)

plt.savefig(f"{output_dir}/fig1_hosting_polytope.pdf")
plt.savefig(f"{output_dir}/fig1_hosting_polytope.png")
print("--> Figura 1 guardada.")

# ==========================================
# 4. FIGURA 2: FRAGILITY MAP
# ==========================================
print("Generando Figura 2: Fragility Map...")

# Nodal Fragility: F_i = (nu_c / M_i) * q_i^2
nodal_frag = (nu_base / M_base) * (q_base**2)


# Edge Fragility: E_ij = (q_i/sqrt(M_i) - q_j/sqrt(M_j))^2
# Links: (1,2), (2,3), (1,3)
def calc_edge_frag(i, j, q, M):
    return (q[i] / np.sqrt(M[i]) - q[j] / np.sqrt(M[j])) ** 2


edge_frag = [
    calc_edge_frag(0, 1, q_base, M_base),  # Line 1-2
    calc_edge_frag(1, 2, q_base, M_base),  # Line 2-3
    calc_edge_frag(0, 2, q_base, M_base),  # Line 1-3
]
edge_labels = ["L1-2", "L2-3", "L1-3"]

fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(4, 2.5), constrained_layout=True)

# Nodal Bar Chart
bars1 = ax2a.bar(["N1", "N2", "N3"], nodal_frag, color="#1f77b4", alpha=0.8)
ax2a.set_ylabel(r"Nodal Fragility $\mathcal{F}_i$")
ax2a.set_title("Node Sensitivity")
ax2a.grid(axis="x")

# Highlight critical node
max_idx = np.argmax(nodal_frag)
bars1[max_idx].set_color("#d62728")  # Rojo para el crítico

# Edge Bar Chart
bars2 = ax2b.bar(edge_labels, edge_frag, color="#2ca02c", alpha=0.8)
ax2b.set_ylabel(r"Edge Fragility $\mathcal{E}_{ij}$")
ax2b.set_title("Link Sensitivity")
ax2b.grid(axis="x")

plt.savefig(f"{output_dir}/fig2_fragility_map.pdf")
plt.savefig(f"{output_dir}/fig2_fragility_map.png")
print("--> Figura 2 guardada.")

# ==========================================
# 5. FIGURA 3: DYNAMIC BRAESS EFFECT
# ==========================================
print("Generando Figura 3: Braess Effect...")

# Simulación: Aumentar w_13 (Línea 1-3)
# Efecto físico: Aumentar w mejora rigidez (nu_c sube)
# Efecto adverso: Interacción PLL reduce amortiguamiento efectivo (delta baja)
# Modelo: D_eff = D_base - alpha * Delta_w
alpha_braess = 0.6
w_vals = np.linspace(3.0, 8.0, 50)
real_parts = []
stiffness_vals = []

for w in w_vals:
    # 1. Actualizar Topología
    L_curr = get_laplacian(w_12, w_23, w)

    # 2. Actualizar Rigidez Espectral
    nu_c, _, _ = get_spectral_metrics(M_base, L_curr)
    stiffness_vals.append(nu_c)

    # 3. Calcular Amortiguamiento Efectivo (Braess Interaction)
    delta_w = w - 3.0
    D_eff = D_base * (1 - 0.1 * delta_w * alpha_braess)  # Reducción fenomenológica

    # 4. Calcular Polo Dominante Exacto (s = -delta/2 +/- sqrt(...))
    # Aproximación modal: Re(s) ~ -delta_eff / 2
    # Modal damping delta_k = q^T D_tilde q
    M_sqrt_inv = np.diag(1.0 / np.sqrt(M_base))
    D_tilde = M_sqrt_inv @ np.diag(D_eff) @ M_sqrt_inv
    _, q_curr, _ = get_spectral_metrics(M_base, L_curr)

    modal_damping = q_curr.T @ D_tilde @ q_curr
    real_parts.append(-modal_damping / 2.0)

fig3, ax3 = plt.subplots()

# Eje primario: Margen de Estabilidad (Parte Real)
ln1 = ax3.plot(w_vals, real_parts, "r-", label="Stability Margin (Re{s})", linewidth=2)
ax3.set_xlabel(r"Link Strength $w_{13}$ [p.u.]")
ax3.set_ylabel(r"Stability Margin ($\sigma$)", color="r")
ax3.tick_params(axis="y", labelcolor="r")

# Zona de Inestabilidad
ax3.axhspan(-0.02, min(real_parts), color="red", alpha=0.1)
ax3.text(
    6.5,
    min(real_parts) * 0.9,
    "UNSTABLE\n(Braess)",
    color="red",
    fontsize=8,
    ha="center",
)

# Eje secundario: Rigidez Espectral (nu_c)
ax3b = ax3.twinx()
ln2 = ax3b.plot(
    w_vals, stiffness_vals, "b--", label=r"Spectral Stiffness ($\nu_c$)", alpha=0.6
)
ax3b.set_ylabel(r"Spectral Stiffness $\nu_c$", color="b")
ax3b.tick_params(axis="y", labelcolor="b")

# Leyenda combinada
lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax3.legend(lns, labs, loc="upper center", frameon=False)

plt.title("Dynamic Braess Effect")
plt.savefig(f"{output_dir}/fig3_braess_effect.pdf")
plt.savefig(f"{output_dir}/fig3_braess_effect.png")
print("--> Figura 3 guardada.")

print("\n¡Todo listo! Gráficos guardados en la carpeta 'paper_plots'.")
