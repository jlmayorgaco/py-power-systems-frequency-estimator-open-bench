import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from scipy.linalg import eigh

# ---------------------------------------------------------
# 1. Definición del Sistema IEEE 13 Nodos (Simplificado)
# ---------------------------------------------------------


def get_ieee13_topology():
    # Nodos y Enlaces estándar del IEEE 13
    # Usaremos una representación simplificada de grafo no dirigido
    G = nx.Graph()

    # Nodos
    nodes = [650, 646, 645, 632, 633, 634, 611, 684, 671, 692, 675, 652, 680]
    G.add_nodes_from(nodes)

    # Enlaces (Edges) con impedancias aproximadas (R+jX -> Peso ~ 1/|Z|)
    # Peso w_ij representa la "fuerza" de la conexión (admitancia magnitud)
    edges = [
        (650, 632, 1.0),
        (632, 645, 1.0),
        (645, 646, 1.0),
        (632, 633, 2.0),  # Línea fuerte
        (633, 634, 0.5),  # Línea larga
        (632, 671, 1.5),
        (671, 680, 1.0),
        (671, 684, 1.0),
        (684, 611, 0.5),
        (684, 652, 0.5),  # Nodo frágil (mencionado en tus papers)
        (671, 692, 1.0),
        (692, 675, 1.0),
    ]

    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    # Posiciones fijas para visualización consistente
    pos = {
        650: (0, 10),
        632: (0, 8),
        645: (2, 8),
        646: (4, 8),
        633: (-2, 7),
        634: (-4, 7),
        671: (0, 5),
        680: (0, 2),
        684: (3, 5),
        611: (5, 4),
        652: (5, 6),
        692: (-3, 5),
        675: (-5, 4),
    }

    return G, pos


# ---------------------------------------------------------
# 2. Configuración de Inercia y Medidas
# ---------------------------------------------------------


def assign_inertia(G):
    # Asignamos inercia M_i
    # Nodo 650 (Subestación) = Alta Inercia
    # Nodos con PV = Media Inercia
    # Nodos de carga = Baja Inercia
    M = {node: 1.0 for node in G.nodes()}  # Base

    M[650] = 100.0  # Grid connection (Infinite bus approx)
    M[632] = 10.0
    M[671] = 10.0
    M[652] = 0.1  # Nodo muy ligero (frágil)
    M[611] = 0.5
    M[634] = 0.5

    return M


def calculate_inertial_measure(G, M):
    # Calcula m_x(y) según la definición del "Paper Teórico"
    measures = {}

    for x in G.nodes():
        neighbors = list(G.neighbors(x))
        if not neighbors:
            measures[x] = {}
            continue

        # Pesos brutos: w_xy / M_y
        raw_weights = {}
        total_w = 0.0

        for y in neighbors:
            w_xy = G[x][y]["weight"]
            m_y = M[y]
            val = w_xy / m_y
            raw_weights[y] = val
            total_w += val

        # Normalizar para obtener probabilidad
        prob_dist = {y: val / total_w for y, val in raw_weights.items()}
        measures[x] = prob_dist

    return measures


# ---------------------------------------------------------
# 3. Cálculo de Curvatura de Ollivier-Ricci (Vía Wasserstein)
# ---------------------------------------------------------


def compute_wasserstein(p_dict, q_dict, G_dist):
    # p_dict, q_dict: Diccionarios {nodo: probabilidad}
    # G_dist: Diccionario de distancias shortest_path entre todos los nodos

    # Unir soportes
    support = list(set(p_dict.keys()) | set(q_dict.keys()))
    n = len(support)

    if n == 0:
        return 0.0

    # Vectores de probabilidad alineados
    p_vec = np.array([p_dict.get(k, 0.0) for k in support])
    q_vec = np.array([q_dict.get(k, 0.0) for k in support])

    # Matriz de costos (distancia grafo entre vecinos)
    C = np.zeros((n, n))
    for i, u in enumerate(support):
        for j, v in enumerate(support):
            C[i, j] = G_dist[u][v]

    # Resolver transporte óptimo con Linprog
    # Min sum(C_ij * x_ij)
    # s.t. sum_j x_ij = p_i, sum_i x_ij = q_j, x >= 0

    c = C.flatten()
    A_eq = []
    b_eq = []

    # Restricciones fila (sum_j x_ij = p_i)
    for i in range(n):
        row = np.zeros((n, n))
        row[i, :] = 1
        A_eq.append(row.flatten())
        b_eq.append(p_vec[i])

    # Restricciones columna (sum_i x_ij = q_j)
    for j in range(n):
        col = np.zeros((n, n))
        col[:, j] = 1
        A_eq.append(col.flatten())
        b_eq.append(q_vec[j])

    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)

    # Bounds x >= 0
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")

    return res.fun  # Distancia Wasserstein W1


def calculate_curvature(G, M):
    measures = calculate_inertial_measure(G, M)
    path_lengths = dict(
        nx.all_pairs_dijkstra_path_length(G, weight="weight")
    )  # Distancia en el grafo (hop o weighted)
    # Para curvatura de grafos standard se usa distancia de salto (hop=1).
    # Pero en variedad Riemanniana ponderada deberíamos usar la inversa del peso.
    # Usaremos distancia de salto (hop distance) para la métrica base d(x,y)=1 como es estándar en ORC discreto.
    shortest_paths = dict(nx.all_pairs_shortest_path_length(G))

    curvatures = {}

    for u, v in G.edges():
        w1 = compute_wasserstein(measures[u], measures[v], shortest_paths)
        # d(x,y) = 1 para vecinos directos en grafo no ponderado por distancia
        kappa = 1 - w1
        curvatures[(u, v)] = kappa

    return curvatures


# ---------------------------------------------------------
# 4. Análisis de Estabilidad (Valores Propios)
# ---------------------------------------------------------


def get_weighted_laplacian_spectrum(G, M):
    # L_w = D - A_weighted
    # Generalized eigenvalue problem: L v = lambda M v  --> L_norm = M^{-1/2} L M^{-1/2}

    nodelist = list(G.nodes())
    n = len(nodelist)

    # Adjacency
    A = nx.to_numpy_array(G, nodelist=nodelist, weight="weight")

    # Inertia matrix
    m_vec = np.array([M[node] for node in nodelist])
    M_mat = np.diag(m_vec)

    # Laplacian
    D = np.diag(np.sum(A, axis=1))
    L = D - A

    # Mass-Normalized Laplacian: L_sys = M^{-1} L
    # Eigenvalues of M^{-1} L are same as M^{-1/2} L M^{-1/2}

    M_inv_sqrt = np.diag(1.0 / np.sqrt(m_vec))
    L_norm = M_inv_sqrt @ L @ M_inv_sqrt

    evals = eigh(L_norm, eigvals_only=True)
    return np.sort(evals)


# ---------------------------------------------------------
# 5. Simulación Principal
# ---------------------------------------------------------

# Caso Base
G_base, pos = get_ieee13_topology()
M_base = assign_inertia(G_base)
kappa_base = calculate_curvature(G_base, M_base)
evals_base = get_weighted_laplacian_spectrum(G_base, M_base)
lambda2_base = evals_base[1] if len(evals_base) > 1 else 0

# Caso Braess (Añadir línea que causa problemas)
# Conectamos un nodo "pesado" (632) con uno "frágil" lejano (684) saltándonos la estructura
G_braess = G_base.copy()
G_braess.add_edge(632, 684, weight=5.0)  # Linea muy fuerte (baja impedancia)
# Esto crea un "shortcut" que puede desviar flujo de probabilidad
kappa_braess = calculate_curvature(G_braess, M_base)
evals_braess = get_weighted_laplacian_spectrum(G_braess, M_base)
lambda2_braess = evals_braess[1]

# ---------------------------------------------------------
# 6. Gráficas
# ---------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Plot Config
cmap = plt.cm.RdYlGn  # Red = Negative Curvature (Fragile), Green = Positive
vmin, vmax = -1.0, 1.0

# --- Plot 1: Caso Base ---
ax = axes[0]
edges = G_base.edges()
colors = [kappa_base[e] for e in edges]
nx.draw_networkx_nodes(G_base, pos, ax=ax, node_color="lightgrey", node_size=500)
nx.draw_networkx_labels(G_base, pos, ax=ax)
lc = nx.draw_networkx_edges(
    G_base,
    pos,
    ax=ax,
    edge_color=colors,
    edge_cmap=cmap,
    width=3,
    edge_vmin=vmin,
    edge_vmax=vmax,
)
ax.set_title(
    f"IEEE 13 Base Topology\n$\lambda_2 = {lambda2_base:.4f}$ (Stable)\nMin Curvature: {min(colors):.2f}"
)
plt.colorbar(lc, ax=ax, label="Ollivier-Ricci Curvature")

# --- Plot 2: Caso Braess ---
ax = axes[1]
edges_b = G_braess.edges()
colors_b = [
    kappa_braess.get(e, kappa_braess.get((e[1], e[0]), 0)) for e in edges_b
]  # Handle undirected key
nx.draw_networkx_nodes(G_braess, pos, ax=ax, node_color="lightgrey", node_size=500)
nx.draw_networkx_labels(G_braess, pos, ax=ax)
lc_b = nx.draw_networkx_edges(
    G_braess,
    pos,
    ax=ax,
    edge_color=colors_b,
    edge_cmap=cmap,
    width=3,
    edge_vmin=vmin,
    edge_vmax=vmax,
)

# Highlight new edge
nx.draw_networkx_edges(
    G_braess,
    pos,
    ax=ax,
    edgelist=[(632, 684)],
    edge_color="blue",
    width=2,
    style="dashed",
)
ax.annotate(
    "New Link (Braess)",
    xy=(1.5, 6.5),
    xytext=(3, 9),
    arrowprops=dict(facecolor="black", shrink=0.05),
)

delta_lambda = lambda2_braess - lambda2_base
status = "DEGRADED" if delta_lambda < 0 else "IMPROVED"
ax.set_title(
    f"After Topological 'Upgrade' (Add 632-684)\n$\lambda_2 = {lambda2_braess:.4f}$ ({status})\nMin Curvature: {min(colors_b):.2f}"
)
plt.colorbar(lc_b, ax=ax, label="Ollivier-Ricci Curvature")

plt.suptitle(
    "Proof of Concept: Thermodynamic Curvature & Braess Effect in IEEE 13 Node System",
    fontsize=16,
)
plt.tight_layout()
plt.savefig("ieee13_curvature_braess.png")
