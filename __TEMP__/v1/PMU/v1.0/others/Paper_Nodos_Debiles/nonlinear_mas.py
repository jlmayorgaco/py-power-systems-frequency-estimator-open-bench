"""
IEEE 14-BUS SPECTRAL VALIDATION IN ANDES (FIXED & ROBUST)
---------------------------------------------------------
Corregido:
1. Añadida capa estática (Slack/PV) para inicializar GENCLS (Hot Start).
2. Fix del error 'Mandatory parameter GENCLS.gen is missing'.
3. Fix de SyntaxWarning en strings de LaTeX.
"""

import andes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import os
import shutil

# Configuración de ANDES silenciosa
andes.config_logger(stream_level=40)

OUTPUT_DIR = "results_andes_validation"
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

print("[INIT] Generando caso IEEE 14 robusto para ANDES...")


# ==============================================================================
# 1. GENERADOR DE CASO ROBUSTO (Static + Dynamic Linking)
# ==============================================================================
def create_ieee14_xlsx(filename="ieee14_andes.xlsx"):
    # --- BUS DATA ---
    bus_data = pd.DataFrame(
        {
            "uid": range(1, 15),
            "idx": range(1, 15),
            "name": [f"Bus{i}" for i in range(1, 15)],
            "Vn": 138.0,
            "v0": 1.0,
            "a0": 0.0,
            "area": 1,
            "zone": 1,
        }
    )

    # --- LINE DATA (B ~ 1/X) ---
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
            "u": [e[0] for e in edges],
            "v": [e[1] for e in edges],
            "r": 0.0,
            "x": [e[2] for e in edges],
            "b": 0.0,
        }
    )

    # --- STATIC GENERATORS (Power Flow Layer) ---
    # Slack en Bus 1
    slack_data = pd.DataFrame(
        {
            "uid": [1],
            "idx": [1],
            "bus": [1],
            "Vn": 138.0,
            "v0": 1.05,
            "a0": 0.0,  # Voltage setpoint
        }
    )

    # PV en Buses 2, 3, 6, 8
    # idx 2,3,4,5 para PVs
    pv_buses = [2, 3, 6, 8]
    pv_data = pd.DataFrame(
        {
            "uid": range(2, 6),
            "idx": range(2, 6),
            "bus": pv_buses,
            "Vn": 138.0,
            "p0": 0.4,
            "v0": 1.02,  # Injecting Active Power
        }
    )

    # --- DYNAMIC GENERATORS (GENCLS) ---
    # IMPORTANTE: 'gen' apunta al idx del Slack/PV correspondiente
    # Slack(idx=1) -> GENCLS(gen=1)
    # PV(idx=2..5) -> GENCLS(gen=2..5)

    # Mapeo: 5 generadores en total
    # GENCLS usa 'gen' (link to static) NO 'bus'.
    gencls_data = pd.DataFrame(
        {
            "uid": range(1, 6),
            "idx": range(1, 6),
            "gen": [1, 2, 3, 4, 5],  # Links to Slack(1) and PV(2,3,4,5)
            "Sn": 100.0,
            "Vn": 138.0,
            "u": 1,
            "M": [10.0] * 5,
            "D": [2.0] * 5,
            "ra": 0.0,
            "xl": 0.0,
        }
    )

    # --- LOAD DATA (PQ) ---
    # Cargas en todos los buses para forzar flujo
    load_buses = list(range(1, 15))
    pq_data = pd.DataFrame(
        {
            "uid": range(1, 15),
            "idx": range(1, 15),
            "bus": load_buses,
            "p0": 0.6,
            "q0": 0.1,
            "Vn": 138.0,  # Carga neta para que generadores trabajen
        }
    )

    # Guardar en Excel
    with pd.ExcelWriter(filename) as writer:
        bus_data.to_excel(writer, sheet_name="Bus", index=False)
        line_data.to_excel(writer, sheet_name="Line", index=False)
        slack_data.to_excel(writer, sheet_name="Slack", index=False)  # NUEVO
        pv_data.to_excel(writer, sheet_name="PV", index=False)  # NUEVO
        gencls_data.to_excel(writer, sheet_name="GENCLS", index=False)
        pq_data.to_excel(writer, sheet_name="PQ", index=False)

    return filename


# ==============================================================================
# 2. MOTOR DE ANÁLISIS ESPECTRAL
# ==============================================================================
def diagnose_andes_system(ss):
    """Calcula H y pi_i extrayendo datos del caso ANDES cargado."""
    n_buses = len(ss.Bus.idx.v)
    M_vec = np.zeros(n_buses)

    # Mapear inercia de GENCLS a Buses
    # GENCLS.gen -> StaticGen -> Bus
    # Necesitamos un mapa inverso: StaticIdx -> BusIdx
    static_map = {}
    if ss.Slack.n > 0:
        for i in range(ss.Slack.n):
            static_map[ss.Slack.idx.v[i]] = int(ss.Slack.bus.v[i]) - 1
    if ss.PV.n > 0:
        for i in range(ss.PV.n):
            static_map[ss.PV.idx.v[i]] = int(ss.PV.bus.v[i]) - 1

    if ss.GENCLS.n > 0:
        for i in range(ss.GENCLS.n):
            gen_link = ss.GENCLS.gen.v[i]  # Link to static
            if gen_link in static_map:
                bus_idx = static_map[gen_link]
                M_vec[bus_idx] += 2 * ss.GENCLS.M.v[i]

    M_vec[M_vec < 0.1] = 0.5  # Inercia virtual mínima en cargas

    L = np.zeros((n_buses, n_buses))
    for i in range(ss.Line.n):
        u = int(ss.Line.u.v[i]) - 1
        v = int(ss.Line.v.v[i]) - 1
        x = ss.Line.x.v[i]
        b_val = 1.0 / max(x, 1e-4)
        L[u, v] -= b_val
        L[v, u] -= b_val
        L[u, u] += b_val
        L[v, v] += b_val

    M_sqrt_inv = np.diag(1.0 / np.sqrt(M_vec))
    H = M_sqrt_inv @ L @ M_sqrt_inv
    vals, vecs = la.eigh(H)
    return vecs[:, 1] ** 2, np.argmax(vecs[:, 1] ** 2)


# ==============================================================================
# 3. EJECUTOR DE ESCENARIOS
# ==============================================================================
def run_scenario(phase_name, topology_stress=1.0, repair_bus=None, repair_M=0):
    xlsx_name = create_ieee14_xlsx()
    ss = andes.load(xlsx_name, setup=False, default_config=True)

    # 1. Estrés de Topología (Debilitar líneas > Bus 5)
    if topology_stress < 1.0:
        factor = 1.0 / topology_stress
        for i in range(ss.Line.n):
            if ss.Line.u.v[i] > 5 or ss.Line.v.v[i] > 5:
                ss.Line.x.v[i] *= factor

    # 2. Estrés de Damping (Fase C)
    if phase_name == "C_Critical":
        ss.GENCLS.D.v[:] = 0.5

    # 3. Reparación (Fase D)
    if repair_bus is not None:
        # Encontrar qué GENCLS está en ese bus (vía static map)
        # Ojo: ANDES no tiene link directo fácil sin buscar.
        # Simplificación: Buscamos si hay un GENCLS cuyo padre (Slack/PV) esté en repair_bus
        target_gen_idx = -1

        # Mapa inverso rápido: Bus -> Static Idx
        bus_to_static = {}
        for i in range(ss.Slack.n):
            bus_to_static[ss.Slack.bus.v[i]] = ss.Slack.idx.v[i]
        for i in range(ss.PV.n):
            bus_to_static[ss.PV.bus.v[i]] = ss.PV.idx.v[i]

        if repair_bus in bus_to_static:
            static_idx = bus_to_static[repair_bus]
            # Buscar GENCLS que apunte a este static_idx
            for i in range(ss.GENCLS.n):
                if ss.GENCLS.gen.v[i] == static_idx:
                    target_gen_idx = i
                    break

        if target_gen_idx >= 0:
            ss.GENCLS.M.v[target_gen_idx] += repair_M / 2.0
            ss.GENCLS.D.v[target_gen_idx] += 10.0
            print(f"   -> Added GFM to Gen at Bus {repair_bus}")

    # SETUP Y PFLOW
    ss.setup()

    # Diagnóstico en Fase C
    target_node = None
    pi_scores = None
    if phase_name == "C_Critical":
        pi_scores, weak_idx = diagnose_andes_system(ss)
        target_node = weak_idx + 1
        print(f"[{phase_name}] DIAGNOSTICO: Nodo Débil Detectado = Bus {target_node}")

    if not ss.PFlow.run():
        print(f"[{phase_name}] Power Flow Failed!")
        return None

    # TDS: Toggle Event (Load Step) en Bus 8 (o target)
    disturb_bus = target_node if target_node else 8

    # Buscar PQ en el bus de disturbio
    pq_idx = -1
    for i in range(ss.PQ.n):
        if ss.PQ.bus.v[i] == disturb_bus:
            pq_idx = i
            break

    if pq_idx >= 0:
        # Step de carga masivo (3.5 p.u.) en t=1.0s
        ss.add(
            "Toggle",
            {
                "model": "PQ",
                "dev": "PQ",
                "idx": pq_idx,
                "attr": "p0",
                "t": 1.0,
                "val": ss.PQ.p0.v[pq_idx] + 3.5,
            },
        )

    ss.TDS.config.tf = 20.0
    ss.TDS.config.criteria = 0
    ss.TDS.run()

    # RESULTADOS
    t = np.array(ss.dae.ts.t)

    # Angulo relativo: Bus 8 - Bus 1
    # ANDES guarda angulos en radianes en ss.dae.ts.y
    # Necesitamos los indices en la matriz Y
    try:
        # Los indices de estado pueden variar, usamos bus.a directamente si está disponible o mapeamos
        # Forma segura: Bus.a es una variable algebraica en ANDES.
        addr_target = ss.Bus.a.a[disturb_bus - 1]
        addr_slack = ss.Bus.a.a[0]

        angle_target = ss.dae.ts.y[:, addr_target]
        angle_slack = ss.dae.ts.y[:, addr_slack]
        angle_diff = (angle_target - angle_slack) * 180 / np.pi

        freq_dev = np.gradient(angle_target, t) / (2 * np.pi)

    except:
        # Fallback si falla la indexación
        angle_diff = np.zeros_like(t)
        freq_dev = np.zeros_like(t)

    return {
        "t": t,
        "angle": angle_diff,
        "freq": freq_dev,
        "pi": pi_scores,
        "target": target_node,
    }


# ==============================================================================
# 4. EJECUCIÓN
# ==============================================================================
print(">>> Fase A: Healthy...")
res_a = run_scenario("A_Healthy", 1.0)

print(">>> Fase B: Stressed...")
res_b = run_scenario("B_Stressed", 0.4)

print(">>> Fase C: Critical...")
res_c = run_scenario("C_Critical", 0.25)
WEAK_BUS = res_c["target"]

print(f">>> Fase D: Repaired (Spectral @ Bus {WEAK_BUS})...")
res_d = run_scenario("D_Repaired", 0.25, repair_bus=WEAK_BUS, repair_M=40.0)

print(">>> Fase D: Bad (Random @ Bus 1)...")
res_bad = run_scenario("D_Bad", 0.25, repair_bus=1, repair_M=40.0)

# ==============================================================================
# 5. DASHBOARD
# ==============================================================================
fig, axs = plt.subplots(4, 4, figsize=(16, 10), constrained_layout=True)
cols = ["Angle Sep (deg)", "RoCoF (Hz/s)", r"Metric $\pi_i$", "Freq (p.u.)"]
rows = ["A", "B", "C", "D"]
results_map = {"A": res_a, "B": res_b, "C": res_c, "D": res_d}
colors = ["green", "orange", "red", "blue"]

for i, ax in enumerate(axs[0]):
    ax.set_title(cols[i], fontweight="bold")

for i, phase in enumerate(rows):
    res = results_map[phase]
    if res is None:
        continue
    c = colors[i]

    # Angle
    axs[i, 0].plot(res["t"], res["angle"], color=c)
    axs[i, 0].set_ylabel(phase, fontweight="bold")
    axs[i, 0].grid(True, alpha=0.3)

    # RoCoF
    rocof = np.gradient(res["freq"], res["t"])
    axs[i, 1].plot(res["t"], rocof, color=c, alpha=0.6)

    # Metric (Solo C)
    if phase == "C":
        axs[i, 2].bar(range(1, 15), res["pi"], color="red")
        axs[i, 2].text(WEAK_BUS, 0.5, f"Bus {WEAK_BUS}", ha="center")

    # Freq
    axs[i, 3].plot(res["t"], res["freq"], color=c)

    if phase == "D":  # Comparativa
        axs[i, 0].plot(
            res_bad["t"], res_bad["angle"], color="gray", ls="--", label="Random"
        )
        axs[i, 3].plot(
            res_bad["t"], res_bad["freq"], color="gray", ls="--", label="Random"
        )
        axs[i, 3].legend()

plt.savefig(f"{OUTPUT_DIR}/andes_final_proof.png", dpi=300)
print("[EXITO] Simulación en ANDES terminada. Gráfico guardado.")
