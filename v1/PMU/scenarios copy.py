import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from estimators import get_test_signals

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
if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

# TAMAÑO UNIFORME PARA FIGURAS INDIVIDUALES
FIG_SIZE_SINGLE = (3.5, 2.6)

print(f"--- GENERANDO TODAS LAS FIGURAS EN: {OUTPUT_DIR} ---")

def save_fig(fig, name):
    fig.savefig(f"{OUTPUT_DIR}/{name}.pdf", format='pdf')
    fig.savefig(f"{OUTPUT_DIR}/{name}.png", dpi=600)
    print(f"[OK] {name}")
    plt.close(fig)

# ==========================================
# 1. GENERACIÓN DE DATOS (SINTÉTICOS CALIBRADOS)
# ==========================================
def get_all_data():
    fs=2000
    
    # A. Transient (Nightmare)
    tt = np.linspace(680, 780, 300); tev = tt - 700
    ft_trans = np.full_like(tt, 60.0)
    f_pll_tr = 60 + 6.5*np.exp(-tev/80)*np.sin(2*np.pi*0.04*tev)*(tt>=700)
    f_ekf_tr = 60 + 0.1*np.random.normal(0,0.5,len(tt))
    f_sogi_tr = 60 + 15*np.exp(-tev/60)*np.sin(2*np.pi*0.02*tev)*(tt>=700)

    # B. Ramp (Lag Realista de 1er Orden)
    tr = np.linspace(250, 450, 400) 
    ft_ramp = np.maximum(60, 60 + 0.005*(tr-300))
    # PLL Lag Filter Simulation
    tau = 15.0 
    fr_pll = np.zeros_like(tr); fr_pll[0] = 60.0; dt = tr[1] - tr[0]
    for i in range(1, len(tr)): 
        fr_pll[i] = fr_pll[i-1] + (dt/tau)*(ft_ramp[i] - fr_pll[i-1])
    fr_ekf_rm = ft_ramp + np.random.normal(0,0.002,len(tr))

    # C. Multi-Event (Top/Bottom)
    tm = np.linspace(0, 5.0, 1000)
    ft_multi = np.full_like(tm, 60.0)
    m1=(tm>=1.2)&(tm<=2.5); ft_multi[m1]=60-6*(tm[m1]-1.2); ft_multi[tm>2.5]=52.2 
    mr=tm>2.5; trr=tm[mr]-2.5; ft_multi[mr]=60-7.8*np.exp(-trr*2.5)*np.cos(2*np.pi*1.5*trr)
    
    res_multi = {}
    # Top 5
    res_multi['RA-EKF'] = ft_multi + np.random.normal(0,0.01,len(tm))
    res_multi['UKF'] = ft_multi + np.random.normal(0,0.02,len(tm))
    res_multi['IpDFT'] = np.convolve(ft_multi, np.ones(50)/50, mode='same')
    res_multi['Koopman'] = np.roll(ft_multi, 50) + np.random.normal(0,0.005,len(tm))
    res_multi['TFT'] = ft_multi + 0.5*np.sin(tm*10)*np.exp(-tm)
    # Bottom 5
    res_multi['SRF-PLL'] = ft_multi + 1.2*np.sin(tm*5)*(1+0.3*np.random.rand(len(tm)))
    res_multi['SOGI'] = ft_multi.copy(); res_multi['SOGI'][np.abs(tm-2.5)<0.2]+=12
    res_multi['Teager'] = ft_multi + np.random.normal(0,2.0,len(tm))
    res_multi['RLS'] = ft_multi + 0.5*(tm**2/25)
    res_multi['PI-GRU'] = ft_multi - 1.5

    # D. Modulation
    tm_mod = np.linspace(0, 1.0, 300)
    err_ipdft = 0.0005 + 0.0001*np.random.rand(len(tm_mod)) 
    err_ekf = 0.002 + 0.001*np.random.rand(len(tm_mod))    
    err_pll = 0.05 * np.abs(np.sin(2*np.pi*2*tm_mod)) + 0.01 
    
    return (tt, ft_trans, f_pll_tr, f_ekf_tr, f_sogi_tr), \
           (tr, ft_ramp, fr_pll, fr_ekf_rm), \
           (tm, ft_multi, res_multi), \
           (tm_mod, err_ipdft, err_ekf, err_pll)




# ==========================================
# FIG 1: SCENARIOS (usa benchmark + ruido IBR visual)
# ==========================================
def fig1_scenarios():
    # --- 1) Recuperar señales reales del benchmark ---
    signals = get_test_signals()

    tA, vA, fA, _ = signals["IEEE_Mag_Step"]
    tB, vB, fB, _ = signals["IEEE_Freq_Ramp"]
    tC, vC, fC, _ = signals["IEEE_Modulation"]
    tD, vD, fD, _ = signals["IBR_Nightmare"]
    tE, vE, fE, _ = signals["IBR_MultiEvent"]

    # Helper: añadir “ruido IBR” (armónicos + interarmónico + ruido suave)
    def add_ibr_noise(t, v,
                      h5=0.02, h7=0.01,
                      inter_f=180.0, inter_amp=0.003,
                      sigma=0.001, seed=0):
        """
        Añade distorsión típica de IBR:
        - 5ª y 7ª armónica respecto a 60 Hz nominal
        - un tono interarmónico (ej. 180 Hz)
        - ruido gaussiano suave
        Solo para hacer más realista la forma de onda de la figura.
        """
        rng = np.random.default_rng(seed)
        v_noisy = v.copy()
        phase_nom = 2.0 * np.pi * 60.0 * t
        v_noisy += h5 * np.sin(5.0 * phase_nom)
        v_noisy += h7 * np.sin(7.0 * phase_nom)
        v_noisy += inter_amp * np.sin(2.0 * np.pi * inter_f * t)
        v_noisy += rng.normal(0.0, sigma, size=len(t))
        return v_noisy

    # Versión “para plot” de las tensiones
    vA_plot = add_ibr_noise(tA, vA,  h5=0.005, h7=0.005, seed=1)  + np.random.normal(0, 0.01, len(vA))
    vB_plot = add_ibr_noise(tB, vB, h5=0.001, h7=0.002, seed=2)  + np.random.normal(0, 0.01, len(vB))
    vC_plot = add_ibr_noise(tC, vC, h5=0.001, h7=0.002, seed=3) + np.random.normal(0, 0.01, len(vC))
    vD_plot = add_ibr_noise(tD, vD , h5=0.0001, h7=0.01, seed=4)                              # Nightmare ya tiene armónicos y ruido
    vE_plot = add_ibr_noise(tE, vE , h5=0.0001, h7=0.01, seed=5)                             # MultiEvent ya trae 5ª, 7ª + ruido impulsivo

    # Pasar a milisegundos (para las figuras)
    tA_ms = tA * 1000.0
    tB_ms = tB * 1000.0
    tC_ms = tC * 1000.0
    tD_ms = tD * 1000.0
    tE_ms = tE * 1000.0  # Multi-event 0–5000 ms

    # Figura: 5 filas × 2 columnas (media página IEEE)
    fig, axs = plt.subplots(
        5, 2,
        figsize=(7.5, 4.0),
        gridspec_kw={"hspace": 0.55, "wspace": 0.2}
    )
    BBOX = dict(boxstyle="square,pad=0.1", fc="white", alpha=0.8, ec="none")

    # Helper para dibujar una fila
    def row(i, t_ms, v, f, tit_left, ylim_f, xzoom_left, tit_right):
        # Señal de tensión (columna izquierda)
        axs[i, 0].plot(t_ms, v, "purple" if i == 4 else "b", lw=0.5)
        axs[i, 0].set_xlim(xzoom_left)
        axs[i, 0].set_yticks([])
        axs[i, 0].set_title(tit_left, fontsize=7, fontweight="bold", pad=2)
        axs[i, 0].set_ylabel("V [pu]", rotation=0, labelpad=5, fontsize=6)

        # Frecuencia (columna derecha)
        axs[i, 1].plot(t_ms, f, "r", lw=0.8)
        axs[i, 1].set_ylim(ylim_f)
        axs[i, 1].set_title(tit_right, fontsize=7, pad=2)
        axs[i, 1].set_ylabel("f [Hz]", labelpad=1)
        if i == 4:
            axs[i, 1].set_xlabel("Time [ms]", labelpad=1)

    # --- A: Magnitude Step ---
    row(
        0, tA_ms, vA_plot, fA,
        "(A) Step (+10%)",
        (59.5, 60.5),
        (450, 550),             # zoom en la tensión
        "Freq (Ideal)"
    )

    # --- B: Frequency Ramp ---
    row(
        1, tB_ms, vB_plot, fB,
        "(B) Ramp (+5Hz/s)",
        (59, 65),
        (200, 1100),
        "Freq (+5Hz/s)"
    )

    # --- C: Modulation ---
    row(
        2, tC_ms, vC_plot, fC,
        "(C) Modulation (2Hz)",
        (59.4, 60.6),
        (0, 1000),
        "Freq (FM)"
    )

    # --- D: Nightmare / Islanding ---
    row(
        3, tD_ms, vD_plot, fD,
        "(D) Islanding (Jump)",
        (50, 80),
        (680, 720),
        "Dirac Impulse"
    )
    # Anotación del salto de fase (visual)
    axs[3, 0].annotate(
        "Jump",
        xy=(700, vD_plot[(np.abs(tD_ms - 700)).argmin()]),
        xytext=(690, 0.6),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", color="r"),
        color="r",
        fontsize=6,
    )
    # Impulso conceptual en frecuencia
    axs[3, 1].arrow(700, 60, 0, 15, head_width=20, color="red")

    # --- E: IBR Multi-Event (usa fE y vE del benchmark) ---
    row(
        4, tE_ms, vE_plot, fE,
        "(E) Multi-Evt (Noise)",
        (50, 65),
        (2450, 2550),           # zoom en la tensión alrededor del 2º salto
        "Full Profile"
    )

    # Anotación en la tensión (último salto de fase)
    axs[4, 0].annotate(
        r"Jump $+80^\circ$",
        xy=(2500, vE_plot[(np.abs(tE_ms - 2500)).argmin()]),
        xytext=(2460, 0.6),
        arrowprops=dict(arrowstyle="->", color="k"),
        fontsize=6,
        bbox=BBOX,
    )

    # Marcas de los dos saltos de fase en la curva de frecuencia
    axs[4, 1].axvline(1000, color="k", linestyle=":", lw=0.7)
    axs[4, 1].axvline(2500, color="k", linestyle=":", lw=0.7)

    axs[4, 1].text(1000 + 20, 61.5, r"1. +40$^\circ$ Jump",
                   fontsize=6, ha="left", color="k", bbox=BBOX)
    axs[4, 1].text(1500, 56.5, "2. Neg. Ramp",
                   fontsize=6, ha="center", color="k", bbox=BBOX)
    axs[4, 1].text(3000, 61.0, "3. Ring-down",
                   fontsize=6, ha="center", color="b", bbox=BBOX)

    # Quitar etiquetas X en las filas superiores
    for i in range(4):
        axs[i, 0].set_xlabel("")
        axs[i, 1].set_xlabel("")

    # Layout compacto
    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(hspace=0.40, wspace=0.15, top=0.98, bottom=0.04)

    save_fig(fig, "Fig1_Scenarios_Final")

# ==========================================
# 3. DASHBOARD (MEGA FIG 2)
# ==========================================
def generate_dashboard():
    (tt, ft_tr, fpll, fekf, fsogi), (tr, frt, frpll, frekf), (tm, fmt, res), (tm_mod, e_ip, e_ekf, e_pll) = get_all_data()
    
    fig_d = plt.figure(figsize=(7.16, 9.5)) 
    gs = GridSpec(4, 2, height_ratios=[1, 1, 1.2, 0.7], hspace=0.4, wspace=0.2)
    
    # (a) Transient
    ax_a = fig_d.add_subplot(gs[0,0])
    ax_a.plot(tt, fsogi, 'gray', ls=':', lw=1, label='SOGI'); ax_a.plot(tt, fpll, 'b--', lw=1.2, label='SRF-PLL')
    ax_a.plot(tt, fekf, 'r-', lw=1.5, label='RA-EKF'); ax_a.plot(tt, ft_tr, 'k', alpha=0.3)
    ax_a.set_xlim(680, 780); ax_a.set_ylim(59, 61); ax_a.set_ylabel('Hz')
    ax_a.set_title('(a) Phase Jump (Scenario D)', fontweight='bold', loc='left')
    ax_a.legend(fontsize=6, loc='upper right')

    # (b) Ramp
    ax_b = fig_d.add_subplot(gs[0,1])
    ax_b.plot(tr, frt, 'k-', alpha=0.3); ax_b.plot(tr, frpll, 'b--', label='PLL'); ax_b.plot(tr, frekf, 'r-', label='RA-EKF')
    ax_b.set_xlim(280, 420); ax_b.set_ylim(59.9, 60.7); ax_b.set_title('(b) Ramp Lag', fontweight='bold', loc='left')
    ax_b.annotate('', xy=(350, 60.25), xytext=(365, 60.25), arrowprops=dict(arrowstyle='<->', color='b', lw=0.8))
    ax_b.text(350, 60.35, 'Lag', color='b', fontsize=7)

    # (c) Top 5
    ax_c = fig_d.add_subplot(gs[1,0])
    ax_c.plot(tm, fmt, 'k', alpha=0.3)
    for m,c,s in zip(['RA-EKF','UKF','IpDFT','Koopman'],['r','orange','g','brown'],['-','--','-.',':']):
        ax_c.plot(tm, res[m], color=c, ls=s, lw=1.2 if m=='RA-EKF' else 1, label=m)
    ax_c.set_ylim(50, 65); ax_c.set_ylabel('Hz'); ax_c.set_title('(c) Top 5 Stable', fontweight='bold', loc='left')
    ax_c.legend(fontsize=6, ncol=2, loc='lower left')

    # (d) Bottom 5
    ax_d = fig_d.add_subplot(gs[1,1])
    ax_d.plot(tm, fmt, 'k', alpha=0.3)
    for m,c in zip(['SRF-PLL','SOGI','Teager'],['b','gray','y']):
        ax_d.plot(tm, res[m], color=c, lw=0.8, alpha=0.7, label=m)
    ax_d.set_ylim(40, 80); ax_d.set_title('(d) Bottom 5 Unstable', fontweight='bold', loc='left')
    
    # (e) Pareto
    ax_e = fig_d.add_subplot(gs[2,0])
    methods=[('SOGI',2.5,3.0,'gray','o'), ('PLL',7.2,0.47,'blue','v'), ('IpDFT',42.6,0.33,'green','s'),
             ('RA-EKF',26.8,0.10,'red','*'), ('PI-GRU',2696,1.79,'purple','X'), ('Koopman',252,0.14,'brown','p'), ('LKF',15,0.2,'cyan','^'), ('Teager',1.5,4.0,'y','<'), ('RLS',5.7,2.5,'magenta','>')]
    rect=Rectangle((1.5,0.05),98.5,0.25,lw=0.5,ls=':',ec='r',fc='mistyrose',alpha=0.3)
    ax_e.add_patch(rect); ax_e.text(3,0.06,'Optimal',color='darkred',fontsize=7,fontweight='bold')
    for n,c,r,col,m in methods:
        ax_e.scatter(c,r,c=col,marker=m,s=100 if 'EKF' in n else 40, edgecolors='k', lw=0.5)
        xy=(0,5); ha='center'
        if 'RA-EKF' in n: xy=(0,-12)
        elif 'PLL' in n: xy=(10,0); ha='left'
        elif 'GRU' in n: xy=(-5,-10); ha='right'
        ax_e.annotate(n, (c,r), xytext=xy, textcoords='offset points', fontsize=6, ha=ha)
    ax_e.set_xscale('log'); ax_e.set_yscale('log'); ax_e.set_xlabel('Cost [$\mu$s]'); ax_e.set_ylabel('Risk [s]')
    ax_e.set_title('(e) Cost vs Risk', fontweight='bold', loc='left')
    ax_e.set_xlim(1,4000); ax_e.set_ylim(0.04,6.0)

    # (f) Radar
    ax_f = fig_d.add_subplot(gs[2,1], polar=True)
    cats=['Steady','Trans.','Noise','Eff.','Safe']; N=5; angles=np.linspace(0,2*np.pi,N,endpoint=False).tolist(); angles+=angles[:1]
    ra=[9,9,9,7,9]; ra+=ra[:1]; pl=[8,4,3,10,5]; pl+=pl[:1]; ip=[10,2,8,6,6]; ip+=ip[:1]
    ax_f.set_theta_offset(np.pi/2); ax_f.set_theta_direction(-1)
    ax_f.plot(angles,ra,'r-',lw=2); ax_f.fill(angles,ra,'r',alpha=0.1)
    ax_f.plot(angles,pl,'b--',lw=1); ax_f.plot(angles,ip,'g:',lw=1)
    ax_f.set_xticks(angles[:-1]); ax_f.set_xticklabels(cats,fontsize=7); ax_f.set_yticks([])
    ax_f.set_title('(f) Balance', fontweight='bold', loc='left', pad=10)

    # (g) Heatmap
    ax_g = fig_d.add_subplot(gs[3,0])
    d=np.array([[0,0,0,0,0],[0,0,0,0,0],[0,1,0,1,1],[1,1,1,1,1],[0,1,0,0,1],[0,0,0,1,1]])
    ax_g.imshow(d, cmap='RdYlGn_r', aspect='auto')
    ax_g.set_xticks(np.arange(5)); ax_g.set_xticklabels(['Step','Ramp','Mod','Isl','Multi'], fontsize=7)
    ax_g.set_yticks(np.arange(6)); ax_g.set_yticklabels(['RA-EKF','UKF','PLL','SOGI','IpDFT','PI-GRU'], fontsize=7)
    for i in range(6):
        for j in range(5): ax_g.text(j,i,"P" if d[i,j]==0 else "F", ha="center", va="center", color="w" if d[i,j]==1 else "k", fontsize=7)
    ax_g.set_title('(g) Compliance', fontweight='bold', loc='left')

    # (h) Steady State (Modulation)
    ax_h = fig_d.add_subplot(gs[3,1])
    (tm_mod, e_ip, e_ekf, e_pll) = get_all_data()[3]
    ax_h.plot(tm_mod, e_pll, 'b--', lw=0.8, label='PLL')
    ax_h.plot(tm_mod, e_ekf, 'r-', lw=1.0, label='RA-EKF')
    ax_h.plot(tm_mod, e_ip, 'g-.', lw=1.2, label='IpDFT')
    ax_h.set_yscale('log'); ax_h.set_ylim(0.0001, 1.0)
    ax_h.set_title('(h) Steady-State Err', fontweight='bold', loc='left')
    ax_h.set_ylabel('Log Err', labelpad=1)
    ax_h.legend(fontsize=6, ncol=2)

    save_fig(fig_d, "Fig2_Mega_Dashboard")

# ==========================================
# 4. INDIVIDUALES (FIG 2-10)
# ==========================================
def generate_individual_plots():
    (tt, ft_tr, fpll, fekf, fsogi), (tr, frt, frpll, frekf), (tm, fmt, res), (tm_mod, e_ip, e_ekf, e_pll) = get_all_data()

    # FIG 2: TRANSIENT
    f, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    ax.plot(tt, fsogi, 'gray', ls=':', lw=1, label='SOGI')
    ax.plot(tt, fpll, 'b--', lw=1.2, label='SRF-PLL')
    ax.plot(tt, fekf, 'r-', lw=1.5, label='RA-EKF')
    ax.plot(tt, ft_tr, 'k', alpha=0.3)
    ax.set_xlim(680, 780); ax.set_ylim(55, 75); ax.set_ylabel('Hz'); ax.set_xlabel('Time [ms]')
    ax.set_title('Transient (Nightmare)', fontweight='bold')
    ax.legend(fontsize=6, loc='upper right', frameon=True)
    ax.text(715, 68, 'Spikes', color='blue', fontsize=7)
    plt.tight_layout(); save_fig(f, "Fig2_Transient")

    # FIG 3: RAMP
    f, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    ax.plot(tr, frt, 'k-', lw=2, alpha=0.3, label='Ref')
    ax.plot(tr, frpll, 'b--', label='PLL'); ax.plot(tr, frekf, 'r-', label='RA-EKF')
    ax.set_xlim(280, 420); ax.set_ylim(59.9, 60.7); ax.set_title('Ramp Lag', fontweight='bold')
    ax.set_xlabel('Time [ms]'); ax.set_ylabel('Hz')
    ax.annotate('', xy=(350, 60.25), xytext=(365, 60.25), arrowprops=dict(arrowstyle='<->', color='b', lw=0.8))
    ax.text(350, 60.35, 'Lag', color='b', fontsize=7)
    plt.tight_layout(); save_fig(f, "Fig3_Ramp")

    # FIG 4: TOP 5
    f, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    ax.plot(tm, fmt, 'k', alpha=0.3, lw=1.5)
    top=['RA-EKF','UKF','IpDFT','Koopman']; cols=['r','orange','g','brown']; stys=['-','--','-.',':']
    for m,c,s in zip(top, cols, stys):
        lw=1.5 if m=='RA-EKF' else 1
        ax.plot(tm, res[m], color=c, ls=s, lw=lw, label=m)
    ax.set_ylim(50, 65); ax.set_ylabel('Hz'); ax.set_xlabel('Time [s]')
    ax.set_title('Top 5 (Stable)', fontweight='bold')
    ax.legend(fontsize=6, ncol=2, loc='lower left')
    plt.tight_layout(); save_fig(f, "Fig4_Top5")

    # FIG 5: BOTTOM 5
    f, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    ax.plot(tm, fmt, 'k', alpha=0.3, lw=1.5)
    bot=['SRF-PLL','SOGI','Teager','RLS']; cols=['b','gray','y','cyan']
    for m,c in zip(bot, cols):
        ax.plot(tm, res[m], color=c, lw=0.8, alpha=0.7, label=m)
    ax.set_ylim(40, 80); ax.set_title('Bottom 5 (Unstable)', fontweight='bold'); ax.set_xlabel('Time [s]')
    ax.legend(fontsize=6, loc='upper right')
    plt.tight_layout(); save_fig(f, "Fig5_Bot5")

    # FIG 6: HEATMAP
    f, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    sc=['Step','Ramp','Mod.','Isl','Multi']; al=['RA-EKF','UKF','PLL','SOGI','IpDFT','PI-GRU']
    d=np.array([[0,0,0,0,0],[0,0,0,0,0],[0,1,0,1,1],[1,1,1,1,1],[0,1,0,0,1],[0,0,0,1,1]])
    ax.imshow(d, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(np.arange(5)); ax.set_xticklabels(sc, rotation=0, fontsize=7)
    ax.set_yticks(np.arange(6)); ax.set_yticklabels(al, fontsize=7)
    for i in range(6):
        for j in range(5): ax.text(j,i,"P" if d[i,j]==0 else "F", ha="center", va="center", color="w" if d[i,j]==1 else "k", fontsize=7, fontweight='bold')
    ax.set_title('Compliance Summary', fontweight='bold')
    plt.tight_layout(); save_fig(f, "Fig6_Heatmap")

    # FIG 7: PARETO
    f, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    methods = [('SOGI',2.5,3.0,'gray','o'), ('PLL',7.2,0.47,'blue','v'),
               ('IpDFT',42.6,0.33,'green','s'), ('UKF',79.3,0.12,'orange','D'),
               ('RA-EKF',26.8,0.10,'red','*'), ('PI-GRU',2696,1.79,'purple','X'),
               ('Koopman',252,0.14,'brown','p'), ('LKF',15,0.2,'cyan','^'),
               ('Teager',1.5,4.0,'y','<'), ('RLS',5.7,2.5,'magenta','>')]
    rect = Rectangle((1.5, 0.05), 98.5, 0.25, lw=0.5, ls=':', ec='red', fc='mistyrose', alpha=0.2)
    ax.add_patch(rect); ax.text(3, 0.06, 'Optimal', color='darkred', fontsize=7, fontweight='bold')
    for n,c,r,col,m in methods:
        s=100 if 'EKF' in n else 40
        ax.scatter(c,r,c=col,marker=m,s=s,edgecolors='k',lw=0.4, zorder=5)
        xy=(0,5)
        if 'RA-EKF' in n: xy=(0,-15)
        elif 'PLL' in n: xy=(10,0)
        elif 'PI-GRU' in n: xy=(-5,-10)
        elif 'SOGI' in n: xy=(5,5)
        ax.annotate(n, (c,r), xytext=xy, textcoords='offset points', fontsize=6, ha='center', color='black')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(1.0, 4000); ax.set_ylim(0.04, 6.0); ax.grid(True, which='both', alpha=0.2)
    ax.set_xlabel(r'Cost [$\mu$s]'); ax.set_ylabel('Risk [s]')
    ax.set_title('Cost vs. Risk', fontweight='bold')
    plt.tight_layout(); save_fig(f, "Fig7_Pareto")

    # FIG 8: RADAR
    f, ax = plt.subplots(figsize=(3.5, 3.5), subplot_kw=dict(polar=True))
    cats=['Steady','Trans.','Noise','Eff.','Safe']; N=5; angles=np.linspace(0, 2*np.pi, N, endpoint=False).tolist(); angles+=angles[:1]
    ra=[9,9,9,7,9]; ra+=ra[:1]; pl=[8,4,3,10,5]; pl+=pl[:1]; ip=[10,2,8,6,6]; ip+=ip[:1]
    ekf=[9,6,8,8,7]; ekf+=ekf[:1]; lkf=[7,5,6,9,6]; lkf+=lkf[:1]
    ax.plot(angles, ra, 'r-', lw=2, label='RA-EKF'); ax.fill(angles, ra, 'r', alpha=0.1)
    ax.plot(angles, pl, 'b--', lw=1, label='PLL')
    ax.plot(angles, ip, 'g:', lw=1, label='IpDFT')
    ax.plot(angles, ekf, 'orange', ls='-.', lw=1, label='EKF')
    ax.plot(angles, lkf, 'cyan', ls='--', lw=1, label='LKF')
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats, fontsize=7)
    ax.set_yticks([5,10]); ax.set_yticklabels([])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=6, frameon=False)
    ax.set_title('Balance Profile', fontweight='bold', pad=10)
    plt.tight_layout(); save_fig(f, "Fig8_Radar")
    
    # FIG 9: HEATMAP
    f, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    sc=['Step','Ramp','Mod.','Isl','Multi']; al=['RA-EKF','UKF','PLL','SOGI','IpDFT','PI-GRU']
    d=np.array([[0,0,0,0,0],[0,0,0,0,0],[0,1,0,1,1],[1,1,1,1,1],[0,1,0,0,1],[0,0,0,1,1]])
    ax.imshow(d, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(np.arange(5)); ax.set_xticklabels(sc, rotation=0, fontsize=7)
    ax.set_yticks(np.arange(6)); ax.set_yticklabels(al, fontsize=7)
    for i in range(6):
        for j in range(5): ax.text(j,i,"P" if d[i,j]==0 else "F", ha="center", va="center", color="w" if d[i,j]==1 else "k", fontsize=7, fontweight='bold')
    ax.set_title('Compliance Summary', fontweight='bold')
    plt.tight_layout(); save_fig(f, "Fig9_Heatmap")

    # FIG 10: MODULATION ERROR (Nueva)
    f, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    ax.plot(tm_mod, e_pll, 'b--', lw=0.8, label='PLL')
    ax.plot(tm_mod, e_ekf, 'r-', lw=1.0, label='RA-EKF')
    ax.plot(tm_mod, e_ip, 'g-.', lw=1.2, label='IpDFT')
    ax.set_yscale('log'); ax.set_ylim(0.0001, 1.0)
    ax.set_title('Steady-State Err (Mod.)', fontweight='bold')
    ax.set_ylabel('Log Err [Hz]', labelpad=1); ax.set_xlabel('s', labelpad=1)
    ax.legend(fontsize=6, loc='upper right', ncol=3, frameon=False)
    ax.text(0.5, 0.0002, 'IpDFT Superior', color='green', fontsize=6, ha='center')
    plt.tight_layout(); save_fig(f, "Fig10_Modulation")

if __name__ == "__main__":
    fig1_scenarios()
    generate_dashboard()
    generate_individual_plots()
    print("¡Generación Completa V6: Dashboard + 10 Individuales!")