# train_pigru.py (FINAL VERSION - FULL SPECTRUM TRAINING)
import os
import math
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader

# Importamos constantes de tu proyecto
from estimators import FS_DSP

# Importamos el modelo avanzado
from pigru_model import PIDRE_Model

# ==============================
# CONFIGURACIÓN
# ==============================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

WINDOW_LEN = 100  # 10 ms
BATCH_SIZE = 128
STEPS_PER_EPOCH = 200  # Aumentado para ver más variedad por época
EPOCHS = 100  # 100 épocas bien cargadas son suficientes
LR = 1e-3

MODEL_PATH = "pi_gru_pmu.pt"
CONFIG_PATH = "pi_gru_pmu_config.json"


# ==============================
# 1. GENERADOR DE DATOS "FULL SPECTRUM"
# ==============================
class SyntheticGenerator(IterableDataset):
    def __init__(self, batch_size=32, window_len=100):
        self.batch_size = batch_size
        self.window_len = window_len
        self.dt = 1.0 / FS_DSP

    def __iter__(self):
        while True:
            yield self.generate_batch()

    def generate_batch(self):
        X = np.zeros((self.batch_size, self.window_len, 1), dtype=np.float32)
        y = np.zeros((self.batch_size,), dtype=np.float32)
        t = np.arange(self.window_len) * self.dt

        for i in range(self.batch_size):
            # Frecuencia base aleatoria (55 - 65 Hz)
            f0 = np.random.uniform(55.0, 65.0)

            # === SELECCIÓN DE MODO ===
            # 0: Steady (Estable + Ruido)
            # 1: Freq Step (Escalón Frecuencia)
            # 2: Freq Ramp (Rampa ROCOF)
            # 3: Ring-down (Oscilación amortiguada) -> Para IBR_MultiEvent
            # 4: Phase Jump (Salto de Fase) -> Para IBR_Nightmare (Target = f0)
            # 5: Amp Fault (Escalón de Amplitud) -> Para IEEE_Mag_Step (Target = f0)

            mode = np.random.choice(
                [0, 1, 2, 3, 4, 5], p=[0.2, 0.15, 0.15, 0.2, 0.15, 0.15]
            )

            # Perfil de frecuencia y fase base
            freq_profile = np.ones_like(t) * f0
            phi_offset = np.zeros_like(t)  # Para saltos de fase
            amp_profile = np.ones_like(t)  # Para fallas de amplitud

            # --- Lógica de Modos ---

            if mode == 1:  # Freq Step
                step_t = np.random.randint(10, self.window_len - 10)
                f_new = f0 + np.random.uniform(-2.0, 2.0)
                freq_profile[step_t:] = f_new

            elif mode == 2:  # Ramp
                rate = np.random.uniform(-5.0, 5.0)
                freq_profile = f0 + rate * t

            elif mode == 3:  # Ringdown (IBR)
                A_osc = np.random.uniform(0.5, 3.0)
                sigma = np.random.uniform(1.0, 6.0)
                w_osc = 2 * np.pi * np.random.uniform(1.0, 5.0)
                osc = A_osc * np.exp(-sigma * t) * np.sin(w_osc * t)
                freq_profile = f0 + osc

            elif mode == 4:  # Phase Jump (CRUCIAL PARA NIGHTMARE)
                # La frecuencia SE MANTIENE en f0, solo cambia la fase.
                # Enseñamos a la red a NO reaccionar a esto.
                jump_t = np.random.randint(10, self.window_len - 10)
                jump_rad = np.deg2rad(np.random.uniform(-90, 90))
                phi_offset[jump_t:] = jump_rad

            elif mode == 5:  # Amp Fault (CRUCIAL PARA MAG_STEP)
                # La frecuencia SE MANTIENE, la amplitud cambia bruscamente.
                # Simula sag, swell o step.
                fault_t = np.random.randint(10, self.window_len - 10)
                # Factor de 0.1 (sag severo) a 1.2 (swell)
                scale = np.random.uniform(0.1, 1.2)
                amp_profile[fault_t:] = scale

            # --- Generación de Señal ---

            target_f = freq_profile[-1]

            # Integrar fase de la frecuencia
            phi_freq = np.cumsum(freq_profile * self.dt) * 2 * np.pi

            # Fase total = Integral(f) + Salto de Fase + Fase Inicial Aleatoria
            phi_total = phi_freq + phi_offset + np.random.uniform(0, 2 * np.pi)

            # Voltaje Base con perfil de amplitud
            v = amp_profile * np.sin(phi_total)

            # --- Corrupción de Señal (IBR Realism) ---

            # 1. Armónicos (aleatorios)
            if np.random.rand() < 0.6:
                v += np.random.uniform(0, 0.05) * np.sin(5 * phi_total)  # 5th
                v += np.random.uniform(0, 0.03) * np.sin(7 * phi_total)  # 7th
                # Interarmónico ocasional (cerca de fundamental)
                if np.random.rand() < 0.3:
                    f_inter = np.random.uniform(20, 40)
                    v += np.random.uniform(0, 0.02) * np.sin(2 * np.pi * f_inter * t)

            # 2. Ruido Gaussiano (Siempre presente)
            noise_lvl = np.random.uniform(0.0001, 0.005)
            v += np.random.normal(0, noise_lvl, size=len(t))

            # 3. Ruido Impulsivo (Spikes)
            if np.random.rand() < 0.25:
                idx_imp = np.random.randint(0, self.window_len)
                v[idx_imp] += np.random.uniform(-0.8, 0.8)  # Picos fuertes

            # --- Normalización (Simulación de Front-End) ---
            # Simulamos el FastRMS dividiendo por el RMS real + un pequeño error
            # Esto hace la red robusta a imperfecciones del AGC
            rms = np.sqrt(np.mean(v**2)) + 1e-6

            # A veces el AGC es perfecto, a veces tiene error (simula transitorio de AGC)
            agc_error = np.random.uniform(0.95, 1.05) if mode == 5 else 1.0
            v_norm = v / (rms * 1.414 * agc_error)

            # Guardar datos
            X[i, :, 0] = v_norm
            # Target Centering: Predecir Delta respecto a 60Hz
            y[i] = target_f - 60.0

        return torch.from_numpy(X), torch.from_numpy(y)


# ==============================
# 2. TRAINING LOOP
# ==============================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training PI-DRE (Full Spectrum) on: {device}")

    # Instanciar Modelo
    model = PIDRE_Model(hidden_dim=128, num_layers=2).to(device)

    # Optimizador y Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS
    )
    criterion = nn.MSELoss()

    # Dataset Infinito
    dataset = SyntheticGenerator(batch_size=BATCH_SIZE, window_len=WINDOW_LEN)
    loader = DataLoader(dataset, batch_size=None)
    iter_loader = iter(loader)

    best_loss = float("inf")
    model.train()

    print(f"[INFO] Starting {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        epoch_loss = 0.0

        for _ in range(STEPS_PER_EPOCH):
            X, y_true = next(iter_loader)
            X, y_true = X.to(device), y_true.to(device)

            optimizer.zero_grad()
            y_pred = model(X)  # Predice Delta

            loss = criterion(y_pred, y_true)
            loss.backward()

            # Gradient Clipping (Vital para GRUs estables)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / STEPS_PER_EPOCH
        rmse = math.sqrt(avg_loss)

        # Logging más limpio
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{EPOCHS} | RMSE: {rmse:.4f} Hz")

        # Guardar siempre el mejor
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_PATH)

    # Guardar configuración para inferencia
    config = {
        "window_len_samples": WINDOW_LEN,
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "target_centered": True,
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)

    print(f"\n[DONE] Best RMSE achieved: {math.sqrt(best_loss):.4f} Hz")
    print(f"[DONE] Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
