# train_pigru.py
import os
import math
import random
import json
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Importa CONSTANTES y FRONT-END desde tu propio código
from estimators import (
    FS_PHYSICS,
    FS_DSP,
    RATIO,
    get_test_signals,
    IIR_Bandpass,
    FastRMS_Normalizer,
)

# ==============================
# 0. CONFIG GLOBAL
# ==============================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

WINDOW_LEN = 100         # 100 muestras @10 kHz -> 10 ms de ventana
BATCH_SIZE = 256
EPOCHS = 40
LR = 1e-3
WEIGHT_DECAY = 1e-5
TRAIN_SPLIT = 0.8        # 80% train / 20% val

MODEL_PATH = "pi_gru_pmu.pt"
CONFIG_PATH = "pi_gru_pmu_config.json"


# ==============================
# 1. DATASET
# ==============================

class FreqWindowDataset(Dataset):
    """
    Dataset de ventanas deslizantes:
      - X: [N, T, 1] donde T = WINDOW_LEN
      - y: [N] frecuencia (Hz) de la última muestra de la ventana
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        X: shape (N, T), y: shape (N,)
        """
        super().__init__()
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]

        # Añadimos el canal de features = 1 -> (N, T, 1)
        self.X = torch.from_numpy(X.astype(np.float32)).unsqueeze(-1)
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def build_dataset(window_len: int = WINDOW_LEN) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construye el dataset global usando TODOS los escenarios de get_test_signals()
    Aplicando el MISMO front-end que el estimador online:
      - IIR_Bandpass
      - FastRMS_Normalizer

    Devuelve:
       X: (N, T)
       y: (N,)
    """
    signals = get_test_signals()

    X_list = []
    y_list = []

    for sc_name, (t_phys, v_ana, f_true, meta) in signals.items():
        # Downsample 1 MHz -> 10 kHz
        v_dsp = v_ana[::RATIO]
        f_dsp = f_true[::RATIO]

        # Front-end propio para este escenario
        bp = IIR_Bandpass()
        agc = FastRMS_Normalizer()

        v_proc = np.zeros_like(v_dsp, dtype=np.float32)
        for i, s in enumerate(v_dsp):
            v_bp = bp.step(float(s))
            v_proc[i] = agc.step(v_bp)

        N = len(v_proc)
        if N <= window_len:
            continue

        # Ventanas deslizantes (stride = 1)
        for i in range(0, N - window_len):
            win = v_proc[i:i + window_len]
            # target = frecuencia de la última muestra de la ventana
            f_target = f_dsp[i + window_len - 1]

            X_list.append(win.astype(np.float32))
            y_list.append(np.float32(f_target))

    X = np.stack(X_list, axis=0)  # (N, T)
    y = np.stack(y_list, axis=0)  # (N,)
    return X, y


# ==============================
# 2. MODELO PI-GRU
# ==============================

class PIGRUModel(nn.Module):
    """
    GRU sencilla:
      Input: (B, T, 1)
      Output: (B,) frecuencia en Hz
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (B, T, 1)
        out, _ = self.gru(x)           # out: (B, T, H)
        last = out[:, -1, :]           # (B, H)
        y = self.fc(last)              # (B, 1)
        return y.squeeze(-1)           # (B,)


# ==============================
# 3. HELPERS DE TRAINING
# ==============================

def train_val_split(X: np.ndarray, y: np.ndarray, train_split: float):
    N = X.shape[0]
    indices = np.arange(N)
    np.random.shuffle(indices)

    train_size = int(N * train_split)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    return X_train, y_train, X_val, y_val


def create_loaders(X: np.ndarray, y: np.ndarray):
    X_train, y_train, X_val, y_val = train_val_split(X, y, TRAIN_SPLIT)

    train_ds = FreqWindowDataset(X_train, y_train)
    val_ds = FreqWindowDataset(X_val, y_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader


# ==============================
# 4. LOOP DE TRAINING
# ==============================

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on: {device}")

    # ---- Dataset
    print("[INFO] Building dataset from synthetic scenarios...")
    X, y = build_dataset(window_len=WINDOW_LEN)
    print(f"[DATA] Total windows: {X.shape[0]}")

    train_loader, val_loader = create_loaders(X, y)

    # ---- Modelo
    model = PIGRUModel(
        input_dim=1,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # ReduceLROnPlateau SIN verbose (para compatibilidad con PyTorch antiguo)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=3,
    )

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # ------------------------
        #   FASE DE TRAIN
        # ------------------------
        model.train()
        train_loss_sum = 0.0
        n_train = 0

        for xb, yb in train_loader:
            xb = xb.to(device)  # (B, T, 1)
            yb = yb.to(device)  # (B,)

            optimizer.zero_grad()
            pred = model(xb)    # (B,)
            loss = criterion(pred, yb)
            loss.backward()
            # Gradient clipping ligero
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = xb.size(0)
            train_loss_sum += loss.item() * batch_size
            n_train += batch_size

        train_loss = train_loss_sum / max(n_train, 1)
        train_rmse = math.sqrt(train_loss)

        # ------------------------
        #   FASE DE VALIDACIÓN
        # ------------------------
        model.eval()
        val_loss_sum = 0.0
        n_val = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                pred = model(xb)
                loss = criterion(pred, yb)

                batch_size = xb.size(0)
                val_loss_sum += loss.item() * batch_size
                n_val += batch_size

        val_loss = val_loss_sum / max(n_val, 1)
        val_rmse = math.sqrt(val_loss)

        # Scheduler en función del loss de validación
        scheduler.step(val_loss)

        print(
            f"[EPOCH {epoch:03d}] "
            f"Train MSE={train_loss:.6f} (RMSE={train_rmse:.4f}) | "
            f"Val MSE={val_loss:.6f} (RMSE={val_rmse:.4f})"
        )

        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(
                f"[SAVE] New best model -> {MODEL_PATH}  "
                f"(ValMSE={val_loss:.6f}, ValRMSE={val_rmse:.4f})"
            )

    # Guardar también config mínima para el estimador online
    config = {
        "window_len_samples": WINDOW_LEN,
        "input_dim": 1,
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.1,
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)

    print("\n[DONE] Training finished.")
    print(f"       Best Val MSE = {best_val_loss:.6f}")
    print(f"       Model saved to: {MODEL_PATH}")
    print(f"       Config saved to: {CONFIG_PATH}")


if __name__ == "__main__":
    train()
