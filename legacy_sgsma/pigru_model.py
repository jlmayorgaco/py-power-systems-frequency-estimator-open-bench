# pigru_model.py

import json
import torch
import torch.nn as nn
import torch.nn.functional as F

# Importamos la clase envoltorio desde estimators.py
# (Asegúrate de que estimators.py esté en la misma carpeta)
from estimators import PIGRU_FreqEstimator

class AttentionBlock(nn.Module):
    """
    Temporal Attention to focus on reliable signal segments
    and ignore impulsive noise spikes.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: (B, T, H)
        # Calcula pesos de atención para cada paso de tiempo
        weights = self.attention(x) # (B, T, 1)
        
        # Suma ponderada de los estados ocultos (Context Vector)
        context = torch.sum(weights * x, dim=1) # (B, H)
        
        return context, weights

class PIDRE_Model(nn.Module):
    """
    Physics-Informed Dynamic Recurrent Estimator (PI-DRE)
    Input:  (B, T, 1) -> Ventana de muestras de voltaje
    Output: (B,)      -> Frecuencia estimada en Hz
    """
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,  # Increased capacity for complex dynamics
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        # 1. Feature extraction
        # Conv1D actúa como un banco de filtros aprendible (suavizado adaptativo)
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(32)
        )
        
        # 2. Recurrent dynamics (GRU)
        self.gru = nn.GRU(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False # Must be False for real-time causality
        )
        
        # 3. Temporal Attention
        self.attn = AttentionBlock(hidden_dim)
        
        # 4. Estimation Head (Predicts Delta Freq)
        self.fc_freq = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        # Aux head for amplitude (useful for Physics Loss, optional for inference)
        self.fc_amp = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus() # Amplitude must be positive
        )

    def forward(self, x):
        # x: (B, T, 1)
        
        # Conv needs (B, C, T)
        x_conv = x.permute(0, 2, 1)
        feat = self.conv(x_conv)
        feat = feat.permute(0, 2, 1) # Back to (B, T, C)
        
        # Recurrent processing
        out, _ = self.gru(feat) # (B, T, H)
        
        # Attention mechanism (Weighted sum of history)
        # Esto permite ignorar outliers (impulsive noise)
        context, _ = self.attn(out) # (B, H)
        
        # Predict Delta Freq (Diferencia respecto a 60Hz)
        delta_freq = self.fc_freq(context).squeeze(-1) # (B,)
        
        # === LOGICA CRÍTICA DE INFERENCIA ===
        # Si estamos entrenando, devolvemos el delta (target centrado en 0).
        # Si estamos en inferencia (main.py), sumamos 60.0 automáticamente.
        if self.training:
            return delta_freq
        else:
            return delta_freq + 60.0

def build_pigru_estimator(
    model_path: str = "pi_gru_pmu.pt",
    config_path: str = "pi_gru_pmu_config.json",
    device: str = None,
):
    """
    Factory function para cargar el modelo y envolverlo en la clase estimadora.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Intentar cargar configuración
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        # Fallback config si no existe el json
        print("[PI-GRU WARNING] Config file not found. Using defaults.")
        cfg = {"window_len_samples": 100}

    # Instanciar arquitectura
    model = PIDRE_Model(
        input_dim=1,
        hidden_dim=cfg.get("hidden_dim", 128),
        num_layers=cfg.get("num_layers", 2),
        dropout=cfg.get("dropout", 0.2),
    )

    # Cargar pesos entrenados
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"[PI-GRU ERROR] Could not load weights from {model_path}: {e}")
        return None

    # Envolver en la clase que usa main.py
    estimator = PIGRU_FreqEstimator(
        model=model,
        window_len_samples=cfg.get("window_len_samples", 100),
        smooth_win=5, # Suavizado bajo (5 muestras) porque la Atención ya filtra ruido
        device=device,
        name="PI-DRE (Advanced)",
    )
    
    return estimator