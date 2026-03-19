from __future__ import annotations
import os
import numpy as np
import pandas as pd
import pytest

from pfebench.scenarios.G4_E18_Chamorro_Event import G4_E18_Chamorro_Event

# =============================================================================
# HELPER: Create Dummy CSV
# =============================================================================


@pytest.fixture
def mock_csv_file(tmp_path):
    """
    Crea un archivo CSV temporal con datos sintéticos para probar la carga.
    """
    # CORRECCIÓN: Aumentamos la resolución a 2000 puntos (2kHz)
    # Antes con 100 puntos (100Hz) teníamos aliasing severo para 60Hz,
    # lo que causaba overshoots locos en la interpolación cúbica.
    t = np.linspace(0, 1.0, 2000)

    f = np.full_like(t, 60.0)
    # Hacemos una rampa de frecuencia
    # f sube a partir de t=0.5
    mask_ramp = t >= 0.5
    f[mask_ramp] = 60.0 + (t[mask_ramp] - 0.5) * 2.0

    # Voltaje Va (Amplitud 100.0)
    va = 100.0 * np.sin(2 * np.pi * 60 * t)
    vb = 100.0 * np.sin(2 * np.pi * 60 * t - 2 * np.pi / 3)
    vc = 100.0 * np.sin(2 * np.pi * 60 * t + 2 * np.pi / 3)

    df = pd.DataFrame({"t": t, "f": f, "va": va, "vb": vb, "vc": vc})

    # Guardar en carpeta temporal
    file_path = tmp_path / "test_chamorro.csv"
    df.to_csv(file_path, index=False)

    return str(file_path)


# =============================================================================
# UNIT TESTS
# =============================================================================


def test_G4_E18_Load_and_Resample(mock_csv_file):
    """
    Prueba que el escenario carga el CSV y re-muestrea correctamente.
    CSV original: 2000 Hz
    Target: 1000 Hz (Downsampling)
    """
    target_fs = 1000.0

    sc = G4_E18_Chamorro_Event(
        csv_path=mock_csv_file, fs_hz=target_fs, col_t="t", col_f="f", col_va="va"
    )
    out = sc.run()

    # 1. Verificar Duración
    # El CSV tiene 1.0s. A 1000Hz, deberíamos tener aprox 1000 muestras.
    assert len(out.t) == pytest.approx(1000, abs=5)

    # 2. Verificar Datos (Frecuencia)
    # Al final del archivo original, f subió a 61.0 Hz
    assert out.f_true[-1] == pytest.approx(61.0, abs=0.1)

    # 3. Verificar Datos (Voltaje)
    assert not np.isnan(out.v).any()

    # Amplitud conservada (Max cerca de 100)
    assert np.max(out.v) > 99.0


def test_G4_E18_MonteCarlo_Tuning(mock_csv_file):
    """
    Verifica que podemos escalar la amplitud y desplazar el tiempo via Monte Carlo.
    """
    sc = G4_E18_Chamorro_Event(
        csv_path=mock_csv_file, fs_hz=1000.0  # Usamos buen sampling rate
    )

    # Caso 1: Escalar amplitud x0.5
    sc.set_montecarlo_tuning({"scale": 0.5, "file": mock_csv_file})
    out1 = sc.run()
    max_v1 = np.max(np.abs(out1.v))

    # En el mock original la amplitud era 100. Ahora debe ser 50.
    # Con 2000Hz de fuente, la interpolación es precisa.
    assert max_v1 == pytest.approx(50.0, rel=0.01)

    # Caso 2: Verificar Shift (opcional, o implícito en que no rompa nada)
    sc.set_montecarlo_tuning(
        {"scale": 1.0, "shift": 0.1, "file": mock_csv_file}  # Desplaza la ventana
    )
    out2 = sc.run()
    # Solo verificamos que corra y tenga datos
    assert len(out2.v) > 0


def test_G4_E18_Missing_File_Error():
    """Asegura que falle con elegancia si el archivo no existe."""
    with pytest.raises(FileNotFoundError):
        sc = G4_E18_Chamorro_Event(csv_path="ruta/imaginaria/no_existe.csv")
        sc.run()
