# tests/test_ipdft.py
import numpy as np

from estimators.basic.ipdft import IpDFT
from scenarios.s1_synthetic.make_clean import make_clean
from utils.pmu.pmu_input import PMU_Input


def test_ipdft_smoke() -> None:
    fs = 5000
    duration = 0.2
    sig, truth = make_clean(f0=60.0, df=0.1, duration=duration, fs=fs)

    # IpDFT ahora recibe config dict
    est = IpDFT(config={"fs": fs, "frame_len": 256, "channel": "V1", "nominal_hz": 60.0})

    # Alimentar snapshots PMU_Input (usar V1; el resto en 0.0)
    out = []
    for n, x in enumerate(sig):
        ts = n / fs
        measures = PMU_Input(
            timestamp=ts,
            V1=float(x),
            V2=0.0,
            V3=0.0,
            I1=0.0,
            I2=0.0,
            I3=0.0,
        )
        y = est.update(measures)
        out.append(y)

    assert len(out) == len(sig)

    # Sanidad básica: la media global cerca de 60 Hz (tolerancia laxa)
    freqs = np.array([o.frequency_hz for o in out], dtype=float)
    mean_freq = float(freqs.mean())
    assert abs(mean_freq - 60.0) < 2.0
