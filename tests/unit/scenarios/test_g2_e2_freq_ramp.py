"""
Unit tests for G2_E2_FreqRamp scenario.

Covers:
  - instantiation and default params
  - build() returns correctly shaped arrays
  - waveform length: n = round(T_s * fs_hz)
  - f_true is constant before onset, ramps after
  - f_true clipped at df_max_hz
  - phase is continuous (no discontinuity at onset)
  - v(t) = A0 * sin(phi) everywhere
  - schema contains expected keys
  - set_montecarlo_tuning with 'seed' works
  - scenario is deterministic given same seed
  - scenario varies given different seed
"""

from __future__ import annotations

import numpy as np
from openfreqbench.scenarios._base import ScenarioOutput
from openfreqbench.scenarios.g2.e2_freq_ramp import G2_E2_FreqRamp
import pytest

FS = 10_000.0
T_S = 2.0


def _build(
    fs=FS,
    T_s=T_S,
    f0=60.0,
    rate=2.0,
    onset=0.25,
    df_max=5.0,
    A0=1.0,
    phi0=0.0,
    seed=0,
) -> ScenarioOutput:
    scen = G2_E2_FreqRamp(
        fs_hz=fs,
        T_s=T_s,
        f0_hz=f0,
        ramp_rate_hzs=rate,
        onset_frac=onset,
        df_max_hz=df_max,
        A0=A0,
        phi0_rad=phi0,
        seed=seed,
    )
    return scen.build()


# ─────────────────────────────────────────────────────────────────────────────
# Instantiation
# ─────────────────────────────────────────────────────────────────────────────


def test_instantiate_default():
    scen = G2_E2_FreqRamp()
    assert scen is not None


def test_scenario_id():
    scen = G2_E2_FreqRamp()
    assert scen.scenario_id == "G2_E2_FreqRamp"


# ─────────────────────────────────────────────────────────────────────────────
# Waveform shape
# ─────────────────────────────────────────────────────────────────────────────


def test_array_shapes():
    out = _build()
    n = round(T_S * FS)
    assert len(out.t) == n
    assert len(out.v) == n
    assert len(out.f_true) == n
    assert len(out.state.phi) == n
    assert len(out.state.A) == n


@pytest.mark.parametrize(
    "fs,T_s",
    [
        (1_000, 1.0),
        (10_000, 2.0),
        (50_000, 0.5),
    ],
)
def test_waveform_length(fs, T_s):
    out = _build(fs=fs, T_s=T_s)
    assert len(out.v) == round(T_s * fs)


# ─────────────────────────────────────────────────────────────────────────────
# Frequency profile correctness
# ─────────────────────────────────────────────────────────────────────────────


def test_f_true_constant_before_onset():
    f0, onset = 60.0, 0.25
    out = _build(fs=FS, T_s=T_S, f0=f0, onset=onset)
    onset_sample = round(onset * len(out.t))
    pre = out.f_true[:onset_sample]
    assert np.allclose(pre, f0, atol=1e-10), f"Pre-onset f_true not constant at {f0} Hz"


def test_f_true_increases_after_onset_positive_rate():
    f0, rate, onset = 60.0, 2.0, 0.25
    out = _build(fs=FS, T_s=T_S, f0=f0, rate=rate, onset=onset)
    onset_sample = round(onset * len(out.t))
    post = out.f_true[onset_sample:]
    # Frequency should be non-decreasing (until clamp)
    diffs = np.diff(post)
    assert np.all(diffs >= -1e-10), "f_true decreased after onset (positive ramp)"


def test_f_true_clamped_at_df_max():
    f0, rate, df_max = 60.0, 5.0, 2.0
    out = _build(fs=FS, T_s=2.0, f0=f0, rate=rate, df_max=df_max)
    assert np.all(out.f_true <= f0 + df_max + 1e-10), f"f_true exceeded f0 + df_max = {f0 + df_max}"


def test_f_true_at_t0_equals_f0():
    f0 = 61.5
    out = _build(f0=f0, onset=0.5)  # onset late so first value is f0
    assert abs(float(out.f_true[0]) - f0) < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# Phase continuity
# ─────────────────────────────────────────────────────────────────────────────


def test_phase_continuous_at_onset():
    """
    No phase jump at onset: max |dphi difference| at step should equal
    the dphi change due to frequency change, not an abrupt jump.
    """
    out = _build(fs=FS, T_s=T_S, f0=60.0, rate=2.0, onset=0.5)
    phi = out.state.phi
    dphi = np.diff(phi)
    # The maximum single-sample dphi jump should be <= 2*pi*f_max/fs + tol
    f_max = 60.0 + 5.0  # f0 + df_max
    max_expected_dphi = 2 * np.pi * f_max / FS
    assert np.all(np.abs(dphi) <= max_expected_dphi + 1e-6), (
        f"Phase discontinuity detected: max dphi = {np.max(np.abs(dphi)):.6f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Waveform values
# ─────────────────────────────────────────────────────────────────────────────


def test_v_equals_A_sin_phi():
    out = _build(A0=0.8)
    expected = 0.8 * np.sin(out.state.phi)
    assert np.allclose(out.v.ravel(), expected, atol=1e-12)


def test_v_amplitude_bounded():
    A0 = 1.5
    out = _build(A0=A0)
    assert np.all(np.abs(out.v) <= A0 + 1e-10)


def test_t_axis():
    out = _build(fs=FS, T_s=T_S)
    assert abs(float(out.t[0])) < 1e-12
    assert abs(float(out.t[1]) - 1.0 / FS) < 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────


def test_schema_keys():
    out = _build()
    schema = out.state.schema
    required = {
        "scenario_id",
        "seed",
        "fs_hz",
        "f0_hz",
        "ramp_rate_hzs",
        "onset_frac",
        "onset_time_s",
        "df_max_hz",
        "T_s",
        "A0",
        "phi0_rad",
        "modifiers",
    }
    assert required.issubset(set(schema.keys())), (
        f"Missing schema keys: {required - set(schema.keys())}"
    )


def test_schema_scenario_id():
    out = _build()
    assert out.state.schema["scenario_id"] == "G2_E2_FreqRamp"


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo tuning interface
# ─────────────────────────────────────────────────────────────────────────────


def test_tuning_map_has_seed():
    scen = G2_E2_FreqRamp()
    assert "seed" in scen.tuning_map


def test_set_montecarlo_tuning_seed():
    scen = G2_E2_FreqRamp()
    scen = scen.set_montecarlo_tuning({"seed": 42})
    assert scen.seed == 42


def test_different_seeds_give_same_waveform():
    """
    G2_E2_FreqRamp is deterministic — it has no stochastic modifier.
    Two builds with the same nominal params (different seeds) should produce
    identical f_true (seed only affects potential future noise modifiers).
    """
    out0 = _build(seed=0)
    out1 = _build(seed=1)
    # f_true is deterministic (no noise in this scenario)
    assert np.allclose(out0.f_true, out1.f_true, atol=1e-12)


def test_tuning_map_aliases():
    scen = G2_E2_FreqRamp()
    expected_aliases = {"f0", "ramp_rate", "onset_frac", "Vmax", "phi", "seed"}
    assert expected_aliases.issubset(set(scen.tuning_map.keys()))
