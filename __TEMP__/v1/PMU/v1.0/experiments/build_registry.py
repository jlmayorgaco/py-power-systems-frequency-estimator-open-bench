# experiments/build_registry.py
from __future__ import annotations

from experiments.registry import MethodRegistry, MethodSpec


def build_default_registry() -> MethodRegistry:
    reg = MethodRegistry()

    reg.register(MethodSpec(name="IpDFT", family="Fourier", factory=factory_ipdft))
    reg.register(MethodSpec(name="PLL", family="PLL", factory=factory_pll))
    reg.register(MethodSpec(name="EKF", family="Kalman", factory=factory_ekf))
    reg.register(MethodSpec(name="EKF2", family="Kalman", factory=factory_ekf2))
    reg.register(MethodSpec(name="SOGI", family="SOGI-FLL", factory=factory_sogi))
    reg.register(MethodSpec(name="RLS", family="Adaptive RLS", factory=factory_rls))
    reg.register(
        MethodSpec(name="Teager", family="Nonlinear Energy", factory=factory_teager)
    )
    reg.register(MethodSpec(name="TFT", family="Time-Frequency", factory=factory_tft))
    reg.register(
        MethodSpec(name="RLS-VFF", family="Adaptive VFF-RLS", factory=factory_vff_rls)
    )
    reg.register(MethodSpec(name="UKF", family="Kalman (UKF)", factory=factory_ukf))
    reg.register(MethodSpec(name="LKF", family="Kalman (LKF)", factory=factory_lkf))
    reg.register(
        MethodSpec(name="Koopman-RKDPmu", family="Koopman", factory=factory_koopman)
    )
    reg.register(MethodSpec(name="PI-GRU", family="Neural", factory=factory_pigru))

    return reg
