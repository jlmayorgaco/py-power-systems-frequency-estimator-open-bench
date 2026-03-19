"""
openfreqbench/estimators/registry.py

EstimatorRegistry: name → class mapping with auto-registration of built-ins.

Built-in estimators by family
──────────────────────────────
  common/
    Baseline_Passthrough

  monophasic/f0_pll/
    SOGI_FLL · SRF_PLL

  monophasic/f1_kalman/
    EKF_Freq · RAEKF

  monophasic/f2_window/
    FFTPeak · IpDFT

  monophasic/f3_recursive/
    ZeroCrossing · RDFT

  (stubs — registered but outputs nominal frequency with valid=False)
  monophasic/f1_kalman/
    UKF

  monophasic/f2_window/
    TFT

  monophasic/f3_recursive/
    RLS · RLS_VFF

  monophasic/f4_data_driven/
    Koopman_RKDPMU · PI_GRU

  (scaffolds — class skeletons, raise NotImplementedError, valid=False)
  monophasic/f0_pll/
    DDSRF_PLL · DSOGI_FLL · EPLL · ANF

  monophasic/f1_kalman/
    LKF · CKF · Adaptive_EKF

  monophasic/f2_window/
    Goertzel · DynamicPhasor

  monophasic/f3_recursive/
    IZC · WLS

  monophasic/f5_parametric/
    Prony · MUSIC · ESPRIT

  monophasic/f6_time_frequency/
    Hilbert_IF

  monophasic/f4_data_driven/ (ML scaffolds)
    GRU_Regressor · LSTM_Regressor · BiLSTM_Regressor · TemporalCNN
    TCN_Freq · Transformer_Freq · EchoState_Reservoir · MLP_Window
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from openfreqbench.estimators.common.base import BaseEstimator


class EstimatorRegistry:
    _registry: Dict[str, Type[BaseEstimator]] = {}

    @classmethod
    def register(cls, estimator_cls: Type[BaseEstimator]) -> Type[BaseEstimator]:
        """Register an estimator class by its SPEC.name."""
        cls._registry[estimator_cls.SPEC.name] = estimator_cls
        return estimator_cls

    @classmethod
    def get(cls, name: str) -> Type[BaseEstimator]:
        if name not in cls._registry:
            raise KeyError(
                f"Estimator {name!r} not registered. "
                f"Available: {sorted(cls._registry)}"
            )
        return cls._registry[name]

    @classmethod
    def build(
        cls,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        # backward-compat alias
        params: Optional[Dict[str, Any]] = None,
    ) -> BaseEstimator:
        """Instantiate a registered estimator with an optional config dict."""
        merged = {**(params or {}), **(config or {})}
        return cls.get(name)(config=merged or None)

    @classmethod
    def list_names(cls) -> List[str]:
        return sorted(cls._registry)

    @classmethod
    def list_descriptors(cls) -> List[Dict[str, Any]]:
        """Full descriptor dict for every registered estimator."""
        return [cls._registry[n].descriptor() for n in sorted(cls._registry)]

    @classmethod
    def list_by_family(cls) -> Dict[str, List[str]]:
        """Names grouped by SPEC.family_path."""
        out: Dict[str, List[str]] = {}
        for name in sorted(cls._registry):
            klass = cls._registry[name]
            fp    = klass.SPEC.family_path or "other"
            out.setdefault(fp, []).append(name)
        return out


# ── Built-in registrations (full implementations) ─────────────────────────────

from openfreqbench.estimators.common.baseline_passthrough import BaselinePassthrough      # noqa: E402
from openfreqbench.estimators.monophasic.f0_pll.sogi_fll  import SOGIFLLEstimator        # noqa: E402
from openfreqbench.estimators.monophasic.f0_pll.srf_pll   import SRFPLLEstimator         # noqa: E402
from openfreqbench.estimators.monophasic.f1_kalman.ekf_freq import EKFFreqEstimator      # noqa: E402
from openfreqbench.estimators.monophasic.f1_kalman.raekf   import RAEKFEstimator          # noqa: E402
from openfreqbench.estimators.monophasic.f2_window.fft_peak import FFTPeakEstimator      # noqa: E402
from openfreqbench.estimators.monophasic.f2_window.ipdft    import IpDFTEstimator        # noqa: E402
from openfreqbench.estimators.monophasic.f3_recursive.zero_crossing import ZeroCrossingEstimator  # noqa: E402
from openfreqbench.estimators.monophasic.f3_recursive.rdft  import RDFTEstimator         # noqa: E402

EstimatorRegistry.register(BaselinePassthrough)
EstimatorRegistry.register(SOGIFLLEstimator)
EstimatorRegistry.register(SRFPLLEstimator)
EstimatorRegistry.register(EKFFreqEstimator)
EstimatorRegistry.register(RAEKFEstimator)
EstimatorRegistry.register(FFTPeakEstimator)
EstimatorRegistry.register(IpDFTEstimator)
EstimatorRegistry.register(ZeroCrossingEstimator)
EstimatorRegistry.register(RDFTEstimator)

# ── Stub registrations (algorithm skeletons — output valid=False) ──────────────
from openfreqbench.estimators.monophasic.f1_kalman.ukf         import UKFEstimator             # noqa: E402
from openfreqbench.estimators.monophasic.f2_window.tft         import TFTEstimator             # noqa: E402
from openfreqbench.estimators.monophasic.f3_recursive.rls      import RLSEstimator             # noqa: E402
from openfreqbench.estimators.monophasic.f3_recursive.rls_vff  import RLSVFFEstimator          # noqa: E402
from openfreqbench.estimators.monophasic.f4_data_driven.koopman_rkdpmu import KoopmanRKDPMUEstimator  # noqa: E402
from openfreqbench.estimators.monophasic.f4_data_driven.pi_gru import PIGRUEstimator          # noqa: E402

EstimatorRegistry.register(UKFEstimator)
EstimatorRegistry.register(TFTEstimator)
EstimatorRegistry.register(RLSEstimator)
EstimatorRegistry.register(RLSVFFEstimator)
EstimatorRegistry.register(KoopmanRKDPMUEstimator)
EstimatorRegistry.register(PIGRUEstimator)

# ── Scaffold registrations (skeletons — raise NotImplementedError, valid=False) ──
from openfreqbench.estimators.monophasic.f0_pll.ddsrf_pll        import DDSRFPLLEstimator          # noqa: E402
from openfreqbench.estimators.monophasic.f0_pll.dsogi_fll        import DSOGIFLLEstimator          # noqa: E402
from openfreqbench.estimators.monophasic.f0_pll.epll             import EPLLEstimator              # noqa: E402
from openfreqbench.estimators.monophasic.f0_pll.anf              import ANFEstimator               # noqa: E402
from openfreqbench.estimators.monophasic.f1_kalman.linear_kalman import LinearKalmanEstimator      # noqa: E402
from openfreqbench.estimators.monophasic.f1_kalman.ckf           import CKFEstimator               # noqa: E402
from openfreqbench.estimators.monophasic.f1_kalman.adaptive_ekf  import AdaptiveEKFEstimator       # noqa: E402
from openfreqbench.estimators.monophasic.f2_window.goertzel      import GoertzelEstimator          # noqa: E402
from openfreqbench.estimators.monophasic.f2_window.dynamic_phasor import DynamicPhasorEstimator   # noqa: E402
from openfreqbench.estimators.monophasic.f3_recursive.interp_zero_crossing import InterpolatedZeroCrossingEstimator  # noqa: E402
from openfreqbench.estimators.monophasic.f3_recursive.windowed_ls import WindowedLeastSquaresEstimator  # noqa: E402
from openfreqbench.estimators.monophasic.f5_parametric.prony     import PronyEstimator             # noqa: E402
from openfreqbench.estimators.monophasic.f5_parametric.music     import MUSICEstimator             # noqa: E402
from openfreqbench.estimators.monophasic.f5_parametric.esprit    import ESPRITEstimator            # noqa: E402
from openfreqbench.estimators.monophasic.f6_time_frequency.hilbert_freq import HilbertFrequencyEstimator  # noqa: E402
from openfreqbench.estimators.monophasic.f4_data_driven.gru_regressor      import GRURegressorEstimator          # noqa: E402
from openfreqbench.estimators.monophasic.f4_data_driven.lstm_regressor     import LSTMRegressorEstimator         # noqa: E402
from openfreqbench.estimators.monophasic.f4_data_driven.bilstm_regressor   import BiLSTMRegressorEstimator       # noqa: E402
from openfreqbench.estimators.monophasic.f4_data_driven.temporal_cnn       import TemporalCNNEstimator           # noqa: E402
from openfreqbench.estimators.monophasic.f4_data_driven.tcn_frequency      import TCNFrequencyEstimator          # noqa: E402
from openfreqbench.estimators.monophasic.f4_data_driven.transformer_regressor import TransformerRegressorEstimator  # noqa: E402
from openfreqbench.estimators.monophasic.f4_data_driven.reservoir_echo     import ReservoirEchoStateEstimator    # noqa: E402
from openfreqbench.estimators.monophasic.f4_data_driven.mlp_window         import MLPWindowRegressorEstimator    # noqa: E402

EstimatorRegistry.register(DDSRFPLLEstimator)
EstimatorRegistry.register(DSOGIFLLEstimator)
EstimatorRegistry.register(EPLLEstimator)
EstimatorRegistry.register(ANFEstimator)
EstimatorRegistry.register(LinearKalmanEstimator)
EstimatorRegistry.register(CKFEstimator)
EstimatorRegistry.register(AdaptiveEKFEstimator)
EstimatorRegistry.register(GoertzelEstimator)
EstimatorRegistry.register(DynamicPhasorEstimator)
EstimatorRegistry.register(InterpolatedZeroCrossingEstimator)
EstimatorRegistry.register(WindowedLeastSquaresEstimator)
EstimatorRegistry.register(PronyEstimator)
EstimatorRegistry.register(MUSICEstimator)
EstimatorRegistry.register(ESPRITEstimator)
EstimatorRegistry.register(HilbertFrequencyEstimator)
EstimatorRegistry.register(GRURegressorEstimator)
EstimatorRegistry.register(LSTMRegressorEstimator)
EstimatorRegistry.register(BiLSTMRegressorEstimator)
EstimatorRegistry.register(TemporalCNNEstimator)
EstimatorRegistry.register(TCNFrequencyEstimator)
EstimatorRegistry.register(TransformerRegressorEstimator)
EstimatorRegistry.register(ReservoirEchoStateEstimator)
EstimatorRegistry.register(MLPWindowRegressorEstimator)
