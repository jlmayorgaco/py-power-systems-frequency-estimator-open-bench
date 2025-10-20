from typing import Any, Dict, Literal

from utils.pmu.pmu_input import PMU_Input
from utils.pmu.pmu_output import PMU_Output

class EstimatorBase:
    """Base class for all frequency estimators, enforcing PMU data contract."""
    
    def __init__(self, config: Any, name: str = "", profile: Literal["P", "M"] = "M") -> None:
        """
        Initializes the estimator with fixed parameters.
        :param fs: Sampling frequency (Hz).
        :param frame_len: Number of samples required for one estimation.
        """
        self.name: str = name
        self.profile: Literal["P", "M"] = profile
        self.memory: Dict[str, Any] = {}
        self.config: Dict[str, Any] = config   

    def reset(self) -> None:
        """Reset internal state (buffers, memory, accumulated sums, etc.)."""
        self.memory.clear()
        pass

    def update(self, measures: PMU_Input) -> PMU_Output:
        """
        Processes a single, time-tagged sample. The estimator must buffer 
        these samples internally until a full frame is available.
        """
        if not isinstance(measures, PMU_Input):
            raise TypeError("update() requires IEEEStandardPMU_Input, a single snapshot.")
            
        # Implementation in derived classes must handle buffering and processing.
        raise NotImplementedError