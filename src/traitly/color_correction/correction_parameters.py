from typing import Dict, Any
from dataclasses import asdict, dataclass, field
import os

from traitly.utils.session_report import AnalysisParameters

@dataclass
class ColorCorrectionParameters(AnalysisParameters):
    apply_color_correction_params: Dict[str, Any] = field(default_factory=dict)

    def _get_sections(self) -> Dict[str, Any]:
        return {
            "APPLY_COLOR_CORRECTION": self.apply_color_correction_params,
        }
