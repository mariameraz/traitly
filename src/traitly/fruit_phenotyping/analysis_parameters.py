# traitly/fruit_phenotyping/analysis_parameters.py
from typing import Dict, Any
from dataclasses import asdict, dataclass, field

from traitly.utils.session_report import AnalysisParameters

@dataclass
class FruitAnalyzerParameters(AnalysisParameters):
    setup_measurements_params: Dict[str, Any] = field(default_factory=dict)
    generate_fruit_mask_params: Dict[str, Any] = field(default_factory=dict)
    enhance_locule_contrast_params: Dict[str, Any] = field(default_factory=dict)
    generate_locule_mask_params: Dict[str, Any] = field(default_factory=dict)
    detect_fruits_params: Dict[str, Any] = field(default_factory=dict)
    analyze_morphology_params: Dict[str, Any] = field(default_factory=dict)
    analyze_color_params: Dict[str, Any] = field(default_factory=dict)

    def _get_sections(self) -> Dict[str, Any]:
        return {
            "SETUP_MEASUREMENTS": self.setup_measurements_params,
            "GENERATE_FRUIT_MASK": self.generate_fruit_mask_params,
            "ENHANCE_LOCULE_CONTRAST": self.enhance_locule_contrast_params,
            "GENERATE_LOCULE_MASK": self.generate_locule_mask_params,
            "DETECT_FRUITS": self.detect_fruits_params,
            "ANALYZE_MORPHOLOGY": self.analyze_morphology_params,
            "ANALYZE_COLOR": self.analyze_color_params,
        }
