# traitly/fruit_phenotyping/analysis_parameters.py
# ============================================================================
# STANDARD LIBRARY
# ============================================================================
from dataclasses import dataclass, field, asdict
from typing import Dict, Any
from datetime import datetime
import importlib.metadata
import sys

# ============================================================================
# INTERNAL IMPORTS
# ============================================================================
from traitly import __version__

# ============================================================================
# Save parameters used and session information as txt and json files
# ============================================================================

@dataclass
class AnalysisMetadata:
    # Create a dictionary for each step
    setup_measurements_params: Dict[str, Any] = field(default_factory=dict)
    generate_fruit_mask_params: Dict[str, Any] = field(default_factory=dict)
    enhance_locule_contrast_params: Dict[str, Any] = field(default_factory=dict)
    generate_locule_mask_params: Dict[str, Any] = field(default_factory=dict)
    detect_fruits_params: Dict[str, Any] = field(default_factory=dict)
    analyze_morphology_params: Dict[str, Any] = field(default_factory=dict)
    analyze_color_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert dicts to easier to read format"""
        return asdict(self)
    
    def to_formatted_string(self) -> str:
        """Return parameters as formated strings."""

        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        lines = [
            "=" * 70,
            "ANALYSIS PROCESSING PARAMETERS",
            "=" * 70,
            f"traitly: v{__version__}",
            f"run date: {date}"
        ]
        
        # Setup Measurements
        if self.setup_measurements_params:
            lines.append("\nSETUP_MEASUREMENTS:")
            for key, value in self.setup_measurements_params.items():
                lines.append(f"   - {key}: {value}")
        
        # Generate Fruit Mask
        if self.generate_fruit_mask_params:
            lines.append("\nGENERATE_FRUIT_MASK:")
            for key, value in self.generate_fruit_mask_params.items():
                lines.append(f"   - {key}: {value}")
        
        # Enhance Locule Contrast
        if self.enhance_locule_contrast_params:
            lines.append("\nENHANCE_LOCULE_CONTRAST:")
            for key, value in self.enhance_locule_contrast_params.items():
                lines.append(f"   - {key}: {value}")
        
        # Generate Locule Mask
        if self.generate_locule_mask_params:
            lines.append("\nGENERATE_LOCULE_MASK:")
            for key, value in self.generate_locule_mask_params.items():
                lines.append(f"   - {key}: {value}")
        
        # Detect Fruits
        if self.detect_fruits_params:
            lines.append("\nDETECT_FRUITS:")
            for key, value in self.detect_fruits_params.items():
                lines.append(f"   - {key}: {value}")
        
        # Analyze Morphology
        if self.analyze_morphology_params:
            lines.append("\nANALYZE_MORPHOLOGY:")
            for key, value in self.analyze_morphology_params.items():
                lines.append(f"   - {key}: {value}")
        
        # Analyze Color
        if self.analyze_color_params:
            lines.append("\nANALYZE_COLOR:")
            for key, value in self.analyze_color_params.items():
                lines.append(f"   - {key}: {value}")
        
        lines.append("\n" + "=" * 70)

        # Version summary
        lines.append("\nDEPENDENCIES:")
        for pkg, version in self.get_package_versions().items():
            lines.append(f"   - {pkg}: {version}")

        return "\n".join(lines)
    
    def save_to_file(self, filepath: str) -> None:
        """Save parameters in .txt."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_formatted_string())

    
    def save_to_json(self, filepath: str) -> None:
        """Save parameters in .json """
        import json
        data = self.to_dict()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def get_package_versions(self) -> Dict[str, str]:
        """Get versions of key packages used in traitly."""
        packages = [
              "opencv-contrib-python",
            "numpy",
            "pandas",
            "scipy",
            "matplotlib",
            "tqdm",
            "psutil",
            "easyocr",
            "PyMuPDF",
            "ultralytics"
        ]
        versions = {'python': sys.version.split()[0]}
        for pkg in packages:
            try:
                versions[pkg] = importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError:
                versions[pkg] = 'not installed'
        return versions


