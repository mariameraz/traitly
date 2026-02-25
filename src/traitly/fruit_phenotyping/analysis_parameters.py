# traitly/fruit_phenotyping/analysis_parameters.py
"""
Analysis metadata tracking for traitly `FruitInternalAnalyzer` and 
`FruitExternalAnalyzer` pipelines.

Provides the :class:`AnalysisParameters` dataclass for capturing and
exporting the processing parameters used in each stage of the fruit
phenotyping pipeline, supporting reproducibility and traceability.
"""

# ============================================================================
# STANDARD LIBRARY
# ============================================================================
from dataclasses import dataclass, field, asdict
from typing import Dict, Any
from datetime import datetime
import importlib.metadata
import sys
import json

# ============================================================================
# INTERNAL IMPORTS
# ============================================================================
from traitly import __version__

# ============================================================================
# Save parameters used and session information as txt and json files
# ============================================================================

@dataclass
class AnalysisParameters:
    """
    Save and report processing parameters for each analysis step.

    Captures the parameters used in each stage of the analysis pipeline
    and provides methods to export them as formatted text or JSON for
    reproducibility and traceability.

    Attributes
    ----------
    setup_measurements_params : Dict[str, Any]
        Parameters used in the setup measurements step.
    generate_fruit_mask_params : Dict[str, Any]
        Parameters used in the fruit mask generation step.
    enhance_locule_contrast_params : Dict[str, Any]
        Parameters used in the locule contrast enhancement step.
    generate_locule_mask_params : Dict[str, Any]
        Parameters used in the locule mask generation step.
    detect_fruits_params : Dict[str, Any]
        Parameters used in the fruit detection step.
    analyze_morphology_params : Dict[str, Any]
        Parameters used in the morphology analysis step.
    analyze_color_params : Dict[str, Any]
        Parameters used in the color analysis step.
    """
    # Create a dictionary for each step
    setup_measurements_params: Dict[str, Any] = field(default_factory=dict)
    generate_fruit_mask_params: Dict[str, Any] = field(default_factory=dict)
    enhance_locule_contrast_params: Dict[str, Any] = field(default_factory=dict)
    generate_locule_mask_params: Dict[str, Any] = field(default_factory=dict)
    detect_fruits_params: Dict[str, Any] = field(default_factory=dict)
    analyze_morphology_params: Dict[str, Any] = field(default_factory=dict)
    analyze_color_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert dataclass fields to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of all analysis parameter fields.
        """
        return asdict(self)
    
    
    def to_formatted_string(self) -> str:
        """
        Return analysis parameters as a readable formatted string.

        Includes version info, run date, per-step parameters, and
        dependency versions.

        Returns
        -------
        str
            Formatted string with all processing parameters and metadata.
        """

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
        """ 
        Save formatted parameters to a plain text file.

        Parameters
        ----------
        filepath : str
            Destination path for the output .txt file.

        Raises
        ------
        OSError
            If the file cannot be created or written to ``filepath``.
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_formatted_string())

    
    def save_to_json(self, filepath: str) -> None:
        """
        Save parameters to a JSON file.

        Parameters
        ----------
        filepath : str
            Destination path for the output .json file.

        Raises
        ------
        OSError
            If the file cannot be created or written to ``filepath``.
        """
        data = self.to_dict()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def get_package_versions(self) -> Dict[str, str]:
        """
        Return installed versions of key dependencies used by traitly.

        Returns
        -------
        Dict[str, str]
            Dictionary mapping package names to their installed version
            strings. Includes Python version under the key ``'python'``.
            Packages not found are reported as ``'not installed'``.
        """
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


