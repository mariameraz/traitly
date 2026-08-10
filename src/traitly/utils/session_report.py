# traitly/utils/session_report.py

"""
Analysis metadata tracking for `fruit_phenotyping` and `color_correction` pipelines.

Provides the :class:`AnalysisParameters` dataclass for capturing and
exporting the processing parameters used in each stage of the fruit
phenotyping pipeline, supporting reproducibility and traceability.
"""

# ============================================================================
# STANDARD LIBRARY
# ============================================================================
import json
from dataclasses import asdict, dataclass, field
from abc import abstractmethod
from typing import Any, Dict
import os

# ============================================================================
# INTERNAL IMPORTS
# ============================================================================
from traitly.utils.environment import (
    get_package_versions,
    get_system_metadata,
    get_session_metadata
)



@dataclass
class AnalysisParameters:
    """
    Base class for analysis parameter tracking and reporting.

    Subclasses must implement :meth:`_get_sections` to define which
    parameter dictionaries are included in the formatted report.
    """

    input_params: Dict[str, Any] = field(default_factory=dict)

    def _get_sections(self) -> Dict[str, Any]:
        """Return ordered dict of section title -> params dict."""
        ...

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
        input_path = self.input_params.get("input_path") if self.input_params else None

        lines = [
            "=" * 70,
            "ANALYSIS PROCESSING PARAMETERS",
            "=" * 70,
        ]

        lines += get_session_metadata(input_path)
        lines += get_system_metadata()

        for title, params in self._get_sections().items():
            if params:
                lines.append(f"\n{title}:")
                for key, value in params.items():
                    lines.append(f"   - {key}: {value}")

        lines.append("\n" + "=" * 70)
        lines.append("\nDEPENDENCIES:")
        for pkg, version in get_package_versions().items():
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
        with open(filepath, "w", encoding="utf-8") as f:
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
        class _SafeEncoder(json.JSONEncoder):
            """
            Convert any object into a string
            """
            def default(self, obj):
                try:
                    return super().default(obj)
                except TypeError:
                    return str(obj)

        data = self.to_dict()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, cls = _SafeEncoder)


def _save_parameters(input_path, parameters, output_path=None):
    if output_path is None:
        output_path = os.path.dirname(input_path)

    output_path = os.path.abspath(output_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    txt_path = os.path.join(output_path, f"{base_name}_parameters.txt")
    parameters.save_to_file(txt_path)

    json_path = os.path.join(output_path, f"{base_name}_parameters.json")
    parameters.save_to_json(json_path)

    print("\n> Parameters saved at:")
    print(f"  - TXT:  {txt_path}")
    print(f"  - JSON: {json_path}")
