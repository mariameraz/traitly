# traitly/__init__.py
"""
Traitly: Phenotyping analysis of fruits in images using computer vision.
"""
import re
from importlib.metadata import version, metadata, PackageNotFoundError

try:
    _meta = metadata("traitly")
    __version__ = version("traitly")
    
    _raw = _meta.get("Author-email") or ""
    # Separar por coma fuera de los < >
    _entries = re.split(r",\s*(?=[^<>]*(?:<|$))", _raw)
    __authors__ = [
        re.match(r"^(.*?)\s*<", a).group(1).strip()
        for a in _entries if "<" in a
    ]
    __author__ = ", ".join(__authors__)

except PackageNotFoundError:
    __version__ = "unknown"
    __authors__ = []
    __author__ = "unknown"


# Import functions from utils
from .utils.basic_functions import (
    load_img,
    plot_img
)

__all__ = [
    '__version__',
    '__author__',
    '__authors__',
    'load_img',
    'plot_img'
]