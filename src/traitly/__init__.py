# traitly/__init__.py

"""
Traitly: Phenotyping analysis of fruits in images using computer vision.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("traitly")
except PackageNotFoundError:
    __version__ = "unknown"

__author__ = "Maria Alejandra Torres Meraz"

# Import functions from utils
from .utils.basic_functions import (
    load_img, 
    plot_img, 
    detect_qr
)
from .utils.convert_pdf import pdf_to_img

__all__ = [
    
    # Version
    '__version__',
    '__author__',
        
    # Util functions
    'load_img', 
    'plot_img', 
    'pdf_to_img', 
    'detect_qr'

]