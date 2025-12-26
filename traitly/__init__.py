# traitly/__init__.py

"""
Traitly: Morphological analysis of fruits in images using computer vision.
"""

__version__ = "0.1.0"
__author__ = "Maria Alejandra Torres Meraz"

# Import functions from utils
from .utils.common_functions import (
    load_img, 
    detect_label_text, 
    detect_img_name, 
    plot_img, 
    pdf_to_img, 
    is_contour_valid,
    validate_dir, 
    detect_qr, 
    detect_size_ref_yolo, 
    px_cm_density, 
    detect_label_box
)

valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}

# Import the main class 
#from .internal_structure.analyzing_image import AnalyzingImage  

# Import core functions
#from .internal_structure import core


__all__ = [
    
    # Version
    '__version__',
    '__author__',

    # Constants
    'valid_extensions',
    
    # Classes
    #'AnalyzingImage', 
    
    # Util functions
    'load_img', 
    'detect_label_text',
    'detect_img_name', 
    'plot_img', 
    'pdf_to_img', 
    'is_contour_valid',
    'validate_dir', 
    'detect_qr', 
    'detect_size_ref_yolo', 
    'px_cm_density',
    'detect_label_box',
    
    # Modules
    #'core'
]