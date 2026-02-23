# traitly/utils/constants.py
"""
Package-level constants for traitly.

Defines accepted file extensions and valid string values for parameters
used across the analysis pipeline to allow centralised validation.
"""

# Supported input image formats for loading
valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}

# Supported output image formats for cv2.imwrite
valid_cv2_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Accepted fruit contour representation modes
valid_contours = {'raw', 'ellipse', 'approx', 'hull', 'circle'}

# Accepted text-label positions relative to the fruit bounding box
label_positions = {'right', 'bottom', 'left', 'top'}

