# traitly/utils/validators.py

# ============================================================================
# STANDARD LIBRARY
# ============================================================================
from typing import Optional
import os
from pathlib import Path
import multiprocessing as mp

# ============================================================================
# THIRD-PARTY
# ============================================================================
import numpy as np

# ============================================================================
# INTERNAL
# ============================================================================
from traitly.utils.constants import valid_extensions

def _validate_color_image(
    img: np.ndarray,
    img_path: Optional[np.ndarray] = None):
    """
    Validate that an image is a non-null 3-channel NumPy array of shape (H, W, 3).

    Parameters
    ----------
    img : np.ndarray
        Input image as a NumPy array of shape (H, W, 3). Accepts any 3-channel format such as BGR, RGB, LAB, HSV.

    Raises
    ------
    ValueError
        If img is None or does not have shape (H, W, 3).
    TypeError
        If img is not a NumPy array.
    """

    if img is None:
        msg = f"Failed to load image: {img_path}." if img_path is not None else "Image cannot be None."
        raise ValueError(msg)

    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected a NumPy image array (np.ndarray), but got {type(img).__name__}. "
                "Make sure to load the image first.")

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(
            f"Expected a 3-channel image with shape (H, W, 3), but got shape {img.shape}. "
            "Make sure the image is not grayscale or RGBA."
        )

def _validate_path_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"The path does not exist: {path}\n"
            f"Verify that the file exists and the path is correct."
        )

def _validate_img_suffix(path):
    abs_path = Path(path)
    if abs_path.suffix.lower() not in valid_extensions:
        raise ValueError(
            f"No valid image format: '{abs_path.suffix.lower()}' -> "
            f"Supported formats are: {valid_extensions}"
        )

def _validate_num_cores(
    num_cores: Optional[int] = None
):
    max_cores = mp.cpu_count()

    if num_cores <= 0:
        num_cores_message = f"    > num_cores: {num_cores} must be at least 1. Using num_cores=1 instead."
        num_cores = 1
    elif num_cores > max_cores:
        num_cores_message = f"    > num_cores: {num_cores} exceeds system cores ({max_cores}). Using {max_cores} instead."
        num_cores = max_cores
    else:
        num_cores_message = None

    return num_cores, num_cores_message
