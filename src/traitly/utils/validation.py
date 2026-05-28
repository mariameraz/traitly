# traitly/utils/validators.py
import numpy as np

def _validate_bgr_image(img):
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
        raise ValueError("Image cannot be None")

    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected a NumPy image array (np.ndarray), but got {type(img).__name__}. "
                "Make sure to load the image first.")

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(
            f"Expected a 3-channel image with shape (H, W, 3), but got shape {img.shape}. "
            "Make sure the image is not grayscale or RGBA."
        )
