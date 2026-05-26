# traitly/color_correction/color_checker.py

# ============================================================================
# STANDARD LIBRARY
# ============================================================================
from typing import Tuple, Optional, TypedDict
# ============================================================================
# THIRD-PARTY LIBRARIES
# ============================================================================
import numpy as np
import cv2
import matplotlib.pyplot as plt

#############################################################
## Detect color checker
#############################################################
## Verify cv2 mcc detector is available
try:
    detector = cv2.mcc.CCheckerDetector.create()
    _MCC_AVAILABLE = True
except AttributeError:
    warnings.warn(
        "Color checker detection is not available with this version of OpenCV. "
        "Install opencv-contrib-python>=4.9 to enable this feature.",
        UserWarning
    )
    _MCC_AVAILABLE = False
except Exception as e:
    warnings.warn(
        f"Color checker detection could not be initialized due to an unexpected error: {e}",
        UserWarning
    )
    _MCC_AVAILABLE = False

## TypedDic for _detect_color_checker docstring
class CheckerCoords(TypedDict):
    x1: int
    y1: int
    x2: int
    y2: int

def _detect_color_checker(
    img: np.ndarray,
    plot: bool = False,
    plot_size: Tuple[int, int] = (5, 5),
    verbose: bool = True,
) -> Optional[Tuple[CheckerCoords, np.ndarray]]:
    """
    Detect a Macbeth Color Checker (MCC24) in a BGR image.

    Only one checker is detected per image. If multiple checkers are present,
    only the first one detected is returned.

    Parameters
    ----------
    img : np.ndarray
        Input image in BGR format with shape (H, W, 3).
    plot : bool, optional
        If True, displays a crop of the detected checker. Default is False.
    plot_size : Tuple[int, int], optional
        Figure size for the plot. Default is (5, 5).
    verbose : bool, optional
        If True, prints detection details to stdout. Default is True.

    Returns
    -------
    Tuple[dict, np.ndarray] or None
        - checker_coords : dict with keys 'x1', 'y1', 'x2', 'y2' in original
          image coordinates.
        - charts : np.ndarray of shape (72, 5) with color data for each patch
          (24 patches x 3 channels). Columns are [n_pixels, mean, std, min, max].
        Returns None if the checker is not detected or MCC is not available.

    Warns
    -----
    UserWarning
        If the color checker is not detected in the image.
    """
    # fast return just in case the mcc detector is not available
    if not _MCC_AVAILABLE:
        checker_coords = None
        warnings.warn("MCC detector not available.", UserWarning)
        return None

    # Verify image input
    if img is None:
        raise ValueError("Image cannot be None")
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected a NumPy image array (np.ndarray), but got {type(img).__name__}. "
                "Make sure to load the image first.")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(
            f"Expected a 3-channel BGR image with shape (H, W, 3), but got shape {img.shape}. "
            "Make sure the image is not grayscale or RGBA."
        )

    # Working only for MCC 24 patches card for now
    # Important: detector expects a BGR image according with cv2 docs
    detector.process(img, cv2.mcc.MCC24)
    checkers = detector.getListColorChecker()

    if not checkers:
        warnings.warn("Color checker not detected.", UserWarning)
        return None

    checker = checkers[0]
    # Draw color patches detected
    cdrawer = cv2.mcc.CCheckerDraw.create(checkers[0])
    cdrawer.draw(img)

    # Save the color mean value for each color patch
    charts = checker.getChartsRGB()

    # Save the coords of the checker box
    box = checker.getBox()


    # Get a rotated bounding box around the checker card and save checker coords
    coords = cv2.boundingRect(np.int32(box))
    h,w = img.shape[:2]

    checker_coords = {
        "x1": max(0, coords[0]),
        "y1": max(0, coords[1]),
        "x2": min(w, coords[0] + coords[2]),
        "y2": min(h, coords[1] + coords[3])
    }

    if verbose:
        print("\n" + "=" * 55)
        print("★ COLOR CARD:")
        print("=" * 55)
        print("> Color checker detected: ")
        print(
            f"    - Coordinates: x1={checker_coords['x1']}, y1={checker_coords['y1']}, "
            f"x2={checker_coords['x2']}, y2={checker_coords['y2']}"
        )

    if plot:
        # Crop the checker region from image (it returns as BGR)
        patch = img[checker_coords['y1']:checker_coords['y2'],
            checker_coords['x1']:checker_coords['x2']]

        plt.figure(figsize = plot_size)
        plt.imshow(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Color checker detected")
        plt.show()

    return checker_coords, charts
