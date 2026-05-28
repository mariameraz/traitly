# traitly/color_correction/color_checker.py

# ============================================================================
# STANDARD LIBRARY
# ============================================================================
from typing import Tuple, Optional, TypedDict, List
import warnings
import re
# ============================================================================
# THIRD-PARTY LIBRARIES
# ============================================================================
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    MaxAbsScaler,
    PolynomialFeatures
)

from sklearn.cross_decomposition import PLSRegression

# ============================================================================
# INTERNAL
# ============================================================================
from traitly.color_correction.color_charts import CHECKER_LAB_D50, CHECKER_PATCH_NAMES
from traitly.utils.validation import _validate_color_image

#############################################################
## Detect color checker
#############################################################
## First, verify cv2 mcc detector is available
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
        - chart : np.ndarray of shape (72, 5) with color data for each patch (24 patches).
          Columns are [n_pixels, mean, std, min, max].
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
    _validate_color_image(img)

    # Working only for MCC 24 patches card for now
    # Important: detector expects a BGR image according with cv2 docs
    detector.process(img, cv2.mcc.MCC24)
    checkers = detector.getListColorChecker()

    #Draw color patches detected
    img_copy = img.copy()
    cdrawer = cv2.mcc.CCheckerDraw.create(checkers[0])
    cdrawer.draw(img_copy)

    if not checkers:
        warnings.warn("Color checker not detected.", UserWarning)
        return None

    checker = checkers[0]

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
        patch = img_copy[checker_coords['y1']:checker_coords['y2'],
            checker_coords['x1']:checker_coords['x2']]

        plt.figure(figsize = plot_size)
        plt.imshow(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Color checker detected")
        plt.show()

    # Save the LAB mean value for each color patch
    chart = checker.getChartsRGB()



    return checker_coords, chart, img_copy

########################################################################
## Extract patch colors from charts and convert them from BGR to LAB   #
########################################################################
def _get_lab_patches(
    detected_chart: np.ndarray,
) -> np.ndarray:

    # Obtain mean colors from the chart array
    means = detected_chart[:, 1]

    # reshape from (72, ) to (1, 24, 3)
    rgb_patches = means.reshape(24, 3).astype(np.float32) / 255

    # reorder rgb to bgr
    bgr_patches = rgb_patches[:, ::-1]

    # Convert RGB to LAB
    lab_patches = cv2.cvtColor(bgr_patches[np.newaxis], cv2.COLOR_BGR2Lab)[0]
    return lab_patches.astype(np.float32)


##############################
## Convert BGR image to LAB  #
##############################

def _img_bgr_to_lab(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    return cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

###########################
# Color correction (PLSR) #
###########################

def _fit_plsr_models(
    detected_lab: np.ndarray,
    reference_lab: np.ndarray = CHECKER_LAB_D50,
    degree: int = 3,
    num_components: int = 11,
    max_iterations: int = 1000,
    scaler = StandardScaler(),
) -> list:
    """
    Adjust 3 PLSR models, each for every LAB channel (L, a, b).

    Parameters
    ----------
    detected_lab : np.ndarray
        Lab values for patches detected from the color checker of shape (24, 3)
    reference_lab : np.ndarray
        Default CHECKER_LAB_D50 of shape (24, 3)
    degree : int
        Polynomial degree (default 3)
    num_components : int
        PLS components (default 11)
    max_iterations : int
        Max iterations for PLSRegression (default 1000)
    scaler : sklearn scaler, optional
        Scaler to use before polynomial expansion. Default is StandardScaler().
        Options are: RobustScaler(), StandardScaler(), MaxAbsScaler(), MinMaxScaler().
        Usage details are in the official documentation of sklearn.preprocessing

    Examples
        --------
        >>> from sklearn.preprocessing import RobustScaler, StandardScaler
        >>> # default
        >>> models = color_correction(detected_lab)
        >>> # with RobustScaler
        >>> models = color_correction(detected_lab, scaler=RobustScaler())
        >>> # with custom parameters
        >>> models = color_correction(detected_lab, scaler=RobustScaler(quantile_range=(25, 75)))
        >>> models = color_correction(detected_lab, scaler=StandardScaler(with_mean=False))
    """

    models = []

    # fit a model for each LAB channel:
    for i in range(3):
        model = make_pipeline(
            scaler, # RobustScaler(), StandardScaler(), MaxAbsScaler(), MinMaxScaler()
            PolynomialFeatures(degree=degree),
            PLSRegression(n_components=num_components, max_iter=max_iterations)
        )
        model.fit(detected_lab, reference_lab[:, i])
        models.append(model)

    return models

##################################
# Convert LAB image to BGR again #
##################################
def _lab_to_bgr(img_lab: np.ndarray) -> np.ndarray:
    lab_clipped = img_lab.copy()
    lab_clipped[..., 0] = np.clip(lab_clipped[..., 0], 0, 100)
    lab_clipped[..., 1] = np.clip(lab_clipped[..., 1], -127, 127)
    lab_clipped[..., 2] = np.clip(lab_clipped[..., 2], -127, 127)

    img_bgr = cv2.cvtColor(lab_clipped, cv2.COLOR_Lab2BGR)

    # use clip to avoid out of range values due conversion
    img_bgr = np.clip(img_bgr * 255, 0, 255).astype(np.uint8)

    return img_bgr


############################################
# Adjust color correction using the models #
############################################

def _apply_color_correction(
    img_lab: np.ndarray,
    models: List
) -> np.ndarray:

    h, w = img_lab.shape[:2]
    flat = img_lab.reshape(-1, 3)

    corrected_lab = np.zeros_like(flat)

    for i in range(3):
        corrected_lab[:, i] = models[i].predict(flat)

    corrected_bgr = _lab_to_bgr(corrected_lab.reshape(h,w, 3))

    return corrected_bgr

##########################################################################
## Calculate delta e value between detected colors and the lab reference
##########################################################################
def _delta_e(
    detected_lab: np.ndarray,
    reference_lab: np.ndarray = CHECKER_LAB_D50,
) -> np.ndarray:
    """
    Get Delta E  for each patch between the detected and the reference LAB values.

    Parameters
    ----------
    detected_lab : np.ndarray
        LAB values of shape (24, 3).
    reference_lab : np.ndarray
        Reference LAB values of shape (24, 3). Default is CHECKER_LAB_D50.

    Returns
    -------
    np.ndarray of shape (24,) with Delta E per patch.
    """
    diff = detected_lab - reference_lab
    return np.sqrt((diff ** 2).sum(axis=1))

def _delta_e_stats(
    corrected_img: np.ndarray,
    detected_lab: Optional[np.ndarray] = None,
    original_img: Optional[np.ndarray] = None,
    verbose: bool = True,
    reference_lab: np.ndarray = CHECKER_LAB_D50,
) -> None:
    if detected_lab is not None:
        # Get delta E before correction (original image)
        delta_e_before = _delta_e(detected_lab)
    else:
        if original_img is not None:
            _, chart_before, _ = _detect_color_checker(original_img.copy(), verbose=False)
            detected_lab_before = _get_lab_patches(chart_before)
            delta_e_before = _delta_e(detected_lab_before)

    # Get delta E after correction
    _, chart_after, _ = _detect_color_checker(corrected_img.copy(), verbose=False)
    detected_lab_after = _get_lab_patches(chart_after)
    delta_e_after = _delta_e(detected_lab_after)

    if verbose:
        print("-" * 55)
        print(f"Mean ΔE before: {delta_e_before.mean():.2f}")
        print(f"Mean ΔE after: {delta_e_after.mean():.2f}")
        print("-" * 55)
        print(f"{'Patch':<17} | ΔE before | ΔE after | Diff")
        print("-" * 55)
        for i, name in enumerate(CHECKER_PATCH_NAMES):
            diff = delta_e_before[i] - delta_e_after[i]
            print(f"{name:<17} | {delta_e_before[i]:9.2f} | {delta_e_after[i]:8.2f} | {diff:6.2f}")

    # Get only patch code
    patch_codes = [re.search(r"[A-F][1-4]", name).group() for name in CHECKER_PATCH_NAMES]

    df = np.column_stack([
            patch_codes,
            delta_e_before,
            delta_e_after,
            delta_e_before - delta_e_after
        ])
    return df
