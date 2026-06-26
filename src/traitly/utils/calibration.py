# traitly/utils/calibration.py
"""
Scale calibration and size-reference detection utilities for traitly.

Provides functions to:

- Estimate pixel-to-centimetre density from known paper dimensions
  (:func:`_img_px_per_cm`) or from detected reference circles
  (:func:`diameter_px_per_cm`).
- Detect circular size-reference objects in an image using a YOLOv8 model
  (:func:`_detect_size_ref_yolo`, :func:`_find_size_ref_circles`).
- Load and cache YOLO models (:func:`_get_yolo_model`) and resolve their
  bundled paths (:func:`_get_package_model_path`).
- Combine all calibration methods into a single wrapper
  (:func:`px_cm_density`).
"""

# ============================================================================
# STANDARD LIBRARY
# ============================================================================
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from importlib.resources import files
import logging
logger = logging.getLogger(__name__)
# ===========================================================================
# THIRD-PARTY LIBRARIES
# ===========================================================================
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

# ============================================================================
# INTERNAL IMPORTS
# ===========================================================================
from .constants import label_positions, valid_cv2_extensions, valid_extensions

##############################################################################
# Calculate px density from physical image dimensions
##############################################################################
paper_sizes = {
    "letter_ansi": (21.6, 27.9),
    "legal_ansi": (21.59, 35.56),
    "a4_iso": (21.0, 29.7),
    "a3_iso": (29.7, 42.0),
}

def _img_px_per_cm(
    img: np.ndarray,
    size: str = "letter_ansi",
    width_cm: Optional[float] = None,
    length_cm: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    """
    Calculate pixel density from an image and known physical dimensions.

    Parameters
    ----------
    img : np.ndarray.
    size : str, optional
        Paper size preset: ``'letter_ansi'``, ``'legal_ansi'``,
        ``'a4_iso'``, or ``'a3_iso'``. Ignored when ``width_cm`` and
        ``length_cm`` are provided. Default is ``'letter_ansi'``.
    width_cm : float or None, optional
        Physical width of the image in centimetres. Default is ``None``.
    length_cm : float or None, optional
        Physical length (height) of the image in centimetres. Default is
        ``None``.

    Returns
    -------
    tuple of float
        ``(px_per_cm_width, px_per_cm_length, used_width_cm,
        used_length_cm)``.

    Raises
    ------
    TypeError
        If ``img`` is not a numpy array.
    ValueError
        If physical dimensions are non-positive, ``width_cm > length_cm``, or ``size`` is not
        a valid predefined option.
    RuntimeError
        If an unexpected calculation error occurs.
    """
    try:
        if size not in ["letter_ansi", "legal_ansi", "a4_iso", "a3_iso"] and (
            width_cm is None or length_cm is None
        ):
            raise ValueError("Provide either valid physical size or custom dimensions")
        if width_cm is not None and (
            not isinstance(width_cm, (int, float)) or width_cm <= 0
        ):
            raise ValueError("width_cm must be positive")
        if length_cm is not None and (
            not isinstance(length_cm, (int, float)) or length_cm <= 0
        ):
            raise ValueError("length_cm must be positive")
        if width_cm is not None and length_cm is not None and width_cm > length_cm:
            raise ValueError("width_cm cannot be greater than length_cm")

        # always h > w
        img_h = max(img.shape[0], img.shape[1])
        img_w = min(img.shape[0], img.shape[1])

        if width_cm is not None and length_cm is not None:
            used_w_cm = width_cm
            used_l_cm = length_cm
        else:
            paper_w, paper_h = paper_sizes[size]
            used_w_cm = min(paper_w, paper_h)
            used_l_cm = max(paper_w, paper_h)

        px_per_cm_w = img_w / used_w_cm
        px_per_cm_l = img_h / used_l_cm

        return px_per_cm_w, px_per_cm_l, used_w_cm, used_l_cm

    except Exception as e:
        raise RuntimeError(f"px_per_cm calculation error: {str(e)}")


##############################################################################
# Obtain px per cm density from the average diameter of reference circles
##############################################################################

def diameter_px_per_cm(
    list_circles: List[Tuple[int, int, int]],
    verbose: bool = False,
    diameter_cm: float = 2.5,
    std_threshold: int = 2,
) -> float:
    """
    Estimate pixel-to-centimetre density from detected circle diameters.

    Filters outlier diameters beyond ``std_threshold`` standard
    deviations from the mean, then computes the mean diameter of the
    remaining circles divided by ``diameter_cm``.

    Parameters
    ----------
    list_circles : list of tuple of int
        List of ``(cx, cy, diameter_px)`` tuples as returned by
        :func:`_detect_size_ref_yolo`.
    verbose : bool, optional
        If True, print filtering statistics and the final density.
        Default is False.
    diameter_cm : float, optional
        Known physical diameter of the reference circles in centimetres.
        Default is 2.5.
    std_threshold : int, optional
        Number of standard deviations used to define the outlier range.
        Default is 2.

    Returns
    -------
    float
        Estimated pixel-to-centimetre density.

    Raises
    ------
    ValueError
        If ``list_circles`` is empty.
    """
    if not list_circles:
        raise ValueError("No circles provided.")

    # Get the diameter values for all the circles in the list
    diameters = np.array([d[2] for d in list_circles], dtype=np.float32)

    # Calculate stats
    mean_val = np.mean(diameters)
    std_val = np.std(diameters)

    # Get outlier ranges from std
    lower = mean_val - std_threshold * std_val
    upper = mean_val + std_threshold * std_val

    keep = (diameters >= lower) & (diameters <= upper)
    filtered_circles = diameters[keep]

    if len(filtered_circles) == 0:
        if verbose:
            print("Warning: Using all the circles (many outliers detected)")
        filtered_circles = diameters

    px_per_cm = np.mean(filtered_circles) / diameter_cm

    if verbose:
        print(
            f"            Circles used: {len(filtered_circles)}/{len(diameters)} (std > {std_threshold})"
        )
        print(
            f"            Mean diameter: {np.mean(filtered_circles):.1f} ± {np.std(filtered_circles):.1f} px"
        )
        print(
            f"\n        . ݁₊ ⊹ . ݁ ⟡ ݁ px/cm density: {px_per_cm:.1f} (diameter_cm: {diameter_cm} cm) ⟡ ݁ . ⊹ ₊ ݁."
        )

    return float(px_per_cm)


##############################################################################
# Load models and obtain their path
##############################################################################

@lru_cache(maxsize=None)
def _get_package_model_path(model_name: str) -> str:
    """
    Resolve the absolute path to a model file bundled with the package.

    Tries :func:`importlib.resources.files` first, then falls back to
    locating the package directory via ``traitly.__file__``.

    Parameters
    ----------
    model_name : str
        Filename of the model (e.g. ``'size_reference.pt'``).

    Returns
    -------
    str
        Absolute path to the model file.

    Raises
    ------
    FileNotFoundError
        If the model file is not found in the expected package location.
    """
    try:
        model_path = files("traitly").joinpath("package_data", "models", model_name)
        logger.debug(f"Resolved model path via importlib.resources: {model_path}")

        return str(model_path)

    except (ImportError, AttributeError):
        import traitly
        package_dir = Path(traitly.__file__).parent
        model_path = package_dir / "package_data" / "models" / model_name

        if not model_path.exists():
            logger.error(f"Model not found at: {model_path}")
            raise FileNotFoundError(
                f"Model not found at: {model_path}\n"
                f"Please ensure the model files are included in the package."
            )

        logger.debug(f"Resolved model path via __file__ fallback: {model_path}")

        return str(model_path)


##############################################################################
# Wrapper: calculate px/cm density from YOLO circles or physical dimensions
##############################################################################

def px_cm_density(
    img: np.ndarray,
    confidence_threshold: float = 0.6,
    plot: bool = False,
    width_cm: Optional[float] = None,
    length_cm: Optional[float] = None,
    diameter_cm: float = 2.5,
    font_size: int = 3,
    physical_size: Optional[str] = None,
    verbose: bool = False,
) -> Union[
    Tuple[Optional[float], np.ndarray],
    Tuple[Optional[float], np.ndarray, Optional[List]],
]:
    """
    Calculate pixel-to-centimetre density using the best available method.

    Tries the following methods in priority order:

    1. **Circle detection** – :func:`_detect_size_ref_yolo` +
       :func:`diameter_px_per_cm`. Used when the YOLO model detects
       reference circles.
    2. **Predetermined size** – :func:`_img_px_per_cm` with ``physical_size``
       preset. Used when ``physical_size`` is provided and no circles
       are detected.
    3. **Custom dimensions** – :func:`_img_px_per_cm` with ``width_cm``
       and ``length_cm``. Used as a fallback when both are provided.
    4. **None** – returns ``None`` when no method succeeds; measurements
       will be in pixels.

    Parameters
    ----------
    img : np.ndarray
        Input BGR image.
    model_path : str or None, optional
        Path to the YOLO weights file. If ``None``, resolved via
        :func:`_get_package_model_path`. Default is ``None``.
    confidence_threshold : float, optional
        YOLO detection confidence threshold forwarded to
        :func:`_detect_size_ref_yolo`. Default is 0.6.
    plot : bool, optional
        If True, display the annotated detection result. Default is
        False.
    width_cm : float or None, optional
        Known physical image width in centimetres for method 3. Default
        is ``None``.
    length_cm : float or None, optional
        Known physical image length in centimetres for method 3. Default
        is ``None``.
    diameter_cm : float, optional
        Reference circle diameter forwarded to :func:`diameter_px_per_cm`.
        Default is 2.5.
    font_size : int, optional
        Font scale for YOLO detection annotations. Default is 3.
    physical_size : str or None, optional
        Paper size preset forwarded to :func:`_img_px_per_cm` for method
        2. Default is ``None``.
    verbose : bool, optional
        If True, print method selection and density results. Default is
        False.

    Returns
    -------
    tuple
        ``(px_cm, annotated_img, roi_contours)``

        Where ``px_cm`` is ``None`` if no calibration method succeeded
        and ``roi_contours`` is a list of contour arrays or ``None``.
    """
    logger.info(f"Calculating px per cm ...")

    # Method 1: circle detection via YOLO
    list_circles, img_annotated, roi_boxes = _detect_size_ref_yolo(
        img,
        model_path=model_path,
        plot=plot,
        font_size=font_size * 0.5,
        confidence_threshold=confidence_threshold,
        return_roi_coords=True,
        yolo_verbose=verbose,
    )

    if list_circles:
        logger.info(f"Method 1 (YOLO): {len(list_circles)} circles detected")

        px_cm = diameter_px_per_cm(
            list_circles,
            verbose=verbose,
            diameter_cm=diameter_cm,
            std_threshold=2
        )
        logger.info(f"px/cm density from circles: {px_cm:.2f}")

        circle_coords = []
        for x1, y1, x2, y2 in roi_boxes:
            contour = np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32
            )
            circle_coords.append(contour)

        return (px_cm, img_annotated, (circle_coords if circle_coords else None))

    # Method 2: predetermined paper size
    if physical_size is not None:
        logger.info(f"Method 2 (Paper predetermined size): {physical_size}")

        px_per_cm_width, px_per_cm_length, _, _ = _img_px_per_cm(img, size=physical_size)
        px_cm = float(np.sqrt(px_per_cm_width * px_per_cm_length))

        logger.info(f"px/cm density for {physical_size}: {px_cm:.2f}")

        return (px_cm, img_annotated, None)

    # Method 3: custom width and length
    if width_cm is not None and length_cm is not None:
        logger.info(f"Method 3 (custom paper size): w: {width_cm}, l: {length_cm}")

        px_per_cm_width, px_per_cm_length, _, _ = _img_px_per_cm(
            img,
            width_cm=width_cm,
            length_cm=length_cm
        )

        px_cm = float(np.sqrt(px_per_cm_width * px_per_cm_length))

        logger.info(f"px/cm density for w={width_cm} and l={length_cm}: {px_cm:.2f}")

        if abs(px_per_cm_width - px_per_cm_length) / px_cm > 0.05: #warning msg if >5% differences
            msg = ( f"px/cm differs between axes: width={px_per_cm_width:.1f}, "
                f"length={px_per_cm_length:.1f}. Check image dimensions.")
            logger.warning(msg)
            if verbose:
                print(f"Warning: {msg}")

        return (px_cm, img_annotated, None)

    # Method 4: no calibration available
    logger.info(f"No calibration available. Values will be returned in pixels.")
    return (None, img_annotated, None)


##############################################################################
# Find black circles in a size reference box
##############################################################################

def _find_size_ref_circles(
    roi_gray: np.ndarray,
    return_debug: bool = False,
    ref_circularity: float = 0.8,
    min_area: int = 50,
) -> Union[
    List[Tuple[int, int, int]],
    Tuple[List[Tuple[int, int, int]], Dict],
]:
    """
    Detect dark circular objects in a grayscale ROI (size references).

    Applies Otsu thresholding and filters contours by area and
    circularity. Each valid contour is fitted with a minimum
    enclosing circle.

    Parameters
    ----------
    roi_gray : np.ndarray
        Grayscale crop of the reference region.
    return_debug : bool, optional
        If True, also return a debug dictionary with intermediate images
        and statistics. Default is False.
    ref_circularity : float, optional
        Minimum circularity score in [0, 1] to accept a contour as a
        circle. Default is 0.7.
    min_area : int, optional
        Minimum contour area. Default is 50 px^2.

    Returns
    -------
    list of tuple of int
        If ``return_debug=False``: list of ``(cx, cy, radius)`` in ROI
        coordinates.
    tuple
        If ``return_debug=True``: ``(circles, debug_dict)`` where
        ``debug_dict`` contains ``'roi_gray'``, ``'binary'``,
        ``'overlay'``, ``'num_contours'``, and ``'num_circles'``.
    """
    h, w = roi_gray.shape

    # apply otsu threshold to detect dark circles
    _, binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circles = []
    valid_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < ref_circularity:
            continue

        (x, y), radius = cv2.minEnclosingCircle(contour)
        circles.append((int(x), int(y), int(radius)))
        valid_contours.append(contour)

    if return_debug:
        overlay = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
        for cx, cy, r in circles:
            cv2.circle(overlay, (cx, cy), r, (0, 0, 255), 2)
            cv2.circle(overlay, (cx, cy), 2, (255, 0, 0), -1)

        return circles, {
            "roi_gray": roi_gray,
            "binary": binary,
            "overlay": overlay,
            "num_contours": len(contours),
            "num_circles": len(circles),
        }

    return circles


##############################################################################
# YOLO model cache
##############################################################################
_YOLO_MODEL_CACHE = {}

def _get_yolo_model(model_path: str) -> "YOLO":
    """
    Load and cache a YOLO model by path.

    Uses a module-level ``_YOLO_MODEL_CACHE`` dict to avoid reloading
    the same model across repeated calls in the same session.

    Parameters
    ----------
    model_path : str
        Absolute path to the YOLO ``.pt`` weights file.

    Returns
    -------
    YOLO
        Loaded and cached YOLO model instance.
    """
    if model_path not in _YOLO_MODEL_CACHE:
        logger.debug(f"Loading YOLO model for the first time: {model_path}")
        from ultralytics import YOLO

        _YOLO_MODEL_CACHE[model_path] = YOLO(model_path)
        logger.info(f"YOLO model loaded successfully: {model_path}")

    else:
        logger.debug(f"Using cached YOLO model: {model_path}")

    return _YOLO_MODEL_CACHE[model_path]


##############################################################################
# Detect size reference circles using YOLOv8
##############################################################################
def _detect_size_ref_yolo(
    img: Optional[np.ndarray] = None,
    model_path: Optional[str] = None,
    img_path: Optional[str] = None,
    confidence_threshold: float = 0.6,
    iou_threshold: float = 0.45,
    show_max_rois: int = 6,
    plot: bool = False,
    plot_size: Tuple[int, int] = (8, 8),
    yolo_verbose: bool = False,
    font_size: int = 1.5,
    plot_roi_analysis: bool = False,
    return_roi_coords: bool = False,
) -> Union[
    Tuple[List[Tuple[int, int, int]], np.ndarray],
    Tuple[
        List[Tuple[int, int, int]],
        np.ndarray,
        Optional[List[Tuple[int, int, int, int]]],
    ],
]:
    """
    Detect size reference circles in an image using a YOLOv8 model.

    Parameters
    ----------

    img : np.ndarray or None, optional
        Input BGR image. Required if ``img_path`` is not provided.
    model_path : str or None, optional
        Path to the YOLO weights file. If ``None``, resolved via
        :func:`_get_package_model_path`.
    img_path : str or None, optional
        Path to the image file. Used if ``img`` is not provided.
    confidence_threshold : float, optional
        Minimum YOLO detection confidence. Default is 0.6.
    iou_threshold : float, optional
        IoU threshold for YOLO NMS. Default is 0.45.
    show_max_rois : int, optional
        Maximum ROIs shown in debug plot. Default is 6.
    plot : bool, optional
        If True, display the annotated image. Default is False.
    plot_size : tuple of int, optional
        Figure size for the result plot. Default is (8, 8).
    yolo_verbose : bool, optional
        If True, print detection details. Default is False.
    font_size : int, optional
        Font scale for annotation labels. Default is 1.5.
    plot_roi_analysis : bool, optional
        If True, display per-ROI debug panel. Default is False.
    return_roi_coords : bool, optional
        If True, also return bounding box coordinates. Default is False.

    Returns
    -------
    tuple
        ``(circles, annotated_img, None)`` or ``(circles, annotated_img, roi_boxes)``
        if ``return_roi_coords=True``. ``circles`` is a list of
        ``(cx, cy, diameter)`` tuples in global image coordinates.

    Raises
    ------
    ValueError
        If neither ``img`` nor ``img_path`` is provided, or no ``model_path``.
    """

    if model_path is not None:
        try:
            model = _get_yolo_model(model_path)

        except Exception as e:
            msg = f"Error loading size model from {model_path}: {e}"
            logger.error(msg)
            print(msg)

            return (None, None, None) if return_roi_coords else (None, None, None)
    else:
        raise ValueError("No model path provided. Please pass 'model_path'.")

    if img is None and img_path is None:
        raise ValueError("No image or image path provided. Please pass either 'img' or 'img_path'.")

    if img is None:
        img = cv2.imread(img_path)

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    results = model(img, conf=0.1, iou=iou_threshold, verbose=False)

    box_detected = False
    list_circles = []
    rois_debug = []
    roi_boxes = []
    img_annotated = None

    pad_x_pct = 0.15
    pad_y_pct = 0.05

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            if yolo_verbose:
                print("> No size reference detected.")

            continue

        filtered_boxes = []
        low_conf_boxes = []

        for box in r.boxes:
            conf = float(box.conf[0].cpu().numpy()) # move to cpu to avoid crash in case tensor is on gpu

            if conf >= confidence_threshold:
                filtered_boxes.append(box)
            else:
                low_conf_boxes.append((box, conf))

        if not filtered_boxes:
            if yolo_verbose and low_conf_boxes:
                print(f"> {len(low_conf_boxes)} box(es) detected but all below confidence threshold {confidence_threshold}")

            continue

        box_detected = True

        if img_annotated is None:
            img_annotated = img.copy()

        if yolo_verbose:
            print("> Reference size detected:")
            print(f"  - Processing {len(filtered_boxes)} box(es) with confidence >={confidence_threshold}:")
            if low_conf_boxes:
                print(f"  - Filtered out {len(low_conf_boxes)} box(es) below threshold:")
                for box_idx, (_, conf_value) in enumerate(low_conf_boxes, 1):
                    print(f"      > Box {box_idx}: confidence = {conf_value:.3f}")

        for i, box in enumerate(filtered_boxes):
            conf = float(box.conf[0].cpu().numpy())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # Clamp to image bounds
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))

            box_width = x2 - x1 + 1
            box_height = y2 - y1 + 1
            padx = int(pad_x_pct * box_width)
            pady = int(pad_y_pct * box_height)

            roi_x1 = max(0, x1 - padx)
            roi_y1 = max(0, y1 - pady)
            roi_x2 = min(w, x2 + padx)
            roi_y2 = min(h, y2 + pady)

            roi_boxes.append((roi_x1, roi_y1, roi_x2, roi_y2))

            roi_gray = gray[roi_y1:roi_y2, roi_x1:roi_x2]

            if yolo_verbose:
                roi_h, roi_w = roi_gray.shape[:2]
                print(f"            Ref {i + 1}: {roi_w}x{roi_h} px, conf: {conf:.3f}")

            if roi_gray.size == 0:
                print("Empty ROI, skipping...")
                continue

            cv2.rectangle(img_annotated, (roi_x1, roi_y1), (roi_x2, roi_y2), (200, 100, 0), 2)
            cv2.putText(
                img_annotated,
                f"Ref {i + 1} ({conf:.2f})",
                (roi_x1 + 5, max(roi_y1 - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size,
                (200, 100, 0),
                3,
                cv2.LINE_AA,
            )

            if plot_roi_analysis:
                circles, dbg = _find_size_ref_circles(roi_gray, return_debug=True, ref_circularity=0.7)
                rois_debug.append({
                    "idx": i + 1,
                    "conf": conf,
                    "roi_box": (roi_x1, roi_y1, roi_x2, roi_y2),
                    "roi_gray": dbg["roi_gray"],
                    "binary": dbg["binary"],
                    "overlay": dbg["overlay"],
                    "num_circles": len(circles),
                })
            else:
                circles = _find_size_ref_circles(roi_gray, return_debug=False, ref_circularity=0.7)

            for cx_roi, cy_roi, radius in circles:
                cx_global = cx_roi + roi_x1
                cy_global = cy_roi + roi_y1
                diameter = 2 * radius

                cv2.circle(img_annotated, (cx_global, cy_global), radius, (0, 0, 255), 5)
                cv2.line(
                    img_annotated,
                    (cx_global - radius, cy_global),
                    (cx_global + radius, cy_global),
                    (255, 139, 99),
                    3,
                )

                text = f"{diameter}px"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 4)[0]
                text_x = cx_global - (text_size[0] // 2)
                text_y = cy_global - 20

                cv2.putText(
                    img_annotated,
                    text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size,
                    (0, 0, 255),
                    4,
                )

                list_circles.append((cx_global, cy_global, diameter))

    if yolo_verbose:
        print(f"\n  - Total circles detected: {len(list_circles)}")

    if plot and img_annotated is not None:
        plt.figure(figsize=plot_size)
        plt.imshow(cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    if plot_roi_analysis and box_detected and rois_debug:
        n = min(len(rois_debug), show_max_rois)
        cols = 3
        rows = n
        plt.figure(figsize=(14, 4 * rows))

        for r_i in range(n):
            item = rois_debug[r_i]
            x1, y1, x2, y2 = item["roi_box"]

            plt.subplot(rows, cols, r_i * cols + 1)
            plt.imshow(item["roi_gray"], cmap="gray")
            plt.title(f"Ref {item['idx']} ({item['conf']:.2f})\nROI: ({x1},{y1})-({x2},{y2})")
            plt.axis("off")

            plt.subplot(rows, cols, r_i * cols + 2)
            plt.imshow(item["binary"], cmap="gray")
            plt.title("Binarization")
            plt.axis("off")

            plt.subplot(rows, cols, r_i * cols + 3)
            plt.imshow(cv2.cvtColor(item["overlay"], cv2.COLOR_BGR2RGB))
            plt.title(f"Overlay (circles: {item['num_circles']})")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

    if img_annotated is None:
        img_annotated = img

    return list_circles, img_annotated, (roi_boxes if roi_boxes else None)
