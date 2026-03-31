# traitly/utils/calibration.py
"""
Scale calibration and size-reference detection utilities for traitly.

Provides functions to:

- Estimate pixel-to-centimetre density from known paper dimensions
  (:func:`img_px_per_cm`) or from detected reference circles
  (:func:`diameter_px_per_cm`).
- Detect circular size-reference objects in an image using a YOLOv8 model
  (:func:`detect_size_ref_yolo`, :func:`find_size_ref_circles`).
- Detect rectangular label regions by contour filtering
  (:func:`detect_label_box`) or by YOLO (:func:`detect_label_box_yolo`).
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


def img_px_per_cm(
    img: np.ndarray,
    size: str = "letter_ansi",
    width_cm: Optional[float] = None,
    length_cm: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    """
    Calculate pixel density from an image and known physical dimensions.

    When ``width_cm`` and ``length_cm`` are provided they map directly
    to ``img.shape[1]`` (width) and ``img.shape[0]`` (height). Otherwise
    a paper size preset from ``size`` is used, auto-oriented to match
    the image aspect ratio.

    Parameters
    ----------
    img : np.ndarray
        Input image (2D or 3D).
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
        If image dimensions are invalid, physical dimensions are
        non-positive, ``width_cm > length_cm``, or ``size`` is not
        a valid preset when custom dimensions are absent.
    RuntimeError
        If an unexpected calculation error occurs.
    """
    try:
        if not isinstance(img, np.ndarray):
            raise TypeError("Input must be a numpy array")
        if img.ndim not in [2, 3]:
            raise ValueError("Image must be 2D (grayscale) or 3D (color)")
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

        paper_sizes = {
            "letter_ansi": (21.6, 27.9),
            "legal_ansi": (21.59, 35.56),
            "a4_iso": (21.0, 29.7),
            "a3_iso": (29.7, 42.0),
        }

        img_height_px = img.shape[0]
        img_width_px = img.shape[1]

        if width_cm is not None and length_cm is not None:
            used_width_cm = width_cm
            used_length_cm = length_cm
        else:
            paper_w, paper_h = paper_sizes[size]
            if img_width_px > img_height_px:
                # Landscape
                used_width_cm = max(paper_w, paper_h)
                used_length_cm = min(paper_w, paper_h)
            else:
                # Portrait
                used_width_cm = min(paper_w, paper_h)
                used_length_cm = max(paper_w, paper_h)

        px_per_cm_width = img_width_px / used_width_cm
        px_per_cm_length = img_height_px / used_length_cm

        return px_per_cm_width, px_per_cm_length, used_width_cm, used_length_cm

    except Exception as e:
        raise RuntimeError(f"Calculation error: {str(e)}")


##############################################################################
# Obtain px per cm density from the average diameter of reference circles
##############################################################################


def diameter_px_per_cm(
    all_circles: List[Tuple[int, int, int]],
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
    all_circles : list of tuple of int
        List of ``(cx, cy, diameter_px)`` tuples as returned by
        :func:`detect_size_ref_yolo`.
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
        If ``all_circles`` is empty.
    """
    if not all_circles:
        raise ValueError("No circles provided.")

    diameters = np.array([d[2] for d in all_circles], dtype=np.float32)

    mean_val = np.mean(diameters)
    std_val = np.std(diameters)

    lower = mean_val - std_threshold * std_val
    upper = mean_val + std_threshold * std_val

    mask = (diameters >= lower) & (diameters <= upper)
    filtered = diameters[mask]

    if len(filtered) == 0:
        if verbose:
            print("Warning: Using all the circles (many outliers detected)")
        filtered = diameters

    px_cm_density = np.mean(filtered) / diameter_cm

    if verbose:
        print(
            f"            Filtered circles: {len(filtered)}/{len(diameters)} (std > {std_threshold})"
        )
        print(
            f"            Mean diameter: {np.mean(filtered):.1f} ± {np.std(filtered):.1f} px"
        )
        print(
            f"\n        . ݁₊ ⊹ . ݁ ⟡ ݁ px/cm density: {px_cm_density:.1f} (diameter_cm: {diameter_cm} cm) ⟡ ݁ . ⊹ ₊ ݁."
        )

    return float(px_cm_density)


##############################################################################
# Load models and obtain their path
##############################################################################


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
        from importlib.resources import files

        model_path = files("traitly").joinpath("package_data", "models", model_name)
        return str(model_path)
    except (ImportError, AttributeError):
        import traitly

        package_dir = Path(traitly.__file__).parent
        model_path = package_dir / "package_data" / "models" / model_name

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at: {model_path}\n"
                f"Please ensure the model files are included in the package."
            )

        return str(model_path)


##############################################################################
# Wrapper: calculate px/cm density from YOLO circles or physical dimensions
##############################################################################


def px_cm_density(
    img: np.ndarray,
    model_path: Optional[str] = None,
    confidence_threshold: float = 0.6,
    plot: bool = False,
    width_cm: Optional[float] = None,
    length_cm: Optional[float] = None,
    diameter_cm: float = 2.5,
    font_size: int = 3,
    physical_size: Optional[str] = None,
    return_coordinates: bool = False,
    verbose: bool = False,
) -> Union[
    Tuple[Optional[float], np.ndarray],
    Tuple[Optional[float], np.ndarray, Optional[List]],
]:
    """
    Calculate pixel-to-centimetre density using the best available method.

    Tries the following methods in priority order:

    1. **Circle detection** – :func:`detect_size_ref_yolo` +
       :func:`diameter_px_per_cm`. Used when the YOLO model detects
       reference circles.
    2. **Predetermined size** – :func:`img_px_per_cm` with ``physical_size``
       preset. Used when ``physical_size`` is provided and no circles
       are detected.
    3. **Custom dimensions** – :func:`img_px_per_cm` with ``width_cm``
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
        :func:`detect_size_ref_yolo`. Default is 0.6.
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
        Paper size preset forwarded to :func:`img_px_per_cm` for method
        2. Default is ``None``.
    return_coordinates : bool, optional
        If True, also return the detected ROI contours. Default is
        False.
    verbose : bool, optional
        If True, print method selection and density results. Default is
        False.

    Returns
    -------
    tuple
        If ``return_coordinates=False``:
            ``(px_cm, annotated_img)``
        If ``return_coordinates=True``:
            ``(px_cm, annotated_img, roi_contours)``

        Where ``px_cm`` is ``None`` if no calibration method succeeded
        and ``roi_contours`` is a list of contour arrays or ``None``.
    """
    if model_path is None:
        model_path = _get_package_model_path("size_reference.pt")

    try:
        _get_yolo_model(model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return (None, None, None) if return_coordinates else (None, None)

    # Method 1: circle detection via YOLO
    if return_coordinates:
        all_circles, img_annotated, roi_boxes = detect_size_ref_yolo(
            img,
            model_path=model_path,
            plot=plot,
            font_size=font_size * 0.5,
            confidence_threshold=confidence_threshold,
            return_roi_coords=True,
            yolo_verbose=verbose,
        )
    else:
        all_circles, img_annotated = detect_size_ref_yolo(
            img,
            model_path=model_path,
            plot=plot,
            font_size=font_size * 0.5,
            confidence_threshold=confidence_threshold,
            yolo_verbose=verbose,
        )

    if all_circles:
        px_cm = diameter_px_per_cm(
            all_circles, verbose=verbose, diameter_cm=diameter_cm, std_threshold=2
        )

        if return_coordinates:
            circle_coords = []
            for x1, y1, x2, y2 in roi_boxes:
                contour = np.array(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32
                )
                circle_coords.append(contour)
            return px_cm, img_annotated, circle_coords if circle_coords else None
        else:
            return px_cm, img_annotated

    # Method 2: predetermined paper size
    valid_sizes = {
        "letter_ansi": (21.6, 27.9),
        "legal_ansi": (21.59, 35.56),
        "a4_iso": (21.0, 29.7),
        "a3_iso": (29.7, 42.0),
    }

    if physical_size is not None:
        if physical_size not in valid_sizes:
            raise ValueError(
                f"Physical_size must be one of {list(valid_sizes.keys())}, "
                f"got '{physical_size}'"
            )
        px_per_cm_width, px_per_cm_length, _, _ = img_px_per_cm(img, size=physical_size)
        px_cm = float(np.sqrt(px_per_cm_width * px_per_cm_length))
        return (
            (px_cm, img_annotated, None)
            if return_coordinates
            else (px_cm, img_annotated)
        )

    # Method 3: custom width and length
    if width_cm is not None and length_cm is not None:
        px_per_cm_width, px_per_cm_length, _, _ = img_px_per_cm(
            img, width_cm=width_cm, length_cm=length_cm
        )
        px_cm = float(np.sqrt(px_per_cm_width * px_per_cm_length))
        return (
            (px_cm, img_annotated, None)
            if return_coordinates
            else (px_cm, img_annotated)
        )

    # Method 4: no calibration available
    return (None, img_annotated, None) if return_coordinates else (None, img_annotated)


##############################################################################
# Detect label ROI by contour filtering
##############################################################################


def detect_label_box(
    imagen_path: Optional[str] = None,
    img: Optional[np.ndarray] = None,
    verbose: Optional[bool] = False,
    plot: Optional[bool] = False,
    max_boxes: int = 10,
) -> List[Dict]:
    """
    Detect rectangular label regions using grayscale thresholding and contour filtering.

    Converts the image to grayscale, applies binary thresholding, finds
    external contours, and filters by area (> 5000 px²) and aspect ratio
    (2 < w/h < 6) to identify label-shaped rectangles. Used as a fallback
    when :func:`detect_label_box_yolo` returns no results.

    Parameters
    ----------
    imagen_path : str or None, optional
        Path to the image file. Used if ``img`` is not provided.
        Default is ``None``.
    img : np.ndarray or None, optional
        BGR image array. Takes precedence over ``imagen_path``. Default
        is ``None``.
    verbose : bool or None, optional
        If True, print the number and details of detected boxes. Default
        is False.
    plot : bool or None, optional
        If True, display the image with detected boxes overlaid. Default
        is False.
    max_boxes : int, optional
        Maximum number of boxes to return. Default is 10.

    Returns
    -------
    list of dict
        List of box dictionaries with keys ``'x'``, ``'y'``,
        ``'width'``, ``'height'``, ``'area'``, ``'aspect_ratio'``.
        Empty list if no boxes are found.

    Raises
    ------
    ValueError
        If neither ``imagen_path`` nor ``img`` is provided, or if the
        image cannot be loaded.
    """
    if imagen_path is not None:
        img = cv2.imread(imagen_path)
    elif img is not None:
        img = img.copy()
    else:
        raise ValueError("Either imagen_path or img must be provided.")

    if img is None:
        raise ValueError(f"Could not load image: {imagen_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []

    for cnt in contours:
        if len(boxes) >= max_boxes:
            break

        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        if area <= 5000:
            continue

        aspect_ratio = w / h

        if 2 < aspect_ratio < 6:
            box_info = {
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "area": area,
                "aspect_ratio": aspect_ratio,
            }
            boxes.append(box_info)

            if plot:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if plot:
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    if verbose:
        print(f"\nTotal boxes found: {len(boxes)}")
        for i, box in enumerate(boxes, 1):
            print(f"Box {i}: {box}")

    return boxes


##############################################################################
# Find black circles in a size reference box
##############################################################################


def find_size_ref_circles(
    roi_gray: np.ndarray,
    return_debug: bool = False,
    ref_circularity: float = 0.7,
    min_area_ratio: float = 0.01,
) -> Union[
    List[Tuple[int, int, int]],
    Tuple[List[Tuple[int, int, int]], Dict],
]:
    """
    Detect dark circular objects in a grayscale ROI.

    Applies Gaussian blur, adaptive thresholding, and morphological
    cleanup, then filters contours by area and circularity. Each
    surviving contour is fitted with a minimum enclosing circle.

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
    min_area_ratio : float, optional
        Minimum contour area as a fraction of the total ROI area. The
        absolute minimum is clamped to 50 px². Default is 0.01.

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
    min_area = max(50, int(min_area_ratio * h * w))

    kernel_size = 3 if min(h, w) < 100 else 5
    blurred = cv2.GaussianBlur(roi_gray, (kernel_size, kernel_size), 0)

    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

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
        from ultralytics import YOLO

        _YOLO_MODEL_CACHE[model_path] = YOLO(model_path)
    return _YOLO_MODEL_CACHE[model_path]


##############################################################################
# Detect size reference circles using YOLOv8
##############################################################################


def detect_size_ref_yolo(
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

    Loads the model via :func:`_get_yolo_model`, detects bounding boxes
    above ``confidence_threshold``, extracts each ROI, and finds circles
    within each ROI via :func:`find_size_ref_circles`. Detected circles
    are converted to global image coordinates and drawn onto a copy of
    the image.

    Parameters
    ----------
    img : np.ndarray or None, optional
        Input BGR image. Required if ``img_path`` is not provided.
    model_path : str or None, optional
        Path to the YOLO weights file. If ``None``, resolved via
        :func:`_get_package_model_path`. Default is ``None``.
    img_path : str or None, optional
        Path to the image file. Used if ``img`` is not provided.
        Default is ``None``.
    confidence_threshold : float, optional
        Minimum YOLO detection confidence to accept a bounding box.
        Default is 0.6.
    iou_threshold : float, optional
        IoU threshold for YOLO NMS. Default is 0.45.
    show_max_rois : int, optional
        Maximum number of ROIs shown in the debug plot. Default is 6.
    plot : bool, optional
        If True, display the annotated image. Default is False.
    plot_size : tuple of int, optional
        Figure size for the main result plot. Default is (8, 8).
    yolo_verbose : bool, optional
        If True, print detection details. Default is False.
    font_size : int, optional
        Font scale for annotation labels. Default is 1.5.
    plot_roi_analysis : bool, optional
        If True, display a per-ROI debug panel with grayscale, binary,
        and overlay views via :func:`find_size_ref_circles`. Default is
        False.
    return_roi_coords : bool, optional
        If True, also return the bounding box coordinates of each
        detected ROI. Default is False.

    Returns
    -------
    tuple
        If ``return_roi_coords=False``:
            ``(circles, annotated_img)``
        If ``return_roi_coords=True``:
            ``(circles, annotated_img, roi_boxes)``

        Where ``circles`` is a list of ``(cx, cy, diameter)`` tuples in
        global image coordinates, and ``roi_boxes`` is a list of
        ``(x1, y1, x2, y2)`` bounding boxes or ``None`` if none were
        detected.

    Raises
    ------
    ValueError
        If neither ``img`` nor ``img_path`` is provided.
    """
    if model_path is None:
        model_path = _get_package_model_path("size_reference.pt")

    try:
        model = _get_yolo_model(model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return (None, None, None) if return_roi_coords else (None, None)

    if img is None and img_path is None:
        raise ValueError(
            "No image or image path provided. Please pass either 'img' or 'img_path'."
        )

    if img_path is not None:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Error loading image from {img_path}")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    results = model(img, conf=0.1, iou=iou_threshold, verbose=False)

    box_detected = False
    all_circles = []
    rois_debug = []
    roi_boxes = []
    img_annotated = None

    pad_x_pct = 0.15
    pad_y_pct = 0.05

    for r in results:
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            if yolo_verbose:
                print("> No size reference detected.")
            continue

        # Filter by confidence threshold
        filtered_boxes = []
        low_conf_boxes = []

        for box in boxes:
            conf = float(box.conf[0].cpu().numpy())
            if conf >= confidence_threshold:
                filtered_boxes.append(box)
            else:
                low_conf_boxes.append((box, conf))

        boxes = filtered_boxes
        box_detected = True

        if img_annotated is None:
            img_annotated = img.copy()

        if yolo_verbose:
            print("> Reference size detected:")
            print(
                f"  - Processing reference box(es) with a confidence "
                f"threshold >={confidence_threshold}:"
            )

            if low_conf_boxes:
                print(
                    f"Filtered out {len(low_conf_boxes)} box(es) below the "
                    f"confidence threshold: {confidence_threshold}"
                )
                for box_idx, (box, conf_value) in enumerate(low_conf_boxes, 1):
                    print(f"    ▸ Box {box_idx}: confidence = {conf_value:.3f}")

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # Clamp to image bounds
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))

            confidence = float(box.conf[0].cpu().numpy())
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
                roi_height, roi_width = roi_gray.shape[:2]
                print(
                    f"            Ref {i + 1}: {roi_width}x{roi_height} px, "
                    f"conf: {confidence:.3f}"
                )

            if roi_gray.size == 0:
                print("Empty ROI, skipping...")
                continue

            # Annotate bounding box on result image
            cv2.rectangle(
                img_annotated, (roi_x1, roi_y1), (roi_x2, roi_y2), (200, 100, 0), 2
            )
            cv2.putText(
                img_annotated,
                f"Ref {i + 1} ({confidence:.2f})",
                (roi_x1 + 5, max(roi_y1 - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size,
                (200, 100, 0),
                3,
                cv2.LINE_AA,
            )

            # Detect circles within ROI
            if plot_roi_analysis:
                circles, dbg = find_size_ref_circles(
                    roi_gray, return_debug=True, ref_circularity=0.7
                )
                rois_debug.append(
                    {
                        "idx": i + 1,
                        "conf": confidence,
                        "roi_box": (roi_x1, roi_y1, roi_x2, roi_y2),
                        "roi_gray": dbg["roi_gray"],
                        "binary": dbg["binary"],
                        "overlay": dbg["overlay"],
                        "num_circles": len(circles),
                    }
                )
            else:
                circles = find_size_ref_circles(
                    roi_gray, return_debug=False, ref_circularity=0.7
                )

            # Convert circle coordinates to global and annotate
            for cx_roi, cy_roi, radius in circles:
                cx_global = cx_roi + roi_x1
                cy_global = cy_roi + roi_y1
                diameter = 2 * radius

                cv2.circle(
                    img_annotated, (cx_global, cy_global), radius, (0, 0, 255), 5
                )

                cv2.line(
                    img_annotated,
                    (cx_global - radius, cy_global),
                    (cx_global + radius, cy_global),
                    (255, 139, 99),
                    3,
                )

                text = f"{diameter}px"
                text_size = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 4
                )[0]
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

                all_circles.append((cx_global, cy_global, diameter))

    if yolo_verbose:
        print(f"\n  - Total circles detected: {len(all_circles)}")
        if not box_detected:
            print(
                "No size reference box detected in the image by YOLO. "
                "Try adjusting confidence threshold or image quality."
            )
        elif len(all_circles) == 0:
            print(
                "No circles detected within the detected size reference boxes. "
                "Try adjusting thresholds or check image quality."
            )

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
            plt.title(
                f"Ref {item['idx']} ({item['conf']:.2f})\nROI: ({x1},{y1})-({x2},{y2})"
            )
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

    if return_roi_coords:
        return all_circles, img_annotated, roi_boxes if roi_boxes else None
    else:
        return all_circles, img_annotated


##############################################################################
# Detect label ROI using YOLOv8
##############################################################################


def detect_label_box_yolo(
    img: np.ndarray,
    model_path: Optional[str] = None,
    conf: float = 0.3,
    plot: bool = False,
) -> Optional[List[Dict]]:
    """
    Detect label ROIs using a YOLOv8 model.

    Loads the label detection model via :func:`_get_yolo_model` and
    converts detected bounding boxes to the same dict format used by
    :func:`detect_label_box`.

    Parameters
    ----------
    img : np.ndarray
        Input BGR image.
    model_path : str or None, optional
        Path to the YOLO label detection weights. If ``None``, resolved
        via :func:`_get_package_model_path` using ``'label.pt'``.
        Default is ``None``.
    conf : float, optional
        Minimum YOLO confidence threshold. Default is 0.3.
    plot : bool, optional
        If True, display the image with detected boxes overlaid. Default
        is False.

    Returns
    -------
    list of dict or None
        List of box dicts with keys ``'x'``, ``'y'``, ``'width'``,
        ``'height'``, ``'area'``, ``'aspect_ratio'``, or ``None`` if no
        boxes are detected or the model fails to load.
    """
    if model_path is None:
        model_path = _get_package_model_path("label.pt")

    try:
        model = _get_yolo_model(model_path)
    except Exception as e:
        print(f"Error loading YOLO model from {model_path}: {e}")
        return None

    results = model(img, conf=conf, verbose=False)

    for r in results:
        boxes = r.boxes

        if boxes is None or len(boxes) == 0:
            return None

        label_boxes = []

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            width = x2 - x1
            height = y2 - y1
            area = width * height
            aspect_ratio = width / height if height > 0 else 0

            label_boxes.append(
                {
                    "x": x1,
                    "y": y1,
                    "width": width,
                    "height": height,
                    "area": area,
                    "aspect_ratio": aspect_ratio,
                }
            )

        if plot:
            plt.figure(figsize=(8, 8))
            img_copy = img.copy()
            for box in label_boxes:
                x, y = box["x"], box["y"]
                w, h = box["width"], box["height"]
                cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.show()

        return label_boxes if label_boxes else None

    return None
