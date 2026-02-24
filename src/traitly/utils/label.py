# traitly/utils/label.py
"""
Label text detection and QR code reading utilities for traitly.

Provides functions to:

- Initialize and cache an EasyOCR reader (:func:`get_easyocr_reader`,
  :func:`get_cached_reader`).
- Extract treatment labels from ROI crops via OCR (:func:`detect_label_text`).
- Decode QR codes from images (:func:`detect_qr`).
- Detect rectangular label bounding boxes by contour filtering
  (:func:`detect_label_box`) or by YOLOv8 (:func:`detect_label_box_yolo`).
"""

# ============================================================================
# STANDARD LIBRARY
# ============================================================================
from typing import Optional, List, Tuple, Union, Dict
import os
from pathlib import Path
from functools import lru_cache

# ===========================================================================
# THIRD-PARTY LIBRARIES
# ===========================================================================
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr

# ===========================================================================
# INTERNAL
# ===========================================================================
from .calibration import _get_package_model_path, _get_yolo_model


##############################################################################
# Detect label text with OCR
##############################################################################

# Module-level cache to avoid reloading the OCR model across calls
_READER_CACHE = {}


def get_easyocr_reader(
    languages: List[str] = ['en', 'es'],
    gpu: bool = False,
) -> 'easyocr.Reader':
    """
    Initialize an EasyOCR reader with optional GPU support.

    Suppresses all stdout and stderr output during initialization.
    Falls back silently to CPU if CUDA is not available when
    ``gpu=True``.

    Parameters
    ----------
    languages : list of str, optional
        Language codes for OCR. Default is ``['en', 'es']``.
    gpu : bool, optional
        If True, attempt to use CUDA GPU acceleration if available. 
        Falls back to CPU if CUDA is not supported. Default is False.

    Returns
    -------
    easyocr.Reader
        Initialized EasyOCR reader instance.
    """
    import sys
    from io import StringIO
    import warnings

    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = StringIO()

    try:
        import easyocr

        if gpu:
            import torch
            if not torch.cuda.is_available():
                print("GPU not available")
                gpu = False
            else:
                print('GPU available')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            reader = easyocr.Reader(languages, quantize=gpu, verbose=True)

    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

    return reader


def get_cached_reader(
    languages: Tuple[str, ...] = ('en', 'es'),
    gpu: bool = False,
) -> 'easyocr.Reader':
    """
    Return a cached EasyOCR reader, initializing it on first call.

    Uses a module-level ``_READER_CACHE`` dict keyed by
    ``(languages, gpu)`` to avoid reloading the model across repeated
    calls in the same session.

    Parameters
    ----------
    languages : tuple of str, optional
        Language codes. Must be a tuple (hashable) for caching.
        Default is ``('en', 'es')``.
    gpu : bool, optional
        If True, use GPU-accelerated reader. Default is False.

    Returns
    -------
    easyocr.Reader
        Cached or newly initialized EasyOCR reader.
    """
    key = (tuple(languages), gpu)

    if key not in _READER_CACHE:
        _READER_CACHE[key] = get_easyocr_reader(list(languages), gpu=gpu)

    return _READER_CACHE[key]


##############################################################################
# Detect label text
##############################################################################

def detect_label_text(
    img: np.ndarray,
    label_roi: List[Dict],
    language: List[str] = ['es', 'en'],
    blur_label: Tuple[int, int] = (11, 11),
    verbose: bool = False,
    gpu: bool = False,
) -> Optional[str]:
    """
    Extract text from the first valid label ROI using EasyOCR.

    Converts the ROI to grayscale, applies Gaussian blur, and runs OCR
    via :func:`get_cached_reader`. Only the first detected word of the
    first valid ROI is returned.

    Parameters
    ----------
    img : np.ndarray
        Full BGR image containing the label region.
    label_roi : list of dict
        List of ROI dicts with keys ``'x'``, ``'y'``, ``'width'``,
        ``'height'``, as returned by :func:`detect_label_box` or
        :func:`detect_label_box_yolo`.
    language : list of str, optional
        OCR language codes forwarded to :func:`get_cached_reader`.
        Default is ``['es', 'en']``.
    blur_label : tuple of int, optional
        Gaussian blur kernel size applied to the ROI before OCR.
        Default is ``(11, 11)``.
    verbose : bool, optional
        If True, print detection results. Default is False.
    gpu : bool, optional
        If True, use GPU for OCR. Default is False.

    Returns
    -------
    str or None
        First detected word from the label, or ``None`` if no valid ROI
        or text is found.
    """
    if not label_roi:
        return None

    # Keep only ROIs that are large enough and fit inside the image
    valid_rois = []
    for i, box in enumerate(label_roi):
        x, y, w, h = box['x'], box['y'], box['width'], box['height']
        if y + h <= img.shape[0] and x + w <= img.shape[1] and h > 10 and w > 10:
            valid_rois.append((i, x, y, w, h))

    if not valid_rois:
        return None

    reader = get_cached_reader(tuple(language), gpu=gpu)

    first_idx, x, y, w, h = valid_rois[0]

    try:
        region = img[y:y + h, x:x + w]
        if region.size == 0:
            return None

        gray    = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        blur    = cv2.GaussianBlur(gray, blur_label, 0)
        results = reader.readtext(blur)

        if results:
            text = results[0][1].split()[0]
            if verbose:
                print(f"Label text found in ROI {first_idx + 1}: '{text}'")
            return text

        if verbose:
            print(f"No label text found in ROI {first_idx + 1}")
        return None

    except Exception as e:
        if verbose:
            print(f"Error processing ROI {first_idx + 1}: {e}")
        return None


##############################################################################
# Detect QR and extract text
##############################################################################

def detect_qr(
    img_path: Optional[str] = None,
    img: Optional[np.ndarray] = None,
    fast_mode: bool = True,
) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """
    Detect a QR code in an image and return its decoded text.

    Uses ``cv2.QRCodeDetector``. When ``fast_mode=True`` and the image
    is larger than 2000 px on any side, the image is downscaled by 0.5
    before detection and detected points are scaled back to original
    resolution.

    Parameters
    ----------
    img_path : str or None, optional
        Path to the image file. Used if ``img`` is not provided.
        Default is ``None``.
    img : np.ndarray or None, optional
        BGR image array. Takes precedence over ``img_path``. Default is
        ``None``.
    fast_mode : bool, optional
        If True, downscale large images before detection for speed.
        Default is True.

    Returns
    -------
    tuple of (str or None, np.ndarray or None)
        ``(qr_text, img)`` where ``qr_text`` is the first word of the
        decoded QR payload, or ``None`` if no QR was detected.
        ``img`` is always returned (original if no QR detected).
    """
    if img is None and img_path:
        img = cv2.imread(img_path)
        if img is None:
            return None, None

    h, w = img.shape[:2]

    gray     = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = cv2.QRCodeDetector()

    if fast_mode and max(h, w) > 2000:
        scale = 0.5
        small = cv2.resize(gray, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
        data, pts, _ = detector.detectAndDecode(small)
        if pts is not None:
            pts = pts * (1 / scale)
    else:
        data, pts, _ = detector.detectAndDecode(gray)

    if pts is not None and data:
        img_color = img.copy()
        pts       = pts[0].astype(int)
        qr_text   = data.split()[0] if data.split() else data
        return qr_text, img_color

    return None, img


##############################################################################
# Detect label bounding box by contour filtering
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
                'x': x, 'y': y, 'width': w, 'height': h,
                'area': area, 'aspect_ratio': aspect_ratio,
            }
            boxes.append(box_info)

            if plot:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if plot:
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    if verbose:
        print(f"\nTotal boxes found: {len(boxes)}")
        for i, box in enumerate(boxes, 1):
            print(f"Box {i}: {box}")

    return boxes


##############################################################################
# Detect label bounding box with YOLOv8
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
        model_path = _get_package_model_path('label.pt')

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

            width        = x2 - x1
            height       = y2 - y1
            area         = width * height
            aspect_ratio = width / height if height > 0 else 0

            label_boxes.append({
                'x': x1, 'y': y1,
                'width': width, 'height': height,
                'area': area, 'aspect_ratio': aspect_ratio,
            })

        if plot:
            plt.figure(figsize=(8, 8))
            img_copy = img.copy()
            for box in label_boxes:
                x, y = box['x'], box['y']
                w, h = box['width'], box['height']
                cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

        return label_boxes if label_boxes else None

    return None