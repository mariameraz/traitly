# traitly/utils/basic_functions.py
"""
General-purpose image utilities for traitly.

Provides functions for loading, displaying, and saving images, extracting
image metadata, validating output directories, and drawing fruit annotations.
Most functions operate on BGR NumPy arrays as returned by ``cv2.imread``.
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

# ============================================================================
# INTERNAL IMPORTS
# ===========================================================================
from .constants import valid_extensions, valid_cv2_extensions, label_positions

##############################################################################
# Load an image
##############################################################################

@lru_cache(maxsize=128)
def _load_img_cached(path: str) -> np.ndarray:
    """
    Load an image from disk into BGR format with LRU caching.

    Parameters
    ----------
    path : str
        Absolute path to the image file. Extension must be in
        ``valid_extensions``.

    Returns
    -------
    np.ndarray
        Loaded BGR image.

    Raises
    ------
    ValueError
        If the file extension is unsupported or the image cannot be
        loaded by ``cv2.imread``.
    """
    path_obj = Path(path)
    if path_obj.suffix.lower() not in valid_extensions:
        raise ValueError(f"Unsupported image format: '{path_obj.suffix.lower()}'")
    
    img = cv2.imread(str(path_obj), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot load image: {path_obj.name}")
    
    return img


def load_img(
    path: str,
    plot: bool = False,
    plot_size: Tuple[int, int] = (20, 10),
    show_axis: bool = False,
    x: Optional[int] = None,
    y: Optional[int] = None,
    w: Optional[int] = None,
    h: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Load an image via :func:`_load_img_cached` and optionally display it.

    Returns a fresh copy of the cached array so callers can modify it
    freely without invalidating the cache.

    Parameters
    ----------
    path : str
        Path to the image file.
    plot : bool, optional
        If True, display the image in RGB. Default is False.
    plot_size : tuple of int, optional
        Figure size for the plot. Default is (20, 10).
    show_axis : bool, optional
        If True, display axis ticks and labels. Default is True.
    x : int, optional
        Left pixel coordinate of the crop region.
    y : int, optional
        Top pixel coordinate of the crop region.
    w : int, optional
        Width of the crop region in pixels.
    h : int, optional
        Height of the crop region in pixels.

    Returns
    -------
    np.ndarray or None
        BGR image array (cropped if x/y/w/h are provided), or ``None`` if loading fails.
    """
    try:
        img = _load_img_cached(path)

        # Crop
        if any(v is not None for v in (x, y, w, h)):
            img_h, img_w = img.shape[:2]

            x0 = x if x is not None else 0
            y0 = y if y is not None else 0
            x1 = x0 + w if w is not None else img_w
            y1 = y0 + h if h is not None else img_h

            img = img[y0:y1, x0:x1]

        # Plot
        if plot:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=plot_size)
            plt.imshow(img_rgb)
            plt.axis('on' if show_axis else 'off')
            plt.show()

        return img.copy()

    except Exception as e:
        print(f"Error loading: {e}")
        return None


##############################################################################
# Detect image name
##############################################################################

def detect_img_name(path_image: str) -> Optional[str]:
    """
    Extract the filename (with extension) from an image path.

    Parameters
    ----------
    path_image : str
        Full or relative path to the image file.

    Returns
    -------
    str or None
        Filename including extension (e.g. ``'fruit_01.jpg'``), or
        ``None`` if extraction fails.
    """
    try:
        if not isinstance(path_image, str):
            raise TypeError('Path input should be of type str')
        
        filename = os.path.basename(path_image)
        return filename if filename else None

    except Exception as e:
        print(f"Error: {e}")
        return None


##############################################################################
# Plotting image on screen
##############################################################################

_PLOT_CACHE = {}

def plot_img(
    img: np.ndarray,
    fig_axis: bool = False,
    plot_size: Tuple[int, int] = (10, 10),
    binary: bool = False,
    cache_key: Optional[str] = None,
    clear_cache: bool = False,
) -> Optional[Tuple]:
    """
    Display an image, optionally reusing a cached figure for performance.

    BGR images are converted to RGB before display. Grayscale and binary
    images are displayed with ``cmap='gray'``. When ``cache_key`` is
    provided, the figure and axes are stored in ``_PLOT_CACHE`` and
    reused on subsequent calls with the same key, avoiding figure
    proliferation in interactive sessions.

    Parameters
    ----------
    img : np.ndarray
        Image to display (BGR, grayscale, or binary).
    fig_axis : bool, optional
        If True, show axis ticks and labels. Default is False.
    plot_size : tuple of int, optional
        Figure size ``(width, height)``. Default is (10, 10).
    binary : bool, optional
        If True, render with ``cmap='gray'`` and nearest interpolation.
        Default is False.
    cache_key : str or None, optional
        Key for figure caching. If provided, the figure is stored and
        reused on subsequent calls. Default is ``None``.
    clear_cache : bool, optional
        If True, clear ``_PLOT_CACHE`` and close all figures, then
        return immediately without plotting. Default is False.

    Returns
    -------
    tuple of (Figure, Axes) or None
        The matplotlib figure and axes, or ``None`` if ``clear_cache``
        is True.
    """
    if clear_cache:
        _PLOT_CACHE.clear()
        plt.close('all')
        return
    
    if cache_key and cache_key in _PLOT_CACHE:
        fig, ax = _PLOT_CACHE[cache_key]
        ax.clear()
    else:
        fig, ax = plt.subplots(figsize=plot_size, num=cache_key if cache_key else None)
        if cache_key:
            _PLOT_CACHE[cache_key] = (fig, ax)
    
    if binary:
        ax.imshow(img, cmap='gray', interpolation='nearest')
    else:
        if len(img.shape) == 3 and img.shape[2] == 3:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                      interpolation='bilinear')
        else:
            ax.imshow(img, interpolation='bilinear')
    
    if not fig_axis:
        ax.axis('off')
    
    plt.tight_layout(pad=0.1)
    plt.draw()
    
    if not plt.isinteractive():
        plt.show(block=False)
    
    fig.canvas.flush_events()
    
    return fig, ax


##############################################################################
# Validate a directory
##############################################################################

def validate_dir(path: str) -> str:
    """
    Ensure the parent directory of ``path`` exists and return the absolute path.

    Parameters
    ----------
    path : str
        File path whose parent directory should be created if absent.
        Supports ``~`` expansion.

    Returns
    -------
    str
        Absolute version of ``path`` with its parent directory guaranteed
        to exist.
    """
    abs_path = os.path.abspath(os.path.expanduser(path))
    dir_path = os.path.dirname(abs_path)
    
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    
    return abs_path


##############################################################################
# Save images
##############################################################################

def save_img(
    self,
    output_path: Optional[str] = None,
    quality: int = 95,
    compression: int = 9,
    output_message: bool = True,
) -> str:
    """
    Save the annotated image to disk.

    Uses ``cv2.imwrite`` with format-appropriate compression settings.
    If ``output_path`` is ``None``, the file is written to a
    ``Results/`` subdirectory next to the source image.

    Parameters
    ----------
    output_path : str or None, optional
        Destination file path including extension. If ``None``, defaults
        to ``<image_dir>/Results/annotated_image.jpg``. Default is
        ``None``.
    quality : int, optional
        JPEG or WebP compression quality in [0, 100]. Default is 95.
    compression : int, optional
        PNG compression level in [0, 9]. Default is 9.
    output_message : bool, optional
        If True, print the saved file path. Default is True.

    Returns
    -------
    str
        Absolute path to the saved image file.

    Raises
    ------
    ValueError
        If no annotated image is available or the file extension is
        not in ``valid_cv2_extensions``.
    IOError
        If ``cv2.imwrite`` fails to write the file.
    """
    if self.morphology_results is None:
        raise ValueError("No image annotated available. Run analyze_morphology() first.")

    image = self.morphology_results

    if output_path is None:
        results_path = os.path.join(self.image_path, "Results")
        os.makedirs(results_path, exist_ok=True)
        output_path = os.path.join(results_path, "annotated_image.jpg")

    ext = os.path.splitext(output_path)[1].lower()

    if ext not in valid_cv2_extensions:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Valid formats: {', '.join(valid_cv2_extensions)}"
        )

    params = []

    if ext in [".jpg", ".jpeg"]:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, compression]
    elif ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, quality]

    success = cv2.imwrite(output_path, image, params)

    if not success:
        raise IOError(f"Failed to save image to '{output_path}'")

    if output_message:
        print(f"Image saved to: {output_path}")

    return output_path


##############################################################################
# Annotate an image
##############################################################################

def annotate_all_fruits(
    fruit_locule_map: Dict[int, List[int]],
    contours: List[np.ndarray],
    annotated_img: np.ndarray,
    img_shape: Tuple[int, int],
    font_scale: float = 2,
    font_thickness: int = 2,
    pericarp_ext_color: Tuple[int, int, int] = (0, 255, 0),
    pericarp_ext_thickness: int = 2,
    locule_thickness: int = 2,
    locule_color: Tuple[int, int, int] = (255, 0, 255),
    label_position: str = 'left',
    margin: int = 10,
    text_color: Tuple[int, int, int] = (0, 0, 0),
    label_background_color: Tuple[int, int, int] = (255, 255, 255),
    label_opacity: float = 0.7,
    verbose: bool = True,
) -> None:
    """
    Draw contours and text annotations for all fruits in a single pass.

    For each fruit in ``fruit_locule_map``, draws the outer fruit
    contour, all locule contours, and a semi-transparent text label
    showing the sequential ID and locule count. Label placement is
    controlled by ``label_position`` and clamped within image boundaries.

    Parameters
    ----------
    fruit_locule_map : dict of {int : list of int}
        Mapping from fruit contour index to list of locule contour
        indices.
    contours : list of np.ndarray
        Full list of all detected contours.
    annotated_img : np.ndarray
        BGR image modified in-place with contours and text labels.
    img_shape : tuple of int
        ``(height, width)`` of the image for boundary clamping.
    font_scale : float, optional
        Font scale for annotation text. Default is 2.
    font_thickness : int, optional
        Thickness of annotation text. Default is 2.
    pericarp_ext_color : tuple of int, optional
        BGR color for fruit contours. Default is ``(0, 255, 0)``.
    pericarp_ext_thickness : int, optional
        Line thickness for fruit contours. Default is 2.
    locule_thickness : int, optional
        Line thickness for locule contours. Default is 2.
    locule_color : tuple of int, optional
        BGR color for locule contours. Default is ``(255, 0, 255)``.
    label_position : str, optional
        Position of the text label relative to the fruit bounding box.
        One of ``'top'``, ``'bottom'``, ``'left'``, ``'right'``.
        Default is ``'left'``.
    margin : int, optional
        Pixel padding around label backgrounds and boundary clamping.
        Default is 10.
    text_color : tuple of int, optional
        BGR color for annotation text. Default is ``(0, 0, 0)``.
    label_background_color : tuple of int, optional
        BGR color for label background rectangles. Default is
        ``(255, 255, 255)``.
    label_opacity : float, optional
        Opacity of label backgrounds in [0, 1]. Default is 0.7.
    verbose : bool, optional
        If True, print warnings for missing or empty contours. Default
        is True.

    Raises
    ------
    ValueError
        If ``label_position`` is not one of the valid options defined in
        ``label_positions``.
    """
    if label_position not in label_positions:
        raise ValueError(
            f"Invalid label position: {label_position}. "
            f"Valid options are: {label_positions}"
        )

    font = cv2.FONT_HERSHEY_SIMPLEX
    (size_w, size_h), _ = cv2.getTextSize("Test", font, font_scale, font_thickness)
    single_line_height = size_h

    img_height, img_width = img_shape
    sequential_id = 1

    for fruit_id, locule_ids in fruit_locule_map.items():

        if fruit_id >= len(contours):
            if verbose:
                print(f"Fruit ID {fruit_id} not in contours list")
            continue

        fruit_contour = contours[fruit_id]
        if fruit_contour is None or len(fruit_contour) == 0:
            if verbose:
                print(f" Empty contour for fruit {fruit_id}")
            continue

        n_locules = len(locule_ids)

        # Draw fruit outer contour
        cv2.drawContours(
            annotated_img, [fruit_contour], -1,
            pericarp_ext_color, pericarp_ext_thickness
        )

        # Draw locule contours
        for locule_id in locule_ids:
            if locule_id >= len(contours):
                if verbose:
                    print(f"Locule ID {locule_id} not in contours list")
                continue

            locule_contour = contours[locule_id]
            if locule_contour is None or len(locule_contour) == 0:
                continue

            cv2.drawContours(
                annotated_img, [locule_contour], -1,
                locule_color, locule_thickness
            )

        # Build label text
        x, y, w, h = cv2.boundingRect(fruit_contour)
        text = f"id {sequential_id}" if n_locules == 0 else f"id {sequential_id}: \n{n_locules} loc"

        (size_w, size_h), _ = cv2.getTextSize("Test", font, font_scale, font_thickness)
        single_line_height = size_h

        num_lines = text.count('\n') + 1
        total_height = (single_line_height * num_lines) + (15 * (num_lines - 1))

        text_width = max([
            cv2.getTextSize(line, font, font_scale, font_thickness)[0][0]
            for line in text.split('\n')
        ])

        # Position label relative to fruit bounding box
        img_height, img_width = img_shape

        if label_position == 'top':
            text_x = max(10, x)
            text_y = max(total_height + 15, y - 15)
        elif label_position == 'bottom':
            text_x = max(10, x)
            text_y = min(img_height - 15, y + h + total_height + 15)
        elif label_position == 'left':
            text_x = max(10, x - text_width - margin * 2 - 15)
            text_y = max(total_height + 15, y + h // 2)
        elif label_position == 'right':
            text_x = min(img_width - text_width - margin * 2 - 10, x + w + 15)
            text_y = max(total_height + 15, y + h // 2)
        else:
            text_x = max(10, x)
            text_y = max(total_height + 15, y - 15)

        # Clamp to image boundaries
        text_x = max(margin, min(text_x, img_width - text_width - margin * 2))
        text_y = max(total_height + margin, min(text_y, img_height - margin))

        # Draw semi-transparent background rectangle
        text_bg_layer = annotated_img.copy()
        cv2.rectangle(
            text_bg_layer,
            (text_x - margin, text_y - total_height - margin),
            (text_x + text_width + margin, text_y + margin),
            label_background_color, -1
        )
        cv2.addWeighted(
            text_bg_layer, label_opacity,
            annotated_img, 1 - label_opacity,
            0, annotated_img
        )

        # Draw each line of text
        for i, line in enumerate(text.split('\n')):
            y_offset = (
                text_y
                - (total_height - single_line_height)
                + (i * (single_line_height + 15))
            )
            cv2.putText(
                annotated_img, line, (text_x, y_offset),
                font, font_scale, text_color,
                font_thickness, cv2.LINE_AA
            )

        sequential_id += 1