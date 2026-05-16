# traitly/fruit_phenotyping/processing.py
"""
Contour processing and pericarp measurement utilities for fruit phenotyping.

Provides functions to extract and transform fruit contours, compute
locule geometry, calculate internal pericarp areas, and estimate radial
pericarp thickness. Designed to be called from
:func:`~traitly.fruit_phenotyping.analysis.analyze_fruits_morphology`
and its sub-functions.

All area and distance outputs are returned in both pixel and centimetre
units via ``px_per_cm``; when ``None`` or invalid, centimetre values
are ``NaN``.
"""

# ============================================================================
# STANDARD LIBRARY
# ============================================================================
from typing import List, Dict, Tuple, Optional
import math
import warnings

# ============================================================================
# THIRD-PARTY LIBRARIES
# ============================================================================
import cv2
import numpy as np
from scipy.spatial import Delaunay, KDTree

# Internal imports
from traitly.utils.constants import label_positions

#################################################################################################
# Calculate fruit centroids
#################################################################################################

def calculate_fruit_centroids(
    contours: List[np.ndarray],
) -> List[Optional[Tuple[float, float]]]:
    """
    Calculate the centroid of each contour in a list.

    Uses ``cv2.moments`` to compute ``(cx, cy)`` for each contour.
    Returns ``None`` for contours whose zeroth moment (area) is zero,
    which occurs for degenerate or empty contours.

    Parameters
    ----------
    contours : list of np.ndarray
        Contours in OpenCV format, as returned by ``cv2.findContours``.

    Returns
    -------
    list of tuple of float or None
        List of length ``len(contours)``. Each entry is a
        ``(cx, cy)`` centroid tuple, or ``None`` if the contour has
        zero area.
    """
    centroids = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            centroids.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
        else:
            centroids.append(None)
    return centroids


#################################################################################################
# Precalculate locules data
#################################################################################################

def precalculate_locules_data(
    contours: List[np.ndarray],
    locules: List[int],
    fruit_centroid: Tuple[float, float],
) -> List[Dict]:
    """
    Precalculate geometric properties for a set of locule contours.

    For each locule index in ``locules``, calcculates area, perimeter,
    centroid, polar coordinates relative to ``fruit_centroid``, and
    circularity using ``cv2.moments`` and ``cv2.arcLength``. Contours
    with zero area (``m00 = 0``) are silently skipped.

    Parameters
    ----------
    contours : list of np.ndarray
        Full list of all detected contours in OpenCV format.
    locules : list of int
        Indices into ``contours`` identifying the locule contours to
        process.
    fruit_centroid : tuple of float
        ``(cx, cy)`` centroid of the parent fruit, used as the origin
        for polar coordinate computation.

    Returns
    -------
    list of dict
        One dictionary per valid locule, with keys:

        - ``'contour_id'`` (int) – index into ``contours``.
        - ``'centroid'`` (tuple of int) – ``(cx, cy)`` of the locule.
        - ``'area'`` (float) – contour area in pixels².
        - ``'perimeter'`` (float) – contour perimeter in pixels.
        - ``'contour'`` (np.ndarray) – original contour points.
        - ``'polar_coord'`` (tuple of float) – ``(angle_rad, radius_px)``
          relative to ``fruit_centroid``, angle in [0, 2π).
        - ``'circularity'`` (float) – ``4π·area / perimeter²`` in [0, 1],
          or ``NaN`` if perimeter is zero.
    """
    locules_data = []
    cx_ref, cy_ref = fruit_centroid

    for locule in locules:
        contour = contours[locule]
        M = cv2.moments(contour)


        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        area = M['m00']
        perimeter = cv2.arcLength(contour, True)

        dx, dy = cx - cx_ref, cy - cy_ref
        angle = math.atan2(dy, dx) % (2 * np.pi)
        radius = math.hypot(dx, dy)

        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
        else:
            circularity = np.nan

        locules_data.append({
            'contour_id': locule,
            'centroid': (cx, cy),
            'area': area, #px
            'perimeter': perimeter, #px
            'contour': contour,
            'polar_coord': (angle, radius),
            'circularity': circularity
        })

    return locules_data


#################################################################################################
# Extract and transform fruit contour
#################################################################################################

def get_fruit_contour(
    contours: List[np.ndarray],
    fruit_id: int,
    contour_mode: str = 'raw',
    epsilon: float = 0.002,
) -> np.ndarray:
    """
    Extract and optionally transform a fruit contour.

    Retrieves the contour at ``contours[fruit_id]`` and applies the
    transformation specified by ``contour_mode``:

    - ``'raw'`` – returns the original contour unchanged.
    - ``'hull'`` – replaces with the convex hull via ``cv2.convexHull``.
    - ``'approx'`` – simplifies with Douglas-Peucker via
      ``cv2.approxPolyDP``, using ``epsilon * perimeter`` as the
      tolerance.
    - ``'ellipse'`` – fits an ellipse via ``cv2.fitEllipse`` and
      converts to a polygon via ``cv2.ellipse2Poly``. Requires at least
      5 contour points.
    - ``'circle'`` – fits the minimum enclosing circle via
      ``cv2.minEnclosingCircle`` and samples 36 equally spaced points.

    Parameters
    ----------
    contours : list of np.ndarray
        Full list of all detected contours in OpenCV format.
    fruit_id : int
        Index into ``contours`` identifying the fruit. Must be in
        ``[0, len(contours) - 1]``.
    contour_mode : str, optional
        Transformation to apply. One of ``'raw'``, ``'hull'``,
        ``'approx'``, ``'ellipse'``, ``'circle'``. Default is ``'raw'``.
    epsilon : float, optional
        Approximation factor for ``'approx'`` mode. Multiplied by the
        contour perimeter to obtain the absolute tolerance. Default is
        0.002.

    Returns
    -------
    np.ndarray
        Transformed contour in OpenCV format ``(N, 1, 2)``.

    Raises
    ------
    ValueError
        If ``contour_mode`` is not one of the supported modes, or if
        ``'ellipse'`` mode is requested on a contour with fewer than 5
        points.
    IndexError
        If ``fruit_id`` is outside ``[0, len(contours) - 1]``.
    """
    valid_modes = ['raw', 'hull', 'approx', 'ellipse', 'circle']
    if contour_mode not in valid_modes:
        raise ValueError(f"contour_mode must be one of {valid_modes}, got '{contour_mode}'")

    if not 0 <= fruit_id < len(contours):
        raise IndexError(f"fruit_id {fruit_id} out of range [0, {len(contours)-1}]")

    fruit_contour = contours[fruit_id]

    if contour_mode == 'hull':
        fruit_contour = cv2.convexHull(fruit_contour)

    elif contour_mode == 'approx':
        peri = cv2.arcLength(fruit_contour, True)
        epsilon = epsilon * peri
        fruit_contour = cv2.approxPolyDP(fruit_contour, epsilon, True)

    elif contour_mode == 'ellipse':
        if len(fruit_contour) < 5:
            raise ValueError("Ellipse fitting requires at least 5 contour points")
        ellipse = cv2.fitEllipse(fruit_contour)
        center = (int(ellipse[0][0]), int(ellipse[0][1]))
        axes = (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2))
        angle = int(ellipse[2])
        fruit_contour = cv2.ellipse2Poly(center, axes, angle, 0, 360, 2).reshape(-1, 1, 2)

    elif contour_mode == 'circle':
        (x, y), radius = cv2.minEnclosingCircle(fruit_contour)
        center = (int(x), int(y))
        radius = int(radius)

        angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
        circle_points = np.column_stack([
            center[0] + radius * np.cos(angles),
            center[1] + radius * np.sin(angles)
        ]).astype(np.int32)

        fruit_contour = circle_points.reshape(-1, 1, 2)

    return fruit_contour


######################################################
# Calculate pericarp thickness using radial sampling #
######################################################

def calculate_pericarp_thickness_radial(
    outer_contour: np.ndarray,
    inner_contour: np.ndarray,
    fruit_centroid: Tuple[float, float],
    img_shape: Tuple[int, int],
    num_rays: int = 180,
    px_per_cm: Optional[float] = None,
) -> Dict[str, float]:
    """
    Estimate pericarp thickness and fruit lobedness using radial ray sampling.

    Crops a tight ROI around ``outer_contour``, rasterizes both contours
    into binary masks, and casts ``num_rays`` from ``fruit_centroid`` at
    evenly spaced angles. For each ray, the outer boundary is located as
    the last pixel inside the outer mask and the inner boundary as the
    first pixel inside the inner mask. Thickness is the difference between
    these two radii. Lobedness is the standard deviation of outer radii
    across all rays, capturing shape irregularity.

    Parameters
    ----------
    outer_contour : np.ndarray
        Processed outer fruit contour in OpenCV format, used to define
        the outer pericarp boundary.
    inner_contour : np.ndarray
        Convex hull of locules (from :func:`get_internal_pericarp_contour`),
        used to define the inner pericarp boundary.
    fruit_centroid : tuple of float
        ``(cx, cy)`` centroid of the fruit, used as the ray origin.
    img_shape : tuple of int
        ``(height, width)`` of the full image, used to clamp the ROI
        boundaries.
    num_rays : int, optional
        Number of evenly spaced rays cast from the centroid. Higher
        values improve accuracy at the cost of speed. Default is 180.
    px_per_cm : float or None, optional
        Pixel-to-centimetre conversion factor. If ``None`` or invalid,
        all distance outputs are in pixels. Default is ``None``.

    Returns
    -------
    dict of {str : float}
        Dictionary with keys suffixed by the active unit (``'cm'`` or
        ``'px'``):

        - ``outer_pericarp_mean_thickness_{unit}`` – mean thickness
          across valid rays.
        - ``outer_pericarp_std_thickness_{unit}`` – standard deviation
          of thickness.
        - ``'outer_pericarp_cv_thickness'`` – coefficient of variation
          as a percentage (unitless).
        - ``fruit_lobedness_{unit}`` – standard deviation of outer
          radii, measuring shape irregularity.

        All values are ``NaN`` if no valid rays could be measured.
    """

    # ROI computation for each fruit
    x, y, w, h = cv2.boundingRect(outer_contour)

    margin = 15
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(img_shape[1], x + w + margin)
    y1 = min(img_shape[0], y + h + margin)

    roi_width = x1 - x0
    roi_height = y1 - y0

    # Vectorized contour shifting
    cx, cy = fruit_centroid
    cx_roi = cx - x0
    cy_roi = cy - y0

    # Create masks more efficiently
    mask_outer = np.zeros((roi_height, roi_width), dtype=np.uint8)
    mask_inner = np.zeros((roi_height, roi_width), dtype=np.uint8)

    # Direct contour drawing without copying if contours are read only
    cv2.drawContours(mask_outer, [outer_contour - [x0, y0]], -1, 255, -1)
    cv2.drawContours(mask_inner, [inner_contour - [x0, y0]], -1, 255, -1)

    # Pre compute all angles and trigonometric values
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    max_search = int(np.ceil(np.sqrt(roi_width**2 + roi_height**2)))
    r_grid = np.arange(1, max_search)

    # Pre allocate arrays
    thicknesses_px = []
    outer_distances_px = []

    # Vectorized computation for all rays
    xs_all = (cx_roi + cos_angles[:, None] * r_grid).astype(int)
    ys_all = (cy_roi + sin_angles[:, None] * r_grid).astype(int)

    # Create validity mask for all points
    valid_mask = (xs_all >= 0) & (xs_all < roi_width) & (ys_all >= 0) & (ys_all < roi_height)

    for i in range(num_rays):
        # Get valid indices for this ray
        valid = valid_mask[i]
        if not np.any(valid):
            continue

        xs_valid = xs_all[i, valid]
        ys_valid = ys_all[i, valid]
        r_valid = r_grid[valid]

        # Get values along ray
        outer_vals = mask_outer[ys_valid, xs_valid]
        inner_vals = mask_inner[ys_valid, xs_valid]

        # Find outer boundary (transition from 255 to 0)
        # Using argmax to find first 0 (faster than where)
        outer_zero_idx = np.argmax(outer_vals == 0)

        if outer_vals[outer_zero_idx] == 0 and outer_zero_idx > 0:
            outer_r = r_valid[outer_zero_idx - 1]
        elif outer_vals[0] == 0:
            continue
        else:
            # All pixels are inside, take the last valid
            outer_r = r_valid[-1]

        outer_distances_px.append(outer_r)

        # Find internal boundary (first 255 in internal mask)
        inner_white_idx = np.argmax(inner_vals == 255)

        if inner_vals[inner_white_idx] == 255:
            inner_r = r_valid[inner_white_idx]
            if outer_r > inner_r:
                thicknesses_px.append(outer_r - inner_r)

    if not thicknesses_px:
        return {
            'outer_pericarp_mean_thickness_cm': np.nan,
            'outer_pericarp_std_thickness_cm': np.nan,
            'outer_pericarp_cv_thickness': np.nan,
            'fruit_lobedness_cm': np.nan
        }

    thicknesses_px = np.array(thicknesses_px)
    lobedness_px = float(np.std(outer_distances_px)) if outer_distances_px else np.nan

    # Convert to cm if needed
    convert_cm = px_per_cm and isinstance(px_per_cm, (int, float)) and px_per_cm > 0

    if convert_cm:
        inv = 1.0 / px_per_cm
        thicknesses = thicknesses_px * inv
        lobedness = lobedness_px * inv
        prefix = 'cm'
    else:
        thicknesses = thicknesses_px
        lobedness = lobedness_px
        prefix = 'px'

    mean_val = np.mean(thicknesses)

    return {
        f'outer_pericarp_mean_thickness_{prefix}': float(mean_val),
        f'outer_pericarp_std_thickness_{prefix}': float(np.std(thicknesses)),
        'outer_pericarp_cv_thickness': float((np.std(thicknesses) / mean_val * 100) if mean_val > 0 else np.nan),
        f'fruit_lobedness_{prefix}': float(lobedness)
    }


#################################################################################################
# Get internal pericarp contour
#################################################################################################

def get_internal_pericarp_contour(
    locules: List[int],
    contours: List[np.ndarray],
    dilation_factor: Optional[float] = None,
    img_shape: Optional[Tuple] = None,
    fruit_id: Optional[int] = None,
) -> np.ndarray:

    if not locules:
        return np.array([])

    all_points = np.vstack([contours[i] for i in locules])

    if dilation_factor: # option 1, dilated mask
        if img_shape is None:
            raise ValueError("img_shape is required when dilation_factor is provided")

        ref = contours[fruit_id] if fruit_id is not None else all_points
        x, y, w, h = cv2.boundingRect(ref)
        pad = 5
        x1, y1 = max(x - pad, 0), max(y - pad, 0)
        x2, y2 = min(x + w + pad, img_shape[1]), min(y + h + pad, img_shape[0])

        roi_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        shifted = [contours[i] - np.array([[[x1, y1]]]) for i in locules]
        for c in shifted:
            cv2.drawContours(roi_mask, [c], -1, 255, -1)

        areas = [cv2.contourArea(contours[i]) for i in locules]
        mean_radius = int(np.sqrt(np.mean(areas) / np.pi) * dilation_factor)
        mean_radius = max(mean_radius, 3)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mean_radius * 2 + 1,) * 2)
        dilated = cv2.dilate(roi_mask, kernel, iterations=2)

        dist = cv2.distanceTransform(dilated, cv2.DIST_L2, 5)
        thresh = dist[roi_mask > 0].min()

        snapped = (dist >= thresh).astype(np.uint8) * 255

        k_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        snapped = cv2.morphologyEx(snapped, cv2.MORPH_CLOSE, k_smooth)

        cnts, _ = cv2.findContours(snapped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            return cv2.convexHull(all_points)

        best = max(cnts, key=cv2.contourArea)
        return (best + np.array([[[x1, y1]]], dtype=np.int32)).astype(np.int32)

    else: # option 2, convex hull
        return cv2.convexHull(all_points)

#################################################################################################
# Get internal pericarp area and draw it
#################################################################################################

def get_internal_pericarp_area(
    locules: List[int],
    contours: List[np.ndarray],
    px_per_cm: Optional[float] = None,
    img: Optional[np.ndarray] = None,
    draw_inner_pericarp: bool = False,
    contour_thickness: int = 2,
    contour_color: Tuple[int, int, int] = (0, 240, 240),
    dilation_factor: Optional[float] = None,
    img_shape: Optional[Tuple] = None,
    fruit_id: Optional[int] = None
) -> Tuple[float, float]:

    if draw_inner_pericarp and img is None:
        raise ValueError("img cannot be None when draw_inner_pericarp=True")

    if not locules:
        return np.nan, np.nan, np.array([])

    hull = get_internal_pericarp_contour(
        locules,
        contours,
        dilation_factor=dilation_factor,
        img_shape=img_shape,
        fruit_id=fruit_id
    )

    if len(hull) == 0:
        return np.nan, np.nan, np.array([])

    if draw_inner_pericarp:
        cv2.drawContours(img, [hull], -1, contour_color, contour_thickness)

    area_px2 = cv2.contourArea(hull)

    if px_per_cm is not None and isinstance(px_per_cm, (int, float)) and px_per_cm > 0:
        area_cm2 = area_px2 / (px_per_cm ** 2)
    else:
        area_cm2 = np.nan

    return area_cm2, area_px2, hull


###########################################################################################
# Simplified image annotation - use when calling FruitInternalAnalyzer.analyze_color() only
###########################################################################################

def annotate_all_fruits(
    fruit_locule_map: Dict[int, List[int]],
    contours: List[np.ndarray],
    annotated_img: np.ndarray,
    font_scale: float = 2,
    font_thickness: int = 2,
    pericarp_ext_color: Tuple[int, int, int] = (0, 255, 0),
    pericarp_ext_thickness: int = 2,
    pericarp_int_color: Tuple[int, int, int] = (255, 255, 0),
    pericarp_int_thickness: int = 2,
    locule_thickness: int = 2,
    locule_color: Tuple[int, int, int] = (255, 0, 255),
    label_position: str = 'left',
    margin: int = 10,
    text_color: Tuple[int, int, int] = (0, 0, 0),
    label_background_color: Tuple[int, int, int] = (255, 255, 255),
    label_opacity: float = 0.7,
    verbose: bool = True,
    dilation_factor: Optional[float] = None
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

        # Draw internal pericarp contour
        internal_per_contour = get_internal_pericarp_contour(
                locules = locule_ids,
                contours = contours,
                img_shape = annotated_img.shape[:2],
                fruit_id = fruit_id,
                dilation_factor = dilation_factor)

        if internal_per_contour is not None and len(internal_per_contour) > 0:
            cv2.drawContours(
                annotated_img, [internal_per_contour], -1,
                pericarp_int_color, pericarp_int_thickness
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

        # Position of the label
        img_height, img_width = annotated_img.shape[:2]

        if label_position == 'top':
            text_x = max(10, x)
            text_y = max(total_height + 15, y - 15)
        elif label_position == 'bottom':
            text_x = max(10, x)
            text_y = min(img_height - 15, y + h + total_height + 15)
        elif label_position == 'left':
            text_x = max(10, x - text_width - margin * 2 - 15)
            text_y = max(total_height + 15, y + h // 2)
        else:
            text_x = min(img_width - text_width - margin * 2 - 10, x + w + 15) # rigth
            text_y = max(total_height + 15, y + h // 2)

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
