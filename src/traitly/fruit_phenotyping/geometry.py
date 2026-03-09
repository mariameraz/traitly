# traitly/fruit_phenotyping/geometry.py
"""
Geometric shape descriptors for fruit phenotyping pipelines.

Provides functions to compute axes, rotated bounding boxes, and
area- and perimeter-based morphological metrics from fruit contours.
Designed to be called from :func:`~traitly.fruit_phenotyping.analysis._calculate_fruit_metrics`.

All functions support dual-unit output (pixels and centimetres) via
``px_per_cm``; when ``None`` or invalid, centimetre values are ``NaN``.
"""
# ============================================================================
# STANDARD LIBRARY
# ============================================================================
from typing import Tuple, Optional

# ============================================================================
# THIRD-PARTY LIBRARIES
# ============================================================================
import cv2
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform

# ============================================================================
# INTERNAL IMPORTS
# ===========================================================================
from .processing import get_fruit_contour

#################################################################################################
# Calculate minor axis (fruit width/length approximation)
#################################################################################################
def calculate_axes(
    fruit_contour: np.ndarray,
    px_per_cm: Optional[float] = None,
    img: Optional[np.ndarray] = None,
    draw_axes: bool = False,
    major_axis_color: Tuple[int, int, int] = (0, 255, 0),
    minor_axis_color: Tuple[int, int, int] = (255, 0, 0),
    axis_thickness: int = 2,
    hull_verts: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float]:
    """
    Calculate the major and minor axes of a fruit contour.

    The major axis is the longest chord of the convex hull, computed via
    pairwise distances using :func:`scipy.spatial.distance.pdist` and
    :class:`scipy.spatial.ConvexHull`. The minor axis is the maximum
    perpendicular width to the major axis, projected across all contour
    points. Axis lines are optionally drawn onto ``img`` in-place.

    Parameters
    ----------
    fruit_contour : np.ndarray
        Contour points of shape ``(N, 1, 2)`` or ``(N, 2)``, as returned
        by ``cv2.findContours`` or :func:`get_fruit_contour`.
    px_per_cm : float or None, optional
        Pixel-to-centimeter conversion factor. If ``None`` or invalid,
        cm values are returned as ``NaN``. Default is ``None``.
    img : np.ndarray or None, optional
        BGR image modified in-place with axis lines when ``draw_axes=True``.
        Default is ``None``.
    draw_axes : bool, optional
        Whether to draw the major and minor axes onto ``img``. Default
        is ``False``.
    major_axis_color : tuple of int, optional
        BGR color for the major axis line. Default is ``(0, 255, 0)``.
    minor_axis_color : tuple of int, optional
        BGR color for the minor axis line. Default is ``(255, 0, 0)``.
    axis_thickness : int, optional
        Line thickness in pixels for both axis lines. Default is 2.
    hull_verts : np.ndarray or None, optional
        Precomputed convex hull vertex indices into ``fruit_contour``.
        If ``None``, the hull is computed internally via
        :class:`scipy.spatial.ConvexHull`. Passing precomputed vertices
        avoids redundant computation when called in a loop. Default is
        ``None``.

    Returns
    -------
    tuple of float
        ``(major_cm, minor_cm, major_px, minor_px)`` where:

        - ``major_cm`` – major axis length in centimetres, or ``NaN`` if
          ``px_per_cm`` is ``None`` or invalid.
        - ``minor_cm`` – minor axis length in centimetres, or ``NaN`` if
          ``px_per_cm`` is ``None`` or invalid.
        - ``major_px`` – major axis length in pixels, or ``NaN`` if the
          contour has fewer than 2 points.
        - ``minor_px`` – minor axis length in pixels, or ``NaN`` / ``0.0``
          for degenerate contours.
    """

    # Reshape and convert contour to float32
    points_px = fruit_contour.reshape(-1, 2).astype(np.float32)
    n = points_px.shape[0] # Number of points in contour
    
    # Early exit if not enough points
    if n < 2: 
        return np.nan, np.nan, np.nan, np.nan
    
    # Major axis calculation
    if hull_verts is None:
        if n >= 3:
            verts = ConvexHull(points_px).vertices
        else:
            verts = np.arange(n)
    else:
        verts = hull_verts

    hull_points = points_px[verts]
    
    if len(hull_points) < 2:
        return np.nan, np.nan, np.nan, np.nan
    
    if len(hull_points) == 2:
        max_dist_px = np.linalg.norm(hull_points[1] - hull_points[0])
        point1_idx, point2_idx = verts[0], verts[1]
    else:
        # Calculate all pairwise distances at once using pdist
        dist_matrix = squareform(pdist(hull_points))
        max_idx = np.unravel_index(dist_matrix.argmax(), dist_matrix.shape)
        point1_idx, point2_idx = verts[max_idx[0]], verts[max_idx[1]]
        max_dist_px = dist_matrix[max_idx]
    
    if max_dist_px == 0:
        return np.nan, np.nan, np.nan, np.nan
    
    # Major axis length in cm calculation
    if px_per_cm is not None and isinstance(px_per_cm, (int, float)) and px_per_cm > 0:
        inv_px_per_cm = 1.0 / px_per_cm
        max_dist_cm = max_dist_px * inv_px_per_cm
    else:
        max_dist_cm = np.nan
    
    # Major axis endpoints
    p1_px = points_px[point1_idx]
    p2_px = points_px[point2_idx]
    
    # Minor axis calculation
    # Vector along major axis
    major_vec = p2_px - p1_px
    major_norm = max_dist_px 
    
    if major_norm < 1e-10:
        min_dist_cm = np.nan if (isinstance(px_per_cm, (int, float)) and px_per_cm > 0) else np.nan
        return max_dist_cm, min_dist_cm, max_dist_px, 0.0
    
    # Calculate perpendicular unit vector
    perp_unit = np.array([-major_vec[1], major_vec[0]], dtype=np.float32) / major_norm
    
    # Vectorized projection calculation
    centered_points = points_px - p1_px
    proj = centered_points @ perp_unit
    
    min_dist_px = proj.max() - proj.min()
    
    if isinstance(px_per_cm, (int, float)) and px_per_cm > 0:
        min_dist_cm = min_dist_px * inv_px_per_cm
    else:
        min_dist_cm = np.nan
    
    # Minor axis endpoints
    idx_min = int(np.argmin(proj))
    idx_max = int(np.argmax(proj))
    p_min_px = points_px[idx_min]
    p_max_px = points_px[idx_max]
    
    # Draw axes if requested
    if draw_axes and img is not None:
        cv2.line(img, 
                 (int(p1_px[0]), int(p1_px[1])), 
                 (int(p2_px[0]), int(p2_px[1])), 
                 major_axis_color, axis_thickness)
        cv2.line(img, 
                 (int(p_min_px[0]), int(p_min_px[1])), 
                 (int(p_max_px[0]), int(p_max_px[1])), 
                 minor_axis_color, axis_thickness)
    
    return max_dist_cm, min_dist_cm, max_dist_px, min_dist_px


#################################################################################################
# Determine rotated bounding box around fruits
#################################################################################################

def rotate_box(
    contour: np.ndarray,
    px_per_cm: Optional[float] = None,
    img: Optional[np.ndarray] = None,
    draw_box: bool = False,
    box_color: Tuple[int, int, int] = (255, 180, 0),
    box_thickness: int = 3,
) -> Tuple[float, float, float, float]:
    """
    Calculate the minimum-area rotated bounding box of a contour.

    Wraps ``cv2.minAreaRect`` and ``cv2.boxPoints`` to compute the tightest
    axis-independent rectangle enclosing ``contour``. The longer side is
    always assigned as the length and the shorter side as the width,
    regardless of orientation. The box is optionally drawn onto ``img``
    in-place via ``cv2.drawContours``.

    Parameters
    ----------
    contour : np.ndarray
        Contour of the object (e.g., fruit) as returned by
        ``cv2.findContours``.
    px_per_cm : float or None, optional
        Pixel-to-centimeter conversion factor. If ``None`` or invalid,
        cm values are returned as ``NaN``. Default is ``None``.
    img : np.ndarray or None, optional
        BGR image modified in-place with the bounding box when
        ``draw_box=True``. Default is ``None``.
    draw_box : bool, optional
        Whether to draw the rotated bounding box onto ``img``. Raises
        ``ValueError`` if ``True`` and ``img`` is ``None``.
        Default is ``False``.
    box_color : tuple of int, optional
        BGR color for the bounding box lines. Default is ``(255, 180, 0)``.
    box_thickness : int, optional
        Line thickness in pixels for the bounding box. Default is 3.

    Returns
    -------
    tuple of float
        ``(box_length_cm, box_width_cm, box_length_px, box_width_px)`` where:

        - ``box_length_cm`` – longer side in centimetres, or ``NaN`` if
          ``px_per_cm`` is ``None`` or invalid.
        - ``box_width_cm`` – shorter side in centimetres, or ``NaN`` if
          ``px_per_cm`` is ``None`` or invalid.
        - ``box_length_px`` – longer side in pixels.
        - ``box_width_px`` – shorter side in pixels.

    Raises
    ------
    ValueError
        If ``draw_box=True`` and ``img`` is ``None``.

    Notes
    -----
    The bounding box is rotation-invariant: it is fitted to the contour's
    orientation, not to the image axes.
    """

    if draw_box and img is None:
        raise ValueError(f"img must be provided when draw_box=True")
    
    # Compute the smallest rotated rectangle that encloses the contour (fruit)
    rotated_rect = cv2.minAreaRect(contour)
    
    # Obtain the width and height in pixels of the computed rectangle
    (center, (width_px, height_px), angle) = rotated_rect
    
    # Convert the rotated box into its 4 corner points
    box_points = cv2.boxPoints(rotated_rect)
    box_points = box_points.astype(int)
    
    # Determine the length (maximum value) and width (minimum value)
    box_length_px = max(width_px, height_px)
    box_width_px = min(width_px, height_px)
    
    if px_per_cm is not None and isinstance(px_per_cm, (int, float)) and px_per_cm > 0:
        inv_px_per_cm = 1.0 / px_per_cm
        box_length_cm = box_length_px * inv_px_per_cm
        box_width_cm = box_width_px * inv_px_per_cm
    else:
        box_length_cm = np.nan
        box_width_cm = np.nan
    
    if draw_box: 
        # Draw the rotated box on the image as a light blue rectangle
        cv2.drawContours(img, [box_points], 0, box_color, box_thickness)
    
    return box_length_cm, box_width_cm, box_length_px, box_width_px


def get_fruit_morphology(
    contour: np.ndarray,
    px_per_cm: Optional[float] = None,
    contour_mode: str = 'raw',
    epsilon: float = 0.002,
) -> dict:
    """
    Calculate comprehensive morphological metrics for a fruit contour.

    Applies an optional contour transformation via :func:`get_fruit_contour`
    (modes ``'hull'``, ``'approx'``, ``'ellipse'``, ``'circle'``), then
    computes area and perimeter with ``cv2.contourArea`` and
    ``cv2.arcLength``. Shape descriptors (circularity, solidity, convexity)
    are derived from the transformed contour and its convex hull.
    When ``contour_mode='hull'``, the hull is reused directly to avoid
    redundant computation.

    Parameters
    ----------
    contour : np.ndarray
        Fruit contour points as returned by ``cv2.findContours``.
        Must contain at least 3 points; shorter contours return all
        ``NaN``.
    px_per_cm : float or None, optional
        Pixel-to-centimeter conversion factor. If ``None`` or invalid,
        all dimensional outputs are in pixels. Default is ``None``.
    contour_mode : str, optional
        Contour transformation applied before metric computation,
        forwarded to :func:`get_fruit_contour`. One of:

        - ``'raw'`` – no transformation (default).
        - ``'hull'`` – convex hull.
        - ``'approx'`` – Douglas-Peucker polygon approximation.
        - ``'ellipse'`` – fitted ellipse.
        - ``'circle'`` – fitted minimum enclosing circle.

    epsilon : float, optional
        Approximation factor forwarded to :func:`get_fruit_contour` when
        ``contour_mode='approx'``. Ignored for other modes. Default is
        0.002.

    Returns
    -------
    dict of {str : float}
        Dictionary with keys suffixed by the active unit:

        - ``fruit_area_{unit}2`` – contour area in cm² or px².
        - ``fruit_perimeter_{unit}`` – contour perimeter in cm or px.
        - ``'fruit_circularity'`` – ``4π·area / perimeter²``, unitless.
        - ``'fruit_solidity'`` – contour area divided by convex hull area,
          unitless.
        - ``'fruit_convexity'`` – hull perimeter divided by contour
          perimeter, unitless.

        All values are ``NaN`` for degenerate contours (fewer than 3
        points, zero area, or zero perimeter).
    """
    # Determine unit
    has_calibration = px_per_cm is not None and isinstance(px_per_cm, (int, float)) and px_per_cm > 0
    unit = 'cm' if has_calibration else 'px'
    unit_area = 'cm2' if has_calibration else 'px2'
    
    # Early exit for invalid contours
    if len(contour) < 3:
        return {
            f'fruit_area_{unit_area}': np.nan,
            f'fruit_perimeter_{unit}': np.nan,
            'fruit_circularity': np.nan,
            'fruit_solidity': np.nan,
            'fruit_convexity': np.nan
        }
    
    if contour_mode == 'raw':
        transformed_contour = contour
    else:
        # Apply contour transformation according to specified mode
        transformed_contour = get_fruit_contour(
            contours=[contour],  # Pass as list with single element
            fruit_id=0,          # Index 0 since there's only one contour
            contour_mode=contour_mode,
        epsilon=epsilon
        )
    
    # Early exit after transformation
    if len(transformed_contour) < 3:
        return {
            f'fruit_area_{unit_area}': np.nan,
            f'fruit_perimeter_{unit}': np.nan,
            'fruit_circularity': np.nan,
            'fruit_solidity': np.nan,
            'fruit_convexity': np.nan
        }
    
    # Calculate area and perimeter on transformed contour
    area_px = cv2.contourArea(transformed_contour)
    perimeter_px = cv2.arcLength(transformed_contour, True)
    
    # Single validation check
    if area_px <= 0 or perimeter_px <= 0:
        return {
            f'fruit_area_{unit_area}': np.nan,
            f'fruit_perimeter_{unit}': np.nan,
            'fruit_circularity': np.nan,
            'fruit_solidity': np.nan,
            'fruit_convexity': np.nan
        }
    
    if has_calibration:
        inv_px_per_cm = 1.0 / px_per_cm
        inv_px_per_cm_sq = inv_px_per_cm * inv_px_per_cm
        area_val = area_px * inv_px_per_cm_sq
        perimeter_val = perimeter_px * inv_px_per_cm
    else:
        area_val = area_px
        perimeter_val = perimeter_px
    
    # Calculate shape metrics using transformed contour
    circularity = (4 * np.pi * area_px) / (perimeter_px ** 2) if perimeter_px > 0 else np.nan
    
    # Reuse hull if contour_mode is 'hull'
    if contour_mode == 'hull':
        hull = transformed_contour  # Already is the convex hull
        hull_area_px = area_px  # Area is already the hull area
        hull_perimeter_px = perimeter_px  # Perimeter is already the hull perimeter
    else:
        hull = cv2.convexHull(transformed_contour)
        hull_perimeter_px = cv2.arcLength(hull, True)
        hull_area_px = cv2.contourArea(hull)
    
    solidity = area_px / hull_area_px if hull_area_px > 0 else np.nan
    
    # Convexity 
    convexity = hull_perimeter_px / perimeter_px if perimeter_px > 0 else np.nan

    return {
        f'fruit_area_{unit_area}': float(area_val),
        f'fruit_perimeter_{unit}': float(perimeter_val),
        'fruit_circularity': float(circularity),
        'fruit_solidity': float(solidity),
        'fruit_convexity': float(convexity)
    }