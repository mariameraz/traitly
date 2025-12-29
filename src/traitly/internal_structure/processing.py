# ============================================================================
# STANDARD LIBRARY
# ============================================================================
from typing import List, Dict, Tuple, Optional
import math

# ============================================================================
# THIRD-PARTY LIBRARIES
# ============================================================================
import cv2
import numpy as np


#################################################################################################
# Determinate inner pericarp area
#################################################################################################

def get_inner_pericarp_area(
    locules: List[int], 
    contours: List[np.ndarray], 
    px_per_cm: Optional[float] = None, 
    img: Optional[np.ndarray] = None,
    draw_inner_pericarp: bool = False, 
    use_ellipse: bool = False, 
    epsilon: float = 0.0001, 
    contour_thickness: int = 2, 
    contour_color: Tuple[int, int, int] = (0, 240, 240)
) -> Tuple[float, float]:
    """
    Calculates and visualizes the inner pericarp area (enclosing locules) using either ellipse fitting 
    or convex hull approximation. Returns the calculated area in both pixels and square centimeters.

    Args:
        locules: Indices of contours in `contours` that correspond to fruit locules.
        contours: Detected contours (as returned by cv2.findContours()).
        px_per_cm: Average pixels per centimeter conversion factor. If None, area_cm2 will be np.nan.
        img: Input BGR image (uint8) where contours will be drawn (if draw_inner_pericarp=True).
        draw_inner_pericarp: If True, draws the contour on `img`.
        use_ellipse: If True, uses ellipse fitting; otherwise uses convex hull.
        epsilon: Smoothing factor as percentage of arc length (range: [0, 1]).
        contour_thickness: Thickness of drawn contours in pixels.
        contour_color: BGR color for contours (default: cyan).

    Returns:
        tuple: (area_cm2, area_px)
            - area_cm2: Calculated area in square centimeters (np.nan if px_per_cm is None).
            - area_px: Calculated area in square pixels.

    Raises:
        ValueError: If `epsilon` is outside [0, 1] or `contours`/`loculi` indices are invalid.
        cv2.error: If OpenCV operations fail (e.g., insufficient points for ellipse fitting).

    Notes:
        - For ellipse fitting: Requires ≥5 contour points (returns area=0 if insufficient).
        - Convex hull: More stable for irregular shapes but may overestimate area.
        - Smoothing (epsilon): Lower values preserve detail; higher values simplify the contour.
        - Color convention: Uses BGR (OpenCV standard) for `contour_color`.
        - Area conversion: Uses average px_per_cm for both dimensions.
    """
    if draw_inner_pericarp and img is None:
        raise ValueError("img cannot be None when draw_inner_pericarp=True")

    if not locules:
        return 0.0, 0.0
        
    all_points = np.vstack([contours[i] for i in locules])
    area_px = 0.0
    
    if use_ellipse and all_points.shape[0] >= 5:
        ellipse = cv2.fitEllipse(all_points.astype(np.float32))
        if draw_inner_pericarp:
            cv2.ellipse(img, ellipse, contour_color, contour_thickness)
        
        a, b = ellipse[1][0] / 2, ellipse[1][1] / 2
        area_px = np.pi * a * b
    else:
        hull = cv2.convexHull(all_points)
        epsilon_val = epsilon * cv2.arcLength(hull, True)
        smoothed_hull = cv2.approxPolyDP(hull, epsilon_val, True)
        if draw_inner_pericarp:
            cv2.drawContours(img, [smoothed_hull], -1, contour_color, contour_thickness)
        area_px = cv2.contourArea(smoothed_hull)
    
    # Handle px_per_cm - ensure it's a valid float, not a dict or other type
    if px_per_cm is not None and not isinstance(px_per_cm, dict) and isinstance(px_per_cm, (int, float)) and px_per_cm > 0:
        area_cm2 = area_px / (px_per_cm ** 2)
    else:
        area_cm2 = np.nan
    
    return area_cm2, area_px


#################################################################################################
# Calculate fruit centroids
#################################################################################################

def calculate_fruit_centroids(contours: List[np.ndarray]) -> List[Optional[Tuple[int, int]]]:
    """
    Calculates the centroid (cx, cy) for each contour in the list.

    Args:
        contours: List of contours (OpenCV format).

    Returns:
        A list containing centroid coordinates (cx, cy) for each contour index. 
        If the contour has zero area, returns None for that position.
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
    centroid: Tuple[int, int]
) -> List[Dict]:
    """ 
    Precalculates and stores geometric data about locules from image contours to optimize further processing.

    Args:
        contours: List of contour points (OpenCV format).
        locules: Indices of contours that represent locules.
        centroid: Reference centroid as a tuple (x, y).
 
    Returns:
        A list of dictionaries, each containing:
            - 'contour_id' (int): Contour identifier.
            - 'centroid' (Tuple[int, int]): (x, y) coordinates of the locule's centroid.
            - 'area' (float): Area of the locule in pixels.
            - 'perimeter' (float): Perimeter of the locule in pixels.
            - 'contour' (np.ndarray): Original contour points.
            - 'polar_coord' (Tuple[float, float]): (angle_in_radians, radius) relative to reference centroid.
            - 'circularity' (float): Circularity in range [0, 1], where 1 = perfect circle.

    Notes:
        - Uses OpenCV moments for centroid calculation.
        - Skips contours with zero area (m00 = 0).
    """
    locules_data = []
    cx_ref, cy_ref = centroid

    for locule in locules:
        contour = contours[locule]
        M = cv2.moments(contour)
    
        if M["m00"] == 0:
            continue
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        dx, dy = cx - cx_ref, cy - cy_ref
        angle = math.atan2(dy, dx) % (2 * np.pi)
        radius = math.hypot(dx, dy)

        locules_data.append({
            'contour_id': locule,
            'centroid': (cx, cy),
            'area': area,
            'perimeter': perimeter,
            'contour': contour,
            'polar_coord': (angle, radius),
            'circularity': (4 * np.pi * area) / (perimeter ** 2)
        })

    return locules_data


#################################################################################################
# Extract and transform fruit contour
#################################################################################################

def get_fruit_contour(
    contours: List[np.ndarray], 
    fruit_id: int, 
    contour_mode: str = 'raw', 
    epsilon_factor: float = 0.0001
) -> np.ndarray:
    """
    Extract and optionally transform a fruit contour.
    
    Args:
        contours: List of contours from cv2.findContours
        fruit_id: Index of the fruit contour to extract
        contour_mode: Transformation mode:
            - 'raw': Original contour (default)
            - 'hull': Convex hull approximation
            - 'approx': Approximate polygon using Douglas-Peucker
            - 'ellipse': Fit an ellipse around the contour
            - 'circle': Fit minimum enclosing circle
        epsilon_factor: Approximation factor for 'approx' mode
        
    Returns:
        Transformed contour points
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
        epsilon = max(1.0, epsilon_factor * peri)
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


#################################################################################################
# Get inner pericarp contour
#################################################################################################

def get_inner_pericarp_contour(
    locules: List[int],
    contours: List[np.ndarray],
    use_ellipse: bool = False,
    epsilon: float = 0.001
) -> np.ndarray:
    """
    Get the inner pericarp contour that encloses all locules.
    
    Args:
        locules: List of locule indices
        contours: List of all contours
        use_ellipse: If True, fit an ellipse instead of using convex hull
        epsilon: Epsilon value for polygon approximation
    
    Returns:
        Inner pericarp contour
    """
    if not locules:
        return np.array([])
    
    all_points = []
    for locule_id in locules:
        contour = contours[locule_id]
        if len(contour) > 0:
            all_points.extend(contour.reshape(-1, 2))
    
    if not all_points:
        return np.array([])
    
    all_points = np.array(all_points)
    
    if use_ellipse and len(all_points) >= 5:
        ellipse = cv2.fitEllipse(all_points)
        inner_contour = cv2.ellipse2Poly(
            (int(ellipse[0][0]), int(ellipse[0][1])),
            (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)),
            int(ellipse[2]),
            0, 360,
            delta=5
        ).reshape(-1, 1, 2)
    else:
        inner_contour = cv2.convexHull(all_points)
    
    if epsilon > 0:
        perimeter = cv2.arcLength(inner_contour, True)
        inner_contour = cv2.approxPolyDP(inner_contour, epsilon * perimeter, True)
    
    return inner_contour


#################################################################################################
# Calculate pericarp thickness using radial sampling
#################################################################################################

# def calculate_pericarp_thickness_radial(
#     outer_contour: np.ndarray, 
#     inner_contour: np.ndarray, 
#     fruit_centroid: Tuple[float, float],
#     img_shape: Tuple[int, int],
#     num_rays: int = 360,
#     px_per_cm: Optional[float] = None
# ) -> Dict[str, float]:
#     """
#     Calcula el grosor del pericarpio usando muestreo radial.
    
#     Args:
#         outer_contour: Contorno exterior del fruto
#         inner_contour: Contorno interior (convex hull de lóculos)
#         fruit_centroid: Centro del fruto (cx, cy)
#         img_shape: Forma de la imagen (height, width)
#         num_rays: Número de rayos a trazar (más = más preciso)
#         px_per_cm: Conversión de píxeles a cm
    
#     Returns:
#         dict con estadísticas: mean, median, std, min, max thickness
#     """
#     cx, cy = fruit_centroid
#     height, width = img_shape[:2]
    
#     # Crear máscaras binarias
#     mask_outer = np.zeros((height, width), dtype=np.uint8)
#     mask_inner = np.zeros((height, width), dtype=np.uint8)
#     cv2.drawContours(mask_outer, [outer_contour], -1, 255, -1)
#     cv2.drawContours(mask_inner, [inner_contour], -1, 255, -1)
    
#     thicknesses_px = []
#     max_search = max(height, width)
#     angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    
#     for angle in angles:
#         dx, dy = np.cos(angle), np.sin(angle)
        
#         # Buscar intersección con contorno exterior
#         outer_dist = 0
#         for r in range(1, max_search):
#             x, y = int(cx + dx * r), int(cy + dy * r)
#             if not (0 <= x < width and 0 <= y < height):
#                 break
#             if mask_outer[y, x] == 0:
#                 outer_dist = r - 1
#                 break
        
#         # Buscar intersección con contorno interior
#         inner_dist = 0
#         for r in range(1, max_search):
#             x, y = int(cx + dx * r), int(cy + dy * r)
#             if not (0 <= x < width and 0 <= y < height):
#                 break
#             if mask_inner[y, x] == 255:
#                 inner_dist = r
#                 break
        
#         if outer_dist > inner_dist > 0:
#             thicknesses_px.append(outer_dist - inner_dist)
    
#     if not thicknesses_px:
#         return {
#             'mean_thickness_cm': np.nan,
#             'median_thickness_cm': np.nan,
#             'std_thickness_cm': np.nan,
#             'min_thickness_cm': np.nan,
#             'max_thickness_cm': np.nan,
#             'cv_thickness': np.nan
#         }
    
#     thicknesses_px = np.array(thicknesses_px)
    
#     if px_per_cm and isinstance(px_per_cm, (int, float)) and px_per_cm > 0:
#         thicknesses_cm = thicknesses_px / px_per_cm
#         mean_val = np.mean(thicknesses_cm)
#         return {
#             'mean_thickness_cm': float(mean_val),
#             'median_thickness_cm': float(np.median(thicknesses_cm)),
#             'std_thickness_cm': float(np.std(thicknesses_cm)),
#             'min_thickness_cm': float(np.min(thicknesses_cm)),
#             'max_thickness_cm': float(np.max(thicknesses_cm)),
#             'cv_thickness': float(np.std(thicknesses_cm) / mean_val * 100)
#         }
#     else:
#         mean_val = np.mean(thicknesses_px)
#         return {
#             'mean_thickness_px': float(mean_val),
#             'median_thickness_px': float(np.median(thicknesses_px)),
#             'std_thickness_px': float(np.std(thicknesses_px)),
#             'min_thickness_px': float(np.min(thicknesses_px)),
#             'max_thickness_px': float(np.max(thicknesses_px)),
#             'cv_thickness': float(np.std(thicknesses_px) / mean_val * 100)
#         }

def calculate_pericarp_thickness_radial(
    outer_contour, inner_contour, fruit_centroid, 
    img_shape, num_rays=180, px_per_cm=None
):
    """
    Calculate pericarp thickness using radial sampling.
    CORRECTED VERSION - Fixed intersection distance calculation.
    
    Args:
        outer_contour: Outer fruit contour
        inner_contour: Inner contour (convex hull of locules)
        fruit_centroid: Fruit center (cx, cy)
        img_shape: Image shape (height, width)
        num_rays: Number of rays to trace (default: 360)
        px_per_cm: Pixel to cm conversion factor
    
    Returns:
        Dict with thickness statistics: mean, median, std, min, max, cv
    """
    cx, cy = fruit_centroid
    height, width = img_shape[:2]
    
    # Create binary masks
    mask_outer = np.zeros((height, width), dtype=np.uint8)
    mask_inner = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask_outer, [outer_contour], -1, 255, -1)
    cv2.drawContours(mask_inner, [inner_contour], -1, 255, -1)
    
    # Precalculate all rays
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    max_search = max(height, width)
    
    # Create radial distance grid
    r_grid = np.arange(1, max_search)
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    
    thicknesses_px = []
    
    # Process rays
    for i, (dx, dy) in enumerate(zip(cos_angles, sin_angles)):
        # Calculate all positions along the ray
        xs = (cx + dx * r_grid).astype(int)
        ys = (cy + dy * r_grid).astype(int)
        
        # Filter valid coordinates
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        xs_valid = xs[valid]
        ys_valid = ys[valid]
        r_valid = r_grid[valid]  # CRITICAL: Keep corresponding radial distances
        
        if len(xs_valid) == 0:
            continue
        
        # Find outer boundary intersection
        outer_vals = mask_outer[ys_valid, xs_valid]
        outer_idx = np.where(outer_vals == 0)[0]
        
        if len(outer_idx) == 0:
            continue
            
        outer_r = r_valid[outer_idx[0] - 1] if outer_idx[0] > 0 else 0
        
        # Find inner boundary intersection
        inner_vals = mask_inner[ys_valid, xs_valid]
        inner_idx = np.where(inner_vals == 255)[0]
        
        if len(inner_idx) == 0:
            continue
            
        inner_r = r_valid[inner_idx[0]]
        
        # Calculate thickness (using actual radial distances)
        if outer_r > inner_r > 0:
            thicknesses_px.append(outer_r - inner_r)
    
    # Handle case with no valid measurements
    if not thicknesses_px:
        return {
            'mean_thickness_cm': np.nan,
            'median_thickness_cm': np.nan,
            'std_thickness_cm': np.nan,
            'min_thickness_cm': np.nan,
            'max_thickness_cm': np.nan,
            'cv_thickness': np.nan
        }
    
    thicknesses_px = np.array(thicknesses_px)
    
    # Convert to cm if calibration available
    if px_per_cm and isinstance(px_per_cm, (int, float)) and px_per_cm > 0:
        thicknesses_cm = thicknesses_px / px_per_cm
        mean_val = np.mean(thicknesses_cm)
        return {
            'mean_thickness_cm': float(mean_val),
            'median_thickness_cm': float(np.median(thicknesses_cm)),
            'std_thickness_cm': float(np.std(thicknesses_cm)),
            'min_thickness_cm': float(np.min(thicknesses_cm)),
            'max_thickness_cm': float(np.max(thicknesses_cm)),
            'cv_thickness': float((np.std(thicknesses_cm) / mean_val * 100) if mean_val > 0 else np.nan)
        }
    else:
        mean_val = np.mean(thicknesses_px)
        return {
            'mean_thickness_px': float(mean_val),
            'median_thickness_px': float(np.median(thicknesses_px)),
            'std_thickness_px': float(np.std(thicknesses_px)),
            'min_thickness_px': float(np.min(thicknesses_px)),
            'max_thickness_px': float(np.max(thicknesses_px)),
            'cv_thickness': float((np.std(thicknesses_px) / mean_val * 100) if mean_val > 0 else np.nan)
        }