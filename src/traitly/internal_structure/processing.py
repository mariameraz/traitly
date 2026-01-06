# traitly/internal_structure/processing.py

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
    centroid: Tuple[int, int] # Fruit centroid (reference)
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
            - 'centroid' (Tuple[int, int]): (x, y) coordinates of the fruit's centroid.
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
    epsilon: float = 0.002
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
        epsilon: Approximation factor for 'approx' mode
        
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


#################################################################################################
# Get inner pericarp contour and area
#################################################################################################

def get_inner_pericarp_contour(
    locules: List[int],
    contours: List[np.ndarray]
) -> np.ndarray:
    """
    Get the convex hull that encloses all locules.
    
    Args:
        locules: List of locule indices
        contours: List of all contours
        
    Returns:
        Convex hull contour enclosing all locules
    """
    if not locules:
        return np.array([])
    
    all_points = np.vstack([contours[i] for i in locules])
    return cv2.convexHull(all_points)


def get_inner_pericarp_area(
    locules: List[int], 
    contours: List[np.ndarray], 
    px_per_cm: Optional[float] = None, 
    img: Optional[np.ndarray] = None,
    draw_inner_pericarp: bool = False, 
    contour_thickness: int = 2, 
    contour_color: Tuple[int, int, int] = (0, 240, 240)
) -> Tuple[float, float]:
    """
    Calculates and visualizes the inner pericarp area.
    
    Returns:
        tuple: (area_cm2, area_px)
    """
    
    if draw_inner_pericarp and img is None:
        raise ValueError("img cannot be None when draw_inner_pericarp=True")
    
    if not locules:
        return 0.0, 0.0
    
    # Reuse the previous ip contour function
    hull = get_inner_pericarp_contour(locules, contours)
    
    if len(hull) == 0:
        return 0.0, 0.0
    
    if draw_inner_pericarp:
        cv2.drawContours(img, [hull], -1, contour_color, contour_thickness)
    
    area_px = cv2.contourArea(hull)
    
    if px_per_cm is not None and isinstance(px_per_cm, (int, float)) and px_per_cm > 0:
        area_cm2 = area_px / (px_per_cm ** 2)
    else:
        area_cm2 = np.nan
    
    return area_cm2, area_px


#################################################################################################
# Calculate pericarp thickness using radial sampling
#################################################################################################

def calculate_pericarp_thickness_radial(
    outer_contour, inner_contour, fruit_centroid, 
    img_shape, num_rays=180, px_per_cm=None
):
    """
    Calculate pericarp thickness using radial sampling.
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
    outer_distances_px = []

    # Process rays
    for dx, dy in zip(cos_angles, sin_angles):
        # Calculate all positions along the ray
        xs = (cx + dx * r_grid).astype(int)
        ys = (cy + dy * r_grid).astype(int)
        
        # Filter valid coordinates
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        xs_valid = xs[valid]
        ys_valid = ys[valid]
        r_valid = r_grid[valid]
        
        if len(xs_valid) == 0:
            continue
        
        # Find outer boundary (last pixel INSIDE mask)
        outer_vals = mask_outer[ys_valid, xs_valid]
        outer_idx = np.where(outer_vals == 0)[0]  # First pixel OUTSIDE
        
        if len(outer_idx) == 0:
            # All pixels are inside, use last valid position
            outer_r = r_valid[-1]
        elif outer_idx[0] == 0:
            # First pixel already outside, skip this ray
            continue
        else:
            # Get last pixel inside (one before first outside pixel)
            outer_r = r_valid[outer_idx[0] - 1]
        
        outer_distances_px.append(outer_r)
        
        # Find inner boundary (first pixel INSIDE inner mask)
        inner_vals = mask_inner[ys_valid, xs_valid]
        inner_idx = np.where(inner_vals == 255)[0]
        
        if len(inner_idx) == 0:
            continue
            
        inner_r = r_valid[inner_idx[0]]
        
        # Calculate thickness
        if outer_r > inner_r:
            thicknesses_px.append(outer_r - inner_r)

    # Calculate lobedness
    lobedness_px = float(np.std(outer_distances_px)) if outer_distances_px else np.nan

    # Handle case with no valid measurements
    if not thicknesses_px:
        return {
            'mean_thickness_cm': np.nan,
            'median_thickness_cm': np.nan,
            'std_thickness_cm': np.nan,
            'min_thickness_cm': np.nan,
            'max_thickness_cm': np.nan,
            'cv_thickness': np.nan,
            'lobedness_cm': np.nan
        }
    
    thicknesses_px = np.array(thicknesses_px)
    
    # Convert to cm if calibration available
    if px_per_cm and isinstance(px_per_cm, (int, float)) and px_per_cm > 0:
        thicknesses_cm = thicknesses_px / px_per_cm
        lobedness = lobedness_px / px_per_cm
        mean_val = np.mean(thicknesses_cm)
        return {
            'mean_thickness_cm': float(mean_val),
            'median_thickness_cm': float(np.median(thicknesses_cm)),
            'std_thickness_cm': float(np.std(thicknesses_cm)),
            'min_thickness_cm': float(np.min(thicknesses_cm)),
            'max_thickness_cm': float(np.max(thicknesses_cm)),
            'cv_thickness': float((np.std(thicknesses_cm) / mean_val * 100) if mean_val > 0 else np.nan),
            'lobedness_cm': float(lobedness)
        }
    else:
        mean_val = np.mean(thicknesses_px)
        return {
            'mean_thickness_px': float(mean_val),
            'median_thickness_px': float(np.median(thicknesses_px)),
            'std_thickness_px': float(np.std(thicknesses_px)),
            'min_thickness_px': float(np.min(thicknesses_px)),
            'max_thickness_px': float(np.max(thicknesses_px)),
            'cv_thickness': float((np.std(thicknesses_px) / mean_val * 100) if mean_val > 0 else np.nan),
            'lobedness_px': float(lobedness_px)
        }