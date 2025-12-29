# traitly/internal_structure/geometry.py

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
# LOCAL/INTERNAL IMPORTS
# ===========================================================================
from .processing import get_fruit_contour


#################################################################################################
# Calculate minor axis (fruit width approximation)
#################################################################################################
def calculate_axes(fruit_contour: np.ndarray, 
                   px_per_cm: Optional[float] = None,  
                   img: Optional[np.ndarray] = None, 
                   draw_axes: bool = False, 
                   major_axis_color: Tuple[int, int, int] = (0, 255, 0), 
                   minor_axis_color: Tuple[int, int, int] = (255, 0, 0), 
                   axis_thickness: int = 2,
                   hull_verts: Optional[np.ndarray] = None): 
    """
    Calculate the minor and major axes of a fruit's contour in centimeters and pixels.
    Uses scipy.pdist for faster distance calculations.
    
    Args:
        fruit_contour: Nx2 or Nx1x2 array of contour points.
        px_per_cm: Average pixel-to-cm conversion factor. If None, measurements in cm will be np.nan.
        img: Image where axes will be drawn if draw_axes=True.
        draw_axes: Whether to draw the axes on the image.
        major_axis_color: BGR color for major axis (default: green).
        minor_axis_color: BGR color for minor axis (default: red).
        axis_thickness: Thickness of axis lines in pixels.
        hull_verts: Pre-computed convex hull vertices (optional, for caching).
    
    Returns:
        tuple: (max_dist_cm, min_dist_cm, max_dist_px, min_dist_px)
            - max_dist_cm: Major axis length in centimeters (np.nan if px_per_cm is None).
            - min_dist_cm: Minor axis length in centimeters (np.nan if px_per_cm is None).
            - max_dist_px: Major axis length in pixels.
            - min_dist_px: Minor axis length in pixels.
    """
    # Reshape and convert contour to float32 (consistent dtype)
    points_px = fruit_contour.reshape(-1, 2).astype(np.float32)
    n = points_px.shape[0]
    
    # Early exit for invalid contours
    if n < 2: 
        return 0.0, 0.0, 0.0, 0.0
    
    ## Major axis calculation - OPTIMIZED with pdist
    if hull_verts is None:
        if n >= 3:
            verts = ConvexHull(points_px).vertices
        else:
            verts = np.arange(n)
    else:
        verts = hull_verts
    
    # Use scipy.pdist 
    hull_points = points_px[verts]
    
    if len(hull_points) < 2:
        return 0.0, 0.0, 0.0, 0.0
    
    if len(hull_points) == 2:
        # Special case: only 2 points
        max_dist_px = np.linalg.norm(hull_points[1] - hull_points[0])
        point1_idx, point2_idx = verts[0], verts[1]
    else:
        # Calculate all pairwise distances at once using pdist
        dist_matrix = squareform(pdist(hull_points))
        max_idx = np.unravel_index(dist_matrix.argmax(), dist_matrix.shape)
        point1_idx, point2_idx = verts[max_idx[0]], verts[max_idx[1]]
        max_dist_px = dist_matrix[max_idx]
    
    if max_dist_px == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    # Major axis length in cm (use multiplication instead of division)
    # Validate px_per_cm is a number before any operations
    if px_per_cm is not None and isinstance(px_per_cm, (int, float)) and px_per_cm > 0:
        inv_px_per_cm = 1.0 / px_per_cm
        max_dist_cm = max_dist_px * inv_px_per_cm
    else:
        max_dist_cm = np.nan
    
    # Major axis endpoints
    p1_px = points_px[point1_idx]
    p2_px = points_px[point2_idx]
    
    ## Minor axis calculation 
    major_vec = p2_px - p1_px
    major_norm = np.linalg.norm(major_vec)
    
    if major_norm < 1e-10:
        min_dist_cm = 0.0 if (isinstance(px_per_cm, (int, float)) and px_per_cm > 0) else np.nan
        return max_dist_cm, min_dist_cm, max_dist_px, 0.0
    
    # Calculate perpendicular unit vector
    perp_unit = np.array([-major_vec[1], major_vec[0]], dtype=np.float32) / major_norm
    
    # Vectorized projection calculation
    #proj = (points_px[:, 0] - p1_px[0]) * perp_unit[0] + \
    #       (points_px[:, 1] - p1_px[1]) * perp_unit[1]
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
def rotate_box(contour: np.ndarray, 
               px_per_cm: Optional[float] = None, 
               img: Optional[np.ndarray] = None, 
               draw_box: bool = False, 
               box_color: Tuple[int, int, int] = (255, 180, 0), 
               box_thickness: int = 3):
    """
    Calculates the rotated bounding box (minimum area rectangle) of a contour.
    OPTIMIZED VERSION - Uses multiplication instead of division for conversions.
    
    Args:
        contour: Contour of the object (e.g., fruit) as returned by cv2.findContours().
        px_per_cm: Average pixels per centimeter conversion factor. If None, dimensions in cm will be np.nan.
        img: BGR image where the bounding box will be drawn (if draw_box=True).
        draw_box: If True, draws the bounding box on `img`.
        box_color: BGR color for the bounding box (default: light blue).
        box_thickness: Thickness of the bounding box lines in pixels.
    
    Returns:
        tuple: (box_length_cm, box_width_cm, box_length_px, box_width_px)
            - box_length_cm: Length (longer side) in centimeters (np.nan if px_per_cm is None).
            - box_width_cm: Width (shorter side) in centimeters (np.nan if px_per_cm is None).
            - box_length_px: Length (longer side) in pixels.
            - box_width_px: Width (shorter side) in pixels.
    
    Notes:
        - The bounding box is axis-independent (rotated to fit the contour tightly).
        - Dimensions are converted to cm using the average px_per_cm.
        - The "length" is always the longer side, and "width" the shorter side, regardless of orientation.
    """
    if draw_box and img is None:
        raise ValueError("img cannot be None when draw_box=True")
    
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
    
    # OPTIMIZED: Use multiplication instead of division (faster)
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


def get_fruit_morphology(contour, px_per_cm=None, contour_mode='raw', epsilon_factor=0.001):
    """
    Calculate comprehensive fruit morphology metrics with contour transformation options.
    OPTIMIZED VERSION - Early exits and hull reuse for better performance.
    
    Args:
        contour (np.ndarray): Fruit contour points
        px_per_cm (float, optional): Pixel to cm conversion factor
        contour_mode (str): Contour transformation mode ('raw', 'hull', 'approx', 'ellipse', 'circle')
        epsilon_factor (float): Epsilon factor for polygon approximation
    
    Returns:
        dict: Dictionary containing fruit morphology metrics
    """
    # OPTIMIZED: Early exit for invalid contours
    if len(contour) < 3:
        return {
            'fruit_area_px': np.nan,
            'fruit_area_cm2': np.nan,
            'fruit_perimeter_px': np.nan,
            'fruit_perimeter_cm': np.nan,
            'fruit_circularity': np.nan,
            'fruit_solidity': np.nan,
            'fruit_compactness': np.nan,
            'fruit_convex_hull_area_px': np.nan
        }
    
    # Apply contour transformation according to specified mode
    transformed_contour = get_fruit_contour(
        contours=[contour],  # Pass as list with single element
        fruit_id=0,          # Index 0 since there's only one contour
        contour_mode=contour_mode,
        epsilon_factor=epsilon_factor
    )
    
    # Early exit after transformation
    if len(transformed_contour) < 3:
        return {
            'fruit_area_px': np.nan,
            'fruit_area_cm2': np.nan,
            'fruit_perimeter_px': np.nan,
            'fruit_perimeter_cm': np.nan,
            'fruit_circularity': np.nan,
            'fruit_solidity': np.nan,
            'fruit_compactness': np.nan,
            'fruit_convex_hull_area_px': np.nan
        }
    
    # Calculate area and perimeter on transformed contour
    area_px = cv2.contourArea(transformed_contour)
    perimeter_px = cv2.arcLength(transformed_contour, True)
    
    # Single validation check
    if area_px <= 0 or perimeter_px <= 0:
        return {
            'fruit_area_px': np.nan,
            'fruit_area_cm2': np.nan,
            'fruit_perimeter_px': np.nan,
            'fruit_perimeter_cm': np.nan,
            'fruit_circularity': np.nan,
            'fruit_solidity': np.nan,
            'fruit_compactness': np.nan,
            'fruit_convex_hull_area_px': np.nan
        }
    
    # OPTIMIZED: Use multiplication instead of division
    if px_per_cm is not None and isinstance(px_per_cm, (int, float)) and px_per_cm > 0:
        inv_px_per_cm = 1.0 / px_per_cm
        inv_px_per_cm_sq = inv_px_per_cm * inv_px_per_cm
        area_cm2 = area_px * inv_px_per_cm_sq
        perimeter_cm = perimeter_px * inv_px_per_cm
    else:
        area_cm2 = np.nan
        perimeter_cm = np.nan
    
    # Calculate shape metrics using transformed contour
    circularity = (4 * np.pi * area_px) / (perimeter_px ** 2) if perimeter_px > 0 else np.nan
    
    # OPTIMIZED: Reuse hull if contour_mode is 'hull'
    if contour_mode == 'hull':
        hull = transformed_contour  # Already is the convex hull
        hull_area_px = area_px  # Area is already the hull area
    else:
        hull = cv2.convexHull(transformed_contour)
        hull_area_px = cv2.contourArea(hull)
    
    solidity = area_px / hull_area_px if hull_area_px > 0 else np.nan
    
    # Compactness (perimeter² / area)
    compactness = (perimeter_px ** 2) / area_px if area_px > 0 else np.nan
    
    return {
        'fruit_area_px': float(area_px),
        'fruit_area_cm2': float(area_cm2),
        'fruit_perimeter_px': float(perimeter_px),
        'fruit_perimeter_cm': float(perimeter_cm),
        'fruit_circularity': float(circularity),
        'fruit_solidity': float(solidity),
        'fruit_compactness': float(compactness),
        'fruit_convex_hull_area_px': float(hull_area_px)
    }