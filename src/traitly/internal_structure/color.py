# traitly/internal_structure/color.py

# ============================================================================
# STANDARD LIBRARY
# ============================================================================
from typing import List, Dict, Tuple, Optional, Any

# ============================================================================
# THIRD-PARTY LIBRARIES
# ============================================================================
import cv2
import numpy as np
from scipy.stats import circmean

def normalize_lab_values(l_values: np.ndarray, a_values: np.ndarray, 
                         b_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize Lab values to standard range."""
    l_values = l_values.astype(np.float32)
    a_values = a_values.astype(np.float32)
    b_values = b_values.astype(np.float32)
    
    l_normalized = (l_values * 100.0) / 255.0
    a_normalized = ((a_values / 255.0) * 255.0) - 128.0
    b_normalized = ((b_values / 255.0) * 255.0) - 128.0
    
    return l_normalized, a_normalized, b_normalized


def normalize_hsv_values(h_values: np.ndarray, s_values: np.ndarray, 
                         v_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize HSV values."""
    h_values = h_values.astype(np.float32)
    s_values = s_values.astype(np.float32)
    v_values = v_values.astype(np.float32)
    
    # Convert H from 0-180 to 0-360
    h_normalized = h_values * 2.0
    # Convert S and V to 0-100 range
    s_normalized = (s_values / 255.0) * 100.0
    v_normalized = (v_values / 255.0) * 100.0
    
    return h_normalized, s_normalized, v_normalized


def calculate_statistics(values: np.ndarray, stat: str = 'mean') -> float:
    """Calculate statistics with option to use mean or median."""
    if len(values) == 0:
        return np.nan
    
    if stat == 'mean':
        return float(np.mean(values))
    elif stat == 'median':
        return float(np.median(values))
    else:
        raise ValueError(f"stat must be 'mean' or 'median', got '{stat}'")


def calculate_std_and_cv(values: np.ndarray) -> Tuple[float, float]:
    """Calculate standard deviation and coefficient of variation."""
    if len(values) == 0:
        return np.nan, np.nan
    
    std = float(np.std(values))
    mean = float(np.mean(values))
    cv = float(std / mean) if mean != 0 else 0.0
    
    return std, cv


def circular_statistics(hue_values: np.ndarray) -> Tuple[float, float]:
    """Calculate circular statistics for hue values."""
    if len(hue_values) == 0:
        return np.nan, np.nan
    
    # Convert from 0-180 to 0-360
    hue_values_normalized = hue_values * 2.0
    
    radians = np.deg2rad(hue_values_normalized)
    x = np.cos(radians)
    y = np.sin(radians)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    r = np.sqrt(mean_x**2 + mean_y**2)
    mean_angle = np.rad2deg(np.arctan2(mean_y, mean_x)) % 360
    
    return float(mean_angle), float(r)


def extract_color_features(
    img: np.ndarray,
    mask: np.ndarray,
    stat: str = 'mean'
) -> Dict[str, float]:
    """
    Extract comprehensive color features from a masked region.
    
    Args:
        img: Input BGR image
        mask: Binary mask (255=foreground, 0=background)
        stat: Statistical measure to use ('mean' or 'median')
    
    Returns:
        Dictionary with color features
    """
    if mask.sum() == 0:
        return _get_nan_color_dict()
    
    # Convert to different color spaces
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Extract channel values
    r_values = img_rgb[:, :, 0][mask == 255]
    g_values = img_rgb[:, :, 1][mask == 255]
    b_values = img_rgb[:, :, 2][mask == 255]
    h_values = img_hsv[:, :, 0][mask == 255]
    s_values = img_hsv[:, :, 1][mask == 255]
    v_values = img_hsv[:, :, 2][mask == 255]
    l_values = img_lab[:, :, 0][mask == 255]
    a_values = img_lab[:, :, 1][mask == 255]
    b_lab_values = img_lab[:, :, 2][mask == 255]
    gray_values = img_gray[mask == 255]
    
    # Normalize values
    h_norm, s_norm, v_norm = normalize_hsv_values(h_values, s_values, v_values)
    l_norm, a_norm, b_norm = normalize_lab_values(l_values, a_values, b_lab_values)
    gray_norm = gray_values.astype(np.float32)
    
    # Circular statistics for Hue
    hue_circular, hue_homogeneity = circular_statistics(h_values)
    
    # Calculate main statistics
    results = {
        f'R_{stat}': calculate_statistics(r_values, stat),
        f'G_{stat}': calculate_statistics(g_values, stat),
        f'B_{stat}': calculate_statistics(b_values, stat),
        f'H_{stat}': calculate_statistics(h_norm, stat),
        f'S_{stat}': calculate_statistics(s_norm, stat),
        f'V_{stat}': calculate_statistics(v_norm, stat),
        f'L_{stat}': calculate_statistics(l_norm, stat),
        f'a_{stat}': calculate_statistics(a_norm, stat),
        f'b_{stat}': calculate_statistics(b_norm, stat),
        f'Gray_{stat}': calculate_statistics(gray_norm, stat),
        f'hue_circular_{stat}': hue_circular,
        'hue_homogeneity': hue_homogeneity
    }
    
    # Calculate a/L ratio
    a_l_ratio = a_norm / (l_norm + 1e-10)
    results[f'a_L_ratio_{stat}'] = calculate_statistics(a_l_ratio, stat)
    
    # RGB ratios
    epsilon = 1e-10
    r_g_ratio = r_values / (g_values + epsilon)
    r_b_ratio = r_values / (b_values + epsilon)
    r_ratio = r_values / (g_values + b_values + epsilon)
    
    results[f'r_g_ratio_{stat}'] = calculate_statistics(r_g_ratio, stat)
    results[f'r_b_ratio_{stat}'] = calculate_statistics(r_b_ratio, stat)
    results[f'r_ratio_{stat}'] = calculate_statistics(r_ratio, stat)
    
    # Standard deviation and CV
    results['R_std'], results['R_cv'] = calculate_std_and_cv(r_values)
    results['G_std'], results['G_cv'] = calculate_std_and_cv(g_values)
    results['B_std'], results['B_cv'] = calculate_std_and_cv(b_values)
    results['H_std'], results['H_cv'] = calculate_std_and_cv(h_norm)
    results['S_std'], results['S_cv'] = calculate_std_and_cv(s_norm)
    results['V_std'], results['V_cv'] = calculate_std_and_cv(v_norm)
    results['L_std'], results['L_cv'] = calculate_std_and_cv(l_norm)
    results['a_std'], results['a_cv'] = calculate_std_and_cv(a_norm)
    results['b_std'], results['b_cv'] = calculate_std_and_cv(b_norm)
    results['Gray_std'], results['Gray_cv'] = calculate_std_and_cv(gray_norm)
    results['a_L_ratio_std'], results['a_L_ratio_cv'] = calculate_std_and_cv(a_l_ratio)
    
    return results


def _get_nan_color_dict() -> Dict[str, float]:
    """Return dictionary with NaN values for empty masks."""
    return {
        'R_mean': np.nan, 'G_mean': np.nan, 'B_mean': np.nan,
        'H_mean': np.nan, 'S_mean': np.nan, 'V_mean': np.nan,
        'L_mean': np.nan, 'a_mean': np.nan, 'b_mean': np.nan,
        'Gray_mean': np.nan, 'hue_circular_mean': np.nan, 'hue_homogeneity': np.nan,
        'a_L_ratio_mean': np.nan, 'r_g_ratio_mean': np.nan,
        'r_b_ratio_mean': np.nan, 'r_ratio_mean': np.nan,
        'R_std': np.nan, 'G_std': np.nan, 'B_std': np.nan,
        'H_std': np.nan, 'S_std': np.nan, 'V_std': np.nan,
        'L_std': np.nan, 'a_std': np.nan, 'b_std': np.nan,
        'Gray_std': np.nan, 'a_L_ratio_std': np.nan,
        'R_cv': np.nan, 'G_cv': np.nan, 'B_cv': np.nan,
        'H_cv': np.nan, 'S_cv': np.nan, 'V_cv': np.nan,
        'L_cv': np.nan, 'a_cv': np.nan, 'b_cv': np.nan,
        'Gray_cv': np.nan, 'a_L_ratio_cv': np.nan
    }


def analyze_fruit_color(
    img: np.ndarray,
    fruit_contour: np.ndarray,
    inner_pericarp_contour: Optional[np.ndarray],
    img_shape: Tuple[int, int],
    stat: str = 'mean',
    locule_contours: Optional[List[np.ndarray]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Analyze color for whole fruit, outer pericarp, and inner pericarp.
    
    Args:
        img: Input BGR image
        fruit_contour: Outer fruit contour
        inner_pericarp_contour: Inner pericarp contour (can be None)
        img_shape: Image shape (height, width)
        stat: Statistical measure ('mean' or 'median')
    
    Returns:
        Dictionary with three keys: 'whole_fruit', 'outer_pericarp', 'inner_pericarp'
    """
    height, width = img_shape[:2]
    
    # Create masks
    mask_fruit = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask_fruit, [fruit_contour], -1, 255, thickness=cv2.FILLED)
    
    # Whole fruit color
    whole_fruit_color = extract_color_features(img, mask_fruit, stat)
    
    # Inner and outer pericarp
    if inner_pericarp_contour is not None and len(inner_pericarp_contour) > 0:
        # Inner pericarp mask (FILLED)
        mask_inner = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(mask_inner, [inner_pericarp_contour], -1, 255, thickness=cv2.FILLED)
        
        # >> NEW: Exclude locules from inner pericarp mask when provided
        if locule_contours is not None:
            for locule_contour in locule_contours:
                if len(locule_contour) > 0:
                    # "Erase" each locule by drawing it as 0 (black)
                    cv2.drawContours(mask_inner, [locule_contour], -1, 0, thickness=cv2.FILLED)
        
        # Outer pericarp = fruit - inner
        mask_outer = mask_fruit.copy()
        mask_outer[mask_inner == 255] = 0
        
        inner_color = extract_color_features(img, mask_inner, stat)
        outer_color = extract_color_features(img, mask_outer, stat)
    else:
        # No inner pericarp detected
        inner_color = _get_nan_color_dict()
        outer_color = _get_nan_color_dict()
    
    return {
        'whole_fruit': whole_fruit_color,
        'outer_pericarp': outer_color,
        'inner_pericarp': inner_color
    }