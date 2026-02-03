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
from scipy.stats import circmean, circstd

from traitly.internal_structure.processing import get_fruit_contour, get_inner_pericarp_contour


def normalize_lab_values(l_values: np.ndarray, 
                         a_values: np.ndarray, 
                         b_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize Lab values to standard range."""
    l_values = l_values.astype(np.float32)
    a_values = a_values.astype(np.float32)
    b_values = b_values.astype(np.float32)
    
    # Rescale L to 0-100, a and b to -128 to 127
    l_normalized = (l_values * 100.0) / 255.0
    a_normalized = a_values - 128.0
    b_normalized = b_values - 128.0
    
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


def circular_mean_and_std_hue(hue_values: np.ndarray, 
                              hue_degree_values: Optional[np.ndarray] = None) -> tuple[float, float]:
    
    """Calculate circular mean and standard deviation for hue values.
    
        Args:
            hue_values: Array of hue values in OpenCV scale [0,179]`
            hue_degree_values: Optional array of hue values in degrees [0,360]
        
        Returns:
            Tuple of (mean_hue_degrees, std_hue_degrees)
    """
    if len(hue_values) == 0:
        return np.nan, np.nan
    if hue_degree_values is not None:
        hue_deg = hue_degree_values
    else:
        # OpenCV H: [0,179] -> grados [0,358]
        hue_deg = hue_values.astype(np.float32) * 2.0
    
    # Convert degrees to radians
    radians = np.deg2rad(hue_deg)

    mean_rad = circmean(radians, high=2*np.pi, low=0)
    std_rad  = circstd(radians,  high=2*np.pi, low=0)

    mean_deg = (np.rad2deg(mean_rad) % 360.0)
    std_deg  = np.rad2deg(std_rad)

    return float(mean_deg), float(std_deg)


# def extract_color_features(
#     img: np.ndarray,
#     mask: np.ndarray,
#     stat: str = 'mean'
# ) -> Dict[str, float]:
#     """
#     Extract comprehensive color features from a masked region.
    
#     Args:
#         img: Input BGR image
#         mask: Binary mask (255=foreground, 0=background)
#         stat: Statistical measure to use ('mean' or 'median')
    
#     Returns:
#         Dictionary with color features
#     """
#     if mask.sum() == 0:
#         return _get_nan_color_dict()
    
#     # Convert to different color spaces
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Extract channel values
#     # RGB
#     r_values = img_rgb[:, :, 0][mask == 255]
#     g_values = img_rgb[:, :, 1][mask == 255]
#     b_values = img_rgb[:, :, 2][mask == 255]
    
#     # HSV
#     h_values = img_hsv[:, :, 0][mask == 255]
#     s_values = img_hsv[:, :, 1][mask == 255]
#     v_values = img_hsv[:, :, 2][mask == 255]
    
#     # Lab
#     l_values = img_lab[:, :, 0][mask == 255]
#     a_values = img_lab[:, :, 1][mask == 255]
#     b_lab_values = img_lab[:, :, 2][mask == 255]
    
#     # Grayscale
#     gray_values = img_gray[mask == 255]
    
#     # Normalize values
#     # Convert H from 0-180 to 0-360, S and V to 0-100
#     h_norm, s_norm, v_norm = normalize_hsv_values(h_values, s_values, v_values)
#     # Conver L from 0-255 to 0-100, a and b to -128 to 127
#     l_norm, a_norm, b_norm = normalize_lab_values(l_values, a_values, b_lab_values)
#     # Grayscale as float32 for calculations 
#     gray_norm = gray_values.astype(np.float32)
    
#     # Circular statistics for Hue
#     hue_circular, hue_homogeneity = circular_mean_and_std_hue(h_values)
    
#     # Calculate main statistics
#     results = {
#         f'R_{stat}': calculate_statistics(r_values, stat),
#         f'G_{stat}': calculate_statistics(g_values, stat),
#         f'B_{stat}': calculate_statistics(b_values, stat),
#         f'H_{stat}': calculate_statistics(h_norm, stat),
#         f'S_{stat}': calculate_statistics(s_norm, stat),
#         f'V_{stat}': calculate_statistics(v_norm, stat),
#         f'L_{stat}': calculate_statistics(l_norm, stat),
#         f'a_{stat}': calculate_statistics(a_norm, stat),
#         f'b_{stat}': calculate_statistics(b_norm, stat),
#         f'Gray_{stat}': calculate_statistics(gray_norm, stat),
#         f'hue_circular_{stat}': hue_circular,
#         'hue_homogeneity': hue_homogeneity
#     }
    

    
#     # Standard deviation and CV
#     results['R_std'], results['R_cv'] = calculate_std_and_cv(r_values)
#     results['G_std'], results['G_cv'] = calculate_std_and_cv(g_values)
#     results['B_std'], results['B_cv'] = calculate_std_and_cv(b_values)
#     results['H_std'], results['H_cv'] = calculate_std_and_cv(h_norm)
#     results['S_std'], results['S_cv'] = calculate_std_and_cv(s_norm)
#     results['V_std'], results['V_cv'] = calculate_std_and_cv(v_norm)
#     results['L_std'], results['L_cv'] = calculate_std_and_cv(l_norm)
#     results['a_std'], results['a_cv'] = calculate_std_and_cv(a_norm)
#     results['b_std'], results['b_cv'] = calculate_std_and_cv(b_norm)
#     results['Gray_std'], results['Gray_cv'] = calculate_std_and_cv(gray_norm)
    
#     return results

def extract_color_features(
    img: np.ndarray,
    mask: np.ndarray,
    stat: str = 'mean',
    color_spaces: str = 'all'
) -> Dict[str, float]:
    """
    Extract comprehensive color features from a masked region.
    Args:
        img: Input BGR image
        mask: Binary mask (255=foreground, 0=background)
        stat: Statistical measure to use ('mean' or 'median')
        color_spaces: Which color spaces to compute ('rgb', 'hsv', 'lab', 'gray', 'all', 
                     or combinations like 'rgb,lab' or 'rgb,hsv,gray')
    Returns:
        Dictionary with color features
    """
    # Get nan for no valid (empty) masks
    if mask.sum() == 0:
        return _get_nan_color_dict()
    
    # Determinate which color spaces to get based on input
    color_spaces = color_spaces.lower().replace(' ', '')
    
    if color_spaces == 'all':
        spaces_to_compute = {'rgb', 'hsv', 'lab', 'gray'}
    else:
        spaces_to_compute = set(color_spaces.split(','))
    
    # Check for valid color spaces
    valid_spaces = {'rgb', 'hsv', 'lab', 'gray'}
    if not spaces_to_compute.issubset(valid_spaces):
        invalid = spaces_to_compute - valid_spaces
        raise ValueError(f"Invalid color spaces: {invalid}. Valid options: {valid_spaces}")
    
    mask_bool = mask == 255
    
    # Convert only the required color spaces
    channels = {}
    
    if 'rgb' in spaces_to_compute:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        r_values = img_rgb[mask_bool, 0]
        g_values = img_rgb[mask_bool, 1]
        b_values = img_rgb[mask_bool, 2]
        channels['R'] = r_values
        channels['G'] = g_values
        channels['B'] = b_values
    
    if 'hsv' in spaces_to_compute:
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_values = img_hsv[mask_bool, 0]
        s_values = img_hsv[mask_bool, 1]
        v_values = img_hsv[mask_bool, 2]
        
        # Normalizar HSV
        h_norm, s_norm, v_norm = normalize_hsv_values(h_values, s_values, v_values)
        channels['H'] = h_norm
        channels['S'] = s_norm
        channels['V'] = v_norm
    
    if 'lab' in spaces_to_compute:
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        l_values = img_lab[mask_bool, 0]
        a_values = img_lab[mask_bool, 1]
        b_lab_values = img_lab[mask_bool, 2]
        
        # Normalizar Lab
        l_norm, a_norm, b_norm = normalize_lab_values(l_values, a_values, b_lab_values)
        channels['L'] = l_norm
        channels['a'] = a_norm
        channels['b'] = b_norm
    
    if 'gray' in spaces_to_compute:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_values = img_gray[mask_bool]
        gray_norm = gray_values.astype(np.float32)
        channels['Gray'] = gray_norm
    
    # Calculate circular statistics for hue (if hsv is required)
    results = {}
    if 'hsv' in spaces_to_compute:
        h_values = img_hsv[mask_bool, 0]
        hue_circular, hue_homogeneity = circular_mean_and_std_hue(h_values)
        results[f'hue_circular_{stat}'] = hue_circular
        results['hue_homogeneity'] = hue_homogeneity
    
    # Calculate main statistics for each channel
    for name, values in channels.items():
        results[f'{name}_{stat}'] = calculate_statistics(values, stat)
        results[f'{name}_std'], results[f'{name}_cv'] = calculate_std_and_cv(values)
    
    return results


def _get_nan_color_dict() -> Dict[str, float]:
    """Return dictionary with NaN values for empty masks."""
    return {
        'R_mean': np.nan, 'G_mean': np.nan, 'B_mean': np.nan, # RGB mean/median
        'H_mean': np.nan, 'S_mean': np.nan, 'V_mean': np.nan, # HSV mean/median
        'hue_circular_mean': np.nan, 'hue_homogeneity': np.nan, # Hue mean/median
        'L_mean': np.nan, 'a_mean': np.nan, 'b_mean': np.nan, # Lab mean/median
        'R_std': np.nan, 'G_std': np.nan, 'B_std': np.nan, # RGB std
        'H_std': np.nan, 'S_std': np.nan, 'V_std': np.nan, # HSV std
        'L_std': np.nan, 'a_std': np.nan, 'b_std': np.nan, # Lab std
        'R_cv': np.nan, 'G_cv': np.nan, 'B_cv': np.nan, # RGB cv
        'H_cv': np.nan, 'S_cv': np.nan, 'V_cv': np.nan, # HSV cv
        'L_cv': np.nan, 'a_cv': np.nan, 'b_cv': np.nan, # Lab cv
        'Gray_mean': np.nan, 'Gray_std': np.nan, 'Gray_cv': np.nan # Gray stats
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


def analyze_all_fruits_color(
    img: np.ndarray,
    contours: List[np.ndarray],
    fruit_locule_map: Dict[int, List[int]],
    stat: str = 'mean',
    color_spaces: str = 'all',
    tissues: str = 'all'  # NUEVO parámetro
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """
    Analyze color for all fruits in the fruit_locule_map.
    
    Args:
        img: Input BGR image
        contours: List of all contours (from cv2.findContours)
        fruit_locule_map: Dictionary mapping fruit_id to list of locule indices
                         Example: {33: [37], 72: [75, 78, 83, 84]}
        stat: Statistical measure ('mean' or 'median')
        color_spaces: Which color spaces to compute ('rgb', 'hsv', 'lab', 'gray', 'all')
        tissues: Which tissues to analyze. Options:
                - 'all': All tissues (whole_fruit, outer_pericarp, inner_flesh, locules)
                - Combinations like: 'whole_fruit,locules' or 'outer_pericarp,inner_flesh'
                - Valid tissues: 'whole_fruit', 'outer_pericarp', 'inner_flesh', 'locules'
    
    Returns:
        Dictionary mapping fruit_id to color analysis:
        {
            fruit_id: {
                'whole_fruit': {color features...},
                'outer_pericarp': {color features...},
                'inner_flesh': {color features...},
                'locules': {color features...}
            },
            ...
        }

    """
    # Procesar el parámetro tissues
    tissues = tissues.lower().replace(' ', '')
    
    if tissues == 'all':
        tissues_to_compute = {'whole_fruit', 'outer_pericarp', 'inner_flesh', 'locules'}
    else:
        tissues_to_compute = set(tissues.split(','))
    
    # Validar tejidos
    valid_tissues = {'whole_fruit', 'outer_pericarp', 'inner_flesh', 'locules'}
    if not tissues_to_compute.issubset(valid_tissues):
        invalid = tissues_to_compute - valid_tissues
        raise ValueError(f"Invalid tissues: {invalid}. Valid options: {valid_tissues}")
    
    height, width = img.shape[:2]
    results = {}
    
    for fruit_id, locule_indices in fruit_locule_map.items():
        fruit_results = {}
        
        # Paso 1: Crear máscara del fruto específico (siempre necesaria como base)
        fruit_contour = get_fruit_contour(contours, fruit_id, contour_mode='raw')
        mask_fruit = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(mask_fruit, [fruit_contour], -1, 255, thickness=cv2.FILLED)
        
        # Paso 2: Extraer color del fruto completo (si se solicita)
        if 'whole_fruit' in tissues_to_compute:
            whole_fruit_color = extract_color_features(img, mask_fruit, stat, color_spaces)
            fruit_results['whole_fruit'] = whole_fruit_color
        
        # Paso 3: Procesar regiones internas si hay lóculos
        if locule_indices:
            # Obtener el hull que rodea todos los lóculos
            inner_flesh_contour = get_inner_pericarp_contour(locule_indices, contours)
            
            if len(inner_flesh_contour) > 0:
                # Crear máscara de lóculos
                mask_locules = np.zeros((height, width), dtype=np.uint8)
                for locule_idx in locule_indices:
                    locule_contour = contours[locule_idx]
                    if len(locule_contour) > 0:
                        cv2.drawContours(mask_locules, [locule_contour], -1, 255, thickness=cv2.FILLED)
                
                # Extraer color de lóculos (si se solicita)
                if 'locules' in tissues_to_compute:
                    if mask_locules.sum() > 0:
                        locules_color = extract_color_features(img, mask_locules, stat, color_spaces)
                    else:
                        locules_color = _get_nan_color_dict()
                    fruit_results['locules'] = locules_color
                
                # Crear máscara del inner flesh (hull - lóculos)
                if 'inner_flesh' in tissues_to_compute:
                    mask_inner_flesh = np.zeros((height, width), dtype=np.uint8)
                    cv2.drawContours(mask_inner_flesh, [inner_flesh_contour], -1, 255, thickness=cv2.FILLED)
                    mask_inner_flesh[mask_locules == 255] = 0  # Excluir lóculos
                    
                    if mask_inner_flesh.sum() > 0:
                        inner_flesh_color = extract_color_features(img, mask_inner_flesh, stat, color_spaces)
                    else:
                        inner_flesh_color = _get_nan_color_dict()
                    fruit_results['inner_flesh'] = inner_flesh_color
                
                # Crear máscara del outer pericarp
                if 'outer_pericarp' in tissues_to_compute:
                    hull_complete = np.zeros((height, width), dtype=np.uint8)
                    cv2.drawContours(hull_complete, [inner_flesh_contour], -1, 255, thickness=cv2.FILLED)
                    mask_outer_pericarp = mask_fruit.copy()
                    mask_outer_pericarp[hull_complete == 255] = 0
                    
                    if mask_outer_pericarp.sum() > 0:
                        outer_pericarp_color = extract_color_features(img, mask_outer_pericarp, stat, color_spaces)
                    else:
                        outer_pericarp_color = _get_nan_color_dict()
                    fruit_results['outer_pericarp'] = outer_pericarp_color
            else:
                # Hull vacío - llenar con NaN los tejidos solicitados
                if 'inner_flesh' in tissues_to_compute:
                    fruit_results['inner_flesh'] = _get_nan_color_dict()
                if 'outer_pericarp' in tissues_to_compute:
                    fruit_results['outer_pericarp'] = _get_nan_color_dict()
                if 'locules' in tissues_to_compute:
                    fruit_results['locules'] = _get_nan_color_dict()
        else:
            # No hay lóculos - llenar con NaN los tejidos solicitados
            if 'inner_flesh' in tissues_to_compute:
                fruit_results['inner_flesh'] = _get_nan_color_dict()
            if 'outer_pericarp' in tissues_to_compute:
                fruit_results['outer_pericarp'] = _get_nan_color_dict()
            if 'locules' in tissues_to_compute:
                fruit_results['locules'] = _get_nan_color_dict()
        
        results[fruit_id] = fruit_results
    
    return results