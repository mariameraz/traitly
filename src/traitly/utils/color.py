# traitly/internal_structure/color.py

# ============================================================================
# STANDARD LIBRARY
# ============================================================================
from typing import List, Dict, Tuple, Optional

# ============================================================================
# THIRD-PARTY LIBRARIES
# ============================================================================
import cv2
import numpy as np
from scipy.stats import circmean, circstd
import matplotlib.pyplot as plt


# ============================================================================
# INTERNAL IMPORTS
# ============================================================================
from traitly.internal_structure.processing import get_fruit_contour, get_inner_pericarp_contour


############################################
# Normalization functions for color spaces #
############################################

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

################################
# Get stats for color channels #
################################

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

###################################################
# Calculate circular mean and std for hue values #
##################################################

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



####################################
# Extract color and get statistics #
####################################
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



#####################################################################
# Extract color for whole fruit, outer pericarp, and inner pericarp #
#####################################################################

# Helper function to return NaN dictionary

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
        # Inner pericarp mask (filled)
        mask_inner = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(mask_inner, [inner_pericarp_contour], -1, 255, thickness=cv2.FILLED)
        
        # Exclude locules from inner pericarp mask when provided
        if locule_contours is not None:
            for locule_contour in locule_contours:
                if len(locule_contour) > 0:
                    # Erase each locule by drawing it as 0 (black)
                    cv2.drawContours(mask_inner, [locule_contour], -1, 0, thickness=cv2.FILLED)
        
        # Outer pericarp (fruit - inner)
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

########################################################
# Analyze color for all fruits in the fruit_locule_map #
########################################################

def analyze_all_fruits_color(
    img: np.ndarray,
    contours: List[np.ndarray],
    fruit_locule_map: Dict[int, List[int]],
    stat: str = 'mean',
    color_spaces: str = 'all',
    tissues: str = 'all' 
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """
    Analyze color for all fruits in the fruit_locule_map.
    """
    tissues = tissues.lower().replace(' ', '')
    
    if tissues == 'all':
        tissues_to_compute = {'whole_fruit', 'outer_pericarp', 'inner_flesh', 'locules'}
    else:
        tissues_to_compute = set(tissues.split(','))
    
    valid_tissues = {'whole_fruit', 'outer_pericarp', 'inner_flesh', 'locules'}
    if not tissues_to_compute.issubset(valid_tissues):
        invalid = tissues_to_compute - valid_tissues
        raise ValueError(f"Invalid tissues: {invalid}. Valid options: {valid_tissues}")
    
    results = {}
    
    for fruit_id, locule_indices in fruit_locule_map.items():
        fruit_results = {}
        
        # Create a ROI around the fruit to optimize processing
        fruit_contour = get_fruit_contour(contours, fruit_id, contour_mode='raw')
        x, y, w, h = cv2.boundingRect(fruit_contour)
        
        roi_img = img[y:y+h, x:x+w]
        
        fruit_contour_roi = fruit_contour.copy()
        fruit_contour_roi[:, :, 0] -= x
        fruit_contour_roi[:, :, 1] -= y
        
        mask_fruit = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask_fruit, [fruit_contour_roi], -1, 255, thickness=cv2.FILLED)
        
        # Get color for whole fruit
        if 'whole_fruit' in tissues_to_compute:
            fruit_results['whole_fruit'] = extract_color_features(
                roi_img, mask_fruit, stat, color_spaces
            )
        
        # Early exit
        if tissues_to_compute == {'whole_fruit'}:
            results[fruit_id] = fruit_results
            continue
        
        # Process inner flesh and outer pericarp
        if locule_indices:
            inner_flesh_contour = get_inner_pericarp_contour(locule_indices, contours)
            
            if len(inner_flesh_contour) > 0:
                inner_flesh_contour_roi = inner_flesh_contour.copy()
                inner_flesh_contour_roi[:, :, 0] -= x
                inner_flesh_contour_roi[:, :, 1] -= y
                
                # Create masks for locules and inner flesh
                if 'locules' in tissues_to_compute or 'inner_flesh' in tissues_to_compute:
                    mask_locules = np.zeros((h, w), dtype=np.uint8)
                    for locule_idx in locule_indices:
                        loc_contour = contours[locule_idx].copy()
                        loc_contour[:, :, 0] -= x
                        loc_contour[:, :, 1] -= y
                        cv2.drawContours(mask_locules, [loc_contour], -1, 255, cv2.FILLED)
            
                if 'locules' in tissues_to_compute:
                    if np.any(mask_locules):
                        fruit_results['locules'] = extract_color_features(
                            roi_img, mask_locules, stat, color_spaces
                        )
                    else:
                        fruit_results['locules'] = _get_nan_color_dict()
                
                if 'inner_flesh' in tissues_to_compute:
                    mask_inner = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(mask_inner, [inner_flesh_contour_roi], -1, 255, cv2.FILLED)
                    mask_inner[mask_locules == 255] = 0
                    
                    if np.any(mask_inner):
                        fruit_results['inner_flesh'] = extract_color_features(
                            roi_img, mask_inner, stat, color_spaces
                        )
                    else:
                        fruit_results['inner_flesh'] = _get_nan_color_dict()
                
                if 'outer_pericarp' in tissues_to_compute:
                    mask_outer = mask_fruit.copy()
                    mask_outer[mask_inner == 255] = 0
                    
                    if np.any(mask_outer):
                        fruit_results['outer_pericarp'] = extract_color_features(
                            roi_img, mask_outer, stat, color_spaces
                        )
                    else:
                        fruit_results['outer_pericarp'] = _get_nan_color_dict()
            else:
                if 'inner_flesh' in tissues_to_compute:
                    fruit_results['inner_flesh'] = _get_nan_color_dict()
                if 'outer_pericarp' in tissues_to_compute:
                    fruit_results['outer_pericarp'] = _get_nan_color_dict()
                if 'locules' in tissues_to_compute:
                    fruit_results['locules'] = _get_nan_color_dict()
        else:
            if 'inner_flesh' in tissues_to_compute:
                fruit_results['inner_flesh'] = _get_nan_color_dict()
            if 'outer_pericarp' in tissues_to_compute:
                fruit_results['outer_pericarp'] = _get_nan_color_dict()
            if 'locules' in tissues_to_compute:
                fruit_results['locules'] = _get_nan_color_dict()
        
        results[fruit_id] = fruit_results
    
    return results


#############################################################
# Create masks for a single fruit and its different tissues #
#############################################################

# Helper function to renumber fruit IDs

def renumber_fruit_locule_map(
    fruit_locule_map: Dict[int, List[int]]
) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    """
    Renumber fruit IDs from 1 to n sequentially.
    
    Args:
        fruit_locule_map: Original mapping of fruit_id -> list of locule_ids
        
    Returns:
        - New mapping with sequential fruit IDs (1, 2, 3, ..., n)
        - Mapping from new fruit_id -> original fruit_id
        
    Example:
        Original: {5: [10, 11], 12: [20, 21], 8: [15]}
        Result map:   {1: [10, 11], 2: [20, 21], 3: [15]}
        ID mapping:  {1: 5, 2: 8, 3: 12}
    """
    # Get original fruit IDs sorted (optional, for consistency)
    original_ids = sorted(fruit_locule_map.keys())
    
    renumbered_map = {}
    fruit_id_map = {}

    for new_id, original_id in enumerate(original_ids, start=1):
        renumbered_map[new_id] = fruit_locule_map[original_id]
        fruit_id_map[new_id] = original_id
    
    return renumbered_map, fruit_id_map

# Create the masks: 
def get_single_fruit_masks_fast(
    img: np.ndarray,
    contours: List[np.ndarray],
    fruit_locule_map: Dict[int, List[int]],
    fruit_id: Optional[int] = None,
    renumber: bool = True
) -> Dict[str, np.ndarray]:
    """
    Obtain masks for the different tissues of a single fruit.
    
    Args:
        fruit_id: id of the fruit to analyze. If None, the first fruit with locules is selected.
        renumber: If True, renumber fruit IDs from 1 to n before selecting.
    
    Returns:
        Dict with masks for:
            - 'whole_fruit': mask of the whole fruit
            - 'outer_pericarp': mask of the outer pericarp
            - 'inner_flesh': mask of the inner flesh
            - 'locules': mask of the locules
            - 'cropped_img': cropped image of the fruit
            - 'bounding_box': bounding box of the fruit in the original image (x, y, w, h)
    """
    
    if not fruit_locule_map:
        raise ValueError("No fruits found in the fruit_locule_map")
    
    # Renumber fruit IDs from 1 to n if requested
    if renumber:
        fruit_locule_map, fruit_id_map = renumber_fruit_locule_map(fruit_locule_map)
    else:
        # Identity mapping if no renumbering
        fruit_id_map = {k: k for k in fruit_locule_map.keys()}
    
    # Select fruit if fruit_id is not provided
    if fruit_id is None:
        # Look for the first fruit with locules
        for fid, locules in fruit_locule_map.items():
            if locules:
                fruit_id = fid
                break
        
        # If don't find any valid fruit, use the first one
        if fruit_id is None:
            fruit_id = list(fruit_locule_map.keys())[0]

    # Map to original fruit ID
    original_fruit_id = fruit_id_map[fruit_id]
    
    # Get the contour of the selected fruit (using original IDs)
    fruit_contour = get_fruit_contour(contours, original_fruit_id)
    if len(fruit_contour) == 0:
        raise ValueError(f"Contour not found for fruit {original_fruit_id}")
    
    # Get the bounding box of the fruit
    x, y, w, h = cv2.boundingRect(fruit_contour)
    
    # Add a small margin around the bounding box
    margin = 10
    x_start = max(0, x - margin)
    y_start = max(0, y - margin)
    x_end = min(img.shape[1], x + w + margin)
    y_end = min(img.shape[0], y + h + margin)
    
    # Cut the image
    cropped_img = img[y_start:y_end, x_start:x_end]
    crop_height, crop_width = cropped_img.shape[:2]
    
    # Adjust fruit contour to the cropped image space
    fruit_contour_adj = fruit_contour.copy()
    fruit_contour_adj[:, :, 0] -= x_start  
    fruit_contour_adj[:, :, 1] -= y_start 
    
    # Create an empty dictionary to hold the masks
    masks = {}
    
    # Create a mask for the whole fruit
    masks['whole_fruit'] = np.zeros((crop_height, crop_width), dtype=np.uint8)
    cv2.drawContours(masks['whole_fruit'], [fruit_contour_adj], -1, 255, cv2.FILLED)
    
    # Get the locule indices for the selected fruit
    # NOTE: locule indices correspond to original contour IDs
    locule_indices = fruit_locule_map.get(fruit_id, [])
    
    if locule_indices:
        # Get the inner pericarp contour
        inner_contour = get_inner_pericarp_contour(locule_indices, contours)
        
        # For the inner pericarp and locules:
        if len(inner_contour) > 0:
            inner_contour_adj = inner_contour.copy()
            inner_contour_adj[:, :, 0] -= x_start
            inner_contour_adj[:, :, 1] -= y_start
            
            # Create an inner area mask
            inner_area = np.zeros((crop_height, crop_width), dtype=np.uint8)
            cv2.drawContours(inner_area, [inner_contour_adj], -1, 255, cv2.FILLED)
            
            # Create a mask for the locules
            masks['locules'] = np.zeros((crop_height, crop_width), dtype=np.uint8)
            for loc_idx in locule_indices:
                loc_contour = contours[loc_idx].copy()
                loc_contour[:, :, 0] -= x_start
                loc_contour[:, :, 1] -= y_start
                cv2.drawContours(masks['locules'], [loc_contour], -1, 255, cv2.FILLED)
            
            # create a mask for the inner_flesh (inner_area - locules)
            masks['inner_flesh'] = cv2.subtract(inner_area, masks['locules'])
            
            # create a mask for the outer_pericarp (whole_fruit - inner_area)
            masks['outer_pericarp'] = cv2.subtract(masks['whole_fruit'], inner_area)
        else:
            # Just in case there is no valid inner contour, all fruit is outer pericarp
            masks['outer_pericarp'] = masks['whole_fruit'].copy()
            masks['inner_flesh'] = np.zeros_like(masks['whole_fruit'])
            masks['locules'] = np.zeros_like(masks['whole_fruit'])
    else:
        # For fruits with no locules, all fruit is outer pericarp too
        masks['outer_pericarp'] = masks['whole_fruit'].copy()
        masks['inner_flesh'] = np.zeros((crop_height, crop_width), dtype=np.uint8)
        masks['locules'] = np.zeros((crop_height, crop_width), dtype=np.uint8)
    
    # Add cropped image and bounding box to the masks dict
    masks['cropped_img'] = cropped_img
    masks['bounding_box'] = (x_start, y_start, x_end - x_start, y_end - y_start)

    # (Optional but useful for debugging)
    masks['fruit_id'] = fruit_id                     # renumbered ID (1..n)
    masks['fruit_id_original'] = original_fruit_id   # original contour ID
    
    return masks

# Option 1 to visualizate single fruit masks - Individual plots for each tissue (binary masks)
def visualize_single_fruit_masks(
    masks: Dict[str, np.ndarray],
    figsize: Tuple[int, int] = (12, 4)
):
    """
    Plot the masks of different tissues of a single fruit.
    """
    tissue_display_names = {
        'whole_fruit': 'Whole Fruit',
        'outer_pericarp': 'Outer Pericarp',
        'inner_flesh': 'Inner Flesh',
        'locules': 'Locules'
    }
    
    # Determine the order of display:
    display_order = ['whole_fruit', 'outer_pericarp', 'inner_flesh', 'locules']
    
    # Get only existing masks
    valid_masks = [m for m in display_order if m in masks]
    
    if not valid_masks:
        print("There are no masks to display.")
        return
    
    n_masks = len(valid_masks)
    
    # Create an array of subplots
    fig, axes = plt.subplots(1, n_masks + 1, figsize=figsize)
    
    if n_masks + 1 == 1:
        axes = [axes]
    
    if 'cropped_img' in masks:
        cropped_img = masks['cropped_img']
        if cropped_img.ndim == 3 and cropped_img.shape[2] == 3:
            img_display = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        else:
            img_display = cropped_img
        
        axes[0].imshow(img_display)
        axes[0].set_title('Original Fruit', fontweight='bold', fontsize=10)
        axes[0].axis('off')
    
    # Show each mask on the plots
    for idx, mask_type in enumerate(valid_masks, 1):
        mask = masks[mask_type]
        axes[idx].imshow(mask, cmap='gray')
        display_name = tissue_display_names.get(mask_type, mask_type.replace('_', ' ').title())
        axes[idx].set_title(display_name, fontweight='bold', fontsize=10)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

# Option 2 to visualizate single fruit masks - Overlay masks on the original image


def visualize_single_fruit_overlay(
    masks: Dict[str, np.ndarray],
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot an overlay of the different tissue masks on the cropped fruit image.
    Args:
        masks: Dict with tissue masks and cropped image.
        figsize: Size of the figure.
    """
    if 'cropped_img' not in masks:
        return
    
    cropped_img = masks['cropped_img'].copy()
    
    # Create the overlay image
    if cropped_img.ndim == 3:
        overlay_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    else:
        overlay_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2RGB)
    
    # Determine colors for each tissue
    tissue_colors = {
        'outer_pericarp': (255, 200, 0, 100),    # Yellow
        'inner_flesh': (255, 100, 100, 100),     #  Light Red
        'locules': (100, 200, 255, 150)          # Light Blue
    }
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot the original cropped image
    axes[0].imshow(overlay_img)
    axes[0].set_title('Original', fontweight='bold')
    axes[0].axis('off')
    
    # Create a copy for overlay
    overlay_display = overlay_img.copy().astype(float) / 255
    
    for tissue, color in tissue_colors.items():
        if tissue in masks and np.any(masks[tissue] > 0):
            mask = masks[tissue] > 0
            if mask.any():
                # Create the colored overlay
                color_rgb = np.array(color[:3]) / 255.0
                alpha = color[3] / 255.0
                
                for c in range(3):
                    overlay_display[:, :, c][mask] = (
                        overlay_display[:, :, c][mask] * (1 - alpha) + 
                        color_rgb[c] * alpha
                    )
    
    # Plot the overlay
    axes[1].imshow(overlay_display)
    axes[1].set_title('Tissue Overlay', fontweight='bold')
    axes[1].axis('off')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=np.array(color[:3])/255, alpha=color[3]/255, 
              label=tissue.replace('_', ' ').title())
        for tissue, color in tissue_colors.items() if tissue in masks
    ]
    
    axes[1].legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.show()