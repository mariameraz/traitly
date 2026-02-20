# traitly/fruit_phenotyping/color_analysis.py

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
from matplotlib.patches import Patch
import pandas as pd

# ============================================================================
# INTERNAL IMPORTS
# ============================================================================
from traitly.fruit_phenotyping.processing import get_internal_pericarp_contour
from traitly.utils.constants import valid_contours

###########################################
# Normalization functions for lab and hsv #
###########################################

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


###################################################
# Calculate circular mean and std for hue values #
##################################################

def circular_mean_and_std_hue(hue_values: np.ndarray,
                              hue_degree_values: Optional[np.ndarray] = None) -> tuple[float, float]:
    if hue_values is None or len(hue_values) == 0:
        return np.nan, np.nan

    hue_deg = hue_degree_values.astype(np.float32) if hue_degree_values is not None \
              else hue_values.astype(np.float32) * 2.0

    mean_deg = circmean(hue_deg, high=360.0, low=0.0)
    std_deg  = circstd(hue_deg,  high=360.0, low=0.0)

    return float(mean_deg), float(std_deg)


####################################
# Extract color and get statistics #
####################################

def extract_color_features(
    img: np.ndarray,
    mask: np.ndarray,
    stat: str = 'mean',
    color_space: str = 'all'
) -> Dict[str, float]:
    stat = stat.lower().strip()
    color_space = color_space.lower().replace(' ', '')

    if color_space == 'all':
        spaces_to_compute = {'rgb', 'lab', 'hsv', 'gray'}
    else:
        spaces_to_compute = set(color_space.split(','))

    valid_spaces = {'rgb', 'lab', 'hsv', 'gray'}
    if not spaces_to_compute.issubset(valid_spaces):
        invalid = spaces_to_compute - valid_spaces
        raise ValueError(f"Invalid color spaces: {invalid}. Valid options: {valid_spaces}")

    if stat not in {'mean', 'median'}:
        raise ValueError("stat must be 'mean' or 'median'")

    suffix = 'mean' if stat == 'mean' else 'median'

    dark_threshold = 15
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    valid_mask = (mask == 255) & (gray_img > dark_threshold)

    if not np.any(valid_mask):
        return _get_nan_color_dict(suffix)

    def _agg_1d(v: np.ndarray) -> float:
        return float(np.mean(v)) if stat == 'mean' else float(np.median(v))

    def _agg_2d(v: np.ndarray) -> np.ndarray:
        return np.mean(v, axis=0) if stat == 'mean' else np.median(v, axis=0)

    out: Dict[str, float] = {}

    if 'rgb' in spaces_to_compute:
        bgr_pixels = img[valid_mask].astype(np.float32)  # Nx3
        rgb_pixels = bgr_pixels[:, ::-1]
        rgb_stat = _agg_2d(rgb_pixels)
        out[f'R_{suffix}'] = float(rgb_stat[0])
        out[f'G_{suffix}'] = float(rgb_stat[1])
        out[f'B_{suffix}'] = float(rgb_stat[2])

    if 'gray' in spaces_to_compute:
        gray_pixels = gray_img[valid_mask].astype(np.float32)
        out[f'Gray_{suffix}'] = _agg_1d(gray_pixels)

    # Lab normalized
    if 'lab' in spaces_to_compute:
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_vals = lab_img[:, :, 0][valid_mask]
        a_vals = lab_img[:, :, 1][valid_mask]
        b_vals = lab_img[:, :, 2][valid_mask]

        l_norm, a_norm, b_norm = normalize_lab_values(l_vals, a_vals, b_vals)

        out[f'L_{suffix}'] = _agg_1d(l_norm)
        out[f'a_{suffix}'] = _agg_1d(a_norm)
        out[f'b_{suffix}'] = _agg_1d(b_norm)

    # Hue in degrees 
    if 'hsv' in spaces_to_compute:
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_vals = hsv_img[:, :, 0][valid_mask].astype(np.float32)  # 0..179
        s_vals = hsv_img[:, :, 1][valid_mask].astype(np.float32)
        v_vals = hsv_img[:, :, 2][valid_mask].astype(np.float32)

        
        h_deg = h_vals * 2.0  # 0..358
        s_norm = (s_vals / 255.0) * 100.0
        v_norm = (v_vals / 255.0) * 100.0

        # Circular mean for hue
        h_cmean, h_cstd = circular_mean_and_std_hue(hue_values=h_vals)
        out[f'H_{suffix}'] = float(h_cmean) # Circular mean
        #out[f'H_{suffix}'] = _agg_1d(h_deg) # Arithmetic mean
        out[f'S_{suffix}'] = _agg_1d(s_norm)
        out[f'V_{suffix}'] = _agg_1d(v_norm)

        out['hue_circular_std'] = float(h_cstd)


        if np.isnan(h_cstd):
            out['hue_homogeneity'] = np.nan
        else:
            out['hue_homogeneity'] = float(np.clip(1.0 - (h_cstd / 180.0), 0.0, 1.0))

    return out


#####################################################################
# Extract color for whole fruit, outer pericarp, and inner pericarp #
#####################################################################

# Helper function to return NaN dictionary
def _get_nan_color_dict(stat_suffix: str = "mean") -> Dict[str, float]:
    """
    Return dictionary with NaN values for empty masks.
    stat_suffix: "mean" or "median":

    {'R_mean': nan,
    'G_mean': nan,
    'B_mean': nan,
    'L_mean': nan,
    'a_mean': nan,
    'b_mean': nan,
    'H_mean': nan,
    'S_mean': nan,
    'V_mean': nan,
    'hue_circular_mean': nan,
    'hue_circular_std': nan,
    'hue_homogeneity': nan,
    'Gray_mean': nan}
    """
    return {
        # RGB (0..255)
        f'R_{stat_suffix}': np.nan, f'G_{stat_suffix}': np.nan, f'B_{stat_suffix}': np.nan,

        # L in (0,100), a,b in (-128,127)
        f'L_{stat_suffix}': np.nan, f'a_{stat_suffix}': np.nan, f'b_{stat_suffix}': np.nan,

        # Hue in degrees (0,360), S,V in (0,100)
        f'H_{stat_suffix}': np.nan, f'S_{stat_suffix}': np.nan, f'V_{stat_suffix}': np.nan,

        # Circular hue stats in degrees
        'hue_circular_mean': np.nan,
        'hue_circular_std': np.nan,
        'hue_homogeneity': np.nan,

        # Gray (0..255)
        f'Gray_{stat_suffix}': np.nan,
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

#########################################
# Helper function to renumber fruit IDs #
#########################################

def renumber_fruit_locule_map(
    fruit_locule_map: Dict[int, List[int]]
) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    """
    Renumber fruit IDs from 1 to n sequentially.
    """
    # Get original fruit IDs sorted (optional, for consistency)
    original_ids = sorted(fruit_locule_map.keys())
    
    renumbered_map = {}
    fruit_id_map = {}

    for new_id, original_id in enumerate(original_ids, start=1):
        renumbered_map[new_id] = fruit_locule_map[original_id]
        fruit_id_map[new_id] = original_id
    
    return renumbered_map, fruit_id_map

########################################################
# Analyze color for all fruits in the fruit_locule_map #
########################################################

def analyze_all_fruits_color(
          img: np.ndarray,
          mask: np.ndarray,
          contours: List[np.ndarray],
          fruit_locule_map: Dict[int, List[int]],
          stat: str = 'mean',
          tissue: str = 'all',
          renumber: bool = True,
          color_space: str = 'all',
          contour_mode: Optional[str] = 'raw'
) -> Dict[int, Dict[str, Dict[str, float]]]:

    """
    Analyze color for all fruits in the fruit_locule_map.

    """

    # Check if inputs are valid:

    if contour_mode not in valid_contours:
            raise ValueError(
                f"Invalid contour_mode: {contour_mode}. Valid options are: {valid_contours}"
            )

    tissue = tissue.lower().replace(' ', '')

    if tissue == 'all':
        tissues_to_extract = {'total_pericarp', 'outer_pericarp', 'internal_pericarp', 'locules'}
    else:
        tissues_to_extract = set(tissue.split(','))

    valid_tissues = {'total_pericarp', 'outer_pericarp', 'internal_pericarp', 'locules'}
    if not tissues_to_extract.issubset(valid_tissues):
        invalid = tissues_to_extract - valid_tissues
        raise ValueError(f"Invalid tissue: {invalid}. Valid options: {valid_tissues}")

    # Renumber fruit ids from 1 to n fruits
    if renumber:
            iter_map, fruit_id_map = renumber_fruit_locule_map(fruit_locule_map)
    else:
            iter_map = fruit_locule_map
            fruit_id_map = {k: k for k in fruit_locule_map.keys()}

    # Precalculate a 'nan' dictionary
    nan_dict = _get_nan_color_dict('mean' if stat == 'mean' else 'median')

    results = {}

    # Then, for each fruit and its locules:
    for fruit_id, locule_indices in iter_map.items():
        fruit_results = {}

        # Use the original fruit_id to extract it from contours
        original_fruit_id = fruit_id_map[fruit_id]

        # Get fruit contour
        
        if original_fruit_id >= len(contours):
            for t in tissues_to_extract:
                fruit_results[t] = nan_dict.copy()

            results[fruit_id] = fruit_results
            continue

        fruit_contour = contours[original_fruit_id]

        # Create a (rotated) bounding-box around the fruit and cut the original image and the binary mask using its ROI
        x, y, w, h = cv2.boundingRect(fruit_contour)

        roi_raw = img[y:y+h, x:x+w]
        roi_mask = mask[y:y+h, x:x+w]

        mask_pericarp = ((roi_mask > 0).astype(np.uint8)) * 255

        mask_locules = None

        # Check if locules are needed. If 'locules' is not in tissue but 'internal_pericarp' is, the locule mask is still required.
        need_locules_mask = bool(locule_indices) and (
            ('locules' in tissues_to_extract) or
            ('internal_pericarp' in tissues_to_extract)
        )

        if need_locules_mask:
            mask_locules = np.zeros((h, w), dtype=np.uint8)
            for locule_idx in locule_indices:
                loc_contour = contours[locule_idx].copy()
                loc_contour[:, :, 0] -= x
                loc_contour[:, :, 1] -= y
                cv2.drawContours(mask_locules, [loc_contour], -1, 255, cv2.FILLED)

            # Only save the results if 'locules' is required (e.g., tissue = 'all' or tissue = 'locules')
            if 'locules' in tissues_to_extract:
                fruit_results['locules'] = (
                    extract_color_features(roi_raw, mask_locules, stat, color_space)
                    if np.any(mask_locules) else nan_dict.copy()
                )

        # Use the original mask (locules = 0, fruit(total_pericarp)= 255)
        if 'total_pericarp' in tissues_to_extract:
            fruit_results['total_pericarp'] = (
                extract_color_features(roi_raw, mask_pericarp, stat, color_space)
                if np.any(mask_pericarp) else nan_dict.copy()
            )

        # Verify that the fruit has valid locules, otherwise, assign NaN results
        if not locule_indices:
                for t in ('outer_pericarp', 'internal_pericarp'):
                    if t in tissues_to_extract:
                        fruit_results[t] = nan_dict.copy()

                results[fruit_id] = fruit_results
                continue

        need_internal_contour = (
             ('internal_pericarp' in tissues_to_extract) or
             ('outer_pericarp' in tissues_to_extract)
        )

        if not need_internal_contour:
            results[fruit_id] = fruit_results
            continue

        internal_flesh_contour = get_internal_pericarp_contour(locule_indices, contours)

        if internal_flesh_contour is None or len(internal_flesh_contour) == 0:
            for t in ('outer_pericarp', 'internal_pericarp'):
                if t in tissues_to_extract:
                    fruit_results[t] = nan_dict.copy()

            results[fruit_id] = fruit_results
            continue

        internal_flesh_contour_roi = internal_flesh_contour.copy()
        internal_flesh_contour_roi[:, :, 0] -= x
        internal_flesh_contour_roi[:, :, 1] -= y

        mask_internal_area = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask_internal_area, [internal_flesh_contour_roi], -1, 255, cv2.FILLED)

        if 'internal_pericarp' in tissues_to_extract:
            mask_internal = mask_internal_area.copy()
            if mask_locules is not None and np.any(mask_locules):
                mask_internal[mask_locules == 255] = 0
            fruit_results['internal_pericarp'] = (
                extract_color_features(roi_raw, mask_internal, stat, color_space)
                if np.any(mask_internal) else nan_dict.copy()
            )

        if 'outer_pericarp' in tissues_to_extract:
            mask_outer = mask_pericarp.copy()
            mask_outer[mask_internal_area == 255] = 0
            fruit_results['outer_pericarp'] = (
                extract_color_features(roi_raw, mask_outer, stat, color_space)
                if np.any(mask_outer) else nan_dict.copy()
            )

        results[fruit_id] = fruit_results

    return results


#############################################################
# Create masks for a single fruit and its different tissue #
#############################################################
def get_single_fruit_masks(
    img: np.ndarray,
    mask: np.ndarray,
    contours: List[np.ndarray],
    fruit_locule_map: Dict[int, List[int]],
    fruit_id: Optional[int] = None,
    renumber: bool = True,
    plot: bool = True,
    plot_size: Tuple[int, int] = (7, 5),
    overlay:bool = False,
    margin: int = 5,
    overlay_legend: bool = True,
    only_fruit: bool = False
) -> Dict[str, np.ndarray]:
    """
    Same goal as get_single_fruit_masks(), but reuses an existing binary mask
    (fruit/total_pericarp = 255, background = 0) cropped to the fruit ROI.

    Returns masks for:
      - 'outer_pericarp'
      - 'internal_pericarp'
      - 'locules'
      - 'total_pericarp'
      - 'cropped_img'
      - 'bounding_box'
      - 'fruit_id'
    """

    if not fruit_locule_map:
        raise ValueError("No fruits found in the fruit_locule_map. Run detect_fruits() first.")

    if renumber:
        iter_map, fruit_id_map = renumber_fruit_locule_map(fruit_locule_map)
    else:
        iter_map = fruit_locule_map
        fruit_id_map = {k: k for k in fruit_locule_map.keys()}

    # Validate fruit_id
    if fruit_id is not None:
        if not isinstance(fruit_id, int):
            raise TypeError(f"fruit_id: {fruit_id} must be an integer.")
        if fruit_id < 1 or fruit_id > len(iter_map):
            raise ValueError(f"fruit_id: {fruit_id} should be between 1 and {len(iter_map)}")

    # If not provided, pick first fruit with locules; else first fruit
    if fruit_id is None:
        for fid, locs in iter_map.items():
            if locs:
                fruit_id = fid
                break
        if fruit_id is None:
            fruit_id = list(iter_map.keys())[0]
    
    # Work with the original ids
    original_fruit_id = fruit_id_map[fruit_id]
    if original_fruit_id >= len(contours):
        raise ValueError(f"Contour not found for fruit {original_fruit_id}")

    fruit_contour = contours[original_fruit_id]
    if fruit_contour is None or len(fruit_contour) == 0:
        raise ValueError(f"Contour not found for fruit {original_fruit_id}")

    # Bounding box + margin
    x, y, w, h = cv2.boundingRect(fruit_contour)
    m = margin if margin is not None else 0

    x_start = max(0, x - m)
    y_start = max(0, y - m)
    x_end = min(img.shape[1], x + w + m)
    y_end = min(img.shape[0], y + h + m)

    # Crop original image and binary mask using the bounding box coords
    cropped_img = img[y_start:y_end, x_start:x_end]
    roi_mask = mask[y_start:y_end, x_start:x_end]

    crop_h, crop_w = cropped_img.shape[:2]

    # Total pericarp (fruit mask without locules)
    total_pericarp = ((roi_mask > 0).astype(np.uint8)) * 255 # Just ensuring roi_mask is a binary mask

    masks: Dict[str, np.ndarray] = {}
    masks["total_pericarp"] = total_pericarp # Save the tp_mask

    locule_indices = iter_map.get(fruit_id, [])

    # No locules case
    if not locule_indices:
        masks["outer_pericarp"] = total_pericarp.copy()
        masks["internal_pericarp"] = np.zeros((crop_h, crop_w), dtype=np.uint8)
        masks["locules"] = np.zeros((crop_h, crop_w), dtype=np.uint8)

    else:
        inner_contour = get_internal_pericarp_contour(locule_indices, contours)

        if inner_contour is None or len(inner_contour) == 0:
            masks["outer_pericarp"] = total_pericarp.copy()
            masks["internal_pericarp"] = np.zeros_like(total_pericarp)
            masks["locules"] = np.zeros_like(total_pericarp)

        else:
            # Adjust internal contour to ROI
            inner_contour_adj = inner_contour.copy()
            inner_contour_adj[:, :, 0] -= x_start
            inner_contour_adj[:, :, 1] -= y_start

            inner_area = np.zeros((crop_h, crop_w), dtype=np.uint8)
            cv2.drawContours(inner_area, [inner_contour_adj], -1, 255, cv2.FILLED)

            # Locules mask
            loc_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
            for loc_idx in locule_indices:
                if loc_idx >= len(contours):
                    continue
                loc_contour = contours[loc_idx]
                if loc_contour is None or len(loc_contour) == 0:
                    continue
                loc_contour = loc_contour.copy()
                loc_contour[:, :, 0] -= x_start
                loc_contour[:, :, 1] -= y_start
                cv2.drawContours(loc_mask, [loc_contour], -1, 255, cv2.FILLED)

            masks["locules"] = loc_mask

            # Internal pericarp = internal area - locules
            masks["internal_pericarp"] = cv2.subtract(inner_area, loc_mask)

            # Outer pericarp = total pericarp - internal area
            masks["outer_pericarp"] = cv2.subtract(total_pericarp, inner_area)

            # Remove locules from total pericarp
            masks["total_pericarp"] = cv2.subtract(total_pericarp, loc_mask)

    masks["cropped_img"] = cropped_img

    # Save bounding_box and fruit_id info (metadata)
    masks["bounding_box"] = (x_start, y_start, x_end - x_start, y_end - y_start)
    masks["fruit_id"] = fruit_id

    if plot:
        if overlay:
            visualize_single_fruit_overlay(masks=masks, 
                                           plot_size=plot_size,
                                           overlay_legend = overlay_legend)
        else:
            visualize_single_fruit_masks(masks=masks, 
                                         plot_size=plot_size, 
                                         only_fruit = only_fruit)

    return masks

#################################################
# Visualize different tissues of a single fruit #
#################################################

def visualize_single_fruit_masks(
    masks: Dict[str, np.ndarray],
    plot_size: Tuple[int, int] = (12, 4),
    only_fruit: Optional[bool] = None
):
    """
    Plot the masks of different tissue of a single fruit.
    """
    
    if only_fruit:
        tissue_display_names = {
        'total_pericarp': 'Total pericarp'
        }

        display_order = ['total_pericarp']
    
        valid_masks = [m for m in display_order if m in masks]

        if not valid_masks:
            print("There are no masks to display.")
            return
        
        n_masks = len(valid_masks)
        
        fig, axes = plt.subplots(1, n_masks + 1, figsize=plot_size)
        
        # Include a plot for the bgr image
        if n_masks + 1 == 1:
            axes = [axes]
        
        if 'cropped_img' in masks:
            fontsize = int(plot_size[0] * 1.3)
            fontsize = max(0.5, min(40, fontsize))

            cropped_img = masks['cropped_img']
            if cropped_img.ndim == 3 and cropped_img.shape[2] == 3:
                img_display = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            else:
                img_display = cropped_img
            
            axes[0].imshow(img_display)
            axes[0].set_title('Original Fruit', fontweight='bold', fontsize=fontsize)
            axes[0].axis('off')
        
        for idx, mask_type in enumerate(valid_masks, 1):
            mask = masks[mask_type]
            axes[idx].imshow(mask, cmap='gray', interpolation = 'nearest')
            display_name = tissue_display_names.get(mask_type, mask_type.replace('_', ' ').title())
            axes[idx].set_title('Fruit Mask', fontweight='bold', fontsize=fontsize)
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    else:
        # Tissue names for visualization
        tissue_display_names = {
            'total_pericarp': 'Total pericarp',
            'outer_pericarp': 'Outer Pericarp',
            'internal_pericarp': 'Internal pericarp',
            'locules': 'Locules'
        }
        
        
        # Ensure plots are always shown in the same order
        display_order = ['total_pericarp', 'outer_pericarp', 'internal_pericarp', 'locules']
        
        valid_masks = [m for m in display_order if m in masks]
        
        if not valid_masks:
            print("There are no masks to display.")
            return
        
        n_masks = len(valid_masks)
        
        fig, axes = plt.subplots(1, n_masks + 1, figsize=plot_size)
        
        # Include a plot for the bgr image
        if n_masks + 1 == 1:
            axes = [axes]
        
        if 'cropped_img' in masks:
            fontsize = int(plot_size[0] * 1.3)
            fontsize = max(0.5, min(40, fontsize))

            cropped_img = masks['cropped_img']
            if cropped_img.ndim == 3 and cropped_img.shape[2] == 3:
                img_display = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            else:
                img_display = cropped_img
            
            axes[0].imshow(img_display)
            axes[0].set_title('Original Fruit', fontweight='bold', fontsize=fontsize)
            axes[0].axis('off')
        
        for idx, mask_type in enumerate(valid_masks, 1):
            mask = masks[mask_type]
            axes[idx].imshow(mask, cmap='gray')
            display_name = tissue_display_names.get(mask_type, mask_type.replace('_', ' ').title())
            axes[idx].set_title(display_name, fontweight='bold', fontsize=fontsize)
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()

###########################################################
# Visualize different tissues overlying on a single fruit #
###########################################################

def visualize_single_fruit_overlay(
    masks: Dict[str, np.ndarray],
    plot_size: Tuple[int, int] = (12, 4),
    overlay_legend: bool = True,
):
    """
    Plot an overlay of the different tissue masks on the cropped fruit image.

    Args:
        masks: Dict with tissue masks and cropped image. Expected key: 'cropped_img'
               Optional keys: 'outer_pericarp', 'internal_pericarp', 'locules'
        plot_size: Size of the figure.
        overlay_legend: If True, show legend explaining overlay colors.
    """
    if "cropped_img" not in masks:
        return

    cropped_img = masks["cropped_img"].copy()

    # Convert to RGB for plotting
    if cropped_img.ndim == 3:
        overlay_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    else:
        overlay_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2RGB)

    # Tissue colors (RGBA) where A is 0-255 alpha
    tissue_colors = {
        "outer_pericarp": (255, 200, 0, 100),       # Yellow
        "internal_pericarp": (255, 100, 100, 100),  # Pink
        "locules": (100, 200, 255, 150),            # Blue
    }

    fig, axes = plt.subplots(1, 2, figsize=plot_size)

    fontsize = int(plot_size[0] * 1.3)

    axes[0].imshow(overlay_img)
    axes[0].set_title("Original Fruit", fontweight="bold", fontsize=fontsize)
    axes[0].axis("off")

    # Create overlay image (float in [0,1])
    overlay_display = overlay_img.astype(np.float32) / 255.0

    # Apply colored overlays
    for tissue, color in tissue_colors.items():
        if tissue in masks and masks[tissue] is not None and np.any(masks[tissue] > 0):
            mask = masks[tissue] > 0
            if mask.any():
                color_rgb = np.array(color[:3], dtype=np.float32) / 255.0
                alpha = float(color[3]) / 255.0

                # Alpha blend
                for c in range(3):
                    overlay_display[:, :, c][mask] = (
                        overlay_display[:, :, c][mask] * (1.0 - alpha)
                        + color_rgb[c] * alpha
                    )

    axes[1].imshow(overlay_display)
    axes[1].set_title("Tissue Overlay", fontweight="bold", fontsize=fontsize)
    axes[1].axis("off")

    # Legend if requested
    if overlay_legend:
        legend_elements = [
            Patch(
                facecolor=np.array(color[:3], dtype=np.float32) / 255.0,
                alpha=float(color[3]) / 255.0,
                label=tissue.replace("_", " ").title(),
            )
            for tissue, color in tissue_colors.items()
            if tissue in masks and masks[tissue] is not None and np.any(masks[tissue] > 0)
        ]

        if legend_elements:
            axes[1].legend(handles=legend_elements, loc="upper right", fontsize=fontsize)

    plt.tight_layout()
    plt.show()


########################################
# Create id annotations for each fruit #
########################################

def annotate_fruits_on_image(
    img: np.ndarray,
    contours: List[np.ndarray],
    fruit_locule_map: Dict[int, List[int]],
    renumber: bool = True,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.6
) -> np.ndarray:
    """
    Annotate the original image with fruit IDs (renumbered if requested)
    and bounding boxes derived from contours.

    Args:
        img: Original image (BGR).
        contours: List of contours (fruit + locules).
        fruit_locule_map: Mapping fruit_id -> locule indices.
        renumber: If True, fruit IDs are renumbered from 1..n.
        color: Bounding box and text color (BGR).
        thickness: Line thickness for bounding box.
        font_scale: Font scale for fruit ID text.

    Returns:
        Annotated image (copy of input).
    """

    annotated = img.copy()

    if not fruit_locule_map:
        return annotated

    # Handle renumbering
    if renumber:
        iter_map, fruit_id_map = renumber_fruit_locule_map(fruit_locule_map)
    else:
        iter_map = fruit_locule_map
        fruit_id_map = {k: k for k in fruit_locule_map.keys()}

    for fruit_id, _ in iter_map.items():
        original_fruit_id = fruit_id_map[fruit_id]

        if original_fruit_id >= len(contours):
            continue

        contour = contours[original_fruit_id]
        if contour is None or len(contour) == 0:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Draw bounding box
        cv2.rectangle(
            annotated,
            (x, y),
            (x + w, y + h),
            color,
            thickness
        )

        # Text position (slightly above the box)
        text = f"Fruit {fruit_id}"
        text_x = x
        text_y = max(0, y - 8)

        cv2.putText(
            annotated,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )

    return annotated



############################################################
# Get pixel-level color histograms for all fruits       #
############################################################

def get_fruit_color_histograms(
    img: np.ndarray,
    hsv_img: Optional[np.ndarray],
    mask: np.ndarray,
    contours: List[np.ndarray],
    fruit_locule_map: Dict[int, List[int]],
    image_name: str = '',
    label: str = '',
    color_space: str = 'all',
    renumber: bool = True,
    dark_threshold: int = 0,
    normalize: bool = False
) -> List[Dict[str, float]]:
    """
    For each fruit, compute the pixel-value histogram (histogram) of each
    color channel over the total_pericarp mask (fruit region, locules excluded).

    """

    # Validate color_space
    color_space = color_space.lower().replace(' ', '')
    spaces = {'rgb', 'lab', 'hsv', 'gray'} if color_space == 'all' else set(color_space.split(','))
    valid_spaces = {'rgb', 'lab', 'hsv', 'gray'}
    if not spaces.issubset(valid_spaces):
        raise ValueError(f"Invalid color_space: {spaces - valid_spaces}. Valid: {valid_spaces}")

    # Define histogram bins ranges per channel

    # For each entry return column_prefix, bin_start, bin_stop_inclusive (bin values are integers after rounding the normalized float pixels)
    channel_specs: Dict[str, Tuple[int, int]] = {}  # ch_name -> (min_val, max_val inclusive)

    if 'rgb'  in spaces:
        channel_specs.update({'R': (0, 255), 'G': (0, 255), 'B': (0, 255)})
    if 'lab'  in spaces:
        channel_specs.update({'L': (0, 100), 'a': (-128, 127), 'b': (-128, 127)})
    if 'hsv'  in spaces:
        channel_specs.update({'H': (0, 359), 'S': (0, 100), 'V': (0, 100)})
    if 'gray' in spaces:
        channel_specs['Gray'] = (0, 255)

    # Pre-build column names in a fixed order
    all_col_names: List[str] = [
        f'{ch}_{i}'
        for ch, (lo, hi) in channel_specs.items()
        for i in range(lo, hi + 1)
    ]

    def _nan_row(fruit_id: int) -> Dict:
        row = {'image_name': image_name, 'label': label, 'fruit_id': fruit_id}
        row.update({col: np.nan for col in all_col_names})
        return row

    def _hist_from_float(values: np.ndarray, lo: int, hi: int, n_pix: int) -> np.ndarray:
        """
        Round normalized float pixel values to int, clamp to [lo, hi],
        and return a count array of length (hi - lo + 1).
        Optionally divide by n_pix if normalize=True.
        """
        rounded = np.round(values).astype(np.int32)
        rounded = np.clip(rounded, lo, hi)
        # Shift so index 0 = lo
        shifted = rounded - lo
        n_bins = hi - lo + 1
        counts = np.bincount(shifted, minlength=n_bins).astype(np.float64)
        if normalize:
            counts /= n_pix
        return counts
    
    if hsv_img is None:
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    else:
        hsv_image = hsv_img

    # Convert full image to each color space once
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lab_full  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  if 'lab' in spaces else None
    hsv_full  = hsv_image  if 'hsv' in spaces else None

    # Renumber fruit IDs
    if renumber:
        iter_map, fruit_id_map = renumber_fruit_locule_map(fruit_locule_map)
    else:
        iter_map = fruit_locule_map
        fruit_id_map = {k: k for k in fruit_locule_map.keys()}

    rows: List[Dict] = []

    for fruit_id, locule_indices in iter_map.items():
        original_fruit_id = fruit_id_map[fruit_id]

        if original_fruit_id >= len(contours):
            rows.append(_nan_row(fruit_id))
            continue

        fruit_contour = contours[original_fruit_id]
        if fruit_contour is None or len(fruit_contour) == 0:
            rows.append(_nan_row(fruit_id))
            continue

        # Bounding box crop
        x, y, w, h = cv2.boundingRect(fruit_contour)
        roi_img  = img      [y:y+h, x:x+w]
        roi_mask = mask     [y:y+h, x:x+w]
        roi_gray = gray_full[y:y+h, x:x+w]

        # Valid pixel mask (total_pericarp, locules excluded)
        valid = (roi_mask > 0) & (roi_gray > dark_threshold)

        if locule_indices:
            loc_mask = np.zeros((h, w), dtype=np.uint8)
            for loc_idx in locule_indices:
                if loc_idx >= len(contours):
                    continue
                lc = contours[loc_idx]
                if lc is None or len(lc) == 0:
                    continue
                lc = lc.copy()
                lc[:, :, 0] -= x
                lc[:, :, 1] -= y
                cv2.drawContours(loc_mask, [lc], -1, 255, cv2.FILLED)
            valid &= (loc_mask == 0)

        if not np.any(valid):
            rows.append(_nan_row(fruit_id))
            continue

        n_pixels = int(valid.sum())
        row: Dict = {'image_name': image_name, 'label': label, 'fruit_id': fruit_id}

        # RGB
        if 'rgb' in spaces:
            bgr_roi = roi_img[valid]  # Nx3
            for ch_name, ch_idx in (('R', 2), ('G', 1), ('B', 0)):
                lo, hi = channel_specs[ch_name]
                counts = _hist_from_float(bgr_roi[:, ch_idx].astype(np.float32), lo, hi, n_pixels)
                for i, v in zip(range(lo, hi + 1), counts):
                    row[f'{ch_name}_{i}'] = float(v)

        # Lab
        if 'lab' in spaces:
            roi_lab = lab_full[y:y+h, x:x+w]
            l_raw = roi_lab[:, :, 0][valid]
            a_raw = roi_lab[:, :, 1][valid]
            b_raw = roi_lab[:, :, 2][valid]
            # Reuse existing normalization function
            l_norm, a_norm, b_norm = normalize_lab_values(l_raw, a_raw, b_raw)
            # L: 0-100, a/b: -128 to 127
            for ch_name, vals in (('L', l_norm), ('a', a_norm), ('b', b_norm)):
                lo, hi = channel_specs[ch_name]
                counts = _hist_from_float(vals, lo, hi, n_pixels)
                for i, v in zip(range(lo, hi + 1), counts):
                    row[f'{ch_name}_{i}'] = float(v)

        # HSV
        if 'hsv' in spaces:
            roi_hsv = hsv_full[y:y+h, x:x+w]
            h_raw = roi_hsv[:, :, 0][valid]
            s_raw = roi_hsv[:, :, 1][valid]
            v_raw = roi_hsv[:, :, 2][valid]
            # Reuse existing normalization: H→0-360, S/V→0-100
            h_norm, s_norm, v_norm = normalize_hsv_values(h_raw, s_raw, v_raw)
            for ch_name, vals in (('H', h_norm), ('S', s_norm), ('V', v_norm)):
                lo, hi = channel_specs[ch_name]
                counts = _hist_from_float(vals, lo, hi, n_pixels)
                for i, v in zip(range(lo, hi + 1), counts):
                    row[f'{ch_name}_{i}'] = float(v)

        # Gray
        if 'gray' in spaces:
            lo, hi = channel_specs['Gray']
            counts = _hist_from_float(roi_gray[valid].astype(np.float32), lo, hi, n_pixels)
            for i, v in zip(range(lo, hi + 1), counts):
                row[f'Gray_{i}'] = float(v)

        rows.append(row)

    return rows


# ============================================================================
# Calculate hue index for fruit color variation (red-yellow)
# ============================================================================

def calculate_hue_index(
    df: pd.DataFrame,
    red_hue_ranges: List[Tuple[int, int]] = [(0, 21), (250, 360)],
    yellow_hue_range: Tuple[int, int] = (40, 80),
    orange_hue_range: Optional[Tuple[int, int]] = (22, 39),
    homogeneity_threshold: float = 0.80,
) -> pd.DataFrame:
    """
    Calculate a red/yellow/orange color index per fruit from Hue histogram columns.

    """
    h_cols_all = [f'H_{i}' for i in range(360)]
    missing = [c for c in h_cols_all if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing Hue columns in DataFrame. "
            "Run analyze_color() with get_color_histogram=True and color_space='hsv' first."
        )

    # Build bin index sets
    red_bins = set()
    for lo, hi in red_hue_ranges:
        red_bins.update(range(lo, hi + 1))

    y_lo, y_hi = yellow_hue_range
    yellow_bins = set(range(y_lo, y_hi + 1))

    use_orange = orange_hue_range is not None
    if use_orange:
        o_lo, o_hi = orange_hue_range
        orange_bins = set(range(o_lo, o_hi + 1))
    else:
        orange_bins = set()

    red_cols    = [f'H_{i}' for i in sorted(red_bins)    if f'H_{i}' in df.columns]
    yellow_cols = [f'H_{i}' for i in sorted(yellow_bins) if f'H_{i}' in df.columns]
    orange_cols = [f'H_{i}' for i in sorted(orange_bins) if f'H_{i}' in df.columns]

    records = []
    for _, row in df.iterrows():
        red_px    = row[red_cols].sum()
        yellow_px = row[yellow_cols].sum()
        orange_px = row[orange_cols].sum() if use_orange else 0

        total_px = red_px + yellow_px + orange_px

        if total_px == 0:
            red_ratio         = np.nan
            yellow_ratio      = np.nan
            orange_ratio      = np.nan if use_orange else None
            color_homogeneity = np.nan
            color_category    = 'unknown'
        else:
            red_ratio    = red_px    / total_px
            yellow_ratio = yellow_px / total_px
            orange_ratio = orange_px / total_px if use_orange else None

            ratios = {'red': red_ratio, 'yellow': yellow_ratio}
            if use_orange:
                ratios['orange'] = orange_ratio

            color_homogeneity = float(max(ratios.values()))
            dominant = max(ratios, key=ratios.get)
            color_category = dominant if color_homogeneity >= homogeneity_threshold else 'mixed'
            dominant_color = dominant

        record = {
            'image_name':        row.get('image_name', ''),
            'fruit_id':          row['fruit_id'],
            'red_pixels':        int(red_px),
            'orange_pixels':     int(orange_px) if use_orange else None,
            'yellow_pixels':     int(yellow_px),
            'total_hue_pixels':  int(total_px),
            'red_ratio':         round(float(red_ratio),    4) if not np.isnan(red_ratio)    else np.nan,
            'orange_ratio':      round(float(orange_ratio), 4) if use_orange and not np.isnan(orange_ratio) else None,
            'yellow_ratio':      round(float(yellow_ratio), 4) if not np.isnan(yellow_ratio) else np.nan,
            'color_homogeneity': round(float(color_homogeneity), 4) if not np.isnan(color_homogeneity) else np.nan,
            'color_category':    color_category,
            'dominant_color':    dominant_color if total_px > 0 else 'unknown',
        }

        # Drop None columns when orange is disabled
        if not use_orange:
            record.pop('orange_pixels')
            record.pop('orange_ratio')

        records.append(record)

    return pd.DataFrame(records)