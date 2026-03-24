# traitly/fruit_phenotyping/color_analysis.py

"""
Color analysis tools for fruit phenotyping pipelines.

Provides functions to extract, normalize, and analyze color features
from fruit images across multiple color spaces (RGB, Lab, HSV, Grayscale).
Supports per-tissue analysis (total pericarp, outer pericarp, internal
pericarp, and locules), pixel-level histograms, and hue-based color
classification.
"""

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


###########################################
# Normalization functions for lab and hsv #
###########################################

def normalize_lab_values(l_values: np.ndarray, 
                         a_values: np.ndarray, 
                         b_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize raw OpenCV Lab channel values to standard ranges.

    Rescales L to [0, 100] and a, b channels to [-128, 127].

    Parameters
    ----------
    l_values : np.ndarray
        Raw L channel values in the OpenCV range [0, 255].
    a_values : np.ndarray
        Raw a channel values in the OpenCV range [0, 255].
    b_values : np.ndarray
        Raw b channel values in the OpenCV range [0, 255].

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Normalized (L, a, b) arrays as float32.
    """
    l_values = l_values.astype(np.float32)
    a_values = a_values.astype(np.float32)
    b_values = b_values.astype(np.float32)
    
    # Rescale L to 0-100, a and b to -128 to 127
    l_normalized = (l_values * 100.0) / 255.0
    a_normalized = a_values - 128.0
    b_normalized = b_values - 128.0
    
    return l_normalized, a_normalized, b_normalized


def normalize_hsv_values(h_values: np.ndarray, 
                         s_values: np.ndarray, 
                         v_values: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize raw OpenCV HSV channel values to standard ranges.

    Converts H from [0, 180] to [0, 360] degrees, and S, V from
    [0, 255] to [0, 100].

    Parameters
    ----------
    h_values : np.ndarray
        Raw hue values in the OpenCV range [0, 180].
    s_values : np.ndarray
        Raw saturation values in the OpenCV range [0, 255].
    v_values : np.ndarray
        Raw value/brightness values in the OpenCV range [0, 255].

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Normalized (H, S, V) arrays as float32.
    """
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
                              hue_degree_values: Optional[np.ndarray] = None
                              ) -> tuple[float, float]:
    """
    Calculate the circular mean and standard deviation of hue values.

    Uses circular statistics to correctly handle the angular wrap-around
    of hue (e.g., values near 0° and 360° are treated as close).

    Parameters
    ----------
    hue_values : np.ndarray
        Raw hue values in the OpenCV range [0, 180]. Used when
        ``hue_degree_values`` is not provided.
    hue_degree_values : np.ndarray, optional
        Hue values already converted to degrees [0, 360]. If provided,
        ``hue_values`` is ignored.

    Returns
    -------
    tuple[float, float]
        Circular mean and circular standard deviation in degrees.
        Returns ``(nan, nan)`` if input is empty or None.
    """

    if hue_values is None or len(hue_values) == 0:
        return np.nan, np.nan
    # Ensure we are always working with [0, 360] degrees 
    hue_deg = hue_degree_values.astype(np.float32) if hue_degree_values is not None \
              else hue_values.astype(np.float32) * 2.0

    mean_deg = circmean(hue_deg, high=360.0, low=0.0)
    std_deg  = circstd(hue_deg,  high=360.0, low=0.0)

    return float(mean_deg), float(std_deg)


####################################
# Extract color and get statistics #
####################################

def extract_color_features(img: np.ndarray,
                            mask: np.ndarray,
                            stat: str = 'mean',
                            color_space: str = 'all',
                            dark_thresh: int = 15
                        ) -> Dict[str, float]:
    
    """
    Extract summary color statistics from masked pixels of an image.

    Calculates mean or median and standard deviation for each channel
    across the valid (non-dark, non-background) pixels within the mask
    region. Supports RGB, Lab, HSV, and Grayscale color spaces. For HSV,
    hue is summarized using circular statistics.

    Parameters
    ----------
    img : np.ndarray
        Input image in BGR format.
    mask : np.ndarray
        Binary mask where 255 indicates pixels to include.
    stat : str, optional
        Summary statistic to compute. Either ``'mean'`` or ``'median'``.
        Default is ``'mean'``.
    color_space : str, optional
        Color spaces to compute. Either ``'all'`` or a comma-separated
        subset of ``'rgb'``, ``'lab'``, ``'hsv'``, ``'gray'``.
        Default is ``'all'``.

    Returns
    -------
    Dict[str, float]
        Dictionary of color features keyed by channel name and statistic
        suffix (e.g., ``'R_mean'``, ``'R_std'``, ``'L_mean'``, ``'H_mean'``).
        Returns NaN values if no valid pixels are found.

    Raises
    ------
    ValueError
        If ``color_space`` contains invalid entries or ``stat`` is not
        ``'mean'`` or ``'median'``.
    """
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

    dark_threshold = dark_thresh
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
        rgb_std = np.std(rgb_pixels, axis=0)
        out['R_std'] = float(rgb_std[0])
        out['G_std'] = float(rgb_std[1])
        out['B_std'] = float(rgb_std[2])

    if 'gray' in spaces_to_compute:
        gray_pixels = gray_img[valid_mask].astype(np.float32)
        out[f'Gray_{suffix}'] = _agg_1d(gray_pixels)
        out['Gray_std'] = float(np.std(gray_pixels))

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
        out['L_std'] = float(np.std(l_norm))
        out['a_std'] = float(np.std(a_norm))
        out['b_std'] = float(np.std(b_norm))

    # Hue in degrees
    if 'hsv' in spaces_to_compute:
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_vals = hsv_img[:, :, 0][valid_mask].astype(np.float32)  
        s_vals = hsv_img[:, :, 1][valid_mask].astype(np.float32)
        v_vals = hsv_img[:, :, 2][valid_mask].astype(np.float32)

        s_norm = (s_vals / 255.0) * 100.0
        v_norm = (v_vals / 255.0) * 100.0

        # Circular mean and std for hue
        h_cmean, h_cstd = circular_mean_and_std_hue(hue_values=h_vals)
        out[f'H_{suffix}'] = float(h_cmean)
        out[f'S_{suffix}'] = _agg_1d(s_norm)
        out[f'V_{suffix}'] = _agg_1d(v_norm)
        out['H_std'] = float(h_cstd)
        out['S_std'] = float(np.std(s_norm))
        out['V_std'] = float(np.std(v_norm))

    return out


#####################################################################
# Extract color for whole fruit, outer pericarp, and inner pericarp #
#####################################################################

# Helper function to return NaN dictionary
def _get_nan_color_dict(stat_suffix: str = "mean"
                        ) -> Dict[str, float]:
    """
    Return a color feature dictionary with NaN values.

    Used as a fallback when a mask contains no valid pixels.

    Parameters
    ----------
    stat_suffix : str, optional
        Suffix used for column naming, either ``'mean'`` or ``'median'``.
        Default is ``'mean'``.

    Returns
    -------
    Dict[str, float]
        Dictionary with NaN values for all color channels across RGB,
        Lab, HSV, and Grayscale color spaces.

    """
    return {
        # Pixel stats
        f'R_{stat_suffix}': np.nan, f'G_{stat_suffix}': np.nan, f'B_{stat_suffix}': np.nan,
        f'L_{stat_suffix}': np.nan, f'a_{stat_suffix}': np.nan, f'b_{stat_suffix}': np.nan,
        f'H_{stat_suffix}': np.nan, f'S_{stat_suffix}': np.nan, f'V_{stat_suffix}': np.nan,
        f'Gray_{stat_suffix}': np.nan,

        # Standard deviation:
        'R_std': np.nan, 'G_std': np.nan, 'B_std': np.nan,
        'L_std': np.nan, 'a_std': np.nan, 'b_std': np.nan,
        'H_std': np.nan, 'S_std': np.nan, 'V_std': np.nan,
        'Gray_std': np.nan
    }


#########################################
# Helper function to renumber fruit IDs #
#########################################

def renumber_fruit_locule_map(fruit_locule_map: Dict[int, List[int]]
                            ) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    """
    Renumber fruit IDs sequentially from 1 to n.

    Parameters
    ----------
    fruit_locule_map : Dict[int, List[int]]
        Original mapping of fruit IDs to lists of locule contour indices.

    Returns
    -------
    Tuple[Dict[int, List[int]], Dict[int, int]]
        A tuple of:
        - Renumbered fruit-locule map with sequential IDs starting at 1.
        - Mapping from new sequential IDs to original fruit IDs.
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

def analyze_all_fruits_color(img: np.ndarray,
                            mask: np.ndarray,
                            contours: List[np.ndarray],
                            fruit_locule_map: Dict[int, List[int]],
                            stat: str = 'mean',
                            tissue: str = 'all',
                            renumber: bool = True,
                            color_space: str = 'all',
                            dilation_factor: Optional[float] = None,
                            dark_threshold: int = 15
                    ) -> Dict[int, Dict[str, Dict[str, float]]]:
    """
    Analyze color features for all fruits in the fruit-locule map.

    Iterates over each fruit, crops it to its bounding box, builds
    tissue-specific masks, and extracts color features per tissue.

    Parameters
    ----------
    img : np.ndarray
        Input image in BGR format.
    mask : np.ndarray
        Binary segmentation mask where fruit pixels are > 0.
    contours : List[np.ndarray]
        List of all contours (fruits and locules).
    fruit_locule_map : Dict[int, List[int]]
        Mapping of fruit contour indices to their locule contour indices.
    stat : str, optional
        Summary statistic: ``'mean'`` or ``'median'``. Default is ``'mean'``.
    tissue : str, optional
        Tissues to analyze. Either ``'all'`` or a comma-separated subset of
        ``'total_pericarp'``, ``'outer_pericarp'``, ``'internal_pericarp'``,
        ``'locules'``. Default is ``'all'``.
    renumber : bool, optional
        If True, fruit IDs are renumbered from 1 to n. Default is True.
    color_space : str, optional
        Color spaces to compute. Either ``'all'`` or a comma-separated
        subset of ``'rgb'``, ``'lab'``, ``'hsv'``, ``'gray'``.
        Default is ``'all'``.

    Returns
    -------
    Dict[int, Dict[str, Dict[str, float]]]
        Nested dictionary keyed by fruit ID, then tissue name, then
        color feature name.

    Raises
    ------
    ValueError
        If ``contour_mode`` or ``tissue`` contains invalid entries.
    """

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
                    extract_color_features(roi_raw, mask_locules, stat, color_space, dark_thresh = dark_threshold)
                    if np.any(mask_locules) else nan_dict.copy()
                )

        # Use the original mask (locules = 0, fruit(total_pericarp)= 255)
        if 'total_pericarp' in tissues_to_extract:
            fruit_results['total_pericarp'] = (
                extract_color_features(roi_raw, mask_pericarp, stat, color_space, dark_thresh = dark_threshold)
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

        internal_flesh_contour = get_internal_pericarp_contour(locule_indices, 
                                                               contours, 
                                                               dilation_factor = dilation_factor,
                                                               img_shape = img.shape[:2],
                                                               fruit_id = original_fruit_id)

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
                extract_color_features(roi_raw, mask_internal, stat, color_space, dark_thresh = dark_threshold)
                if np.any(mask_internal) else nan_dict.copy()
            )

        if 'outer_pericarp' in tissues_to_extract:
            mask_outer = mask_pericarp.copy()
            mask_outer[mask_internal_area == 255] = 0
            fruit_results['outer_pericarp'] = (
                extract_color_features(roi_raw, mask_outer, stat, color_space, dark_thresh = dark_threshold)
                if np.any(mask_outer) else nan_dict.copy()
            )

        results[fruit_id] = fruit_results

    return results


#############################################################
# Create masks for a single fruit and its different tissue #
#############################################################

def get_single_fruit_masks(img: np.ndarray,
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
                            only_fruit: bool = False,
                            dilation_factor: Optional[float] = None,
                        ) -> Dict[str, np.ndarray]:
    """
    Generate tissue masks for a single fruit cropped to its bounding box.

    Builds binary masks for each tissue region of the selected fruit and
    optionally visualizes them.

    Parameters
    ----------
    img : np.ndarray
        Input image in BGR format.
    mask : np.ndarray
        Binary segmentation mask where fruit pixels are white [255].
    contours : List[np.ndarray]
        List of all contours (fruits and locules).
    fruit_locule_map : Dict[int, List[int]]
        Mapping of fruit contour indices to their locule contour indices.
    fruit_id : int, optional
        ID of the fruit to analyze. If None, the first fruit with locules
        is selected automatically.
    renumber : bool, optional
        If True, fruit IDs are renumbered from 1 to n. Default is True.
    plot : bool, optional
        If True, display tissue masks. Default is True.
    plot_size : Tuple[int, int], optional
        Figure size for visualization. Default is (7, 5).
    overlay : bool, optional
        If True, display an overlay visualization instead of individual masks.
        Default is False.
    margin : int, optional
        Pixel margin to add around the fruit bounding box. Default is 5.
    overlay_legend : bool, optional
        If True, include a legend in the overlay plot. Default is True.
    only_fruit : bool, optional
        If True, display only the total pericarp mask. Default is False.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing masks and metadata with keys:
        ``'total_pericarp'``, ``'outer_pericarp'``, ``'internal_pericarp'``,
        ``'locules'``, ``'cropped_img'``, ``'bounding_box'``, ``'fruit_id'``.

    Raises
    ------
    ValueError
        If ``fruit_locule_map`` is empty, ``fruit_id`` is out of range,
        or no contour is found for the requested fruit.
    TypeError
        If ``fruit_id`` is not an integer.
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
        inner_contour = get_internal_pericarp_contour(locule_indices, 
                                                      contours, 
                                                      fruit_id = original_fruit_id,
                                                      img_shape = img.shape[:2],
                                                      dilation_factor = dilation_factor
                                                      )

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

def visualize_single_fruit_masks(masks: Dict[str, np.ndarray],
                                plot_size: Tuple[int, int] = (12, 4),
                                only_fruit: Optional[bool] = None
                                ):
    """
    Display individual binary masks for each tissue of a single fruit.

    Parameters
    ----------
    masks : Dict[str, np.ndarray]
        Dictionary of tissue masks as returned by :func:`get_single_fruit_masks`.
        Expected keys: ``'total_pericarp'``, ``'outer_pericarp'``,
        ``'internal_pericarp'``, ``'locules'``, ``'cropped_img'``.
    plot_size : Tuple[int, int], optional
        Figure size for the matplotlib plot. Default is (12, 4).
    only_fruit : bool, optional
        If True, display only the total pericarp mask alongside the
        original image. Default is None (show all tissues).
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

def visualize_single_fruit_overlay(masks: Dict[str, np.ndarray],
                                plot_size: Tuple[int, int] = (12, 4),
                                overlay_legend: bool = True
):
    """
    Display a color overlay of tissue masks on the cropped fruit image.

    Applies semi-transparent color overlays for each tissue region
    on top of the original fruit image using alpha blending.

    Parameters
    ----------
    masks : Dict[str, np.ndarray]
        Dictionary of tissue masks as returned by :func:`get_single_fruit_masks`.
        Must contain ``'cropped_img'``. Optional tissue keys:
        ``'outer_pericarp'``, ``'internal_pericarp'``, ``'locules'``.
    plot_size : Tuple[int, int], optional
        Figure size for the matplotlib plot. Default is (12, 4).
    overlay_legend : bool, optional
        If True, display a legend identifying tissue colors. Default is True.
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
    Annotate an image with fruit IDs and bounding boxes.

    Parameters
    ----------
    img : np.ndarray
        Input image in BGR format.
    contours : List[np.ndarray]
        List of all contours (fruits and locules).
    fruit_locule_map : Dict[int, List[int]]
        Mapping of fruit contour indices to their locule contour indices.
    renumber : bool, optional
        If True, fruit IDs are renumbered from 1 to n. Default is True.
    color : Tuple[int, int, int], optional
        BGR color for bounding boxes and text. Default is green (0, 255, 0).
    thickness : int, optional
        Line thickness for bounding boxes. Default is 2.
    font_scale : float, optional
        Font scale for fruit ID labels. Default is 0.6.

    Returns
    -------
    np.ndarray
        Annotated copy of the input image in BGR format.
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
    Create pixel-level color histograms for all fruits.

    For each fruit, extracts the pixel distribution across each color
    channel for the total pericarp region (or total fruit region if 
    external analysis) only. 
    
    Each bin in the histogram corresponds to one integer intensity value
    within the channel's valid range.

    Parameters
    ----------
    img : np.ndarray
        Input image in BGR format.
    hsv_img : np.ndarray or None
        Precomputed HSV image. If None, it is created internally.
    mask : np.ndarray
        Binary segmentation mask where fruit pixels are > 0.
    contours : List[np.ndarray]
        List of all contours (fruits and locules).
    fruit_locule_map : Dict[int, List[int]]
        Mapping of fruit contour indices to their locule contour indices.
    image_name : str, optional
        Image identifier included in each output row. Default is ``''``.
    label : str, optional
        Sample label included in each output row. Default is ``''``.
    color_space : str, optional
        Color spaces to include. Either ``'all'`` or a comma-separated
        subset of ``'rgb'``, ``'lab'``, ``'hsv'``, ``'gray'``.
        Default is ``'all'``.
    renumber : bool, optional
        If True, fruit IDs are renumbered from 1 to n. Default is True.
    dark_threshold : int, optional
        Grayscale intensity threshold below which pixels are excluded.
        Default is 0 (no exclusion).
    normalize : bool, optional
        If True, bin counts are divided by the total number of valid pixels.
        Default is False.

    Returns
    -------
    List[Dict[str, float]]
        One dictionary per fruit with keys ``'image_name'``, ``'label'``,
        ``'fruit_id'``, and one key per histogram bin (e.g., ``'R_0'``
        through ``'R_255'``). Returns NaN values for fruits with no
        valid pixels.

    Raises
    ------
    ValueError
        If ``color_space`` contains invalid entries.
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
    Calculate a red/yellow/orange color index per fruit from hue histograms.

    Classifies each fruit into a dominant color category based on the
    proportion of pixels falling within defined hue ranges. A fruit is
    labeled with its dominant color only if that color exceeds the
    homogeneity threshold; otherwise it is labeled ``'mixed'``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing hue histogram columns (``'H_0'`` through
        ``'H_359'``) as produced by :func:`get_fruit_color_histograms`.
    red_hue_ranges : List[Tuple[int, int]], optional
        List of (min, max) hue degree ranges considered red.
        Default is ``[(0, 21), (250, 360)]``.
    yellow_hue_range : Tuple[int, int], optional
        (min, max) hue degree range considered yellow. Default is ``(40, 80)``.
    orange_hue_range : Tuple[int, int] or None, optional
        (min, max) hue degree range considered orange. Set to None to
        disable orange classification. Default is ``(22, 39)``.
    homogeneity_threshold : float, optional
        Minimum dominant color ratio required to assign a single color
        category. Default is 0.80.

    Returns
    -------
    pd.DataFrame
        One row per fruit with columns: ``'image_name'``, ``'fruit_id'``,
        ``'red_pixels'``, ``'yellow_pixels'``, ``'orange_pixels'`` (if enabled),
        ``'total_hue_pixels'``, ``'red_ratio'``, ``'yellow_ratio'``,
        ``'orange_ratio'`` (if enabled), ``'color_homogeneity'``,
        ``'color_category'``, and ``'dominant_color'``.

    Raises
    ------
    ValueError
        If required hue histogram columns are missing from ``df``.
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