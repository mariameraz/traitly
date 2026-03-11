# traitly/fruit_phenotyping/mask.py
"""
Mask generation and fruit detection utilities for fruit phenotyping pipelines.

Provides functions to segment fruits from HSV images, enhance locule
contrast, detect and filter fruit contours, and merge nearby locules.
Designed to be called from
:class:`~traitly.fruit_phenotyping.internal_analysis.FruitInternalAnalyzer`.
"""
# ============================================================================
# STANDARD LIBRARY
# ============================================================================
from typing import Optional, Tuple, List, Dict

# ============================================================================
# THIRD-PARTY LIBRARIES
# ============================================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap

# ============================================================================
# INTERNAL IMPORTS
# ===========================================================================
from ..utils.basic_functions import plot_img

#################################################################################################
# Create fruit mask
#################################################################################################

def create_mask(
    img_hsv: np.ndarray,
    lower_hsv: Optional[Tuple[int, int, int]] = None,
    upper_hsv: Optional[Tuple[int, int, int]] = None,
    n_iteration: int = 1,
    kernel_blur: Optional[int] = None,
    kernel_open: Optional[int] = None,
    kernel_close: Optional[int] = None,
    canny_min: Optional[int] = None,
    canny_max: Optional[int] = None,
    plot: bool = True,
    plot_size: Tuple[int, int] = (5, 5),
    background_color: Optional[str] = None,
    fill_holes: bool = False,
    apply_convex_hull: bool = False,
) -> np.ndarray:
    """
    Generate a binary mask segmenting foreground objects from an HSV image.

    Applies HSV color thresholding to isolate the background, then inverts
    to obtain the foreground mask. The following refinement steps are applied
    in order when their parameters are provided:

    1. Morphological opening via ``kernel_open`` — removes small noise.
    2. Morphological closing via ``kernel_close`` — fills small holes.
    3. Gaussian blur via ``kernel_blur`` — smooths edges.
    4. Canny edge detection via ``canny_min`` / ``canny_max`` — recovers
       detail lost by blurring; edges are OR-combined with the mask.
    5. Hole filling via :func:`fill_holes_to_mask` when ``fill_holes=True``.
    6. Convex hull per contour via :func:`apply_convex_hull_to_mask` when
       ``apply_convex_hull=True``.

    Parameters
    ----------
    img_hsv : np.ndarray
        Input image in HSV format (uint8, H: 0–180, S: 0–255, V: 0–255).
    lower_hsv : tuple of int or None, optional
        Lower HSV bound for background thresholding. If ``None`` and
        ``background_color`` is also ``None``, defaults to
        ``[0, 0, 0]``. Default is ``None``.
    upper_hsv : tuple of int or None, optional
        Upper HSV bound for background thresholding. If ``None`` and
        ``background_color`` is also ``None``, defaults to
        ``[180, 250, 50]``. Default is ``None``.
    n_iteration : int, optional
        Number of iterations for morphological operations. Default is 1.
    kernel_blur : int or None, optional
        Odd kernel size for Gaussian blur. If ``None``, blur is skipped.
        Default is ``None``.
    kernel_open : int or None, optional
        Odd kernel size for morphological opening. If ``None``, opening
        is skipped. Default is ``None``.
    kernel_close : int or None, optional
        Odd kernel size for morphological closing. If ``None``, closing
        is skipped. Default is ``None``.
    canny_min : int or None, optional
        Lower threshold for Canny edge detection. Must be provided together
        with ``canny_max``. Default is ``None``.
    canny_max : int or None, optional
        Upper threshold for Canny edge detection. Must be provided together
        with ``canny_min``. Default is ``None``.
    plot : bool, optional
        If True, display the final mask. Default is True.
    plot_size : tuple of int, optional
        Figure size for the mask plot. Default is (5, 5).
    background_color : str or None, optional
        Preset background color that overrides ``lower_hsv`` and
        ``upper_hsv``. One of ``'blue'``, ``'white'``, or ``'black'``.
        Default is ``None``.
    fill_holes : bool, optional
        If True, fill enclosed holes in the mask via
        :func:`fill_holes_to_mask`. Default is False.
    apply_convex_hull : bool, optional
        If True, replace each contour with its convex hull via
        :func:`apply_convex_hull_to_mask`. Default is False.

    Returns
    -------
    np.ndarray
        Binary mask of shape ``(H, W)`` with dtype uint8, where 255
        indicates foreground (fruit) and 0 indicates background.

    Raises
    ------
    TypeError
        If ``img_hsv`` is not a numpy array.
    ValueError
        If ``img_hsv`` is not a 3-channel uint8 array, kernel sizes are
        not positive odd integers, ``canny_min >= canny_max``, only one
        of ``canny_min`` / ``canny_max`` is provided, ``lower_hsv`` has
        values greater than ``upper_hsv``, or ``background_color`` is
        not a supported preset.
    RuntimeError
        If the initial mask creation fails or an OpenCV error occurs.
    """
    try:
       
        # Validation img
        if not isinstance(img_hsv, np.ndarray):
            raise TypeError("Input image must be a numpy array")
            
        if img_hsv.ndim != 3 or img_hsv.shape[2] != 3:
            raise ValueError("Image must be in HSV format (3 channels)")
        
        if img_hsv.dtype != np.uint8:
            raise ValueError("HSV image must be uint8 type (0-180 for H, 0-255 for S/V)")
        
        # Validate kernels: 
        if not isinstance(n_iteration, int) or n_iteration < 1:
            raise ValueError("n_iteration must be a positive integer")
        
        if kernel_open is not None:
            if not isinstance(kernel_open, int) or kernel_open < 1 or kernel_open % 2 == 0:
                raise ValueError("kernel_open must be a positive odd integer")
        
        if kernel_close is not None:
            if not isinstance(kernel_close, int) or kernel_close < 1 or kernel_close % 2 == 0:
                raise ValueError("kernel_close must be a positive odd integer")
        
        if kernel_blur is not None:
            if not isinstance(kernel_blur, int) or kernel_blur < 1 or kernel_blur % 2 == 0:
                raise ValueError("blur_kernel must be a positive odd integer")
        
        # Validate canny:
        if (canny_min is None) != (canny_max is None):
            raise ValueError("Both canny_min and canny_max must be provided together or both None")
        
        if canny_min is not None and canny_max is not None:
            if not isinstance(canny_min, int) or not isinstance(canny_max, int):
                raise ValueError("canny_min and canny_max must be integers")
            if canny_min >= canny_max:
                raise ValueError("canny_min must be < canny_max")
        
        # Set default HSV values for black/dark backgrounds if not provided
        
        if background_color is not None:
            if background_color == 'blue':
                lower_hsv = np.array([90, 100, 80], dtype=np.uint8)
                upper_hsv = np.array([130, 255, 255], dtype=np.uint8)
            elif background_color == 'white':
                lower_hsv = np.array([0, 0, 100], dtype=np.uint8)   
                upper_hsv = np.array([180, 50, 255], dtype=np.uint8)
            elif background_color == 'black':
                lower_hsv = np.array([0, 0, 0], dtype=np.uint8)
                upper_hsv = np.array([180, 250, 50], dtype=np.uint8)
            else:
                raise ValueError(f"Invalid background_color: {background_color}. Use: {{'blue', 'white', 'black'}}")

        if lower_hsv is None:
            lower_hsv = np.array([0, 0, 0], dtype=np.uint8)
        elif isinstance(lower_hsv, (list, tuple)): 
            lower_hsv = np.array(lower_hsv, dtype=np.uint8)
            
        if upper_hsv is None:
            upper_hsv = np.array([180, 250, 50], dtype=np.uint8)
        elif isinstance(upper_hsv, (list, tuple)):
            upper_hsv = np.array(upper_hsv, dtype=np.uint8)

        # Validate hsv thresh
        if not isinstance(lower_hsv, np.ndarray) or lower_hsv.shape != (3,):
            raise ValueError("lower_hsv must be a numpy array with shape (3,)")
        if not isinstance(upper_hsv, np.ndarray) or upper_hsv.shape != (3,):
            raise ValueError("upper_hsv must be a numpy array with shape (3,)")
            
        if (lower_hsv > upper_hsv).any():
            raise ValueError("All values in lower_hsv must be <= corresponding values in upper_hsv")

        
        # Create binary mask where [lower_hsv, upper_hsv] are white (255) (background) 
        # and others black (0) (fruits/label)
        mask_background = cv2.inRange(img_hsv, lower_hsv, upper_hsv) 
        if mask_background is None:
            raise RuntimeError("Failed to create initial mask")

        # Invert the binary mask to focus on foreground objects (fruits/label)
        mask = cv2.bitwise_not(mask_background) 

        # Creates an elliptical kernel for morphological operations:
        if kernel_open is not None:        
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_open, kernel_open)) 
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=n_iteration)

        if kernel_close is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_close, kernel_close))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=n_iteration)
        
        if kernel_blur is not None:
            mask = cv2.GaussianBlur(mask, (kernel_blur, kernel_blur), 0)
        
        edges = None
        if canny_min is not None and canny_max is not None: 
            mask_canny = mask if kernel_blur is not None else cv2.GaussianBlur(mask, (5,5), 0)
            edges = cv2.Canny(mask_canny, canny_min, canny_max)

        if edges is not None:
            final_mask = cv2.bitwise_or(mask, edges)
        else:
            final_mask = mask

        if fill_holes:
            final_mask = fill_holes_to_mask(final_mask)
        
        if apply_convex_hull:
            final_mask = apply_convex_hull_to_mask(final_mask)

        if plot:
            plot_img(final_mask, 
                     fig_axis=False, 
                     plot_size=plot_size)

        return final_mask
        
    except cv2.error as e:
        raise RuntimeError(f"OpenCV error: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {str(e)}")
    

#################################################################################################
# Fill contour holes with floodfill
#################################################################################################
def fill_holes_to_mask(mask: np.ndarray) -> np.ndarray:
    """
    Fill enclosed holes in a binary mask using flood fill.

    Flood-fills from the top-left corner of a padded copy to identify
    all background-connected regions, then inverts and OR-combines with
    the original mask to close interior holes.

    Parameters
    ----------
    mask : np.ndarray
        2D binary mask (any dtype). Non-zero pixels are treated as
        foreground.

    Returns
    -------
    np.ndarray
        Binary mask of the same shape with dtype uint8, where all
        enclosed holes are filled (set to 255).

    Raises
    ------
    ValueError
        If ``mask`` is not a 2D array.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    m = (mask > 0).astype(np.uint8) * 255

    h, w = m.shape
    flood = m.copy()

    ff_mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(flood, ff_mask, (0, 0), 255)

    flood_inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(m, flood_inv)
    return filled

#################################################################################################
# Close contours slit with convex hull 
#################################################################################################

def apply_convex_hull_to_mask(
    mask: np.ndarray,
    min_area: int = 50,
    contours: Optional[Dict] = None,
) -> np.ndarray:
    """
    Replace each contour in a binary mask with its convex hull.

    Finds external contours via ``cv2.findContours`` (unless precomputed
    contours are provided), filters by ``min_area``, and draws filled
    convex hulls onto a blank canvas.

    Parameters
    ----------
    mask : np.ndarray
        2D binary mask (any dtype). Non-zero pixels are treated as
        foreground.
    min_area : int, optional
        Minimum contour area in pixels to include. Smaller contours are
        skipped. Default is 50.
    contours : dict or None, optional
        Precomputed contours to use instead of running
        ``cv2.findContours``. If ``None``, contours are detected
        internally. Default is ``None``.

    Returns
    -------
    np.ndarray
        Binary mask of the same shape and dtype uint8, where each
        qualifying contour region is replaced by its filled convex hull.
    """
    m = (mask > 0).astype(np.uint8) * 255

    if contours is None:
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out = np.zeros_like(m)
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        hull = cv2.convexHull(c)
        cv2.drawContours(out, [hull], -1, 255, thickness=-1)
    return out

#################################################################################################
# Find fruits using size/shape thresholds
#################################################################################################
def find_fruits(
    binary_mask: np.ndarray,
    min_locule_area: int = 50,
    min_locules_per_fruit: int = 1,
    min_fruit_area: Optional[int] = None,
    max_fruit_area: Optional[int] = None,
    min_circularity: float = 0.4,
    max_circularity: float = 1.0,
    rescale_factor: Optional[float] = None,
) -> Tuple[List[np.ndarray], Dict[int, List[int]]]:
    """
    Detect fruit contours and map each fruit to its internal locules.

    Uses ``cv2.RETR_TREE`` hierarchy when ``min_locules_per_fruit > 0``
    to associate child contours (locules) with their parent fruits.
    Uses ``cv2.RETR_EXTERNAL`` otherwise for efficiency. Contours are
    filtered by area, circularity, and aspect ratio. When
    ``rescale_factor`` is provided, the mask is downscaled before
    detection and contours are scaled back to full resolution.

    Parameters
    ----------
    binary_mask : np.ndarray
        2D binary mask (uint8) where foreground pixels are non-zero.
    min_locule_area : int, optional
        Minimum contour area in pixels to accept a child contour as a
        locule. Default is 50.
    min_locules_per_fruit : int, optional
        Minimum number of valid locules required to retain a fruit.
        Set to 0 to disable locule filtering. Default is 1.
    min_fruit_area : int or None, optional
        Minimum contour area in pixels to accept a top-level contour as
        a fruit. If ``None``, no lower bound is applied. Default is
        ``None``.
    max_fruit_area : int or None, optional
        Maximum contour area in pixels. If ``None``, no upper bound is
        applied. Default is ``None``.
    min_circularity : float, optional
        Minimum circularity score in [0, 1]. Default is 0.4.
    max_circularity : float, optional
        Maximum circularity score in [0, 1]. Default is 1.0.
    rescale_factor : float or None, optional
        Factor in (0, 1] to downscale the mask before detection. Area
        thresholds are adjusted automatically. If ``None`` or 1, no
        rescaling is applied. Default is ``None``.

    Returns
    -------
    contours : list of np.ndarray
        All detected contours (including locules), indexed consistently
        with ``fruit_locule_map``.
    fruit_locule_map : dict of {int : list of int}
        Mapping from fruit contour index to list of locule contour
        indices. Fruits with fewer than ``min_locules_per_fruit`` locules
        are excluded.

    Raises
    ------
    ValueError
        If ``binary_mask`` is not a 2D uint8 array, area thresholds are
        non-positive or inverted, circularity range is invalid,
        ``rescale_factor`` is outside (0, 1], or locule count is
        negative.
    """

    min_aspect_ratio = 0.3
    max_aspect_ratio = 3
    
    # Validation
    if not isinstance(binary_mask, np.ndarray) or binary_mask.dtype != np.uint8:
        raise ValueError("binary_mask must be uint8 numpy array")
    
    if len(binary_mask.shape) != 2:
        raise ValueError("binary_mask must be 2D array")
    
    if rescale_factor is not None and not (0 < rescale_factor <= 1):
        raise ValueError('rescale_factor must be in range (0, 1]')
    
    if min_locule_area <= 0 or min_locules_per_fruit < 0:
        raise ValueError("Area and locule count must be positive")
    
    if min_fruit_area is not None and min_fruit_area <= 0:
        raise ValueError("min_fruit_area must be positive")
    
    if max_fruit_area is not None and max_fruit_area <= 0:
        raise ValueError("max_fruit_area must be positive")
    
    if min_fruit_area is not None and max_fruit_area is not None:
        if min_fruit_area > max_fruit_area:
            raise ValueError("min_fruit_area cannot be greater than max_fruit_area")
    
    if not (0 <= min_circularity <= max_circularity <= 1):
        raise ValueError("Circularity: 0 ≤ min ≤ max ≤ 1")
    
    if not (0 < min_aspect_ratio <= max_aspect_ratio):
        raise ValueError("Aspect ratio: 0 < min ≤ max")

    # rescale image (if requested)
    should_rescale = rescale_factor is not None and rescale_factor < 1
    
    if should_rescale:
        original_h, original_w = binary_mask.shape
        new_w = int(original_w * rescale_factor)
        new_h = int(original_h * rescale_factor)
        resized_mask = cv2.resize(binary_mask, (new_w, new_h), 
                                  interpolation=cv2.INTER_NEAREST)
        
        adjusted_min_locule_area = int(min_locule_area * (rescale_factor ** 2))
        adjusted_min_fruit_area = (int(min_fruit_area * (rescale_factor ** 2)) 
                                  if min_fruit_area is not None else None)
        adjusted_max_fruit_area = (int(max_fruit_area * (rescale_factor ** 2)) 
                                  if max_fruit_area is not None else None)
        
        scale_x = original_w / new_w
        scale_y = original_h / new_h
    else:
        resized_mask = binary_mask
        adjusted_min_locule_area = min_locule_area
        adjusted_min_fruit_area = min_fruit_area
        adjusted_max_fruit_area = max_fruit_area

    # Contours detected (hierarchy)
    needs_locules = min_locules_per_fruit > 0
    
    if needs_locules:
        contours, hierarchy = cv2.findContours(
            resized_mask,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE 
        )
        
        if not contours or hierarchy is None:
            return [], {}
        
        hierarchy = hierarchy[0]
        is_top_level = hierarchy[:, 3] == -1
        
    else:
        contours, _ = cv2.findContours(
            resized_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return [], {}
        
        n_contours = len(contours)
        
        hierarchy = np.full((n_contours, 4), -1, dtype=np.int32)
        
        if n_contours > 1:
            hierarchy[:-1, 0] = np.arange(1, n_contours)  # next
            hierarchy[1:, 1] = np.arange(n_contours - 1)   # prev
        
        is_top_level = np.ones(n_contours, dtype=bool)

    # Calculate shape/size metrics
    n_contours = len(contours)
    
    areas = np.zeros(n_contours, dtype=np.float64)
    perimeters = np.zeros(n_contours, dtype=np.float64)
    aspect_ratios = np.zeros(n_contours, dtype=np.float64)

    for i, contour in enumerate(contours):
        areas[i] = cv2.contourArea(contour)
        perimeters[i] = cv2.arcLength(contour, True)
    
        rect = cv2.minAreaRect(contour)
        w, h = rect[1]
        if min(w, h) > 0:
            aspect_ratios[i] = max(w, h) / min(w, h)

    with np.errstate(divide='ignore', invalid='ignore'):
        circularities = (4 * np.pi * areas) / (perimeters ** 2)
        circularities = np.nan_to_num(circularities, nan=0.0, posinf=0.0, neginf=0.0)

    # Create a mask for only filtered fruits
    filters = np.ones(n_contours, dtype=bool)
    
    # Apply filters
    filters &= is_top_level
    
    if adjusted_min_fruit_area is not None:
        filters &= (areas >= adjusted_min_fruit_area)
    if adjusted_max_fruit_area is not None:
        filters &= (areas <= adjusted_max_fruit_area)
    
    filters &= (circularities >= min_circularity) & (circularities <= max_circularity)
    filters &= (aspect_ratios >= min_aspect_ratio) & (aspect_ratios <= max_aspect_ratio)
    
    valid_fruit_indices = np.where(filters)[0]

    # Build fruit-locule map
    if needs_locules:
        
        parents = hierarchy[:, 3]
        
        # For each fruit, find their child 
        fruit_locules_map = {}
        
        if len(valid_fruit_indices) > 0:
            for fruit_idx in valid_fruit_indices:
             
                child_mask = (parents == fruit_idx)
                
                valid_locules_mask = child_mask & (areas >= adjusted_min_locule_area)
                locule_indices = np.where(valid_locules_mask)[0]
                
                if len(locule_indices) >= min_locules_per_fruit:
                    fruit_locules_map[int(fruit_idx)] = locule_indices.tolist()
    else:
        
        fruit_locules_map = dict.fromkeys(valid_fruit_indices.astype(int), [])

    # Rescale image 
    if should_rescale:
        scale_factors = np.array([[scale_x, scale_y]], dtype=np.float32)
        
        contours = [
            (contour.astype(np.float32) * scale_factors).astype(np.int32)
            for contour in contours
        ]
    
    return contours, fruit_locules_map


#################################################################################################
# Merge close locules
#################################################################################################

def merge_locules_func(
    locules_indices: List[int],
    contours: List[np.ndarray],
    min_distance: int = 0,
    max_distance: int = 50,
    min_area: int = 10,
) -> List[np.ndarray]:
    """
    Merge spatially close locule contours into single contours.

    Filters locules by ``min_area``, computes pairwise centroid distances
    via :func:`scipy.spatial.distance.pdist`, and merges pairs whose
    actual point-to-polygon distance falls in
    ``(min_distance, max_distance)``. Merged contours are approximated
    with :func:`cv2.approxPolyDP` to reduce point count.

    Parameters
    ----------
    locules_indices : list of int
        Indices into ``contours`` identifying the locule contours to
        process.
    contours : list of np.ndarray
        Full list of all detected contours.
    min_distance : int, optional
        Minimum point-to-polygon distance in pixels for two locules to
        be eligible for merging. Default is 0.
    max_distance : int, optional
        Maximum point-to-polygon distance in pixels for merging.
        Locule pairs farther apart than this are kept separate. Default
        is 50.
    min_area : int, optional
        Minimum contour area in pixels. Locules below this threshold are
        discarded before merging. Default is 10.

    Returns
    -------
    list of np.ndarray
        List of merged (or unchanged) contour arrays. Empty if
        ``locules_indices`` is empty or no valid locules remain after
        area filtering.
    """

    if not locules_indices:
        return []
    
    # Filter valid contours and compute centroids
    valid_locules = []
    valid_contours = []
    centroids = []
    
    for i in locules_indices:
        if len(contours[i]) > 0 and cv2.contourArea(contours[i]) > min_area:
            valid_locules.append(i)
            valid_contours.append(contours[i])
            
            # Compute centroid
            M = cv2.moments(contours[i])
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                centroids.append((cx, cy))
            else:
                centroids.append(None)
    
    if not valid_locules:
        return []
    
    # Build centroid distance matrix
    centroids_valid = [(i, c) for i, c in enumerate(centroids) if c is not None]
    
    if len(centroids_valid) < 2:
        # Not enough valid centroids, return original contours
        return valid_contours
    
    # Extract valid centroid coordinates
    centroid_indices = [idx for idx, _ in centroids_valid]
    centroid_coords = np.array([c for _, c in centroids_valid])
    
    # Compute pairwise centroid distances
    from scipy.spatial.distance import pdist, squareform
    centroid_distances = squareform(pdist(centroid_coords))
    
    # Merge locules based on distance thresholds
    merged = [False] * len(valid_locules)
    result_locules = []
    
    for i in range(len(valid_locules)):
        if not merged[i]:
            current_contour = valid_contours[i]
            
            # Skip empty contours
            if len(current_contour) == 0:
                continue
            
            merged[i] = True
            to_merge = [current_contour]
            
            # Pre filter candidates using centroid distances
            if centroids[i] is not None:
                try:
                    centroid_idx = centroid_indices.index(i)
                    
                    close_mask = centroid_distances[centroid_idx] <= (max_distance * 3)
                    close_indices = np.where(close_mask)[0]

                    candidates = [centroid_indices[idx] for idx in close_indices 
                                 if idx != centroid_idx]
                except ValueError:
                    # Centroid not in valid list, check all
                    candidates = range(i+1, len(valid_locules))
            else:
                # No valid centroid, check all remaining locules
                candidates = range(i+1, len(valid_locules))
            
            # Check each candidate for actual merge eligibility
            for j in candidates:
                if j <= i or merged[j]:
                    continue
                
                other_contour = valid_contours[j]
                
                # Skip empty contours
                if len(other_contour) == 0:
                    continue
                
                min_dist = float('inf')
                for point in other_contour[::2, 0, :]:
                    dist = cv2.pointPolygonTest(
                        current_contour, 
                        (float(point[0]), float(point[1])), 
                        True
                    )
                    if dist < min_dist:
                        min_dist = dist
                        # Early exit if distance is already too close
                        if min_dist <= 0:
                            break
                
                if min_distance < abs(min_dist) < max_distance:
                    to_merge.append(other_contour)
                    merged[j] = True
            
            # Merge contours if multiple were found
            if len(to_merge) > 1:
                try:
                    merged_contour = np.vstack(to_merge)
                    epsilon = 0.001 * cv2.arcLength(merged_contour, True)
                    merged_loculus = cv2.approxPolyDP(merged_contour, epsilon, True)
                    
                    # Verify merged contour is valid
                    if len(merged_loculus) > 0:
                        result_locules.append(merged_loculus)
                except:
                    # If merge fails, keep original contour
                    result_locules.append(current_contour)
            else:
                # No merge needed, keep original
                result_locules.append(current_contour)
    
    return result_locules


#################################################
# Convert image to lab and apply clahe contrast #
#################################################

def _ensure_uint8(L: np.ndarray) -> np.ndarray:
    """
    Convert an array to uint8, handling float [0, 1] and other ranges.

    Parameters
    ----------
    L : np.ndarray
        Input array. If float with values in [0, 1], scaled to [0, 255].
        If in another range, clipped to [0, 255] before conversion.

    Returns
    -------
    np.ndarray
        Array of dtype uint8.
    """
    if L.dtype != np.uint8:
        # If in float [0, 1], scale to [0, 255]
        if L.max() <= 1.0:
            L = (L * 255).astype(np.uint8)
        else:
            # If in another range, clip and convert
            L = np.clip(L, 0, 255).astype(np.uint8)
    return L

def gamma_contrast(
    L: np.ndarray,
    gamma: float = 1.0,
    plot: bool = False,
) -> np.ndarray:
    """
    Apply gamma correction to a luminance channel.

    Normalizes ``L`` to [0, 1], raises to the power of ``gamma``, and
    scales back to [0, 255].

    Parameters
    ----------
    L : np.ndarray
        Grayscale luminance channel (2D). Converted to uint8 via
        :func:`_ensure_uint8` before processing.
    gamma : float, optional
        Gamma exponent. Values below 1 brighten shadows; values above 1
        darken highlights; 1 applies no change. Default is 1.0.
    plot : bool, optional
        If True, display the corrected channel. Default is False.

    Returns
    -------
    np.ndarray
        Gamma-corrected luminance channel with dtype uint8.
    """

    # Normalize to [0, 1]
    L = _ensure_uint8(L)
    L_norm = L / 255.0
    
    # Apply gamma correction
    L_corrected = np.power(L_norm, gamma)

    # Convert back to [0, 255]
    l_gamma_corrected = (L_corrected * 255).astype(np.uint8)
    
    if plot:
        plot_img(l_gamma_corrected)

    return l_gamma_corrected

def sigmoid_contrast(
    L: np.ndarray,
    gain: float = 10,
    cutoff: float = 0.5,
) -> np.ndarray:
    """
    Apply sigmoidal contrast enhancement to a luminance channel.

    Normalizes ``L`` to [0, 1], applies the sigmoid function
    ``1 / (1 + exp(-gain * (x - cutoff)))``, renormalizes to [0, 1],
    and scales back to [0, 255].

    Parameters
    ----------
    L : np.ndarray
        Grayscale luminance channel (2D). Converted to uint8 via
        :func:`_ensure_uint8` before processing.
    gain : float, optional
        Steepness of the sigmoid curve. Recommended range is 5–20.
        Default is 10.
    cutoff : float, optional
        Midpoint of the sigmoid in normalized [0, 1] space. Recommended
        range is 0.3–0.7. Default is 0.5.

    Returns
    -------
    np.ndarray
        Sigmoid-enhanced luminance channel with dtype uint8.
    """

    L = _ensure_uint8(L)
    L_norm = L / 255.0
    
    # Apply sigmoid transformation
    L_sigmoid = 1 / (1 + np.exp(-gain * (L_norm - cutoff)))
    
    # Renormalize to [0, 1]
    L_sigmoid = (L_sigmoid - L_sigmoid.min()) / (L_sigmoid.max() - L_sigmoid.min())
    
    return (L_sigmoid * 255).astype(np.uint8)

def exp_transform(
    L: np.ndarray,
    c: float = 1.0,
) -> np.ndarray:
    """
    Apply an exponential transformation to a luminance channel.

    Normalizes ``L`` to [0, 1], applies ``expm1(c * x)``, rescales the
    result to [0, 255]. Expands high-value regions more than low-value
    regions.

    Parameters
    ----------
    L : np.ndarray
        Grayscale luminance channel (2D). Converted to uint8 via
        :func:`_ensure_uint8` before processing.
    c : float, optional
        Exponential coefficient controlling the expansion intensity.
        Default is 1.0.

    Returns
    -------
    np.ndarray
        Exponentially transformed luminance channel with dtype uint8.
    """

    L = _ensure_uint8(L)
    L_norm = L / 255.0
    
    # Apply exponential transformation
    L_exp = np.expm1(c * L_norm)
    L_exp = (L_exp / L_exp.max() * 255)
    
    return L_exp.astype(np.uint8)

def apply_contrast(
    img: np.ndarray,
    contrast_method: str = 'gamma',
    gamma: float = 1.5,
    gain: float = 5,
    cutoff: float = 0.5,
    c: float = 0.5,
    plot: bool = False,
    plot_size: Tuple[int, int] = (5, 5),
    compare: bool = False,
    kernel_blur: int = 1,
    clip_limit: Optional[int] = None,
    tile_grid_size: int = 12,
) -> np.ndarray:
    """
    Apply contrast enhancement to the L channel of a BGR image.

    Converts ``img`` to LAB color space, extracts the L channel, and
    applies the selected contrast method via :func:`gamma_contrast`,
    :func:`sigmoid_contrast`, or :func:`exp_transform`. Optionally
    applies median blur and CLAHE afterward. When ``compare=True``, all
    three methods are computed and displayed side by side before
    returning the result of the selected method.

    Parameters
    ----------
    img : np.ndarray
        Input image in BGR format (3-channel uint8).
    contrast_method : str or None, optional
        Enhancement method: ``'gamma'``, ``'sigmoid'``, ``'exp'``, or
        ``'none'``. Default is ``'gamma'``.
    gamma : float or None, optional
        Gamma exponent forwarded to :func:`gamma_contrast`. Default is
        1.5.
    gain : float or None, optional
        Sigmoid gain forwarded to :func:`sigmoid_contrast`. Default is 5.
    cutoff : float or None, optional
        Sigmoid cutoff forwarded to :func:`sigmoid_contrast`. Default is
        0.5.
    c : float or None, optional
        Exponential coefficient forwarded to :func:`exp_transform`.
        Default is 0.5.
    plot : bool or None, optional
        If True and ``compare=False``, display a side-by-side comparison
        of the original and transformed L channel. Default is False.
    plot_size : tuple of int or None, optional
        Figure size for plots. Default is (5, 5).
    compare : bool or None, optional
        If True, compute all three methods and display them together.
        The selected ``contrast_method`` is still returned. Default is
        False.
    kernel_blur : int or None, optional
        Odd kernel size for median blur applied after contrast
        enhancement. Set to 1 or ``None`` to skip. Default is 1.
    clip_limit : int or None, optional
        CLAHE clip limit applied after contrast and blur. If ``None``,
        CLAHE is skipped. Default is ``None``.
    tile_grid_size : int or None, optional
        CLAHE tile grid size. Used only when ``clip_limit`` is set.
        Default is 12.

    Returns
    -------
    np.ndarray
        Transformed L channel as a 2D uint8 array.

    Raises
    ------
    TypeError
        If ``img`` is not a numpy array.
    ValueError
        If ``img`` is not a 3-channel array or ``contrast_method`` is
        not one of the supported options.
    """
    # Validate input
    if not isinstance(img, np.ndarray):
        raise TypeError("Input image must be a numpy array")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Image must be in BGR format (3 channels)")

    # Validate method early (avoid extra work if wrong)
    if contrast_method not in ('gamma', 'sigmoid', 'exp', 'none'):
        raise ValueError("contrast_method must be one of ['gamma', 'sigmoid', 'exp', 'none']")
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Extract L channel
    l_channel = lab[:, :, 0]

    # If compare=True, compute all 3 once (and reuse for selected output)
    l_gamma = l_sigmoid = l_exp = None
    if compare:
        l_gamma = gamma_contrast(l_channel, gamma=gamma, plot=False)
        l_sigmoid = sigmoid_contrast(l_channel, gain=gain, cutoff=cutoff)
        l_exp = exp_transform(l_channel, c=c)

        plt.figure(figsize=plot_size)
        
        plt.subplot(2, 2, 1)
        plt.imshow(l_channel, cmap='gray')
        plt.title("Original")
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.imshow(l_gamma, cmap='gray')
        plt.title(f"Gamma (γ={gamma})")
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.imshow(l_sigmoid, cmap='gray')
        plt.title(f"Sigmoid (gain={gain}, cutoff={cutoff})")
        plt.axis('off')
        
        plt.subplot(2, 2, 4)
        plt.imshow(l_exp, cmap='gray')
        plt.title(f"Exponential (c={c})")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    # Apply transformation to the L channel using the selected method
    if contrast_method == 'gamma':
        l_transformed = l_gamma if compare else gamma_contrast(l_channel, gamma=gamma, plot=False)
    elif contrast_method == 'sigmoid':
        l_transformed = l_sigmoid if compare else sigmoid_contrast(l_channel, gain=gain, cutoff=cutoff)
    elif contrast_method == 'exp':
        l_transformed = l_exp if compare else exp_transform(l_channel, c=c)
    else:  # 'none'
        l_transformed = l_channel.copy()

    # Apply median blur (if kernel 1 skip to save time)
    if kernel_blur is not None and kernel_blur > 1:
        l_transformed = cv2.medianBlur(l_transformed, kernel_blur)
    
    # Apply CLAHE only if clip_limit is specified
    if clip_limit is not None:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, 
                               tileGridSize=(tile_grid_size, tile_grid_size))
        l_transformed = clahe.apply(l_transformed)
    
    # If plot=True and compare=False, display only the selected method
    if plot and not compare:
        plt.figure(figsize=plot_size)
        
        plt.subplot(1, 2, 1)
        plt.imshow(l_channel, cmap='gray')
        plt.title("Original L Channel")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(l_transformed, cmap='gray')
        
        # Build title with processing info
        if contrast_method == 'none':
            title = "L Channel (no contrast)"
        else:
            title = f"L Channel ({contrast_method})"
        
        if clip_limit is not None:
            title += f" + CLAHE({clip_limit})"
        
        plt.title(title)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return l_transformed

##########################################
# Generate histagram plots for L channel #
##########################################

def generate_l_channel_histogram(
    l_transformed: np.ndarray,
    fruit_mask: np.ndarray,
    otsu_offset: int = 0,
    plot_size: Tuple[int, int] = (14, 4),
) -> None:

    pixels = l_transformed[fruit_mask > 0]

    otsu_val, _ = cv2.threshold(l_transformed, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    otsu_adjusted = int(otsu_val + otsu_offset)

    kde = gaussian_kde(pixels, bw_method=0.1)
    x = np.linspace(0, 255, 500)
    kde_scaled = kde(x) * len(pixels) * (255 / 100)

    width = plot_size[0]
    title_fontsize  = int(np.clip(8 + width * 0.4, 10, 24))
    label_fontsize  = int(np.clip(6 + width * 0.2,  8, 18))
    tick_fontsize   = int(np.clip(5 + width * 0.2,  7, 14))
    legend_fontsize = int(np.clip(4 + width * 0.2,  7, 13))

    intensity_cmap = LinearSegmentedColormap.from_list('gray_bar', ['black', 'white'])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)

    locules  = pixels[pixels <= otsu_adjusted]
    pericarp = pixels[pixels >  otsu_adjusted]

    fig, axes = plt.subplots(1, 2, figsize=plot_size)
    fig.subplots_adjust(bottom=0.28)

    # L adjusted plot
    axes[0].hist(pixels, bins=100, color='thistle', edgecolor='none', alpha=0.6)
    axes[0].plot(x, kde_scaled, color="#094A95", linewidth=1.8)
    axes[0].set_xlabel('L value (0–255)', fontsize=label_fontsize, labelpad=8)
    axes[0].set_ylabel('Pixel count', fontsize=label_fontsize)
    axes[0].set_title('L channel distribution', fontsize=title_fontsize, fontweight='bold')
    axes[0].tick_params(axis='both', labelsize=tick_fontsize)
    axes[0].grid(alpha=0.3)

    # Otsu plot
    axes[1].hist(locules,  bins=100, color='tomato',    edgecolor='none',
                 alpha=0.7, label=f'Locules (≤{otsu_adjusted})')
    axes[1].hist(pericarp, bins=100, color='thistle', edgecolor='none',
                 alpha=0.7, label=f'Pericarp (>{otsu_adjusted})')
    axes[1].plot(x, kde_scaled, color="#094A95", linewidth=1.8)
    axes[1].axvline(otsu_adjusted, color='black', linestyle='--',
                    linewidth=1.5, label=f'Otsu: {otsu_adjusted}')
    axes[1].set_xlabel('L value (0–255)', fontsize=label_fontsize, labelpad=8)
    axes[1].set_ylabel('Pixel count', fontsize=label_fontsize)
    axes[1].set_title('L channel distribution (Otsu split)', fontsize=title_fontsize, fontweight='bold')
    axes[1].tick_params(axis='both', labelsize=tick_fontsize)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=legend_fontsize)

    # Pixel intensity bars
    x1, y1 = plot_size
    bar_h = 0.03 
    bar_y = 0.025 * plot_size[1]

    for ax in axes:
        pos = ax.get_position()
        bar_ax = fig.add_axes([pos.x0, bar_y, pos.width, bar_h])
        bar_ax.imshow(gradient, aspect='auto', cmap=intensity_cmap,
                      extent=[0, 255, 0, 1])
        bar_ax.set_xticks([])
        bar_ax.set_yticks([])
        for spine in bar_ax.spines.values():
            spine.set_visible(False)

    plt.show()

#######################
# Create locule masks #
#######################

def create_mask_locules(
    l_transformed: np.ndarray,
    fruit_mask: np.ndarray,
    kernel_close: Optional[int] = None,
    thresh_min: int = 100,
    kernel_open: Optional[int] = None,
    kernel_blur: Optional[int] = None,
    erosion_px: int = 0,
    use_otsu: bool = False,
    otsu_offset: int = 0,
    min_fruit_area: int = 5000,
    min_locule_area: int = 50,
    invert_locules: bool = False,
    plot: bool = False,
    plot_size: Tuple[int, int] = (15, 5),
) -> np.ndarray:
    """
    Generate a fused binary mask containing fruits with their internal locules.

    Thresholds ``l_transformed`` to extract locule regions, applies
    optional morphological refinement, and retains only locules that fall
    inside large fruit contours identified from the inverted locule mask.
    The locule mask is then fused with ``fruit_mask`` so that locule
    cavities appear as foreground holes within each fruit.

    Parameters
    ----------
    l_transformed : np.ndarray
        Contrast-enhanced L channel (2D uint8) as returned by
        :func:`apply_contrast`.
    fruit_mask : np.ndarray
        Binary fruit mask (2D uint8) as returned by :func:`create_mask`.
    kernel_close : int or None, optional
        Odd kernel size for morphological closing applied to the thresholded
        locule mask. If ``None``, closing is skipped. Default is ``None``.
    thresh_min : int, optional
        Lower threshold for ``cv2.THRESH_BINARY_INV`` binarization of
        ``l_transformed``. Default is 100.
    kernel_open : int or None, optional
        Odd kernel size for morphological opening applied after closing.
        If ``None``, opening is skipped. Default is ``None``.
    kernel_blur : int or None, optional
        Odd kernel size for Gaussian blur applied after morphological
        operations. If ``None``, blur is skipped. Default is ``None``.
    erosion_px : int, optional
        Erosion radius in pixels applied to ``fruit_mask`` before masking
        locules. Useful to exclude false locules detected at the fruit
        border. Set to 0 to skip. Default is 0.
    use_otsu : bool, optional
        If True, ignore ``thresh_min`` and compute the threshold automatically
        using Otsu's method. Default is False.
    otsu_offset : int, optional
        Offset added to the Otsu threshold. Positive values capture more
        pixels, negative values less. Only used when ``use_otsu=True``.
        Default is 0.
    min_fruit_area : int, optional
        Minimum contour area in pixels to classify a region as a fruit
        during mask fusion. Smaller contours are ignored. Default is 5000.
    min_locule_area : int, optional
        Minimum area in pixels to retain a locule region. Smaller blobs
        are removed after morphological operations. Set to 0 to skip.
        Default is 50.
    invert_locules : bool, optional
        If True, invert the locule mask within fruit regions before
        fusion. Useful when locules are brighter than the surrounding
        pericarp. Default is False.
    plot : bool, optional
        If True, display the L channel, cleaned locule mask, and final
        fused mask side by side. Default is False.
    plot_size : tuple of int, optional
        Figure size for the three-panel plot. Default is (15, 5).

    Returns
    -------
    np.ndarray
        Binary mask (uint8) with fruit regions fused with their internal
        locule cavities.

    Raises
    ------
    TypeError
        If ``l_transformed`` or ``fruit_mask`` is not a numpy array.
    """

    # Validate input
    if not isinstance(l_transformed, np.ndarray):
        raise TypeError("l_transformed must be a numpy array")
    if not isinstance(fruit_mask, np.ndarray):
        raise TypeError("fruit_mask must be a numpy array")

    # Apply threshold to get locules
    if use_otsu:
        otsu_val, locule_mask = cv2.threshold(l_transformed, 0, 255,
                                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if otsu_offset != 0:
            _, locule_mask = cv2.threshold(l_transformed, otsu_val + otsu_offset, 255,
                                           cv2.THRESH_BINARY_INV)
    else:
        locule_mask = cv2.inRange(l_transformed, 0, thresh_min)

    # Apply morphological closing if specified
    if kernel_close is not None:
        kernel_cl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                              (kernel_close, kernel_close))
        locule_mask = cv2.morphologyEx(locule_mask, cv2.MORPH_CLOSE, kernel_cl)

    # Apply morphological opening if specified
    if kernel_open is not None:
        kernel_op = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                              (kernel_open, kernel_open))
        locule_mask = cv2.morphologyEx(locule_mask, cv2.MORPH_OPEN, kernel_op)

    # Apply Gaussian blur if specified
    if kernel_blur is not None:
        locule_mask = cv2.GaussianBlur(locule_mask, (kernel_blur, kernel_blur), 0)

    # Find all contours
    inv_locule_mask = cv2.bitwise_not(locule_mask)
    contours, hierarchy = cv2.findContours(inv_locule_mask, cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_SIMPLE)

    # Create black mask and fill ONLY fruits (large contours without parent)
    fruits_only_mask = np.zeros_like(locule_mask)

    if contours and hierarchy is not None:
        h0 = hierarchy[0]
        for i, cnt in enumerate(contours):
            if h0[i][3] == -1 and cv2.contourArea(cnt) > min_fruit_area:
                cv2.drawContours(fruits_only_mask, [cnt], -1, 255, -1)

    # Erode fruit mask to exclude border locules
    if erosion_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (erosion_px * 2 + 1, erosion_px * 2 + 1)
        )
        fruit_mask_eroded = cv2.erode(fruit_mask.copy(), kernel, iterations=1)
    else:
        fruit_mask_eroded = fruit_mask

    # Invert locules mask if requested
    if invert_locules:
        locule_mask_clean = cv2.bitwise_and(
            cv2.bitwise_not(locule_mask),
            fruit_mask_eroded
        )
    else:
        locule_mask_clean = cv2.bitwise_and(locule_mask, fruit_mask_eroded)

    # Filter small locules
    if min_locule_area > 0:
        contours_loc, _ = cv2.findContours(locule_mask_clean, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
        filtered = np.zeros_like(locule_mask_clean)
        for cnt in contours_loc:
            if cv2.contourArea(cnt) >= min_locule_area:
                cv2.drawContours(filtered, [cnt], -1, 255, -1)
        locule_mask_clean = filtered

    # Fuse fruit mask with locules mask
    mask_fruits_inv = cv2.bitwise_not(fruit_mask)
    mask_fruits_inv[locule_mask_clean == 255] = 255
    final_mask = cv2.bitwise_not(mask_fruits_inv)

    if plot:
        plt.figure(figsize=plot_size)

        plt.subplot(1, 3, 1)
        plt.imshow(l_transformed, cmap='gray')
        plt.title('L* contrast applied')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(locule_mask_clean, cmap='gray')
        title = "Locules mask" + (" (inverted)" if invert_locules else "")
        plt.title(title)
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(final_mask, cmap='gray')
        plt.title("Final mask (Fruits + Locules)")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    return final_mask

######################################################################
# Create a scatter plot to visualize pixel colors (HSV) on the image #
######################################################################

def generate_scatter_plot(
    img_hsv: np.ndarray,
    img_rgb: np.ndarray,
    sample_size: int = 10000,
    plot_size: Tuple[int, int] = (18, 5),
) -> None:
    """
    Display a scatterplot of pixel colors in HSV channel pairs.

    Randomly samples ``sample_size`` pixels from ``img_hsv`` and plots
    three pairwise HSV scatterplots (H vs S, H vs V, S vs V), colored
    by their true RGB values from ``img_rgb``. Useful for choosing HSV
    thresholds before calling :func:`create_mask`.

    Parameters
    ----------
    img_hsv : np.ndarray
        Image in HSV format (3-channel uint8), H: 0–180, S: 0–255,
        V: 0–255.
    img_rgb : np.ndarray
        Corresponding image in RGB format (3-channel uint8), used to
        color the scatter points.
    sample_size : int, optional
        Number of pixels to sample randomly. Capped at the total number
        of pixels. Default is 10000.
    plot_size : tuple of int, optional
        Figure size ``(width, height)`` for the three-panel plot.
        Font sizes scale with ``plot_size[0]``. Default is (18, 5).

    Returns
    -------
    None
    """
    
    # Reuse HSV image
    h, s, v = cv2.split(img_hsv)
    
    # Sample a random number of pixels
    indices = np.random.choice(h.size, min(sample_size, h.size), replace=False) 

    h_sample = h.ravel()[indices]
    s_sample = s.ravel()[indices]
    v_sample = v.ravel()[indices]

    # Reuse RGB image
    rgb_sample = img_rgb.reshape(-1, 3)[indices] / 255.0
    
    # Create subplot canvas
    fig, axes = plt.subplots(1, 3, figsize=plot_size)

    # Precalculate font size based on plot width size
    width = plot_size[0]
    title_fontsize = int(np.clip(8 + width * 0.6, 10, 24)) 
    label_fontsize = int(np.clip(6 + width * 0.4, 8, 18))
    tick_fontsize = int(np.clip( 5 + width * 0.3, 7, 14))

    # H vs S (RGB colored pixels)
    axes[0].scatter(h_sample, s_sample, c=rgb_sample, alpha=0.6, s=10)
    axes[0].set_xlabel('Hue (0-180)', fontsize = label_fontsize)
    axes[0].set_ylabel('Saturation (0-255)', fontsize = label_fontsize)
    axes[0].set_title('H vs S', fontweight='bold', fontsize = title_fontsize)
    axes[0].tick_params(axis='both', labelsize=tick_fontsize)
    axes[0].grid(alpha=0.3)

    # H vs V (RGB colored pixels)
    axes[1].scatter(h_sample, v_sample, c=rgb_sample, alpha=0.6, s=10)
    axes[1].set_xlabel('Hue (0-180)', fontsize = label_fontsize)
    axes[1].set_ylabel('Value (0-255)', fontsize = label_fontsize)
    axes[1].set_title('H vs V', fontweight='bold', fontsize = title_fontsize)
    axes[1].tick_params(axis='both', labelsize=tick_fontsize)
    axes[1].grid(alpha=0.3)

    # S vs V (RGB colored pixels)
    axes[2].scatter(s_sample, v_sample, c=rgb_sample, alpha=0.6, s=10)
    axes[2].set_xlabel('Saturation (0-255)', fontsize = label_fontsize)
    axes[2].set_ylabel('Value (0-255)', fontsize = label_fontsize)
    axes[2].set_title('S vs V', fontweight='bold', fontsize = title_fontsize)
    axes[2].tick_params(axis='both', labelsize=tick_fontsize)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


## Interactive editor

def interactive_mask_editor(mask: np.ndarray, original_img: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Interactive mask editor. Draw polygons to add (white) or remove (black) regions.

    Controls:
        Left click          : add polygon point (both panels)
        Right click drag    : pan
        W                   : fill polygon WHITE (add region)
        B                   : fill polygon BLACK (remove region)
        Enter               : apply current polygon
        Z                   : undo last edit
        C                   : clear current polygon points
        + / =               : zoom in
        - / _               : zoom out
        T                   : toggle original image overlay opacity (10% steps)
        Q                   : quit and SAVE changes
        ESC                 : quit and DISCARD all changes
    """
    edited = mask.copy()
    history = [mask.copy()]
    points = []
    mode = 'white'
    show_original = original_img is not None
    overlay_alpha = 0.4

    zoom = 1.0
    pan_x, pan_y = 0.0, 0.0
    is_panning = False
    pan_start = (0, 0)
    pan_origin = (0.0, 0.0)

    img_h, img_w = mask.shape

    # Same size for both panels
    PANEL_W = 700
    PANEL_H = 700

    if original_img is not None:
        if original_img.ndim == 2:
            orig_bgr = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
        else:
            orig_bgr = original_img.copy()
            if orig_bgr.dtype != np.uint8:
                orig_bgr = (orig_bgr * 255).astype(np.uint8)
        orig_bgr = cv2.resize(orig_bgr, (img_w, img_h))
    else:
        orig_bgr = None

    window = 'Mask Editor  |  W/B=mode  Enter=apply  Z=undo  C=clear  +/-=zoom  RightDrag=pan  T=overlay  Q=save  ESC=discard'

    def clamp_pan():
        nonlocal pan_x, pan_y
        max_pan_x = max(0.0, img_w - img_w / zoom)
        max_pan_y = max(0.0, img_h - img_h / zoom)
        pan_x = max(0.0, min(pan_x, max_pan_x))
        pan_y = max(0.0, min(pan_y, max_pan_y))

    def screen_to_img(sx, sy):
        """convert screen coords to image coords"""
        ix = int(sx / PANEL_W * (img_w / zoom) + pan_x)
        iy = int(sy / PANEL_H * (img_h / zoom) + pan_y)
        return (max(0, min(ix, img_w - 1)), max(0, min(iy, img_h - 1)))

    def img_to_screen(ix, iy):
        """Convert image coords to screen coords"""
        sx = int((ix - pan_x) * zoom / img_w * PANEL_W)
        sy = int((iy - pan_y) * zoom / img_h * PANEL_H)
        return (sx, sy)

    def render_panel_mask(img):
        """Render left panel"""
        x1 = int(pan_x)
        y1 = int(pan_y)
        x2 = int(pan_x + img_w / zoom)
        y2 = int(pan_y + img_h / zoom)
        x1, x2 = max(0, x1), min(img_w, x2)
        y1, y2 = max(0, y1), min(img_h, y2)

        preview = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if len(points) >= 3:
            poly = np.array(points, dtype=np.int32)
            overlay = preview.copy()
            cv2.fillPoly(overlay, [poly],
                         (0, 180, 0) if mode == 'white' else (0, 0, 200))
            preview = cv2.addWeighted(preview, 0.5, overlay, 0.5, 0)

        panel = cv2.resize(preview[y1:y2, x1:x2], (PANEL_W, PANEL_H),
                           interpolation=cv2.INTER_NEAREST)

        color = (100, 200, 100) if mode == 'white' else (100, 100, 255)
        screen_pts = [img_to_screen(*p) for p in points]

        for spt in screen_pts:
            cv2.circle(panel, spt, 5, color, -1)
        if len(screen_pts) > 1:
            for i in range(len(screen_pts) - 1):
                cv2.line(panel, screen_pts[i], screen_pts[i + 1], color, 2)
            cv2.line(panel, screen_pts[-1], screen_pts[0], color, 1)

        cv2.putText(panel,
                    f'Mode: {"ADD (white)" if mode == "white" else "REMOVE (black)"}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(panel, f'Points: {len(points)}   Zoom: {zoom:.1f}x',
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(panel, 'MASK  |  Q=save  ESC=discard',
                    (10, PANEL_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        return panel

    def render_panel_original(img):
        """Render right panel"""
        x1 = int(pan_x)
        y1 = int(pan_y)
        x2 = int(pan_x + img_w / zoom)
        y2 = int(pan_y + img_h / zoom)
        x1, x2 = max(0, x1), min(img_w, x2)
        y1, y2 = max(0, y1), min(img_h, y2)

        panel = cv2.resize(orig_bgr[y1:y2, x1:x2], (PANEL_W, PANEL_H),
                           interpolation=cv2.INTER_LINEAR)

        # Mask with independent polygon
        mask_current = img.copy()
        if len(points) >= 3:
            poly = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask_current, [poly], 255 if mode == 'white' else 0)

        mask_overlay = cv2.cvtColor(mask_current[y1:y2, x1:x2], cv2.COLOR_GRAY2BGR)
        mask_overlay = cv2.resize(mask_overlay, (PANEL_W, PANEL_H),
                                  interpolation=cv2.INTER_NEAREST)
        panel = cv2.addWeighted(panel, 1 - overlay_alpha, mask_overlay, overlay_alpha, 0)

        color = (100, 200, 100) if mode == 'white' else (100, 100, 255)
        screen_pts = [img_to_screen(*p) for p in points]

        for spt in screen_pts:
            cv2.circle(panel, spt, 5, color, -1)
        if len(screen_pts) > 1:
            for i in range(len(screen_pts) - 1):
                cv2.line(panel, screen_pts[i], screen_pts[i + 1], color, 2)
            cv2.line(panel, screen_pts[-1], screen_pts[0], color, 1)

        cv2.putText(panel, f'Mask overlay: {int(overlay_alpha * 100)}%',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(panel, 'ORIGINAL',
                    (10, PANEL_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        return panel

    def draw_state(img):
        left = render_panel_mask(img)
        if not show_original:
            return left
        right = render_panel_original(img)
        separator = np.full((PANEL_H, 4, 3), 80, dtype=np.uint8)
        return np.hstack([left, separator, right])

    def mouse_callback(event, x, y, flags, param):
        nonlocal is_panning, pan_start, pan_origin, pan_x, pan_y
        nx = x
        if show_original and x > PANEL_W + 4:
            nx = x - PANEL_W - 4

        if event == cv2.EVENT_LBUTTONDOWN:
            img_pt = screen_to_img(nx, y)
            points.append(img_pt)
            cv2.imshow(window, draw_state(edited))

        elif event == cv2.EVENT_RBUTTONDOWN:
            is_panning = True
            pan_start = (x, y)
            pan_origin = (pan_x, pan_y)

        elif event == cv2.EVENT_MOUSEMOVE and is_panning:
            
            dx = (x - pan_start[0]) / PANEL_W * (img_w / zoom)
            dy = (y - pan_start[1]) / PANEL_H * (img_h / zoom)
            pan_x = pan_origin[0] - dx
            pan_y = pan_origin[1] - dy
            clamp_pan()
            cv2.imshow(window, draw_state(edited))

        elif event == cv2.EVENT_RBUTTONUP:
            is_panning = False

    win_w = PANEL_W * 2 + 4 if show_original else PANEL_W
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, mouse_callback)

    result = mask.copy()

    while True:
        cv2.imshow(window, draw_state(edited))
        key = cv2.waitKey(20) & 0xFF

        if key == ord('w'):
            mode = 'white'

        elif key == ord('b'):
            mode = 'black'

        elif key == ord('c'):
            points.clear()

        elif key == ord('z'):
            if len(history) > 1:
                history.pop()
                edited = history[-1].copy()
            points.clear()

        elif key == ord('t') and orig_bgr is not None:
            overlay_alpha = round((overlay_alpha + 0.1) % 1.1, 1)
            if overlay_alpha > 1.0:
                overlay_alpha = 0.1

        elif key in (ord('+'), ord('=')):
            zoom = min(zoom * 1.3, 10.0)
            clamp_pan()

        elif key in (ord('-'), ord('_')):
            zoom = max(zoom / 1.3, 1.0)
            if zoom == 1.0:
                pan_x, pan_y = 0.0, 0.0

        elif key == 13 and len(points) >= 3:  # Enter
            history.append(edited.copy())
            poly = np.array(points, dtype=np.int32)
            cv2.fillPoly(edited, [poly], 255 if mode == 'white' else 0)
            points.clear()

        elif key == ord('q'): 
            result = edited.copy()
            break

        elif key == 27: # ESC
            result = mask.copy()
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1) 
    return result