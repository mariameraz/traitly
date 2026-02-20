# traitly/fruit_phenotyping/mask.py

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

# ============================================================================
# INTERNAL IMPORTS
# ===========================================================================
from ..utils.basic_functions import plot_img

#################################################################################################
# Create fruit mask
#################################################################################################

def create_mask(
    img_hsv: np.ndarray,
    lower_hsv: Optional[Tuple[int,int,int]] = None, 
    upper_hsv: Optional[Tuple[int,int,int]] = None,
    n_iteration: int = 1,
    kernel_blur: Optional[int] = None, 
    kernel_open: Optional[int] = None,
    kernel_close: Optional[int] = None, 
    canny_min: Optional[int] = None,
    canny_max: Optional[int] = None,
    plot: bool = True, 
    plot_size: Tuple[int,int] = (5,5),
    background_color: Optional[str] = None,
    fill_holes: bool = False,
    apply_convex_hull: bool = False
) -> np.ndarray: 
    """
    Creates a binary mask to segment objects from an HSV image using color 
    thresholding, morphological operations and edge detection.
    
    PIPELINE:
        1. HSV color thresholding
        2. Invert mask (background → foreground)
        3. Opening (optional) - removes small noise
        4. Closing (optional) - fills small holes
        5. Gaussian blur (optional) - smooths edges
        6. Canny edge detection (optional)
        7. Combine mask with edges
    
    Arguments:
        img_hsv: Image in HSV format (H: 0-180, S: 0-255, V: 0-255)
        lower_hsv: Lower HSV bound for background (default: [0,0,0])
        upper_hsv: Upper HSV bound for background (default: [180,250,50])
        n_iteration: Iterations for morphological operations (default: 1)
        kernel_blur: Kernel size for Gaussian blur (must be odd). If None, skipped.
        kernel_open: Kernel size for opening (must be odd). If None, skipped.
        kernel_close: Kernel size for closing (must be odd). If None, skipped.
        canny_min: Lower threshold for Canny. Requires canny_max. If None, skipped.
        canny_max: Upper threshold for Canny. Requires canny_min. If None, skipped.
        plot: Whether to display the resulting mask                                                   
        plot_size: Figure size for plotting (width, height)
        
    Returns:
        Binary mask as 2D numpy array (uint8)
    
    Raises:
        TypeError: If input types are incorrect
        ValueError: If parameters are invalid
        RuntimeError: If image processing fails
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
        background_color_list = {
            'blue', 'white', 'black'
        }
        if background_color is not None:
            if background_color == 'blue':
                lower_hsv = np.array([90, 100, 80], dtype=np.uint8)
                upper_hsv = np.array([130, 255, 255], dtype=np.uint8)
            elif background_color == 'white':
                lower_hsv = np.array([0, 0, 85], dtype=np.uint8)   
                upper_hsv = np.array([180, 66, 255], dtype=np.uint8)
            elif background_color == 'black':
                lower_hsv = np.array([0, 0, 0], dtype=np.uint8)
                upper_hsv = np.array([180, 250, 50], dtype=np.uint8)
            else:
                raise ValueError(f"Invalid background_color: {background_color}. Use: {background_color_list}")

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
def apply_convex_hull_to_mask(mask: np.ndarray, min_area: int = 50, contours: Optional[Dict] = None) -> np.ndarray:
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
    rescale_factor: Optional[float] = None
) -> Tuple[List[np.ndarray], Dict[int, List[int]]]:
    """                 
    Detects fruit contours in a binary mask using morphological filtering and maps 
    fruits to their internal cavities (locules).
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

def merge_locules_func(locules_indices, 
                       contours, 
                       min_distance=0, 
                       max_distance=50, 
                       min_area=10):
    """
    Merge close locules based on proximity.
    
    Args:
        locules_indices (list): Indices of locule contours to process
        contours (list): List of all contours
        min_distance (int): Minimum distance threshold for merging (default: 0)
        max_distance (int): Maximum distance threshold for merging (default: 50)
        min_area (int): Minimum contour area to consider valid (default: 10)
    
    Returns:
        list: List of merged contour arrays
    
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
    """Ensures the image is in uint8 format."""
    if L.dtype != np.uint8:
        # If in float [0, 1], scale to [0, 255]
        if L.max() <= 1.0:
            L = (L * 255).astype(np.uint8)
        else:
            # If in another range, clip and convert
            L = np.clip(L, 0, 255).astype(np.uint8)
    return L

def gamma_contrast(L: np.ndarray, 
                   gamma: float = 1.0, 
                   plot: bool = False) -> np.ndarray:
    """
    Applies gamma correction to enhance contrast.
    
    Args:
        L: Luminance channel (grayscale image)
        gamma: Gamma value for correction
            - gamma < 1: Brightens shadows (expands low values)
            - gamma > 1: Darkens highlights (compresses high values)
            - gamma = 1: No change
        plot: If True, displays the result
    
    Returns:
        Gamma-corrected luminance channel
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

def sigmoid_contrast(L: np.ndarray, 
                     gain: float = 10, 
                     cutoff: float = 0.5) -> np.ndarray:
    """
    Applies sigmoidal contrast enhancement.
    
    Args:
        L: Luminance channel (grayscale image)
        gain: Intensity of the contrast (5-20 recommended)
        cutoff: Central point of the sigmoid (0.3-0.7 recommended)
    
    Returns:
        Sigmoid-transformed luminance channel
    """
    L = _ensure_uint8(L)
    L_norm = L / 255.0
    
    # Apply sigmoid transformation
    L_sigmoid = 1 / (1 + np.exp(-gain * (L_norm - cutoff)))
    
    # Renormalize to [0, 1]
    L_sigmoid = (L_sigmoid - L_sigmoid.min()) / (L_sigmoid.max() - L_sigmoid.min())
    
    return (L_sigmoid * 255).astype(np.uint8)

def exp_transform(L: np.ndarray, c: float = 1.0) -> np.ndarray:
    """
    Applies exponential transformation: expands high values.
    
    Args:
        L: Luminance channel (grayscale image)
        c: Exponential coefficient (controls expansion intensity)
    
    Returns:
        Exponentially transformed luminance channel
    """
    L = _ensure_uint8(L)
    L_norm = L / 255.0
    
    # Apply exponential transformation
    L_exp = np.expm1(c * L_norm)
    L_exp = (L_exp / L_exp.max() * 255)
    
    return L_exp.astype(np.uint8)

def apply_contrast(img: np.ndarray, 
                   contrast_method: Optional[str] = 'gamma',
                   gamma: Optional[float] = 1.5,
                   gain: Optional[float] = 5,
                   cutoff: Optional[float] = 0.5,
                   c: Optional[float] = 0.5,
                   plot: Optional[bool] = False,
                   plot_size: Optional[Tuple[int, int]] = (5, 5),
                   compare: Optional[bool] = False,
                   kernel_blur: Optional[int] = 1,
                   clip_limit: Optional[int] = None,
                   tile_grid_size: Optional[int] = 12) -> np.ndarray:
    """
    Applies contrast transformation to the L channel of a LAB image.
    
    Args:
        img: Image in BGR format
        contrast_method: Contrast method to apply ('gamma', 'sigmoid', 'exp', or 'none')
        gamma: Parameter for gamma_contrast (default: 1.5)
        gain: Parameter for sigmoid_contrast (default: 5)
        cutoff: Parameter for sigmoid_contrast (default: 0.5)
        c: Parameter for exp_transform (default: 0.5)
        plot: If True, displays the result of the selected method
        plot_size: Figure size for plotting (default: (12, 12))
        compare: If True, visually compares all 3 methods (overrides plot)
        kernel_blur: Median blur kernel size (default: 1, must be odd)
        clip_limit: CLAHE clip limit (default: None = no CLAHE applied)
        tile_grid_size: CLAHE tile grid size (default: 12)
    
    Returns:
        Transformed L channel (2D numpy array)
    
    Raises:
        TypeError: If input image is not a numpy array
        ValueError: If image format is invalid or contrast_method is unknown
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

def create_mask_locules(l_transformed,
                        fruit_mask,
                        kernel_close=None,
                        thresh_min=100,
                        thresh_max=255,
                        kernel_open=None,
                        min_fruit_size=5000,
                        invert_locules=False,
                        plot=False,
                        plot_size=(15, 5)):
    """
    Creates a fused mask containing fruits with their internal locules.
    
    Args:
        l_transformed: Transformed L channel from LAB color space
        fruit_mask: Binary mask of fruits (from create_mask)
        kernel_close: Kernel size for closing operation (optional, None = no closing)
        thresh_min: Minimum threshold value for binarization (default: 100)
        thresh_max: Maximum threshold value for binarization (default: 255)
        kernel_open: Kernel size for opening operation (optional, None = no opening)
        min_fruit_size: Minimum area to consider a contour as a fruit (default: 5000)
        invert_locules: If True, inverts locules mask before fusion (default: False)
        plot: If True, displays the masks (default: False)
        plot_size: Figure size for plotting (default: (15, 5))
    
    Returns:
        Binary mask with fruits and internal locules fused (numpy array)
    """
    # Validate input
    if not isinstance(l_transformed, np.ndarray):
        raise TypeError("l_transformed must be a numpy array")
    if not isinstance(fruit_mask, np.ndarray):
        raise TypeError("fruit_mask must be a numpy array")
    
    # Apply threshold to get locules
    _, locule_mask = cv2.threshold(l_transformed, thresh_min, thresh_max, 
                                   cv2.THRESH_BINARY_INV)
    
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
    
    # Find all contours
    inv_locule_mask = cv2.bitwise_not(locule_mask)
    contours, hierarchy = cv2.findContours(inv_locule_mask, cv2.RETR_CCOMP, 
                                          cv2.CHAIN_APPROX_SIMPLE)
    
    # Create black mask and fill ONLY fruits (large contours without parent)
    fruits_only_mask = np.zeros_like(locule_mask)
    
    if contours and hierarchy is not None:
        h0 = hierarchy[0]
        for i, cnt in enumerate(contours):
            # If no parent (external contour) and large (fruit, not noise)
            if h0[i][3] == -1 and cv2.contourArea(cnt) > min_fruit_size:
                # Fill this fruit completely (includes all internal structures)
                cv2.drawContours(fruits_only_mask, [cnt], -1, 255, -1)
    
    # Invert locules mask if requested
    if invert_locules:
        # Invert only within fruits (keep background black)
        locule_mask_clean = cv2.bitwise_and(
            cv2.bitwise_not(locule_mask),
            fruits_only_mask
        )
    else:
        locule_mask_clean = cv2.bitwise_and(locule_mask, fruits_only_mask)
    
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

def generate_scatter_plot(img_hsv: np.ndarray = None, 
                          img_rgb: np.ndarray = None, 
                          sample_size: int = 10000,
                          plot_size = (18,5)):
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
    axes[1].tick_params(axis='both', labelsize=tick_fontsize)
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
    axes[1].tick_params(axis='both', labelsize=tick_fontsize)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()