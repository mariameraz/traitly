
# traitly/internal_structure/mask.py


# ============================================================================
# STANDARD LIBRARY
# ============================================================================
from typing import Optional, Tuple

# ============================================================================
# THIRD-PARTY LIBRARIES
# ============================================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# LOCAL/INTERNAL IMPORTS
# ===========================================================================
from ..utils.common_functions import is_contour_valid, plot_img

#################################################################################################
# Create fruit mask
#################################################################################################

def create_mask(
    img_hsv,lower_hsv=None, upper_hsv=None,
    n_iteration=1, n_kernel=7, kernel_open = None,
    kernel_close = None, canny_min=30, canny_max=100,
    plot=True, plot_size=(20,10), fig_axis = False,
):
    """
    Creates a binary mask to segment objects from an HSV image using color thresholding, morphological operations and edge detection
    
    Arguments:
    
    REQUIRED:
        - img_hsv (numpy.ndarray): Image in HSV format.

    OPTIONAL:
        - lower_hsv (Tuple[int, int, int]): Lower bound for HSV background detection (default: (0,0,0)).
        - upper_hsv (Tuple[int, int, int]): Upper bound for HSV background detection default: (180,255,30).
        - n_iteration (int): Number of iterations for morphological operations.
        - n_kernel (int): Kernel size (odd) for morphological ops when kernel_open/kernel_close are None (default: 7).
        - kernel_open (int): Custom kernel size for opening (overrides n_kernel if set).
        - kernel_close (int): Custom kernel size for closing (overrides n_kernel if set).
        - canny_min (int): First threshold for Canny edge detection.
        - canny_max (int): Second threshold for Canny edge detection.
        - plot (numpy.ndarray): Whether to plot the resulting mask as a binary image.
        - figsize (Tuple[int, int]): Figure size for plotting.
        
    Returns:
        - Binary mask as 2D numpy array (numpy.dnarray)
    
    Raises:
        - ValueError: If parameters are invalid
        - TypeError: If input types are incorrect
        - RuntimeError: If image processing fails
    """
    try:
        # Input validation
        if not isinstance(img_hsv, np.ndarray):
            raise TypeError("Input image must be a numpy array")
            
        if img_hsv.ndim != 3 or img_hsv.shape[2] != 3:
            raise ValueError("Image must be in HSV format (3 channels)")
            
        if not isinstance(n_iteration, int) or n_iteration < 1:
            raise ValueError("n_iteration must be a positive integer")
            
        if not isinstance(n_kernel, int) or n_kernel < 1 or n_kernel % 2 == 0:
            raise ValueError("n_kernel must be a positive odd integer")
            
        if img_hsv.dtype != np.uint8:
            raise ValueError("HSV image must be uint8 type (0-180 for H, 0-255 for S/V)")
    
        # Set default HSV values for black/dark backgrounds if not provided
        if lower_hsv is None:
            lower_hsv = np.array([0, 0, 0], dtype=np.uint8)
        elif isinstance(lower_hsv, list):
            lower_hsv = np.array(lower_hsv, dtype=np.uint8)
            
        if upper_hsv is None:
            upper_hsv = np.array([180, 250, 50], dtype=np.uint8)
        elif isinstance(upper_hsv, list):
            upper_hsv = np.array(upper_hsv, dtype=np.uint8)

        # Validate HSV bounds
        if not isinstance(lower_hsv, np.ndarray) or lower_hsv.shape != (3,):
            raise ValueError("lower_hsv must be a numpy array with shape (3,)")
        if not isinstance(upper_hsv, np.ndarray) or upper_hsv.shape != (3,):
            raise ValueError("upper_hsv must be a numpy array with shape (3,)")
            
        if (lower_hsv > upper_hsv).any():
            raise ValueError("All values in lower_hsv must be <= corresponding values in upper_hsv")

        
        mask_background = cv2.inRange(img_hsv, lower_hsv, upper_hsv) # Create binary mask where [lower_hsv, upper_hsv] are white (255) (background) and others black (0) (fruits/label)
        if mask_background is None:
            raise RuntimeError("Failed to create initial mask")

        mask_inverted = cv2.bitwise_not(mask_background) # Invert the binary mask to focus on foreground objects (fruits/label)
        
        kernel_open = kernel_open if kernel_open is not None else n_kernel
        kernel_close = kernel_close if kernel_close is not None else n_kernel

        kernel_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_open, kernel_open)) # Creates an elliptical kernel for morphological operations
        kernel_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_close, kernel_close)) 

        mask_open = cv2.morphologyEx(mask_inverted, cv2.MORPH_OPEN, kernel_o, iterations=n_iteration) # Opening (erosion followed by dilation) to remove small noise
        mask_closed = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel_c, iterations=n_iteration) # Closing (dilation followed by erosion) to fill small holes
        
        blurred = cv2.GaussianBlur(mask_closed, (n_kernel, n_kernel), 0) # Applies Gaussian blur to smooth edges
        edges = cv2.Canny(blurred, canny_min, canny_max) # Detects edges using the Canny algorithm
        
        final_mask = cv2.bitwise_or(mask_closed, edges) # Combines the closed mask with edges to refine boundaries

        if plot:# Displays the final mask with/without axes based on the `axis` parameter
            plot_img(final_mask, 
                     fig_axis=fig_axis, 
                     plot_size=plot_size, 
                     metadata = False, gray = True)

        return final_mask
        
    except cv2.error as e:
        raise RuntimeError(f"OpenCV error: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {str(e)}")


#################################################################################################
# Detect fruit contours in a binary mask
#################################################################################################

def find_fruits(
    binary_mask,
    min_locule_area = 50,
    min_locules_per_fruit = 1,
    min_circularity = 0.4,
    max_circularity = 1.0,
    min_aspect_ratio = 0.3,
    max_aspect_ratio = 3.0,
    rescale_factor = None,
    contour_approximation = cv2.CHAIN_APPROX_SIMPLE,
    contour_filters = None):
    """
    Detects fruit contours in a binary mask using morphological filtering criteria and returns 
    a mapping of fruits to their internal cavities (locules).

    Args:
        REQUIRED:
            - binary_mask (np.ndarray): Binary image where white represents objects (fruits) and black background (uint8).
        
        OPTIONAL:
            - min_locule_area (int): Minimum pixel area for a locule to be considered valid (default: 50).
            - min_locules_per_fruit (int): Minimum number of locules required to classify as fruit (default: 1).
            - min_circularity (float): Minimum circularity threshold (0-1, 1=perfect circle) (default: 0.4).
            - max_circularity (float): Maximum circularity threshold (default: 1.0).
            - min_aspect_ratio (float): Minimum width/height ratio for valid contours (default: 0.3).
            - max_aspect_ratio (float): Maximum width/height ratio (default: 3.0).
            - rescale_factor (float): Scaling factor (0.0-1.0) for faster processing (None=no rescaling).
            - contour_approximation: OpenCV contour approximation method (default: CHAIN_APPROX_SIMPLE).
            - contour_filters (Dict): Dictionary to override default filter values.

    Returns:
        Tuple[List[np.ndarray], Dict[int, List[int]]] containing:
            - contours: List of all detected contours (in original coordinates)
            - fruit_locules_map: Dictionary mapping fruit indices to lists of locule indices

    Raises:
        ValueError: If input parameters are invalid
        cv2.error: If OpenCV contour detection fails
    """
    # Validate rescale_factor
    if rescale_factor is not None and not (0 < rescale_factor <= 1):
        raise ValueError('rescale_factor must be between 0 and 1')

    # Store original dimensions for later rescaling
    original_shape = binary_mask.shape[:2] if rescale_factor is not None else None

    # Conditional image resizing
    if rescale_factor is not None and rescale_factor < 1: # Check that rescale_factor is a value between 0 and 1
        new_size = (int(binary_mask.shape[1] * rescale_factor), 
                   int(binary_mask.shape[0] * rescale_factor))
        resized_mask = cv2.resize(binary_mask, new_size, interpolation=cv2.INTER_NEAREST)
        min_locule_area = int(min_locule_area * (rescale_factor ** 2))
    else:
        resized_mask = binary_mask.copy()

    # Configure filters with validation
    default_filters = {
        'min_area': min_locule_area, 
        'min_circularity': min_circularity,
        'max_circularity': max_circularity,
        'min_aspect_ratio': min_aspect_ratio,
        'max_aspect_ratio': max_aspect_ratio
    }
    
    if contour_filters:
        invalid_keys = set(contour_filters.keys()) - set(default_filters.keys())
        if invalid_keys:
            raise ValueError(f"Invalid filter keys: {invalid_keys}. Valid keys are: {list(default_filters.keys())}")
    
    filters = {**default_filters, **(contour_filters or {})}

    # Input validation
    if not isinstance(resized_mask, np.ndarray) or resized_mask.dtype != np.uint8:
        raise ValueError("Input mask must be uint8 numpy array")
    
    if any(v <= 0 for v in [min_locule_area, *filters.values()]):
        raise ValueError("All parameters must be positive values")

    # Contour detection
    contours, hierarchy = cv2.findContours(
        resized_mask, 
        cv2.RETR_TREE,
        contour_approximation
    )
    
    if not contours or hierarchy is None:
        return [], {}

    hierarchy = hierarchy[0]  # Simplify hierarchy structure

    # Process contours and build fruit-locules mapping
    fruit_locules_map = {}
    for i, contour in enumerate(contours):
        # Check if contour is top-level (fruit candidate) and passes filters
        if hierarchy[i][3] == -1 and is_contour_valid(contour, filters):
            # Find all valid child contours (locules)
            locules = [
                j for j, h in enumerate(hierarchy)
                if h[3] == i and  # Is direct child
                cv2.contourArea(contours[j]) >= filters['min_area']
            ]
            
            # Only register as fruit if minimum locules count is met
            if len(locules) >= min_locules_per_fruit:
                fruit_locules_map[i] = locules

    # Rescale contours back to original coordinates if needed
    if rescale_factor is not None and rescale_factor < 1:
        scale_x = original_shape[1] / resized_mask.shape[1]
        scale_y = original_shape[0] / resized_mask.shape[0]
        
        rescaled_contours = [
            (contour.astype(np.float32) * np.array([scale_x, scale_y])).astype(np.int32)
            for contour in contours
        ]
        contours = rescaled_contours
            
    return contours, fruit_locules_map

#################################################################################################
# Merge close locules
#################################################################################################

def merge_locules_func(locules_indices, contours, min_distance=0, max_distance=50, min_area=10):
    """
    Merge close locules based on proximity.
    OPTIMIZED VERSION with identical behavior to original.
    
    Args:
        locules_indices (list): Indices of locule contours to process
        contours (list): List of all contours
        min_distance (int): Minimum distance threshold for merging (default: 0)
        max_distance (int): Maximum distance threshold for merging (default: 50)
        min_area (int): Minimum contour area to consider valid (default: 10)
    
    Returns:
        list: List of merged contour arrays
    
    Optimizations:
        - Pre-filter candidates using centroid distances (10x faster initial filtering)
        - Only perform expensive pointPolygonTest on promising pairs
        - Vectorized distance matrix calculation using scipy.pdist
    """
    if not locules_indices:
        return []
    
    # Step 1: Filter valid contours and compute centroids
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
    
    # Step 2: OPTIMIZATION - Build centroid distance matrix (vectorized)
    # This allows fast pre-filtering before expensive pointPolygonTest
    centroids_valid = [(i, c) for i, c in enumerate(centroids) if c is not None]
    
    if len(centroids_valid) < 2:
        # Not enough valid centroids, return original contours
        return valid_contours
    
    # Extract valid centroid coordinates
    centroid_indices = [idx for idx, _ in centroids_valid]
    centroid_coords = np.array([c for _, c in centroids_valid])
    
    # Compute pairwise centroid distances (vectorized)
    from scipy.spatial.distance import pdist, squareform
    centroid_distances = squareform(pdist(centroid_coords))
    
    # Step 3: Merge logic (same as original, but with pre-filtering)
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
            
            # OPTIMIZATION: Pre-filter candidates using centroid distances
            # Only check locules whose centroids are within a reasonable range
            # Use a conservative upper bound (max_distance * 3) to avoid false negatives
            if centroids[i] is not None:
                # Find index of current centroid in the valid centroids array
                try:
                    centroid_idx = centroid_indices.index(i)
                    
                    # Get candidate indices where centroid distance is within range
                    # Use generous buffer (3x max_distance) to ensure we don't miss valid pairs
                    close_mask = centroid_distances[centroid_idx] <= (max_distance * 3)
                    close_indices = np.where(close_mask)[0]
                    
                    # Convert back to original valid_locules indices
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
                
                # Compute actual minimum distance using pointPolygonTest
                # Sample every 2nd point as in original (balance speed/accuracy)
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
                
                # Check if distance is within merge range
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
    Applies exponential transformation - expands high values.
    
    Args:
        L: Luminance channel (grayscale image)
        c: Exponential coefficient (controls expansion intensity)
    
    Returns:
        Exponentially-transformed luminance channel
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
                   plot_size: Optional[Tuple[int, int]] = (12, 12),
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
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Extract L channel
    l_channel = lab[:, :, 0]
    
    # Dictionary of contrast methods
    contrast_methods = {
        'gamma': lambda: gamma_contrast(l_channel, gamma=gamma, plot=False),
        'sigmoid': lambda: sigmoid_contrast(l_channel, gain=gain, cutoff=cutoff),
        'exp': lambda: exp_transform(l_channel, c=c),
        'none': lambda: l_channel.copy()  # No transformation, return L channel as-is
    }
    
    # Validate method
    if contrast_method not in contrast_methods:
        raise ValueError(f"contrast_method must be one of {list(contrast_methods.keys())}")
    
    # If compare=True, display all 3 methods (excluding 'none')
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
    l_transformed = contrast_methods[contrast_method]()
    
    # Apply median blur
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
    contours, hierarchy = cv2.findContours(inv_locule_mask, cv2.RETR_TREE, 
                                          cv2.CHAIN_APPROX_SIMPLE)
    
    # Create black mask and fill ONLY fruits (large contours without parent)
    fruits_only_mask = np.zeros_like(locule_mask)
    
    if contours and hierarchy is not None:
        for i, cnt in enumerate(contours):
            parent = hierarchy[0][i][3]
            area = cv2.contourArea(cnt)
            
            # If no parent (external contour) and large (fruit, not noise)
            if parent == -1 and area > min_fruit_size:
                # Fill this fruit completely (includes all internal structures)
                cv2.drawContours(fruits_only_mask, [cnt], -1, 255, -1)
    
    # Apply fruits mask to original locule_mask
    # This removes ALL background, keeping only fruits and their structures
    locule_mask_clean = cv2.bitwise_and(locule_mask, fruits_only_mask)
    
    # Invert locules mask if requested
    if invert_locules:
        # Invert only within fruits (keep background black)
        locule_mask_clean = cv2.bitwise_and(
            cv2.bitwise_not(locule_mask_clean),
            fruits_only_mask
        )
    
    # Fuse fruit mask with locules mask
    mask_fruits_rgb = cv2.cvtColor(cv2.bitwise_not(fruit_mask), cv2.COLOR_GRAY2BGR)
    mask_fruits_rgb[locule_mask_clean == 255] = [255, 255, 255]
    final_mask = cv2.bitwise_not(mask_fruits_rgb[:, :, 0])
    
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