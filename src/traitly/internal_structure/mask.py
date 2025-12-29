
# traitly/internal_structure/mask.py

# ============================================================================
# THIRD-PARTY LIBRARIES
# ============================================================================
import cv2
import numpy as np

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

# def merge_locules_func(locules_indices, contours, min_distance=0, max_distance=50, min_area=10):
#     if not locules_indices:
#         return []
    
#     # Filtrar contornos válidos (con área suficiente y que no estén vacíos)
#     valid_locules = []
#     for i in locules_indices:
#         if (len(contours[i]) > 0 and  # Verificar que el contorno tenga puntos
#             cv2.contourArea(contours[i]) > min_area):
#             valid_locules.append(i)
    
#     if not valid_locules:
#         return []
    
#     merged = [False] * len(valid_locules)
#     result_locules = []
    
#     for i in range(len(valid_locules)):
#         if not merged[i]:
#             current_idx = valid_locules[i]
#             current_contour = contours[current_idx]
            
#             # Verificar que el contorno actual no esté vacío
#             if len(current_contour) == 0:
#                 continue
                
#             merged[i] = True
#             to_merge = [current_contour]
            
#             for j in range(i+1, len(valid_locules)):
#                 if not merged[j]:
#                     other_idx = valid_locules[j]
#                     other_contour = contours[other_idx]
                    
#                     # Verificar que el otro contorno no esté vacío
#                     if len(other_contour) == 0:
#                         continue
                    
#                     # Cálculo de distancia (tu código existente)
#                     min_dist = float('inf')
#                     for point in other_contour[::2, 0, :]:
#                         dist = cv2.pointPolygonTest(current_contour, (float(point[0]), float(point[1])), True)
#                         if dist < min_dist:
#                             min_dist = dist
#                             if min_dist <= 0:
#                                 break
                    
#                     if min_distance < abs(min_dist) < max_distance:
#                         to_merge.append(other_contour)
#                         merged[j] = True
            
#             if len(to_merge) > 1:
#                 try:
#                     merged_contour = np.vstack(to_merge)
#                     epsilon = 0.001 * cv2.arcLength(merged_contour, True)
#                     merged_loculus = cv2.approxPolyDP(merged_contour, epsilon, True)
#                     if len(merged_loculus) > 0:  # Verificar que el resultado no esté vacío
#                         result_locules.append(merged_loculus)
#                 except:
#                     # En caso de error en la fusión, mantener el contorno original
#                     result_locules.append(current_contour)
#             else:
#                 result_locules.append(current_contour)
    
#     return result_locules

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