
# traitly/internal_structure/mask.py


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
from ..utils.common_functions import is_contour_valid, plot_img

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
    plot: bool = True,  # ← Cambio aquí
    plot_size: Tuple[int,int] = (5,5)
) -> np.ndarray:  # ← Cambio aquí
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
        ### Validation ###################################################################

        # VALIDA INPUT: 
        if not isinstance(img_hsv, np.ndarray):
            raise TypeError("Input image must be a numpy array")
            
        if img_hsv.ndim != 3 or img_hsv.shape[2] != 3:
            raise ValueError("Image must be in HSV format (3 channels)")
        
        if img_hsv.dtype != np.uint8:
            raise ValueError("HSV image must be uint8 type (0-180 for H, 0-255 for S/V)")
        
        # VALIDATE KERNELS (open/close/blur): 
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
        
        # VALIDATE CANNY PARAMETERS:
        if (canny_min is None) != (canny_max is None):
            raise ValueError("Both canny_min and canny_max must be provided together or both None")
        
        if canny_min is not None and canny_max is not None:
            if not isinstance(canny_min, int) or not isinstance(canny_max, int):
                raise ValueError("canny_min and canny_max must be integers")
            if canny_min >= canny_max:
                raise ValueError("canny_min must be < canny_max")
        
        #####################################################################################        

        # Set default HSV values for black/dark backgrounds if not provided
        if lower_hsv is None:
            lower_hsv = np.array([0, 0, 0], dtype=np.uint8)
        elif isinstance(lower_hsv, (list, tuple)):  # ← Cambio aquí
            lower_hsv = np.array(lower_hsv, dtype=np.uint8)
            
        if upper_hsv is None:
            upper_hsv = np.array([180, 250, 50], dtype=np.uint8)
        elif isinstance(upper_hsv, (list, tuple)):  # ← Cambio aquí
            upper_hsv = np.array(upper_hsv, dtype=np.uint8)

        # VALIDATE HSV THRESH VALUES
        if not isinstance(lower_hsv, np.ndarray) or lower_hsv.shape != (3,):
            raise ValueError("lower_hsv must be a numpy array with shape (3,)")
        if not isinstance(upper_hsv, np.ndarray) or upper_hsv.shape != (3,):
            raise ValueError("upper_hsv must be a numpy array with shape (3,)")
            
        if (lower_hsv > upper_hsv).any():
            raise ValueError("All values in lower_hsv must be <= corresponding values in upper_hsv")

        ### Processing image #################################################################
        
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

        if plot:
            plot_img(final_mask, 
                     fig_axis=False, 
                     plot_size=plot_size, 
                     metadata=False, 
                     gray=True)

        return final_mask
        
    except cv2.error as e:
        raise RuntimeError(f"OpenCV error: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {str(e)}")


#################################################################################################
# Detect fruit contours in a binary mask
#################################################################################################

def find_fruits(
    binary_mask: np.ndarray,
    min_locule_area: int = 50,
    min_locules_per_fruit: int = 1,
    min_circularity: float = 0.4,
    max_circularity: float = 1.0,
    min_aspect_ratio: float = 0.3,
    max_aspect_ratio: float = 3.0,
    rescale_factor: Optional[float] = None
) -> Tuple[List[np.ndarray], Dict[int, List[int]]]:
    """                 
    Detects fruit contours in a binary mask using morphological filtering and maps 
    fruits to their internal cavities (locules).
    
    This function identifies fruit structures by detecting outer contours that contain 
    smaller internal contours (locules/seed cavities). Applies multiple morphological 
    filters to distinguish valid fruits from noise.

    Pipeline:
        1. Optional rescaling for speed (processes smaller image)
        2. Contour detection with hierarchy
        3. Vectorized computation of morphological features
        4. Vectorized filtering of valid fruits
        5. Fruit-locule mapping
        6. Rescale contours back to original coordinates

    Arguments:
        binary_mask: Binary image (uint8) where white (255) represents fruits, 
            black (0) background. Must be a 2D array.
        min_locule_area: Minimum area in square pixels (px²) for a valid locule.
            **Specified in original image coordinates.** Automatically scaled 
            when rescale_factor is used.
        min_locules_per_fruit: Minimum number of valid locules required to classify 
            as fruit. Set to 0 to detect fruits without visible locules.
        min_circularity: Minimum circularity (0-1, where 1.0 = perfect circle).
            Formula: 4π*Area / Perimeter²
        max_circularity: Maximum circularity threshold.
        min_aspect_ratio: Minimum width/height ratio for valid contours.
            Uses rotation-invariant minAreaRect (not affected by object orientation).
        max_aspect_ratio: Maximum width/height ratio for valid contours.
        rescale_factor: Image scaling factor (0.0, 1.0] for faster processing.
            Example: 0.5 = 50% size (~4× faster).
            **Note:** All processing is done on scaled image, then contours are 
            rescaled back to original coordinates.
    
    Returns:
        contours: List of all detected contours in **original image coordinates**.
            Each contour is (N, 1, 2) array.
        fruit_locules_map: Maps fruit indices to their locule indices.
            Example: {0: [1, 2], 5: [6]} means contour 0 contains locules 1 and 2.

    Raises:
        ValueError: Invalid input parameters
        cv2.error: OpenCV contour detection failure
    """

    # INPUT VALIDATION
    if not isinstance(binary_mask, np.ndarray) or binary_mask.dtype != np.uint8:
        raise ValueError("binary_mask must be uint8 numpy array")
    
    if len(binary_mask.shape) != 2:
        raise ValueError("binary_mask must be 2D array")
    
    if rescale_factor is not None and not (0 < rescale_factor <= 1):
        raise ValueError('rescale_factor must be in range (0, 1]')
    
    if min_locule_area <= 0 or min_locules_per_fruit < 0:
        raise ValueError("Area and locule count must be positive")
    
    if not (0 <= min_circularity <= max_circularity <= 1):
        raise ValueError("Circularity: 0 ≤ min ≤ max ≤ 1")
    
    if not (0 < min_aspect_ratio <= max_aspect_ratio):
        raise ValueError("Aspect ratio: 0 < min ≤ max")

    # PREPROCESSING: RESCALE IMAGE (if requested)
    should_rescale = rescale_factor is not None and rescale_factor < 1
    
    if should_rescale:
        original_h, original_w = binary_mask.shape
        new_w = int(original_w * rescale_factor)
        new_h = int(original_h * rescale_factor)
        resized_mask = cv2.resize(binary_mask, (new_w, new_h), 
                                  interpolation=cv2.INTER_NEAREST)
        
        # Area scales with square of linear scale factor (2D geometry)
        adjusted_min_area = int(min_locule_area * (rescale_factor ** 2))
        
        # Pre-compute scale factors for final coordinate transformation
        scale_x = original_w / new_w
        scale_y = original_h / new_h
    else:
        resized_mask = binary_mask
        adjusted_min_area = min_locule_area

    # CONTOUR DETECTION WITH HIERARCHY
    contours, hierarchy = cv2.findContours(
        resized_mask,
        cv2.RETR_TREE,  # Get full tree hierarchy (parent-child relationships)
        cv2.CHAIN_APPROX_SIMPLE 
    )
    
    if not contours or hierarchy is None:
        return [], {}

    hierarchy = hierarchy[0]  # (1, N, 4) to (N, 4)
    
    # Normize contours for downstream processing
    normalized_contours = [
        cnt.reshape(-1, 2).astype(np.float32) for cnt in contours
    ]
    
    
    # COMPUTE BASIC GEOMETRICS
    # Area and perimeter
    areas = np.array([cv2.contourArea(c) for c in contours])
    perimeters = np.array([cv2.arcLength(c, True) for c in contours])
    
    # Circularity
    with np.errstate(divide='ignore', invalid='ignore'):
        circularities = (4 * np.pi * areas) / (perimeters ** 2)
        circularities = np.nan_to_num(circularities, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Aspect ratios using rotated bounding boxes
    min_area_rects = [cv2.minAreaRect(c) for c in contours]
    aspect_ratios = np.array([
        max(w, h) / min(w, h) if min(w, h) > 0 else 0.0
        for _, (w, h), _ in min_area_rects
    ])

    # IDENTIFY VALID FRUITS
    # Potential fruits: no parent (only top level contours)
    is_top_level = hierarchy[:, 3] == -1
    
    # Apply morphological filters
    passes_area = areas >= adjusted_min_area #Filter by area
    passes_circularity = (circularities >= min_circularity) & (circularities <= max_circularity) # Filter by circularity
    passes_aspect = (aspect_ratios >= min_aspect_ratio) & (aspect_ratios <= max_aspect_ratio) # Filter by aspect ratio
    
    # Combine all filters (vectorized boolean operations)
    valid_fruits_mask = is_top_level & passes_area & passes_circularity & passes_aspect # Compare ALL the filters
    
    # Get indices of valid fruits
    valid_fruit_indices = np.where(valid_fruits_mask)[0]
    
    # FRUIT-LOCULE MAPPING
    fruit_locules_map = {}
    
    for fruit_idx in valid_fruit_indices:
        # Find all child contours (locules)
        # NOTE: A contour is a child if hierarchy[i, 3] == fruit_idx
        is_child_of_fruit = hierarchy[:, 3] == fruit_idx
        
        # Filter locules by minimum area
        valid_locules_mask = is_child_of_fruit & (areas >= adjusted_min_area) # Using previously calculated area
        locule_indices = np.where(valid_locules_mask)[0].tolist()
        
        # Only keep fruit if it has minimum required locules (default = 1)
        if len(locule_indices) >= min_locules_per_fruit:
            fruit_locules_map[int(fruit_idx)] = locule_indices
    
    # RESCALE CONTOURS BACK TO ORIGINAL COORDINATES
    if should_rescale:
        scale_factors = np.array([[scale_x, scale_y]], dtype=np.float32)
        
        contours = [
            (c.astype(np.float32) * scale_factors).astype(np.int32)
            for c in contours
        ]
    
    return contours, fruit_locules_map


#################################################################################################
# Merge close locules
#################################################################################################

def merge_locules_func(locules_indices, contours, min_distance=0, max_distance=50, min_area=10):
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
    
    # Step 2: Build centroid distance matrix (vectorized)
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
    
    # Step 3: Merge locules based on distance thresholds
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
            
            # Pre-filter candidates using centroid distances
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
    L_corrected = np.power(L_norm, gamma) # L_norm ** gamma, so no linear transformation

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
    contours, hierarchy = cv2.findContours(inv_locule_mask, cv2.RETR_CCOMP, 
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
    
    # Invert locules mask if requested
    if invert_locules:
        # Invert only within fruits (keep background black)
        locule_mask_clean = cv2.bitwise_and(
            cv2.bitwise_not(locule_mask_clean),
            fruits_only_mask
        )
    else:
        locule_mask_clean = cv2.bitwise_and(locule_mask, fruits_only_mask)
    
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