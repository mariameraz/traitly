# traitly/utils/basic_functions.py

# ============================================================================
# STANDARD LIBRARY
# ============================================================================
from typing import Optional, List, Tuple, Union, Dict
import os
from pathlib import Path
from functools import lru_cache

# ===========================================================================
# THIRD-PARTY LIBRARIES
# ===========================================================================
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr
from ultralytics import YOLO


# ============================================================================
# INTERNAL IMPORTS
# ===========================================================================
from .constants import valid_extensions, valid_cv2_extensions, label_positions


##############################################################################
# Load an image
##############################################################################

@lru_cache(maxsize=128)
def _load_img_cached(path):
    """Load image into the system and save cache."""
    path_obj = Path(path)
    if path_obj.suffix.lower() not in valid_extensions:
        raise ValueError(f"Unsupported image format: '{path_obj.suffix.lower()}'")
    
    img = cv2.imread(str(path_obj), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot load image: {path_obj.name}")
    
    return img

def load_img(path, plot=False, plot_size=(20, 10)):
    """Wrapper: load image (cache) + plot it."""
    try:
        img = _load_img_cached(path)
        
        if plot:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=plot_size)
            plt.imshow(img_rgb)
            plt.axis('off')
            plt.show()
        
        return img.copy()
        
    except Exception as e:
        print(f"Error loading: {e}")
        return None

##############################################################################
# Evaluate if a contour is valid using geometric thresholds
##############################################################################

def is_contour_valid(contour, filters=None):
    """Versión optimizada con precomputación."""
    # Filtros por defecto como tupla (inmutable, más rápido)
    default_filters = (
        300,    # min_area
        0.6,    # min_circularity  
        1.0,    # max_circularity
        0.4,    # min_aspect_ratio
        1.0     # max_aspect_ratio
    )
    
    if filters is None:
        min_area, min_circ, max_circ, min_ar, max_ar = default_filters
    else:
        # Usar get() para evitar KeyError
        min_area = filters.get('min_area', 300)
        min_circ = filters.get('min_circularity', 0.6)
        max_circ = filters.get('max_circularity', 1.0)
        min_ar = filters.get('min_aspect_ratio', 0.4)
        max_ar = filters.get('max_aspect_ratio', 1.0)
    
    # Calcular área primero (la más barata)
    area = cv2.contourArea(contour)
    if area < min_area:
        return False
    
    # Calcular perímetro solo si pasa área
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False
    
    # Calcular circularidad
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    if not (min_circ <= circularity <= max_circ):
        return False
    
    # Calcular aspect ratio solo si pasa circularidad
    _, (w, h), _ = cv2.minAreaRect(contour)
    if w == 0 or h == 0:
        return False
    
    aspect_ratio = min(w, h) / max(w, h)
    return min_ar <= aspect_ratio <= max_ar


#####################################
# Load models and obtain their path #
#####################################

def _get_package_model_path(model_name: str) -> str:
    """Get absolute path to model file in package_data"""
    try:
        # Try importlib.resources (Python 3.9+)
        from importlib.resources import files
        model_path = files('traitly').joinpath('package_data', 'models', model_name)
        return str(model_path)
    except (ImportError, AttributeError):
        # Fallback for older Python or development mode
        import traitly
        package_dir = Path(traitly.__file__).parent
        model_path = package_dir / 'package_data' / 'models' / model_name
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at: {model_path}\n"
                f"Please ensure the model files are included in the package."
            )
        
        return str(model_path)

#################################################################################################
# Detect label text with OCR
#################################################################################################

# 1. Helper functios: Cache to avoid loading OCR model multiple times
_READER_CACHE = {} 

def get_easyocr_reader(languages=['en', 'es'], gpu=False, verbose = False):
    """
    Initialize EasyOCR reader with optional GPU support.
    
    Args:
        languages: List of language codes for OCR
        gpu: Whether to use GPU (only works with CUDA, not Apple MPS)
    
    Returns:
        EasyOCR Reader instance
    
    Notes:
        - GPU only works on systems with NVIDIA CUDA
        - Apple Silicon Macs (M1/M2/M3) don't support CUDA
        - Falls back to CPU silently if GPU unavailable
    """
    # Only import them if OCR detection needed
    import sys
    from io import StringIO
    import warnings
    
    # Suppress all output during initialization
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = StringIO()
    
    try:
        import easyocr
        
        # Check if GPU is actually available (only for CUDA)
        if gpu:
            import torch
            if not torch.cuda.is_available():
                # Silently fall back to CPU if CUDA not available
                print("GPU not available")
                gpu = False
            else:
                print('GPU available')
        
        # Suppress warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            reader = easyocr.Reader(languages, quantize=gpu, verbose=True)
    
    finally:
        # Always restore stdout/stderr
        sys.stdout, sys.stderr = old_stdout, old_stderr
    
    return reader

def get_cached_reader(languages=('en', 'es'), gpu=False):
    """
    Return cached EasyOCR reader for given languages and GPU setting.
    
    Args:
        languages: Tuple of language codes
        gpu: Whether to use GPU
    
    Returns:
        Cached EasyOCR Reader instance
    """
    # Include GPU in cache key to avoid conflicts
    key = (tuple(languages), gpu)
    
    if key not in _READER_CACHE:
        _READER_CACHE[key] = get_easyocr_reader(list(languages), gpu=gpu)
    
    return _READER_CACHE[key]

#####################
# Detect label text #
#####################

def detect_label_text(img: np.ndarray, 
                           label_roi: List[Dict], 
                           language: List[str] = ['es', 'en'],
                           blur_label: Tuple[int, int] = (11, 11),
                           verbose: bool = False,
                           gpu: bool = False,
                           batch_size: int = 4) -> Optional[str]:
    """Detect label text using OCR. When multiple labels, only the first ROI is processed."""
    
    if not label_roi:
        return None
    
    # Filter valid ROIs
    valid_rois = []
    for i, box in enumerate(label_roi):
        x, y, w, h = box['x'], box['y'], box['width'], box['height']
        if (y + h <= img.shape[0] and x + w <= img.shape[1] and h > 10 and w > 10): # min roi size
            valid_rois.append((i, x, y, w, h))
    
    if not valid_rois:
        return None
    
    # Start reader with cache 
    reader = get_cached_reader(tuple(language), gpu=gpu)
    
    # Process only the first label 
    first_idx, x, y, w, h = valid_rois[0]
    
    try:
        # Processing only first ROI
        region = img[y:y+h, x:x+w]
        if region.size == 0:
            return None
        
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, blur_label, 0)
        
        results = reader.readtext(blur)
        if results:
            # text = " ".join([r[1] for r in results])
            # text = text.split()[0]
            
            text = results[0][1].split()[0]
            if verbose:
                print(f"Label text found in ROI {first_idx+1}: '{text}'")
            return text
        
        if verbose:
            print(f"No label text found in ROI {first_idx+1}")
        return None
        
    except Exception as e:
        if verbose: 
            print(f"Error processing ROI {first_idx+1}: {e}")
        return None
    
#################################################################################################
# Detect image name
#################################################################################################

def detect_img_name(path_image):
    try:
        if not isinstance(path_image, str):
            raise TypeError('Path input should be of type str')
        
        # Only return the filename (with extension if present)
        filename = os.path.basename(path_image)
        
        return filename if filename else None
    
    except Exception as e:
        print(f"Error: {e}")
        return None
    

#################################################################################################
# Plotting image on screen
#################################################################################################

_PLOT_CACHE = {}

def plot_img(img, fig_axis=False, plot_size=(10, 10), binary =False, 
                 cache_key=None, clear_cache=False):
    """
    Plot images 
    
    """
    if clear_cache:
        _PLOT_CACHE.clear()
        plt.close('all')
        return
    
    if cache_key and cache_key in _PLOT_CACHE:
        fig, ax = _PLOT_CACHE[cache_key]
        ax.clear()
    else:
        fig, ax = plt.subplots(figsize=plot_size, num=cache_key if cache_key else None)
        if cache_key:
            _PLOT_CACHE[cache_key] = (fig, ax)
    
    if binary:
        ax.imshow(img, cmap='gray', interpolation='nearest') 
    else:
        if len(img.shape) == 3 and img.shape[2] == 3:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 
                     interpolation='bilinear')
        else:
            ax.imshow(img, interpolation='bilinear')
    
    if not fig_axis:
        ax.axis('off')
    
    plt.tight_layout(pad=0.1)  
    plt.draw()
    
    if not plt.isinteractive():
        plt.show(block=False)
    
    fig.canvas.flush_events()
    
    return fig, ax

##########################
## Validate a directoty ##
##########################

def validate_dir(path):
    """
    Ensure the directory exists and return the absolute path.
    
    Args:
        path (str): File path to check
        
    Returns:
        str: Absolute path with ensured directory existence
    """
    # Convert to absolute path and expand user directory (e.g. ~/file.txt)
    abs_path = os.path.abspath(os.path.expanduser(path))
    
    # Extract directory portion from the absolute path
    dir_path = os.path.dirname(abs_path)
    
    # Create directory hierarchy if it doesn't exist and path contains directories
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    
    return abs_path

################################
## Detect QR and extract text ##
################################

def detect_qr(img_path: Optional[str] = None, 
                  img: Optional[np.ndarray] = None,
                  fast_mode: bool = True) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """Detect QR and return text"""
    
    if img is None and img_path:
        img = cv2.imread(img_path)
        if img is None:
            return None, None

    h, w = img.shape[:2]
    
    # Only if img is BGR, convert to gray
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    detector = cv2.QRCodeDetector()    
    
    if fast_mode and max(h, w) > 2000: # To speed process, reduce image size when at least one size is > 2000 px 
        scale = 0.5
        small = cv2.resize(gray, None, fx=scale, fy=scale, 
                          interpolation=cv2.INTER_AREA)
        data, pts, _ = detector.detectAndDecode(small)
        if pts is not None:
            pts = pts * (1/scale)  
    else:
        data, pts, _ = detector.detectAndDecode(gray)
    
    # Create a copy of img only if QR is detected
    if pts is not None and data:
        img_color = img.copy()
        pts = pts[0].astype(int)
        #cv2.polylines(img_color, [pts], True, (0, 255, 0), 2)
        qr_text = data.split()[0] if data.split() else data
        return qr_text, img_color
    
    # If QR no detected, then, return original image
    return None, img

#################### New functions for pixel/cm estimation ##############################
#### Version: Nov/2025

#############################################
## Detect size reference (ROI) using YOLOv8
#############################################

## Cache yolo 

_YOLO_MODEL_CACHE = {} # Avoid loaded the model multiple times

def _get_yolo_model(model_path: str):
    """Cache YOLO models to avoid reloading"""
    if model_path not in _YOLO_MODEL_CACHE:
        from ultralytics import YOLO
        _YOLO_MODEL_CACHE[model_path] = YOLO(model_path)
    return _YOLO_MODEL_CACHE[model_path]

def detect_size_ref_yolo(
        img: Optional[np.ndarray] = None,
        model_path: Optional[str] = None,
        img_path: Optional[str] = None,
        confidence_threshold: float = 0.6,
        iou_threshold: float = 0.45,
        show_max_rois: int = 6,
        plot: bool = False,
        plot_size: Tuple[int, int] = (8, 8),
        yolo_verbose: bool = False,
        font_size: int = 1.5,
        plot_roi_analysis: bool = False,
        return_roi_coords: bool = False
) -> Union[Tuple[List[Tuple[int, int, int]], np.ndarray], 
           Tuple[List[Tuple[int, int, int]], np.ndarray, List[Tuple[int, int, int, int]]]]:
    """
    Detect size reference circles using YOLOv8 model.
    """

    if model_path is None:
        model_path = _get_package_model_path('size_reference.pt')
    
    try:
        model = _get_yolo_model(model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None if not return_roi_coords else (None, None, None)

    
    # Load image
    if img is None and img_path is None:
        raise ValueError("No image or image path provided. Please pass either 'img' or 'img_path'.")

    if img_path is not None:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Error loading image from {img_path}")
    
    # Extract dimensions once
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # YOLO detection
    results = model(img, conf=0.1, iou=iou_threshold, verbose=False)

    # Initialize variables
    box_detected = False
    all_circles = []
    rois_debug = []
    roi_boxes = []
    
    # Only create annotated image if we need it (when boxes are detected)
    img_annotated = None

    for r in results:
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            if yolo_verbose:
                print("> No size reference detected.")
                continue
        
        # Filter boxes by confidence threshold
        filtered_boxes = []
        low_conf_boxes = []

        for box in boxes:
            conf = float(box.conf[0].cpu().numpy())
            if conf >= confidence_threshold:
                filtered_boxes.append(box)
            else:
                low_conf_boxes.append((box, conf))

        # Report filtered boxes
        if len(low_conf_boxes) > 0:
            print(f"Filtered out {len(low_conf_boxes)} box(es) below the confidence threshold: {confidence_threshold}")
            for box_idx, (box, conf_value) in enumerate(low_conf_boxes, 1):
                print(f"    ▸ Box {box_idx}: confidence = {conf_value:.3f}")

        boxes = filtered_boxes

        box_detected = True
        
        # Now that we know we have boxes, create annotated image
        if img_annotated is None:
            img_annotated = img.copy()

        if yolo_verbose:
            print('> Reference size detected:') 
            print(f"  - Processing reference box(es) with a confidence threshold >={confidence_threshold}:")
        
        # Pre-calculate margin percentages
        pad_x_pct = 0.15
        pad_y_pct = 0.05
        
        for i, box in enumerate(boxes):
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # Clamp coordinates
            x1 = max(0, min(x1, w-1))
            x2 = max(0, min(x2, w-1))
            y1 = max(0, min(y1, h-1))
            y2 = max(0, min(y2, h-1))
            
            confidence = float(box.conf[0].cpu().numpy())

            # Calculate margin
            box_width = x2 - x1 + 1
            box_height = y2 - y1 + 1
            padx = int(pad_x_pct * box_width)
            pady = int(pad_y_pct * box_height)

            # Calculate ROI with margin
            roi_x1 = max(0, x1 - padx)
            roi_y1 = max(0, y1 - pady)
            roi_x2 = min(w, x2 + padx)
            roi_y2 = min(h, y2 + pady)
            
            roi_boxes.append((roi_x1, roi_y1, roi_x2, roi_y2))

            # Extract ROI from grayscale
            roi_gray = gray[roi_y1:roi_y2, roi_x1:roi_x2]

            if yolo_verbose:
                roi_height, roi_width = roi_gray.shape[:2]
                print(f"            Ref {i+1}: {roi_width}x{roi_height} px, conf: {confidence:.3f}")

            if roi_gray.size == 0:
                print("Empty ROI, skipping...")
                continue
            
            # Draw bounding box on annotated image
            cv2.rectangle(img_annotated, (roi_x1, roi_y1), (roi_x2, roi_y2), (200, 100, 0), 2)
            cv2.putText(
                img_annotated,
                f"Ref {i+1} ({confidence:.2f})",
                (roi_x1 + 5, max(roi_y1 - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, (200, 100, 0), 3, cv2.LINE_AA
            )

            # Find circles in ROI
            if plot_roi_analysis:
                circles, dbg = find_size_ref_circles(roi_gray, return_debug=True, ref_circularity=0.7)
                rois_debug.append({
                    "idx": i+1,
                    "conf": confidence,
                    "roi_box": (roi_x1, roi_y1, roi_x2, roi_y2),
                    "roi_gray": dbg["roi_gray"],
                    "binary": dbg["binary"],
                    "overlay": dbg["overlay"],
                    "num_circles": len(circles)
                })
            else:
                circles = find_size_ref_circles(roi_gray, return_debug=False, ref_circularity=0.7)

            # Convert circle coordinates to global and draw
            for (cx_roi, cy_roi, radius) in circles:
                cx_global = cx_roi + roi_x1
                cy_global = cy_roi + roi_y1
                diameter = 2 * radius
                
                # Draw circle
                cv2.circle(img_annotated, (cx_global, cy_global), radius, (0, 0, 255), 5)
                
                
                # Draw diameter line
                line_start_x = cx_global - radius
                line_end_x = cx_global + radius
                cv2.line(img_annotated, (line_start_x, cy_global), 
                         (line_end_x, cy_global), (255, 139, 99), 3)
                
                # Draw diameter text (centered above line)
                text = f"{diameter}px"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 4)[0]
                text_x = cx_global - (text_size[0] // 2)
                text_y = cy_global - 20
                
                cv2.putText(img_annotated, text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), 4)

                # Store circle data
                all_circles.append((cx_global, cy_global, diameter))

        # Report results
        if yolo_verbose:
            print(f"\n  - Total circles detected: {len(all_circles)}")
        
            if not box_detected:
                print("No size reference box detected in the image by YOLO. Try adjusting confidence threshold or image quality.")
            elif len(all_circles) == 0:
                print("No circles detected within the detected size reference boxes. Try adjusting thresholds or check image quality.")

        # Plot main result if requested
        if plot and img_annotated is not None:
            plt.figure(figsize=plot_size)
            plt.imshow(cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

        # Plot ROI analysis if requested
        if plot_roi_analysis and box_detected and len(rois_debug) > 0:
            n = min(len(rois_debug), show_max_rois)
            cols = 3
            rows = n
            plt.figure(figsize=(14, 4*rows))
            
            for r_i in range(n):
                item = rois_debug[r_i]
                
                # ROI Gray
                plt.subplot(rows, cols, r_i*cols + 1)
                plt.imshow(item["roi_gray"], cmap='gray')
                x1, y1, x2, y2 = item["roi_box"]
                plt.title(f'Ref {item["idx"]} ({item["conf"]:.2f})\nROI: ({x1},{y1})-({x2},{y2})')
                plt.axis('off')
                
                # Binary
                plt.subplot(rows, cols, r_i*cols + 2)
                plt.imshow(item["binary"], cmap='gray')
                plt.title('Binarization')
                plt.axis('off')
                
                # Overlay
                plt.subplot(rows, cols, r_i*cols + 3)
                plt.imshow(cv2.cvtColor(item["overlay"], cv2.COLOR_BGR2RGB))
                plt.title(f'Overlay (circles: {item["num_circles"]})')
                plt.axis('off')
            
            plt.tight_layout()
            plt.show()

    # Return appropriate values based on whether boxes were detected
    if img_annotated is None:
        img_annotated = img  # Return original if nothing was detected
    
    if return_roi_coords:
        return all_circles, img_annotated, roi_boxes if roi_boxes else None
    else:
        return all_circles, img_annotated

##############################################
## Find black circles in size reference box ##
##############################################

def find_size_ref_circles(roi_gray, return_debug=False, ref_circularity=0.7, 
                              min_area_ratio=0.01):
    
    h, w = roi_gray.shape
    min_area = max(50, int(min_area_ratio * h * w))
    
    # Preprocess image
    kernel_size = 3 if min(h, w) < 100 else 5
    blurred = cv2.GaussianBlur(roi_gray, (kernel_size, kernel_size), 0)
    
    # Apply threshold to detect dark circles
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    circles = []
    valid_contours = []
    
    # For all the circles detected:
    for contour in contours:
        # Filter by minimun area
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # Filter by circularity
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < ref_circularity:
            continue
        
        # Adjust a circle contour
        (x, y), radius = cv2.minEnclosingCircle(contour)
        circles.append((int(x), int(y), int(radius)))
        valid_contours.append(contour)
    
    if return_debug:
        # Return circles overlay 
        overlay = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
        for (cx, cy, r) in circles:
            cv2.circle(overlay, (cx, cy), r, (0, 0, 255), 2)
            cv2.circle(overlay, (cx, cy), 2, (255, 0, 0), -1)
        
        return circles, {
            "roi_gray": roi_gray,
            "binary": binary,
            "overlay": overlay,
            "num_contours": len(contours),
            "num_circles": len(circles)
        }
    
    return circles

##########################
## Calculate px density ##
##########################

def img_px_per_cm(img, size='letter_ansi', width_cm=None, length_cm=None):
    """
    Calculates pixel density (pixels/cm) from an image and physical dimensions.
    
    When custom dimensions are provided, they map directly to image dimensions:
    - width_cm corresponds to img.shape[1] (image width)
    - length_cm corresponds to img.shape[0] (image height)
    """
    try:
        # Input validation
        if not isinstance(img, np.ndarray):
            raise TypeError("Input must be a numpy array")
        if img.ndim not in [2, 3]:
            raise ValueError("Image must be 2D (grayscale) or 3D (color)")
        if size not in ['letter_ansi', 'legal_ansi', 'a4_iso', 'a3_iso'] and (width_cm is None or length_cm is None):
            raise ValueError("Provide either valid physical size or custom dimensions")
        if width_cm is not None and (not isinstance(width_cm, (int, float)) or width_cm <= 0):
            raise ValueError("width_cm must be positive")
        if length_cm is not None and (not isinstance(length_cm, (int, float)) or length_cm <= 0):
            raise ValueError("length_cm must be positive")
        if width_cm > length_cm:
            raise ValueError("width_cm cannot be greater than length_cm")
        if length_cm < width_cm:
            raise ValueError("length_cm cannot be less than width_cm")
        
        # Scanner paper sizes
        paper_sizes = {
            'letter_ansi': (21.6, 27.9),
            'legal_ansi': (21.59, 35.56),
            'a4_iso': (21.0, 29.7),
            'a3_iso': (29.7, 42.0)
        }
        
        # Get image dimensions (standard OpenCV format)
        img_height_px = img.shape[0]  # Height (rows)
        img_width_px = img.shape[1]   # Width (columns)
        
        # Get physical dimensions
        if width_cm is not None and length_cm is not None:
            used_width_cm = width_cm
            used_length_cm = length_cm
        else:
            # Use paper size (auto-orient to match image)
            paper_w, paper_h = paper_sizes[size]
            if img_width_px > img_height_px:
                # Landscape image
                used_width_cm = max(paper_w, paper_h)
                used_length_cm = min(paper_w, paper_h)
            else:
                # Portrait image
                used_width_cm = min(paper_w, paper_h)
                used_length_cm = max(paper_w, paper_h)
        
        # Calculate density (direct mapping)
        px_per_cm_width = img_width_px / used_width_cm
        px_per_cm_length = img_height_px / used_length_cm
        
        return px_per_cm_width, px_per_cm_length, used_width_cm, used_length_cm
    
    except Exception as e:
        raise RuntimeError(f"Calculation error: {str(e)}")

##############################################################################    
## Obtain the px per cm density using the avg diameter of reference circles ##
##############################################################################

def diameter_px_per_cm(all_circles, verbose=False, diameter_cm=2.5, std_threshold=2):
    
    if not all_circles:
        raise ValueError('No circles provided.')
    
    # Get diameters array
    diameters = np.array([d[2] for d in all_circles], dtype=np.float32)
    
    # Calculate diameter stats 
    mean_val = np.mean(diameters)
    std_val = np.std(diameters)
    
    # Filter outliers based on std thresh
    lower = mean_val - std_threshold * std_val
    upper = mean_val + std_threshold * std_val
    
    mask = (diameters >= lower) & (diameters <= upper)
    filtered = diameters[mask]
    
    if len(filtered) == 0:
        if verbose:
            print("Warning: Using all the circles (many outliers detected)")
        filtered = diameters
    
    # Calculate px per cm density
    px_cm_density = np.mean(filtered) / diameter_cm
    
    if verbose:
        print(f"            Range: [{lower:.1f}, {upper:.1f}] px")
        print(f"            Filtered circles: {len(filtered)}/{len(diameters)} (std >  {std_threshold})")
        print(f"\n        . ݁₊ ⊹ . ݁ ⟡ ݁ px/cm density: {px_cm_density:.1f} (diameter_cm: {diameter_cm} cm) ⟡ ݁ . ⊹ ₊ ݁.")
    
    return float(px_cm_density)  

###################################################################################
## Wrapper: Calculate px to cm density from yolo or physical dimensions methods ##
###################################################################################

def px_cm_density(img, model_path: Optional[str] = None,
                  confidence_threshold: float = 0.6, 
                  plot=False,
                  width_cm: Optional[float] = None, 
                  length_cm: Optional[float] = None, 
                  diameter_cm: float = 2.5,
                  font_size: int = 3,
                  physical_size: Optional[str] = None,
                  return_coordinates: bool = False,
                  verbose = False) -> Union[Optional[float], Tuple[Optional[float], Optional[List]]]:
    """
    Calculate pixel-to-centimeter density using various reference methods.
    Always returns the average px/cm for both axes.
    
    Priority order:
    1. Circle detection (YOLO model)
    2. Predetermined physical size
    3. Provided width/length dimensions
    4. None (measurements in pixels)
    
    Args:
        img (np.ndarray): Input image
        model_path (str): Path to YOLO detection model
        confidence_threshold (float): Confidence threshold for detection
        plot (bool): Whether to plot detection results
        width_cm (float, optional): Image width in centimeters
        length_cm (float, optional): Image length in centimeters
        diameter_cm (float): Reference circle diameter in cm
        ref_circularity (float): Minimum circularity for valid circles
        physical_size (str, optional): Predetermined image size
        return_coordinates (bool): If True, return (px_cm, circle_coords) tuple
    
    Returns:
        If return_coordinates=False: float or None (px/cm density)
        If return_coordinates=True: tuple (px/cm density or None, list of circle contours or None)
    """
    # Method 1: Try circle detection

    # Load model (cached)
    if model_path is None:
        model_path = _get_package_model_path('size_reference.pt')
    
    try:
        model = _get_yolo_model(model_path) # Only to check if model exist
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None if not return_coordinates else (None, None, None)

    if return_coordinates:
        all_circles, img_annotated, roi_boxes = detect_size_ref_yolo(
            img, 
            model_path=model_path, 
            plot=plot, 
            font_size=font_size * 0.5,
            confidence_threshold=confidence_threshold,
            return_roi_coords=True,
            yolo_verbose=verbose
        )
    else:
        all_circles, img_annotated = detect_size_ref_yolo(
            img, 
            model_path=model_path, 
            plot=plot, 
            font_size=font_size * 0.5,
            confidence_threshold=confidence_threshold,
            yolo_verbose=verbose
        )
    
    if all_circles:
        #print('Using circle detection method:')
        #print(f'  - Reference diameter (cm): {diameter_cm}')
        #print(f'  - Reference circularity threshold: {ref_circularity}')
        px_cm = diameter_px_per_cm(all_circles, verbose=verbose, 
                                    diameter_cm=diameter_cm, std_threshold=2)
        
        if return_coordinates:
            # Convert ROI boxes to contours
            circle_coords = []
            for (x1, y1, x2, y2) in roi_boxes:
                contour = np.array([
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2]
                ], dtype=np.int32)
                circle_coords.append(contour)
            
            return px_cm, img_annotated, circle_coords if circle_coords else None
        else:
            return px_cm, img_annotated
    
    # Method 2: Use predetermined physical size
    valid_sizes = {
        'letter_ansi': (21.6, 27.9),
        'legal_ansi': (21.59, 35.56),
        'a4_iso': (21.0, 29.7),
        'a3_iso': (29.7, 42.0)
    }
    
    if physical_size is not None:
        if physical_size not in valid_sizes:
            raise ValueError(f"Physical_size must be one of {list(valid_sizes.keys())}, got '{physical_size}'")
        
        #print(f'No circles detected. Using predetermined image size: {physical_size}')
        px_per_cm_width, px_per_cm_length, _, _ = img_px_per_cm(img, size=physical_size)
        #px_cm = (px_per_cm_width + px_per_cm_length) / 2
        px_cm = float(np.sqrt(px_per_cm_width * px_per_cm_length))
        #print(f"Estimated px/cm: {px_cm:.2f}")
        
        if return_coordinates:
            return px_cm, img_annotated, None
        else:
            return px_cm, img_annotated
    
    # Method 3: Use provided width and length dimensions
    if width_cm is not None and length_cm is not None:
        #print('No circles detected. Using provided image dimensions (width/length in cm) as fallback.')
        px_per_cm_width, px_per_cm_length, _, _ = img_px_per_cm(img, width_cm=width_cm, 
                                                                  length_cm=length_cm)
        px_cm = np.sqrt(px_per_cm_width * px_per_cm_length)
        #print(f"Estimated px/cm from provided dimensions: {px_cm:.2f}")
        
        if return_coordinates:
            return px_cm, img_annotated, None
        else:
            return px_cm, img_annotated
    
    # Method 4: No reference available
    #print("No size references detected and no physical dimensions provided for px/cm estimation.")
    #print(">>>>>> Measurements will be reported in PIXELS <<<<<<")
    
    if return_coordinates:
        return None, img_annotated, None
    else:
        return None, img_annotated


######################## 
# Detect label roi
######################## 

def detect_label_box(imagen_path: Optional[str] = None, 
                     img: Optional[np.ndarray] = None,
                     verbose: Optional[bool] = False, 
                     plot: Optional[bool] = False,
                     max_boxes: int = 10) -> List[Dict]: 
    """Optimized label box detection"""
    
    if imagen_path is not None:
        img = cv2.imread(imagen_path)
    elif img is not None:
        img = img.copy()
    else:
        raise ValueError("Either imagen_path or img must be provided.")
    
    if img is None:
        raise ValueError(f"Could not load image: {imagen_path}")
    
    # Single conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    
    # ⚡ Pre-calculate to avoid repeated operations
    for cnt in contours:
        if len(boxes) >= max_boxes:
            break
        
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        
        # Quick reject before expensive division
        if area <= 5000:
            continue
        
        aspect_ratio = w / h
        
        if 2 < aspect_ratio < 6:
            box_info = {
                'x': x, 'y': y, 'width': w, 'height': h,
                'area': area, 'aspect_ratio': aspect_ratio
            }
            boxes.append(box_info)
            
            if plot:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    if plot:
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    
    if verbose:
        print(f"\nTotal boxes found: {len(boxes)}")
        for i, box in enumerate(boxes, 1):
            print(f"Box {i}: {box}")
    
    return boxes


##############################
# Detect label box with yolo #
##############################

def detect_label_box_yolo(img: np.ndarray, 
                          model_path: Optional[str] = None,
                          conf: float = 0.3, plot: bool = False) -> Optional[List[Dict]]:
    """Optimized with model caching"""
    
    if model_path is None:
        model_path = _get_package_model_path('label.pt')
    
    try:
        model = _get_yolo_model(model_path)
    except Exception as e:
        print(f"Error loading YOLO model from {model_path}: {e}")
        return None
    
    results = model(img, conf=conf, verbose=False)
    
    for r in results:
        boxes = r.boxes
        
        if boxes is None or len(boxes) == 0:
            return None
        
        # Convert YOLO detections to label_roi format
        label_boxes = []
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # Calculate box properties
            width = x2 - x1
            height = y2 - y1
            area = width * height
            aspect_ratio = width / height if height > 0 else 0
            
            # Create dict in same format as detect_label_box
            box_info = {
                'x': x1,
                'y': y1,
                'width': width,
                'height': height,
                'area': area,
                'aspect_ratio': aspect_ratio
            }
            
            label_boxes.append(box_info)
        if plot:
            plt.figure(figsize = (8,8))
            img_copy = img.copy()
            for box in label_boxes:
                x, y = box['x'], box['y']
                w, h = box['width'], box['height']
                cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
            plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

        return label_boxes if label_boxes else None
    
    return None

###################
### Save images ###
###################

def save_img(self,
             output_path: Optional[str] = None,
             quality: int = 95,
             compression: int = 9,
             output_message: bool = True):

    if self.morphology_results is None:
        raise ValueError("No image annotated available. Run analyze_morphology() first.")

    image = self.morphology_results  

    if output_path is None:
        results_path = os.path.join(self.image_path, "Results")
        os.makedirs(results_path, exist_ok=True)
        output_path = os.path.join(results_path, "annotated_image.jpg")

    # Validate image extension
    ext = os.path.splitext(output_path)[1].lower()

    if ext not in valid_cv2_extensions:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Valid formats: {', '.join(valid_cv2_extensions)}"
        )

    params = []

    if ext in [".jpg", ".jpeg"]:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]

    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, compression]

    elif ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, quality]

    success = cv2.imwrite(output_path, image, params)

    if not success:
        raise IOError(f"Failed to save image to '{output_path}'")

    if output_message:
        print(f"Image saved to: {output_path}")

    return output_path



#####################
# Annotate an image #
#####################

def annotate_all_fruits(
    fruit_locule_map: Dict[int, List[int]],
    contours: List[np.ndarray],
    annotated_img: np.ndarray,
    img_shape: Tuple[int, int],
    font_scale = 2,
    font_thickness = 2,
    pericarp_ext_color = (0,255,0),
    pericarp_ext_thickness = 2,
    locule_thickness = 2,
    locule_color = (255,0,255),
    label_position = 'left',
    margin = 10, 
    text_color = (0,0,0), 
    label_background_color = (255,255,255),
    label_opacity = 0.7, verbose: bool = True


) -> None:
    """Draw contours and annotations for ALL fruits in one pass."""
    
    # if not fruit_locule_map:
    #     print("No fruits_locule map to annotate")
    #     return
    # if not contours:
    #     print("No contours to annotate")

    if label_position not in label_positions:
        raise ValueError(f"Invalid label position: {label_position}. Valid options are: {label_positions}") 
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    (size_w, size_h), _ = cv2.getTextSize("Test", font, font_scale, font_thickness)
    single_line_height = size_h
    
    img_height, img_width = img_shape
    sequential_id = 1
    

    
    # For each fruit:
    for fruit_id, locule_ids in fruit_locule_map.items():
        
        # validate contours
        if fruit_id >= len(contours):
            print(f"Fruit ID {fruit_id} not in contours list")
            continue
        
        fruit_contour = contours[fruit_id]
        if fruit_contour is None or len(fruit_contour) == 0:
            print(f" Empty contour for fruit {fruit_id}")
            continue
        
        n_locules = len(locule_ids)
        
        # Draw fruit contour (green)
        cv2.drawContours(annotated_img, [fruit_contour], -1, pericarp_ext_color, pericarp_ext_thickness)
        
        # Draw locule contours (pink)
        for locule_id in locule_ids:
            if locule_id >= len(contours):
                print(f"Locule ID {locule_id} not in contours list")
                continue
            
            locule_contour = contours[locule_id]
            if locule_contour is None or len(locule_contour) == 0:
                continue
            
            cv2.drawContours(annotated_img, [locule_contour], -1, locule_color, locule_thickness)
        
        # Create label (fruit id + n locules)
        x, y, w, h = cv2.boundingRect(fruit_contour)
        if n_locules == 0:
            text = f"id {sequential_id}"
        else:
            text = f"id {sequential_id}: \n{n_locules} loc"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        (size_w, size_h), _ = cv2.getTextSize("Test", font, font_scale, font_thickness)
        
        single_line_height = size_h
        
        num_lines = text.count('\n') + 1
        total_height = (single_line_height * num_lines) + (15 * (num_lines - 1))

        
        # Calculate max text width
        text_width = max([
            cv2.getTextSize(line, font, font_scale, font_thickness)[0][0]
            for line in text.split('\n')
        ])
        
        # Calculate position based on label_position
        img_height, img_width = img_shape
        
        if label_position == 'top':
            text_x = max(10, x)
            text_y = max(total_height + 15, y - 15)
        elif label_position == 'bottom':
            text_x = max(10, x)
            text_y = min(img_height - 15, y + h + total_height + 15)
        elif label_position == 'left':
            text_x = max(10, x - text_width - margin * 2 - 15)
            text_y = max(total_height + 15, y + h // 2)
        elif label_position == 'right':
            text_x = min(img_width - text_width - margin * 2 - 10, x + w + 15)
            text_y = max(total_height + 15, y + h // 2)
        else:
            text_x = max(10, x)
            text_y = max(total_height + 15, y - 15)
        
        # Ensure text stays within bounds
        text_x = max(margin, min(text_x, img_width - text_width - margin * 2))
        text_y = max(total_height + margin, min(text_y, img_height - margin))
        
        # Draw background
        text_bg_layer = annotated_img.copy()

        cv2.rectangle(
            text_bg_layer,
            (text_x - margin, text_y - total_height - margin),
            (text_x + text_width + margin, text_y + margin),
            label_background_color, -1
        )
        cv2.addWeighted(
            text_bg_layer,              
            label_opacity,
            annotated_img,              
            1 - label_opacity,
            0,
            annotated_img              
        )

        # Draw text
        for i, line in enumerate(text.split('\n')):
            y_offset = text_y - (total_height - single_line_height) + \
                    (i * (single_line_height + 15))
            cv2.putText(
                annotated_img, line, (text_x, y_offset),
                font, font_scale, text_color, 
                font_thickness, cv2.LINE_AA
            )

        sequential_id = sequential_id +1


