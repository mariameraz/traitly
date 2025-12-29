import cv2
import os
import numpy as np
import easyocr
import warnings
import matplotlib.pyplot as plt
#from pdf2image import convert_from_path

from typing import Optional, List, Tuple, Union, Dict

# Read QR code
#from pyzbar.pyzbar import decode


# New modules included for px density (yolo model)
from ultralytics import YOLO
import statistics

import sys
from io import StringIO
import warnings

##############################################################################
valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}
##############################################################################


##############################################################################
# Load an image
##############################################################################

def load_img(path, plot=True, plot_size=(20, 10), fig_axis=True):
    """Load a BGR image from file and validate its format.
    
    Args:
        path (str): Full path to the image file.
        plot (bool, optional): Whether to display the image after loading. 
                              Defaults to True.
        plot_size (tuple, optional): Figure size for display in inches (width, height). 
                                    Defaults to (20, 10).
        fig_axis (bool, optional): Whether to show axis when plotting. 
                                  Defaults to True.

    Returns:
        numpy.ndarray or None: Loaded image as a numpy array in BGR format, 
                              or None if loading failed.

    Raises:
        ValueError: If the file extension is not valid or image cannot be loaded.

    """
    try:
        valid_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}  # BGR only
        ext = os.path.splitext(path)[1].lower()  # Obtain image extension

        if ext not in valid_extensions: 
            raise ValueError("Image format not valid (expected .jpg/.jpeg/.png/.tiff)")

        img = cv2.imread(path)  # Read image in BGR format
        if img is None:
            raise ValueError(f"Cannot load image: {os.path.basename(path)}")
        
        if plot:
            plot_img(img, metadata=False, fig_axis=fig_axis, plot_size=plot_size)

        return img

    except Exception as e:
        print(f"Error loading image: {e}")
        return None

##############################################################################
# Evaluate if a contour is valid using geometric thresholds
##############################################################################

def is_contour_valid(contour, filters=None):
    """Evaluates if a contour meets all geometric criteria.
    
    Args:
        REQUIRED:
        - contour (np.ndarray): Input contour to evaluate (shape: [N, 1, 2]).
        
        OPTIONAL:
        - filters (Dict): Dictionary containing filter thresholds with keys:
            - min_area (int): Minimum area threshold (default: 300)
            - min_circularity (float): Minimum circularity threshold (default: 0.7)
            - max_circularity (float): Maximum circularity threshold (default: 1.0)
            - min_aspect_ratio (float): Minimum aspect ratio threshold (default: 0.8)
            - max_aspect_ratio (float): Maximum aspect ratio threshold (default: 1.0)
            
    Returns:
        bool: True if contour passes all filters, False otherwise
    """
    # Valores por defecto
    default_filters = {
        'min_area': 300,
        'min_circularity': 0.6,
        'max_circularity': 1.0,
        'min_aspect_ratio': 0.4,
        'max_aspect_ratio': 1.0
    }
    
    # Combinar filtros proporcionados con valores por defecto
    if filters is None:
        filters = default_filters
    else:
        # Actualizar solo las claves proporcionadas, mantener las demás por defecto
        filters = {**default_filters, **filters}
    
    area = cv2.contourArea(contour)
    if area < filters['min_area']:
        return False
        
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False
        
    circularity = (4 * np.pi * area) / (perimeter ** 2) 
    _, (w, h), _ = cv2.minAreaRect(contour)
    aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0
    
    return (filters['min_circularity'] <= circularity <= filters['max_circularity'] and
            filters['min_aspect_ratio'] <= aspect_ratio <= filters['max_aspect_ratio'])


#################################################################################################
# Detect label text
#################################################################################################


# _READER_CACHE = {}

# def get_easyocr_reader(languages=['en', 'es']):
#     """Initialize EasyOCR reader with GPU (falls back to CPU if unavailable), suppressing all messages."""
#     old_stdout, old_stderr = sys.stdout, sys.stderr
#     sys.stdout = sys.stderr = StringIO()
    
#     try:
#         import easyocr
#         # Suppress all warnings including CUDA warnings
#         with warnings.catch_warnings():
#             warnings.simplefilter('ignore')
#             reader = easyocr.Reader(languages, gpu=True, verbose=False)
#     finally:
#         sys.stdout, sys.stderr = old_stdout, old_stderr
    
#     return reader

# def get_cached_reader(languages=('en', 'es')):
#     """Return cached reader."""
#     key = tuple(languages)
#     if key not in _READER_CACHE:
#         _READER_CACHE[key] = get_easyocr_reader(list(languages))
#     return _READER_CACHE[key]

    
# def detect_label_text(img: np.ndarray, 
#                       label_roi: List[Dict], 
#                       language: List[str] = ['es', 'en'],
#                       blur_label: Tuple[int, int] = (11, 11),
#                       verbose: bool = False) -> Optional[str]:
#     """
#     Extract text from detected label regions using OCR.
    
#     Args:
#         img: Input image (BGR format from OpenCV)
#         label_roi: List of label boxes from detect_label_box().
#                    Each dict contains: x, y, width, height, area, aspect_ratio
#         language: Languages for OCR detection. Default is ['es', 'en'].
#         blur_label: Gaussian blur kernel size. Default is (11, 11).
#         verbose: Print debug information. Default is False.
    
#     Returns:
#         Detected label text, or None if no text found in any region.
#     """
#     # Validate input
#     if label_roi is None or len(label_roi) == 0:
#         if verbose:
#             print("No label regions provided")
#         return None
    
#     if verbose:
#         print(f"Processing {len(label_roi)} label box(es)...")
    
#     # Initialize OCR reader (attempts GPU, falls back to CPU silently)
#     reader = get_cached_reader(tuple(language))
    
#     # Try OCR on each detected label region
#     for i, box in enumerate(label_roi):
#         x = box['x']
#         y = box['y']
#         w = box['width']
#         h = box['height']
        
#         if verbose:
#             print(f"\nProcessing box {i+1}/{len(label_roi)}:")
#             print(f"   Position: ({x}, {y}), Size: {w}x{h}")
        
#         try:
#             # Validate coordinates are within image bounds
#             if y + h > img.shape[0] or x + w > img.shape[1]:
#                 if verbose:
#                     print(f"   - Box out of image bounds, skipping")
#                 continue
            
#             # Extract label region from image
#             label_region = img[y:y+h, x:x+w]
            
#             # Validate extracted region
#             if label_region.size == 0:
#                 if verbose:
#                     print(f"   - Empty region, skipping")
#                 continue
            
#             # Preprocess: Convert to grayscale
#             gray = cv2.cvtColor(label_region, cv2.COLOR_BGR2GRAY)
            
#             # Apply Gaussian blur to reduce noise
#             blur = cv2.GaussianBlur(gray, blur_label, 0)
            
#             # Run OCR on the preprocessed region
#             results = reader.readtext(blur)
            
#             if results:
#                 # Extract text from OCR results and join with spaces
#                 label_text = " ".join([result[1] for result in results])
                
#                 if verbose:
#                     print(f"   - Text found: '{label_text}'")
#                     print(f"     Confidence scores: {[result[2] for result in results]}")
                
#                 return label_text  # Return immediately when text is found
            
#             else:
#                 if verbose:
#                     print(f"   - No text detected in this box")
        
#         except Exception as e:
#             if verbose:
#                 print(f"   - Error processing box {i+1}: {e}")
#             continue
    
#     # No text found in any box
#     if verbose:
#         print("\nNo text could be extracted from any label box")
    
#     return None

# Reemplaza estas funciones en common_functions.py (líneas ~265-305)

_READER_CACHE = {}

def get_easyocr_reader(languages=['en', 'es'], gpu=False):
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
                gpu = False
        
        # Suppress warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            reader = easyocr.Reader(languages, gpu=gpu, verbose=False)
    
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

    
def detect_label_text(img: np.ndarray, 
                      label_roi: List[Dict], 
                      language: List[str] = ['es', 'en'],
                      blur_label: Tuple[int, int] = (11, 11),
                      verbose: bool = False,
                      gpu: bool = False) -> Optional[str]:  # ← Nuevo parámetro
    """
    Extract text from detected label regions using OCR.
    
    Args:
        img: Input image (BGR format from OpenCV)
        label_roi: List of label boxes from detect_label_box().
                   Each dict contains: x, y, width, height, area, aspect_ratio
        language: Languages for OCR detection. Default is ['es', 'en'].
        blur_label: Gaussian blur kernel size. Default is (11, 11).
        verbose: Print debug information. Default is False.
        gpu: Whether to attempt GPU acceleration (only works with CUDA). Default is False.
    
    Returns:
        Detected label text, or None if no text found in any region.
    """
    # Validate input
    if label_roi is None or len(label_roi) == 0:
        if verbose:
            print("No label regions provided")
        return None
    
    if verbose:
        print(f"Processing {len(label_roi)} label box(es)...")
    
    # Initialize OCR reader with GPU setting
    reader = get_cached_reader(tuple(language), gpu=gpu)
    
    # Try OCR on each detected label region
    for i, box in enumerate(label_roi):
        x = box['x']
        y = box['y']
        w = box['width']
        h = box['height']
        
        if verbose:
            print(f"\nProcessing box {i+1}/{len(label_roi)}:")
            print(f"   Position: ({x}, {y}), Size: {w}x{h}")
        
        try:
            # Validate coordinates are within image bounds
            if y + h > img.shape[0] or x + w > img.shape[1]:
                if verbose:
                    print(f"   - Box out of image bounds, skipping")
                continue
            
            # Extract label region from image
            label_region = img[y:y+h, x:x+w]
            
            # Validate extracted region
            if label_region.size == 0:
                if verbose:
                    print(f"   - Empty region, skipping")
                continue
            
            # Preprocess: Convert to grayscale
            gray = cv2.cvtColor(label_region, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blur = cv2.GaussianBlur(gray, blur_label, 0)
            
            # Run OCR on the preprocessed region
            results = reader.readtext(blur)
            
            if results:
                # Extract text from OCR results and join with spaces
                label_text = " ".join([result[1] for result in results])
                
                if verbose:
                    print(f"   - Text found: '{label_text}'")
                    print(f"     Confidence scores: {[result[2] for result in results]}")
                
                return label_text  # Return immediately when text is found
            
            else:
                if verbose:
                    print(f"   - No text detected in this box")
        
        except Exception as e:
            if verbose:
                print(f"   - Error processing box {i+1}: {e}")
            continue
    
    # No text found in any box
    if verbose:
        print("\nNo text could be extracted from any label box")
    
    return None

#################################################################################################
# Detect image name
#################################################################################################

def detect_img_name(path_image):
    try:
        if not isinstance(path_image, str):
            raise TypeError('Path input should be of type str')
        
        extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}

        filename = os.path.basename(path_image)
        name, ext = os.path.splitext(filename)

        if ext.lower() not in extensions:
            warnings.warn("Warning: File extension is not a valid image format.")
        return name
    
    except Exception as e:
        print(f"Error: {e}")
        return None
    


#################################################################################################
# Plotting image on screen
#################################################################################################

def plot_img(img, fig_axis=False, plot_size=(10, 10), label_text='None', 
             img_name='None', title_fontsize=12, title_location='center', 
             metadata=True, gray = False):
    """
    Plots an image with customizable display options.
    
    Args:
        img (numpy.ndarray): Input image in BGR format
        fig_axis (bool): Whether to show axis (default: False)
        plot_size (tuple): Figure size (width, height) in inches (default: (10, 10))
        label_text (str): Text label for the title (default: 'None')
        img_name (str): Image name for the title (default: 'None')
        title_fontsize (int): Font size for the title (default: 12)
        title_location (str): Title location ('left', 'center', 'right') (default: 'center')
        metadata (bool): If True, suppresses title display (default: False)
    """
    plt.figure(figsize=plot_size)
    if gray:
        plt.imshow(img, cmap='gray') 
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if metadata:
        plt.title(f"{img_name}: {label_text}", fontsize=title_fontsize, loc=title_location)
    
    plt.tight_layout()
    
    if not fig_axis:
        plt.axis('off')
    
    plt.show()


#################################################################################################
# Converting PDF pages to JPEG images
#################################################################################################


def validate_dir(path):
    """
    Ensure the directory exists and return the absolute path.
    
    Args:
        path (str): File path to check
        
    Returns:
        str: Absolute path with ensured directory existence
    """
    # Convert to absolute path and expand user directory (e.g., ~/file.txt)
    abs_path = os.path.abspath(os.path.expanduser(path))
    
    # Extract directory portion from the absolute path
    dir_path = os.path.dirname(abs_path)
    
    # Create directory hierarchy if it doesn't exist and path contains directories
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    
    return abs_path


# def detect_qr(img_path: Optional[str] = None, img: Optional[np.ndarray] = None) -> Tuple[Optional[str], Optional[np.ndarray]]:
#     """
#     Detects QR codes in an image
#     Args:
#         img_path: Path to the image (optional if img is provided)
#         img: Image as numpy array (optional if img_path is provided)
#     Returns:
#         Tuple with (qr_text, image_with_rectangle)
#     """
#     # Valid extensions
#     valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}
    
#     # Validate that at least one argument is present
#     if img_path is None and img is None:
#         raise ValueError("You must provide either img_path or img")
    
#     # If path is provided, validate and load image
#     if img_path:
#         # Validate file exists
#         if not os.path.exists(img_path):
#             raise FileNotFoundError(f"Image does not exist: {img_path}")
        
#         # Validate extension
#         file_ext = os.path.splitext(img_path)[1].lower()
#         if file_ext not in valid_extensions:
#             raise ValueError(f"Invalid format. Use: {', '.join(sorted(valid_extensions))}")
        
#         # Load image with OpenCV (already in BGR format)
#         img = cv2.imread(img_path)
        
#         # Validate image loaded correctly
#         if img is None:
#             raise ValueError(f"Could not load image: {img_path}")
        
    
    
#     # Validate it's a valid image
#     if not isinstance(img, np.ndarray):
#         raise TypeError("img must be a numpy array")
    
#     # Decode QR codes
#     decoded_objects = decode(img)
    
#     # Initialize text variable
#     qr_text = None
    
#     # Extract text and draw rectangle for each QR found
#     for obj in decoded_objects:
#         # Get the bounding box coordinates of the QR code
#         x, y, w, h = obj.rect
        
#         # Draw a green rectangle around the QR code
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
#         # Extract text (take the first QR found)
#         if qr_text is None:
#             qr_text = obj.data.decode('utf-8')
    
#     # If no QR was found
#     if qr_text is not None:
#         # Take only the first word (before first space)
#         qr_text = qr_text.split()[0] if qr_text.split() else qr_text
    
#     return qr_text, img

def detect_qr(img_path: Optional[str] = None, img: Optional[np.ndarray] = None) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """Optimized QR detection with early stopping"""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}
    
    if img_path is None and img is None:
        raise ValueError("You must provide either img_path or img")
    
    if img_path is not None:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image does not exist: {img_path}")
        
        file_ext = os.path.splitext(img_path)[1].lower()
        if file_ext not in valid_extensions:
            raise ValueError(f"Invalid format. Use: {', '.join(sorted(valid_extensions))}")
        
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
    
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a numpy array")
    
    detector = cv2.QRCodeDetector()
    
    # Convert to grayscale ONCE
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ⚡ OPTIMIZATION: Try strategies in order of success rate
    # Most QR codes work with original or adaptive, so try those first
    preprocessing_strategies = [
        ("original", gray),
        ("adaptive", None),  # Compute lazily
        ("otsu", None),
        ("clahe", None),
        ("blur_thresh", None)
    ]
    
    qr_text: Optional[str] = None
    points = None
    
    for strategy_name, processed_img in preprocessing_strategies:
        # ⚡ Lazy computation - only process if needed
        if processed_img is None:
            if strategy_name == "adaptive":
                processed_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                       cv2.THRESH_BINARY_INV, 11, 2)
            elif strategy_name == "otsu":
                _, processed_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif strategy_name == "clahe":
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                processed_img = clahe.apply(gray)
            elif strategy_name == "blur_thresh":
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                _, processed_img = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        
        data, pts, _ = detector.detectAndDecode(processed_img)
        
        if pts is not None and data:
            qr_text = data
            points = pts
            break  # ⚡ Early exit
    
    # Draw rectangle if QR found
    if points is not None and qr_text:
        pts = points[0].astype(int)
        x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
        x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
        
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        qr_text = qr_text.split()[0] if qr_text.split() else qr_text
    
    return qr_text, img

#################### New functions for pixel/cm estimation ##############################
#### Version: Nov/2025

#############################################
## Detect size reference (ROI) using YOLOv8
#############################################

## Cache yolo 

_YOLO_MODEL_CACHE = {}

def _get_yolo_model(model_path: str):
    """Cache YOLO models to avoid reloading"""
    if model_path not in _YOLO_MODEL_CACHE:
        from ultralytics import YOLO
        _YOLO_MODEL_CACHE[model_path] = YOLO(model_path)
    return _YOLO_MODEL_CACHE[model_path]


    

def detect_size_ref_yolo(
        img: Optional[np.ndarray] = None,
        model_path: str = 'traitly/package_data/models/size_reference.pt',
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
    Optimized version with reduced memory copies and lazy computations.
    """
    
    # Load model
    # try:
    #     model = YOLO(model_path)
    # except Exception as e:
    #     print(f"Error loading model from {model_path}: {e}")
    #     return
    
    # Load model (cached)
    try:
        model = _get_yolo_model(model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return
    
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
            print("No size reference detected.")
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
                print(f"  - Box {box_idx}: confidence = {conf_value:.3f}")

        boxes = filtered_boxes

        if len(boxes) == 0:
            print("No size reference boxes above confidence threshold. Try adjusting the threshold or image quality.")
            continue

        box_detected = True
        
        # Now that we know we have boxes, create annotated image
        if img_annotated is None:
            img_annotated = img.copy()

        if yolo_verbose:
            print(f"Processing {len(boxes)} high-confidence (>={confidence_threshold}) size reference box(es):")
        
        # Pre-calculate padding percentages
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

            # Calculate padding
            box_width = x2 - x1 + 1
            box_height = y2 - y1 + 1
            padx = int(pad_x_pct * box_width)
            pady = int(pad_y_pct * box_height)

            # Calculate ROI with padding
            roi_x1 = max(0, x1 - padx)
            roi_y1 = max(0, y1 - pady)
            roi_x2 = min(w, x2 + padx)
            roi_y2 = min(h, y2 + pady)
            
            roi_boxes.append((roi_x1, roi_y1, roi_x2, roi_y2))

            # Extract ROI from grayscale
            roi_gray = gray[roi_y1:roi_y2, roi_x1:roi_x2]

            if yolo_verbose:
                roi_height, roi_width = roi_gray.shape[:2]
                print(f"  - Ref {i+1}: {roi_width}x{roi_height} px, conf: {confidence:.3f}")

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
                cv2.circle(img_annotated, (cx_global, cy_global), 10, (255, 0, 0), -1)
                
                # Draw diameter line
                line_start_x = cx_global - radius
                line_end_x = cx_global + radius
                cv2.line(img_annotated, (line_start_x, cy_global), 
                         (line_end_x, cy_global), (0, 255, 0), 3)
                
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
            print(f"\nTotal circles detected: {len(all_circles)}")
        
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
    


def find_size_ref_circles(roi_gray, return_debug=False, ref_circularity=0.7):
    """Optimized circle detection with vectorized operations"""
    
    # Preprocessing
    blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Morphology
    k = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=1)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # ⚡ Calculate min_area once
    h, w = roi_gray.shape[:2]
    min_area = max(50, int(0.01 * h * w))
    
    # ⚡ VECTORIZED: Filter all contours at once
    circles = []
    valid_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        per = cv2.arcLength(contour, True)
        if per == 0:
            continue
        
        circularity = 4 * np.pi * area / (per * per)
        if circularity > ref_circularity:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            circles.append((int(x), int(y), int(radius)))
            valid_contours.append(contour)
    
    if return_debug:
        # ⚡ Only create overlay if needed
        overlay = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
        for (cx, cy, r) in circles:
            cv2.circle(overlay, (cx, cy), r, (0, 0, 255), 5)
            cv2.circle(overlay, (cx, cy), 5, (0, 0, 255), -1)
        
        return circles, {"roi_gray": roi_gray, "binary": binary, "overlay": overlay}
    else:
        return circles
    
def diameter_px_per_cm(all_circles: List[Tuple[int, int, int]], verbose: bool = False, 
                       diameter_cm: float = 2.5, std_threshold: float = 2):
    """Optimized diameter calculation with numpy"""
    
    if not all_circles:
        raise ValueError('No circles provided. The circles list is empty or not specified.')
    
    # ⚡ Convert to numpy array for vectorized operations
    all_diameters = np.array([d[2] for d in all_circles], dtype=float)
    
    circles_mean = np.mean(all_diameters)
    std_dev = np.std(all_diameters)
    
    # ⚡ Vectorized filtering
    lower_limit = circles_mean - std_threshold * std_dev
    upper_limit = circles_mean + std_threshold * std_dev
    
    mask = (all_diameters >= lower_limit) & (all_diameters <= upper_limit)
    filtered = all_diameters[mask]
    
    if len(filtered) == 0:
        if verbose:
            print("Warning: All circles were filtered as outliers. Using full dataset for calculation.")
        filtered = all_diameters
    
    # ⚡ Single mean calculation
    px_cm_density = np.mean(filtered) / diameter_cm
    
    if verbose:
        print(f"  - Diameter range (mean ± {std_threshold}): {lower_limit:.2f} px to {upper_limit:.2f} px")
        print(f"  - Total circles after removing outliers (std < 2): {len(filtered)}")
        print(f"  - Mean diameter of filtered circles: {np.mean(filtered):.2f} px")
        print(f"\n >>> Estimated px/cm density: {px_cm_density:.2f} px/cm (Reference diameter: {diameter_cm} cm)")
    
    return px_cm_density


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
            # USE DIRECTLY - respect user's specification
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



def px_cm_density(img, model_path='/Users/alejandra/Documents/GitHub/Morpho/Morpho/morphoslicer/utils/Models/References_Model.pt', 
                  confidence_threshold: float = 0.6, 
                  plot=False,
                  width_cm: Optional[float] = None, 
                  length_cm: Optional[float] = None, 
                  diameter_cm: float = 2.5,
                  font_size: int = 3,
                  ref_circularity: float = 0.7,
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
        px_cm = diameter_px_per_cm(all_circles, verbose=True, 
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
        px_cm = (px_per_cm_width + px_per_cm_length) / 2
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
        px_cm = (px_per_cm_width + px_per_cm_length) / 2
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
                     max_boxes: int = 10) -> List[Dict]:  # ⚡ NEW: limit results
    """Optimized label box detection"""
    
    if imagen_path is not None:
        img = cv2.imread(imagen_path)
    elif img is not None:
        img = img.copy()
    else:
        raise ValueError("Either imagen_path or img must be provided.")
    
    if img is None:
        raise ValueError(f"Could not load image: {imagen_path}")
    
    # ⚡ Single conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    
    # ⚡ Pre-calculate to avoid repeated operations
    for cnt in contours:
        if len(boxes) >= max_boxes:  # ⚡ Early stopping
            break
        
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        
        # ⚡ Quick reject before expensive division
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




def create_mask(self, n_kernel: int = 5, plot: bool = False, plot_size: Tuple[int, int] = (5, 5), 
                stamp: bool = False, plot_axis: bool = False, n_iteration: int = 1, 
                canny_min: int = 30, canny_max: int = 100, lower_hsv: Optional[List[int]] = None,
                upper_hsv: Optional[List[int]] = None, locules_filled: bool = False, 
                min_locule_size: int = 300, n_blur: int = 11, clip_limit: int = 4, 
                tile_grid_size: int = 8, remove_roi: bool = True) -> None:  # ← Nuevo parámetro
    """
    Create a mask for fruit detection and segmentation.
    
    Args:
        ... (parámetros existentes)
        remove_roi: Whether to remove label and reference ROIs from mask
    """
    if stamp:
        self.img_inverted = cv2.bitwise_not(self.img)
    else:
        self.img_inverted = self.img.copy()
    
    # Create base mask - only calculate once
    self.mask = create_mask(
        self.img_inverted,
        n_kernel=n_kernel, 
        n_iteration=n_iteration,
        plot=False,
        plot_size=plot_size,
        fig_axis=plot_axis,
        canny_max=canny_max,
        canny_min=canny_min,
        lower_hsv=lower_hsv,
        upper_hsv=upper_hsv
    )
    
    if locules_filled:
            # Use the already calculated mask instead of recalculating
            base_mask = self.mask.copy()
            
            # Fill fruit contours
            contours, _ = cv2.findContours(base_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cv2.drawContours(base_mask, [cnt], -1, 255, -1)
                
            # Convert image to Lab for locule processing
            lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
            l_channel = lab[:,:,0]

            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
            l_clahe = clahe.apply(l_channel)

            _, locule_mask = cv2.threshold(l_clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            locule_mask = cv2.medianBlur(locule_mask, n_blur)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n_kernel, n_kernel))
            opened = cv2.morphologyEx(locule_mask, cv2.MORPH_OPEN, kernel)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

            # Detect only internal contours (locules)
            inv_closed = cv2.bitwise_not(closed)
            contours, hierarchy = cv2.findContours(inv_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            mask_lobules_only = np.zeros_like(closed)
            
            for i, cnt in enumerate(contours):
                if hierarchy[0][i][3] != -1 and cv2.contourArea(cnt) > min_locule_size:
                    cv2.drawContours(mask_lobules_only, [cnt], -1, 255, -1)
            
            mask_lobules_only = cv2.medianBlur(mask_lobules_only, n_blur)

            # Overlap fruits mask with locule mask
            mask_fruits_rgb = cv2.cvtColor(cv2.bitwise_not(base_mask), cv2.COLOR_GRAY2BGR)
            mask_fruits_rgb[mask_lobules_only == 255] = [255, 255, 255]
            
            self.mask_fruits = base_mask.copy()
            self.mask = cv2.bitwise_not(mask_fruits_rgb[:,:,0])
    
    # ============= NUEVO: Eliminar ROIs =============
    if remove_roi:
        self.mask = self._remove_rois_from_mask(self.mask)
    # ================================================

    if plot:
        plt.figure(figsize=plot_size)
        plt.imshow(cv2.cvtColor(self.img_inverted, cv2.COLOR_BGR2RGB))
        plt.axis('off')

    return None


def _remove_rois_from_mask(self, mask: np.ndarray) -> np.ndarray:
    """Optimized ROI removal with batch operations"""
    
    # ⚡ No copy if no ROIs to remove
    if not ((hasattr(self, 'label_roi') and self.label_roi) or 
            (hasattr(self, 'ref_roi') and self.ref_roi)):
        return mask
    
    mask_clean = mask.copy()
    
    # ⚡ Batch process all label ROIs
    if hasattr(self, 'label_roi') and self.label_roi:
        for box in self.label_roi:
            x, y, w, h = box['x'], box['y'], box['width'], box['height']
            mask_clean[y:y+h, x:x+w] = 0
    
    # ⚡ Batch process all reference ROIs
    if hasattr(self, 'ref_roi') and self.ref_roi:
        # Use single fillPoly call if possible
        cv2.fillPoly(mask_clean, self.ref_roi, 0)
    
    return mask_clean


##############################
# Detect label box with yolo #
##############################

def detect_label_box_yolo(img: np.ndarray, 
                          model_path: str = 'traitly/package_data/models/label.pt',
                          conf: float = 0.3, plot: bool = False) -> Optional[List[Dict]]:
    """Optimized with model caching"""
    
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