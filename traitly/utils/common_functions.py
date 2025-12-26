import cv2
import os
import numpy as np
import easyocr
import warnings
import matplotlib.pyplot as plt
#from pdf2image import convert_from_path

from typing import Optional, List, Tuple, Union, Dict
import fitz  # PyMuPD

# Read QR code
from pyzbar.pyzbar import decode
import re

# New modules included for px density (yolo model)
from ultralytics import YOLO
import statistics





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

def detect_label_text(img: np.ndarray, 
                      label_roi: List[Dict], 
                      language: List[str] = ['es', 'en'],
                      blur_label: Tuple[int, int] = (11, 11),
                      verbose: bool = False,
                      use_gpu: bool = False) -> Optional[str]:  # Nuevo parámetro
    """
    Extract text from detected label regions using OCR.
    
    This function receives label box coordinates from detect_label_box() and 
    applies OCR to extract text from those specific regions.
    
    Args:
        img: Input image (BGR format from OpenCV)
        label_roi: List of label boxes from detect_label_box().
                   Each dict contains: x, y, width, height, area, aspect_ratio
        language: Languages for OCR detection. Default is ['es', 'en'].
        blur_label: Gaussian blur kernel size. Default is (11, 11).
        verbose: Print debug information. Default is False.
        use_gpu: Try to use GPU if available. Default is True.
    
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
    
    # ========== AUTO-DETECT GPU AVAILABILITY ==========
    gpu_available = False
    if use_gpu:
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            if verbose and gpu_available:
                print(f"GPU detected: {torch.cuda.get_device_name(0)}")
                print(f"CUDA version: {torch.version.cuda}")
        except ImportError:
            if verbose:
                print("PyTorch not found. Install with: pip install torch torchvision")
    
    # Initialize OCR reader with GPU if available
    try:
        if language:
            reader = easyocr.Reader(language, gpu=gpu_available)
        else:
            reader = easyocr.Reader(['en'], gpu=gpu_available)
        
        if verbose:
            print(f"Using {'GPU' if gpu_available else 'CPU'} for OCR")
    
    except Exception as e:
        if verbose:
            print(f"Error initializing OCR reader: {e}")
            print("Falling back to CPU mode")
        reader = easyocr.Reader(['en'], gpu=False)
    # ==================================================
    
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


def pdf_to_img(pdf_path: str, dpi: int = 300, output_dir: Optional[str] = None, 
               n_threads: Optional[int] = None, output_message: bool = True,
               qr_label: bool = False) -> List[str]:
    """
    Converts a PDF file (or all PDFs in a folder) to JPEG images (one per page).
    
    Args:
        pdf_path: Path to the input PDF file or folder containing PDF files.
        dpi: Conversion resolution (dots per inch).
        output_dir: Directory to save the images. If None, creates 'images_from_pdf' in the same folder as the PDF.
        n_threads: Number of threads for parallel processing. If None, uses 1 thread.
        output_message: Whether to print progress messages.
        qr_label: If True, detects QR codes and renames images with QR text. Falls back to default naming if no QR detected.
    
    Returns:
        List of paths to the generated and renamed image files.
    
    Raises:
        ValueError: If the input file is not a valid PDF or folder contains no PDFs.
        RuntimeError: If the conversion process fails.
    """
    # Check if path is a directory or file
    if os.path.isdir(pdf_path):
        # Find all PDF files in the directory
        pdf_files = [os.path.join(pdf_path, f) for f in os.listdir(pdf_path) 
                     if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(pdf_path, f))]
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in directory: {pdf_path}")
        
        if output_message:
            print(f"Extracting images may take a few minutes... ⋆✧｡٩(ˊᗜˋ )و✧*｡")
            print(f"Processing {len(pdf_files)} PDF files...")
        
        # Process all PDFs without individual messages
        all_saved_paths = []
        for pdf_file in pdf_files:
            paths = pdf_to_img(pdf_file, dpi=dpi, output_dir=output_dir, 
                             n_threads=n_threads, output_message=False, qr_label=qr_label)
            all_saved_paths.extend(paths)
        
        # Print final summary
        if output_message:
            final_output_dir = output_dir if output_dir else os.path.join(pdf_path, 'images_from_pdf')
            print(f"{len(all_saved_paths)} images saved in: {final_output_dir}")
        
        return all_saved_paths
    
    # Input validation for single file
    if not os.path.isfile(pdf_path):
        raise ValueError(f"File not found: {pdf_path}")
    
    if not pdf_path.lower().endswith('.pdf'):
        raise ValueError("Input file must be a PDF (.pdf extension)")
    
    # Set up output paths
    pdf_dir = os.path.dirname(pdf_path)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    if output_dir is None:
        output_dir = os.path.join(pdf_dir, 'images_from_pdf')
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        if output_message:
            print("Extracting images may take a few minutes... ⋆✧｡٩(ˊᗜˋ )و✧*｡")
        
        # Convert PDF to images using PyMuPDF
        doc = fitz.open(pdf_path)
        
        # Calculate zoom factor from DPI (72 is the base DPI in PDF)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        
        # Save images
        saved_paths = []
        used_names = {}  # Track used names to avoid duplicates
        
        for i in range(len(doc)):
            page = doc[i]
            pix = page.get_pixmap(matrix=mat)
            
            # Default image name
            img_name = f"{pdf_name}_page{i+1}.jpg"
            output_path = os.path.join(output_dir, img_name)
            
            # Save with default name first
            pix.save(output_path)
            
            # If qr_label is True, try to detect QR and rename
            if qr_label:
                qr_text, _ = detect_qr(img_path=output_path)
                
                # Debug print
                if output_message:
                    print(f"Page {i+1} - QR detected: {qr_text}")
                
                # If QR detected and has valid text
                if qr_text and qr_text != 'No QR code detected':
                    # Sanitize QR text for filename
                    sanitized_name = _sanitize_filename(qr_text)
                    
                    if output_message:
                        print(f"  Sanitized name: {sanitized_name}")
                    
                    # Handle duplicate names
                    if sanitized_name in used_names:
                        used_names[sanitized_name] += 1
                        final_name = f"{sanitized_name}_{used_names[sanitized_name]}.jpg"
                    else:
                        used_names[sanitized_name] = 0
                        final_name = f"{sanitized_name}.jpg"
                    
                    # Rename file
                    new_path = os.path.join(output_dir, final_name)
                    
                    if output_message:
                        print(f"  Renaming: {img_name} -> {final_name}")
                    
                    os.rename(output_path, new_path)
                    output_path = new_path
            
            saved_paths.append(output_path)
        
        doc.close()
        
        if output_message and not qr_label:
            print(f"{len(saved_paths)} images saved in: {output_dir}")
        
        return saved_paths
    
    except Exception as e:
        error_msg = f"PDF conversion error: {str(e)}"
        if output_message:
            print(error_msg)
        raise RuntimeError(error_msg) from e


def _sanitize_filename(text: str, max_length: int = 100) -> str:
    """
    Sanitizes text to be a valid filename.
    
    Args:
        text: Text to sanitize
        max_length: Maximum length for the filename
    
    Returns:
        Sanitized filename (without extension)
    """
    # Remove or replace invalid characters for filenames
    # Keep alphanumeric, spaces, hyphens, and underscores
    sanitized = re.sub(r'[^\w\s-]', '_', text)
    
    # Replace multiple spaces/underscores with single underscore
    sanitized = re.sub(r'[\s_]+', '_', sanitized)
    
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    
    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    # If empty after sanitization, return default
    if not sanitized:
        sanitized = "unnamed"
    
    return sanitized


def detect_qr(img_path: Optional[str] = None, img: Optional[np.ndarray] = None) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """
    Detects QR codes in an image
    Args:
        img_path: Path to the image (optional if img is provided)
        img: Image as numpy array (optional if img_path is provided)
    Returns:
        Tuple with (qr_text, image_with_rectangle)
    """
    # Valid extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}
    
    # Validate that at least one argument is present
    if img_path is None and img is None:
        raise ValueError("You must provide either img_path or img")
    
    # If path is provided, validate and load image
    if img_path:
        # Validate file exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image does not exist: {img_path}")
        
        # Validate extension
        file_ext = os.path.splitext(img_path)[1].lower()
        if file_ext not in valid_extensions:
            raise ValueError(f"Invalid format. Use: {', '.join(sorted(valid_extensions))}")
        
        # Load image with OpenCV (already in BGR format)
        img = cv2.imread(img_path)
        
        # Validate image loaded correctly
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
    
    
    # Validate it's a valid image
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a numpy array")
    
    # Decode QR codes
    decoded_objects = decode(img)
    
    # Initialize text variable
    qr_text = None
    
    # Extract text and draw rectangle for each QR found
    for obj in decoded_objects:
        # Get the bounding box coordinates of the QR code
        x, y, w, h = obj.rect
        
        # Draw a green rectangle around the QR code
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Extract text (take the first QR found)
        if qr_text is None:
            qr_text = obj.data.decode('utf-8')
    
    # If no QR was found
    if qr_text is not None:
        # Take only the first word (before first space)
        qr_text = qr_text.split()[0] if qr_text.split() else qr_text
    
    return qr_text, img




#################### New functions for pixel/cm estimation ##############################
#### Version: Nov/2025

#############################################
## Detect size reference (ROI) using YOLOv8
#############################################

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
    
    ## 1. Default method: Detect ROIs with YOLOv8
    
    # Load the model
    try:
        model = YOLO(model_path)
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
        
    img = img.copy()
        
    # Extracting image dimensions
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Yolo detection
    results = model(img, conf=0.1, iou=iou_threshold, verbose=False)

    box_detected = False

    # Empty list to store all detected circles and ROI boxes
    all_circles = []
    rois_debug = []
    roi_boxes = []  # NEW: Store ROI bounding boxes

    for r in results:
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            print("No size reference detected.")
            continue
        
        filtered_boxes = []
        low_conf_boxes = []

        for box in boxes:
            conf = float(box.conf[0].cpu().numpy())
            if conf >= confidence_threshold:
                filtered_boxes.append(box)
            else:
                low_conf_boxes.append((box, conf))

        if len(low_conf_boxes) > 0:
            print(f"Filtered out {len(low_conf_boxes)} box(es) below the confidence threshold: {confidence_threshold}")
            for box_idx, (box, conf_value) in enumerate(low_conf_boxes, 1):
                print(f"  - Box {box_idx}: confidence = {conf_value:.3f}")                

        boxes = filtered_boxes

        if len(boxes) == 0:
            print("No size reference boxes above confidence threshold. Try adjusting the threshold or image quality.")
            continue

        box_detected = True

        if yolo_verbose:
            print(f"Processing {len(boxes)} high-confidence (≥{confidence_threshold}) size reference box(es):")
            
        for i, box in enumerate(boxes):

            # Bounding box de YOLO
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w-1))
            y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h-1))
            confidence = float(box.conf[0].cpu().numpy())

            # Increase the space between the box content and its ROI (15% padding)
            padx = int(0.15 * (x2 - x1 + 1))
            pady = int(0.05 * (y2 - y1 + 1))

            roi_x1 = max(0, x1 - padx)
            roi_y1 = max(0, y1 - pady)
            roi_x2 = min(w, x2 + padx)
            roi_y2 = min(h, y2 + pady)
            
            # NEW: Store ROI coordinates
            roi_boxes.append((roi_x1, roi_y1, roi_x2, roi_y2))

            # Extract ROI
            roi_gray = gray[roi_y1:roi_y2, roi_x1:roi_x2]

            if yolo_verbose:
                roi_height, roi_width = roi_gray.shape[:2]
                print(f"  - Ref {i+1}: {roi_width}×{roi_height} px, conf: {confidence:.3f}")

            if roi_gray.size == 0:
                print("Empty ROI, skipping...")
                continue
            
            # Draw the bounding box on the original image
            cv2.rectangle(img, (roi_x1, roi_y1), (roi_x2, roi_y2), (200, 100, 0), 2)
            cv2.putText(
                img,
                f"Ref {i+1} ({confidence:.2f})",
                (roi_x1 + 5, max(roi_y1 - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, (200, 100, 0), 3, cv2.LINE_AA
            )

            # Find circles in the ROI
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

            # Convert circle coordinates to global image coordinates
            for (cx_roi, cy_roi, radius) in circles:
                cx_global = cx_roi + roi_x1
                cy_global = cy_roi + roi_y1

                diameter = 2*radius
                
                cv2.circle(img, (cx_global, cy_global), radius, (0, 0, 255), 5)
                cv2.circle(img, (cx_global, cy_global), 10, (255, 0, 0), -1)
                
                # Draw diameter line
                end_x_radius = cx_global + radius
                end_y_radius = cy_global 

                cv2.line(img, (cx_global - radius, cy_global), 
                         (end_x_radius, end_y_radius), (0, 255, 0), 3)
                
                # Draw diameter text above the line, centered
                text = f"{diameter}px"

                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 4)[0]
                text_width = text_size[0]
                
                # Get center of the line
                start_x = cx_global - radius
                center_x = (start_x + end_x_radius) // 2
                center_y = end_y_radius
                
                # Adjust text position to be centered above the line
                text_x = center_x - (text_width // 2)
                text_y = center_y - 20
                
                cv2.putText(img, text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), 4)

                # Save circle results
                all_circles.append((cx_global, cy_global, diameter))

        # Confirm detection
        if yolo_verbose:
            print(f"\nTotal circles detected: {len(all_circles)}")
            
        if not box_detected:
            print("No size reference box detected in the image by YOLO. Try adjusting confidence threshold or image quality.")
        elif len(all_circles) == 0:
            print("No circles detected within the detected size reference boxes. Try adjusting thresholds or check image quality.")       

        if plot:
            plt.figure(figsize=plot_size)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')

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

    if return_roi_coords:
        return all_circles, img, roi_boxes
    else:
        return all_circles, img


def find_size_ref_circles(roi_gray, return_debug=False, ref_circularity=0.7):

    circles = []

    # Pre processing image
    blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)

    # Image binarization
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Close/Open morphology
    k = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=1)

    # Get contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtering contours by area and circularity
    h, w = roi_gray.shape[:2]
    min_area = max(50, int(0.01 * h * w))

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

    # Draw contours and circles on a color version of the ROI
    overlay = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)

    for (cx, cy, r) in circles:
        cv2.circle(overlay, (cx, cy), r, (0, 0, 255), 5)
        cv2.circle(overlay, (cx, cy), 5, (0, 0, 255), -1)

    if return_debug:
        return circles, {"roi_gray": roi_gray, "binary": binary, "overlay": overlay}
    else:
        return circles


def diameter_px_per_cm(all_circles: List[Tuple[int, int, int]], verbose: bool = False, diameter_cm: float = 2.5, std_threshold: float = 2):

    if not all_circles:
        raise ValueError('No circles provided. The circles list is empty or not specified.')

    all_diameters = []
    for d in all_circles:
        diameter = d[2]
        all_diameters.append(diameter)

    circles_mean = statistics.mean(all_diameters)
    std_dev = statistics.stdev(all_diameters)

    lower_limit = circles_mean - std_threshold * std_dev
    upper_limit = circles_mean + std_threshold * std_dev

    outliers = [d for d in all_diameters if d < lower_limit or d > upper_limit]
    filtered = [d for d in all_diameters if d >= lower_limit and d <= upper_limit]

    if not filtered:  
        if verbose:
            print("Warning: All circles were filtered as outliers. Using full dataset for calculation.") 
        filtered = all_diameters
    
    # Calculate pixel per centimeter density
    px_cm_density = statistics.mean(filtered) / diameter_cm

    if verbose:
        #print(f"Circle count: {len(all_diameters)}:")
        #print(f"  - Mean diameter: {circles_mean:.2f} px; Standard deviation: {std_dev:.2f} px.")
        print(f"  - Diameter range (mean ± {std_threshold}): {lower_limit:.2f} px to {upper_limit:.2f} px")
        #print(f"  - Outliers removed (std > 2): {outliers}")
        print(f"  - Filtered diameters count (std < 2): {len(filtered)}; mean diameter: {statistics.mean(filtered):.2f} px.")
        #print(f"  - Filtered mean diameter: {statistics.mean(filtered):.2f} px.")
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
                     verbose: Optional[bool] = False, plot: Optional[bool] = False) -> List[Dict]:  # ← Cambiar aquí
    """
    Detects text boxes in an image
    Args:
        imagen_path: Path to the image
        img: Numpy array with the image
        verbose: Print debug information
    Returns:
        List of boxes, each box is a dict with keys: x, y, width, height, area, aspect_ratio
    """
    # Read image
    if imagen_path is not None:
        img = cv2.imread(imagen_path)
    elif img is not None:
        img = img.copy()
    else:
        raise ValueError("Either imagen_path or img must be provided.") 
    
    if img is None:
        raise ValueError(f"Could not load image: {imagen_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Binarize
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # List to store box coordinates
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = w / h
        
        # Filter by size and aspect ratio
        if area > 5000 and 2 < aspect_ratio < 6:  # Horizontal rectangle
            # Save coordinates
            box_info = {
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'area': area,
                'aspect_ratio': aspect_ratio
            }
            boxes.append(box_info)

            if plot:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
    if plot:
        plt.figure(figsize = (8,8))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    if verbose:
        print(f"\nTotal boxes found: {len(boxes)}")
        print("\nAll coordinates:")
        for i, box in enumerate(boxes, 1):
            print(f"Box {i}: {box}")
        
        # Access individual boxes
        if boxes:
            first_box = boxes[0]
            print(f"\nFirst box X: {first_box['x']}")
            print(f"First box Y: {first_box['y']}")
    
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
    """
    Remove label and reference ROIs from mask by setting them to black (0).
    
    Args:
        mask: Input mask to clean
        
    Returns:
        Cleaned mask with ROIs removed
    """
    mask_clean = mask.copy()
    
    # Remove label ROIs (rectangles)
    if hasattr(self, 'label_roi') and self.label_roi:
        for box in self.label_roi:
            x, y = box['x'], box['y']
            w, h = box['width'], box['height']
            # Set rectangle area to black (0)
            mask_clean[y:y+h, x:x+w] = 0
    
    # Remove reference ROI (polygon)
    if hasattr(self, 'ref_roi') and self.ref_roi:
        for roi in self.ref_roi:
            # Fill polygon with black (0)
            cv2.fillPoly(mask_clean, [roi], 0)
    
    return mask_clean