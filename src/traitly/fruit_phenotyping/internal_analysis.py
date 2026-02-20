# traitly/fruit_phenotyping/internal_analysis.py

# ============================================================================
# STANDARD LIBRARY
# ============================================================================
import os
from io import StringIO
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, List, Dict, Tuple, Any
import warnings
from datetime import datetime
from pathlib import Path
import json

# ============================================================================
# THIRD-PARTY LIBRARIES
# ============================================================================
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import psutil

# ============================================================================
# INTERNAL IMPORTS
# ============================================================================
from .mask import (create_mask, 
                   find_fruits, 
                   apply_contrast, 
                   create_mask_locules, 
                   generate_scatter_plot)

from .fruit_config import analyze_fruits_morphology
from ..utils.basic_functions import (load_img,
                                      detect_qr,
                                      detect_label_box_yolo,
                                      detect_label_box,
                                      detect_label_text,
                                      px_cm_density,
                                      detect_img_name,
                                      plot_img,
                                      annotate_all_fruits)

from traitly import __version__
from ..utils.constants import valid_extensions
from .results_image import ResultsImage
from .color_analysis import (get_single_fruit_masks, 
                             analyze_all_fruits_color, 
                             get_fruit_color_histograms)
from .analysis_parameters import AnalysisMetadata

##########################################################################################
# Ignore warnings from torch 
##########################################################################################

warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', message='Using CPU')

##########################################################################################
# Worker function for parallel processing 
##########################################################################################

def _process_image_worker(img_path: str,
                          config: Dict, analyze_morphology: bool, analyze_color: bool):
    t0 = time.time()
    try:
        analyzer = FruitInternalAnalyzer(img_path)
        analyzer.load_image(plot=False)
        df_morphology, df_color, error_dict, n_fruits, annotated_img = analyzer.process_single_file(
            config=config,
            json_path=None,
            analyze_morphology=analyze_morphology,
            analyze_color=analyze_color,
            save_image=False
        )
        elapsed  = time.time() - t0
        filename = os.path.basename(img_path)
        return df_morphology, df_color, error_dict, n_fruits, annotated_img, filename, elapsed
    except Exception as e:
        return None, None, {
            'filename': os.path.basename(img_path),
            'status':   f'Error: {str(e)}'
        }, 0, None, os.path.basename(img_path), time.time() - t0


##########################################################################################
# Initializig class
##########################################################################################

class FruitInternalAnalyzer:
    """Class for analyzing fruit images with morphological measurements."""
    
    def __init__(self, image_path: str):
        """
        Initialize the image analyzer.
        
        Args:
            image_path: Path to image file or directory
        """
        ## Verify image path exists
        # Assign the path first
        self.img_path = image_path
        
        # Then verify if it was provided and exists
        if self.img_path is not None:
            if not os.path.exists(self.img_path):
                raise FileNotFoundError(
                    f"The path does not exist: {self.img_path}\n"
                    f"Verify that the file exists and the path is correct."
                )         

        # load_img
        self.is_directory = os.path.isdir(image_path)
        self.img = None
        self.img_copy = None
        self.img_shape = None
        self.img_rgb = None
        self.img_hsv = None
        self.l_transformed = None

        # setup_measurements
        self.ref_roi = None
        self.label_roi = None
        self.checker_roi = None
        self.label_text = None
        self.label_id = None
        self.img_name = None
        
        # create_mask
        self.mask_fruit = None
        self.mask_locules = None
        self.contours = None
        self.fruit_locule_map = None
        
        # analyze fruits
        self.px_per_cm = None  
        self.results = None

        # save metadata
        self.parameters = AnalysisMetadata() 
        self.is_metadata_saved = True


    ##########################################################################################
    ## Load and display an image 
    ##########################################################################################
    def load_image(self, 
                   plot: Optional[bool] = True, 
                   plot_size: Optional[Tuple[int, int]] = (5, 5), 
                   ) -> None:
        """
        Load and display the image.
        
        Args:
            plot: Display the image (Optional, default = True)
            plot_size: Figure size for plotting (default: (5,5))

        """

        if self.img_path is None:
            raise ValueError(  
                f"No image loaded."
                "Run FruitInternalAnalyzer('path/to/your/image.jpg') first."
            )
        
        path = Path(self.img_path)
        if path.suffix.lower() not in valid_extensions:
            raise ValueError(f"No valid image format: '{path.suffix.lower()}' -> "
                             f"Supported formats are: {valid_extensions}")
        
        self.img = load_img(self.img_path, plot = plot, plot_size = plot_size)

        if self.img is None:
            raise ValueError(f"Failed to load image: {self.img_path}."
                             "The file may be corrupted or not in a supported format.")

        self.img_shape = self.img.shape[:2]
        self.img_copy = self.img.copy()
        self.img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        self.img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.img_name = detect_img_name(self.img_path)

        return None

    ##########################################################################################
    ## Detect label text and ROI 
    ##########################################################################################
    def setup_label(
        self,
        verbose: bool = True,
        detect_label: bool = False,
        language_label: List[str] = ["es", "en"],
        blur_label: Tuple[int, int] = (11, 11),
        gpu: bool = True,
        skip_qr: bool = False,
        skip_label_detection: bool = False,
    ) -> None:
        """
        Detect:
            - QR (Optional skip_qr, default = False)
            - Label ROI (YOLO -> shape threshold) (Optional skip_label_detection, default = False)
            - Label text by OCR (only if there is a label ROI y no QR text was detected)

        Return:
            self.label_text
            self.label_roi
        """


        if verbose:
            print("\n" + "=" * 55)
            print("★ LABEL DETECTION:")
            print("=" * 55)

        if not detect_label:
            # Try to detect label roi
            self.label_roi = detect_label_box_yolo(img=self.img, plot=False, conf=0.4)
            if self.label_roi is None or len(self.label_roi) == 0:
                self.label_roi = detect_label_box(img=self.img, verbose=False, plot=False)
            
            # No text detection
            self.label_text = "No label detected"
            
            if verbose:
                if self.label_roi and len(self.label_roi) > 0:
                    print("> Label detection: SKIPPED (detect_label=False)")
            
            return None

        # QR (optional)
        qr_text = None
        if not skip_qr:
            qr_start = time.time()
            qr_text, self.img = detect_qr(img=self.img)
            if verbose and qr_text is not None and "No QR" not in str(qr_text):
                print(f"    > QR Code detected: {qr_text} ({time.time()-qr_start:.2f}s)")
        else:
            if verbose:
                print("    > QR detection: SKIPPED")

        # ROI + OCR (optional)
        if not skip_label_detection:
            label_start = time.time()

            # 1. YOLO
            self.label_roi = detect_label_box_yolo(img=self.img, plot=False, conf=0.4)

            # 2. Detect with ROI + OCR
            if self.label_roi is None or len(self.label_roi) == 0:
                self.label_roi = detect_label_box(img=self.img, verbose=False, plot=False)

            # Keeping for debugging and testing running time only
            #if verbose:
            #    print(f"    > Label text ROI detection: {time.time()-label_start:.2f}s")

            # OCR if only ROI detected but no QR
            if self.label_roi and len(self.label_roi) > 0 and qr_text is None:
                ocr_start = time.time()

                # silent qr output
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                try:
                    self.label_text = detect_label_text(
                        img=self.img,
                        label_roi=self.label_roi,
                        language=language_label,
                        blur_label=blur_label,
                        verbose=False,
                        gpu=gpu
                    )
                finally:
                    sys.stdout = old_stdout

                if verbose and self.label_text:
                    print(f"    > Label text detected: {self.label_text}   (OCR: {time.time()-ocr_start:.2f}s)")
            else:
                # If qr detected, use this as label_text
                if qr_text is not None:
                    self.label_text = qr_text
                else:
                    self.label_text = "No label detected"
                    
                    
        else:
            self.label_roi = None
            self.label_text = qr_text if qr_text is not None else "No label detected"

        if self.label_text is None:
            self.label_text = "No label detected"
        
        if self.label_text == 'No label detected':
            if verbose:
                print("> No label detected.")
                print("     - Use skip_label_detection=True to disable label detection.")
            
        return None
    
    ##########################################################################################
    ## Detect size reference 
    ##########################################################################################

    def setup_calibration(
        self,
        verbose: bool = True,
        confidence: float = 0.6,
        font_size: int = 3,
        width_cm: Optional[float] = None,
        length_cm: Optional[float] = None,
        diameter_cm: Optional[float] = None,
        fast_calibration: bool = False,
    ) -> None:
        """
        Detect and save:
            self.px_per_cm
            self.img_annotated
            self.ref_roi

        When fast_calibration=True, use width_cm y length_cm instead of looking
        for the size reference box with YOLO
        """

        h, w, _ = self.img.shape

        if verbose:
            print("\n" + "=" * 55)
            print("★ REFERENCE SIZE:")
            print("=" * 55)

        # Default diameter
        using_default_diameter = False
        if diameter_cm is None:
            diameter_cm = 2.5
            using_default_diameter = True

        #calibration_start = time.time()

        if fast_calibration and width_cm and length_cm:
            # Fast method: use physical dimensions 
            self.px_per_cm = np.sqrt((w * h) / (width_cm * length_cm))
            #self.img_annotated = self.img.copy()
            self.ref_roi = None

            if verbose:
                print("> Size reference detection: SKIPPED.")
        else:
            
            self.px_per_cm, self.img_copy, self.ref_roi = px_cm_density(
                self.img_copy,
                confidence_threshold=confidence,
                plot=False,
                font_size=font_size,
                verbose=verbose,
                width_cm=width_cm,
                length_cm=length_cm,
                diameter_cm=diameter_cm,
                return_coordinates=True,
            )

        if self.ref_roi is not None:
            if verbose and using_default_diameter:
                print("\nNote: Default reference diameter (2.5 cm) applied.")
                print("        Specify diameter_cm to override this value.")
        else:
            if width_cm is not None and length_cm is not None:
                if verbose:
                    print("> Using provided physical dimensions:")
                    print(f"    - width_cm:  {width_cm} cm")
                    print(f"    - length_cm: {length_cm} cm")
                    print(f"\n        . ݁₊ ⊹ . ݁ ⟡ ݁ px/cm density: {self.px_per_cm:.2f} ⟡ ݁ . ⊹ ₊ ݁.")
                    
            else:
                if verbose:
                    print("> Using provided physical dimensions:")
                    print(f"    - width_cm:  None")
                    print(f"    - length_cm: None")             
                    print("\n        . ݁₊ ⊹ . ݁ ⟡ ݁ Measurements will be returned in PIXEL units ⟡ ݁ . ⊹ ₊ ݁.")

        return None

    ##########################################################################################
    # Wrapper: label + size reference detection 
    ##########################################################################################
    def setup_measurements(
        self,
        plot_reference: bool = False,
        plot_color_checker: bool = False,
        font_size: int = 3,
        confidence: float = 0.6,
        detect_label: bool = False,
        verbose: bool = True,
        plot_size: Tuple[int, int] = (5, 5),
        language_label: List[str] = ["es", "en"],
        width_cm: Optional[float] = None,
        length_cm: Optional[float] = None,
        diameter_cm: Optional[float] = None,
        gpu: bool = False,
        skip_qr: bool = False,
        fast_calibration: bool = False,
        detect_color_checker: bool = False,
        scale_factor: float = 0.5
    ) -> None:
        """
        Detect label text and calculate px_per_cm using:
            - setup_label()
            - setup_calibration()

        Saves: 
                - self.label_text
                - self.label_roi
                - self.ref_roi
                - self.px_per_cm
        Returns:
            None
        """
        if self.img is None:
            raise ValueError("No image loaded. Run load_img() first.")
        if scale_factor > 1 or scale_factor < 0.1 :
            raise ValueError(
                    f"scale_factor: {scale_factor} must be > 0.1 and ≤ 1."
                )
        metadata = self.is_metadata_saved

        if metadata:
            self.parameters.setup_measurements_params = {
            'width_cm': width_cm,
            'length_cm': length_cm,
            'diameter_cm': diameter_cm,
            'fast_calibration': fast_calibration,
            'skip_qr': skip_qr,
            'detect_label': detect_label,
            'language_label': language_label,
            'confidence': confidence,
            'gpu': gpu,
            'font_size': font_size,
            'detect_color_checker': detect_color_checker,
            'scale_factor': scale_factor
            }


        # 1) label
        self.setup_label(
            detect_label = detect_label,
            verbose = verbose,
            language_label=language_label,
            gpu=gpu,
            skip_qr=skip_qr
        )

        # 2) calibration
        self.setup_calibration(
            verbose = verbose,
            confidence=confidence,
            font_size=font_size,
            width_cm=width_cm,
            length_cm=length_cm,
            diameter_cm=diameter_cm,
            fast_calibration=fast_calibration,
        )

        if detect_color_checker:
            self.detect_color_checker(verbose = verbose,
                                      plot = plot_color_checker,
                                      scale_factor = scale_factor)

        # Plot 
        if plot_reference and self.ref_roi:
            
            h_img, w_img = self.img.shape[:2]
            margin = 5 # px
            
            # Determine orientation and plot size 
            is_portrait = h_img > w_img
            n = len(self.ref_roi)
            
            if is_portrait:
                nrows, ncols = n,1
                figsize = (plot_size[0], plot_size[1] * n)
            else:
                nrows, ncols = 1, n
                figsize = (plot_size[0], plot_size[1]) 

            plt.figure(figsize=figsize)

            # Plot all the reference boxes detected
            for i, ref_contour in enumerate(self.ref_roi, 1):

                x, y, w, h = cv2.boundingRect(ref_contour)
                # Add the margin
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(w_img, x + w + margin)
                y2 = min(h_img, y + h + margin)

                roi_ref_img = self.img_copy[y1:y2, x1:x2]
                
                plt.subplot(nrows, ncols, i)
                plt.imshow(cv2.cvtColor(roi_ref_img, cv2.COLOR_BGR2RGB))
                plt.axis("off")
                plt.title(f"Ref {i}")

            plt.tight_layout()
            plt.show()

        return None
    
    ##########################################################################################
    #                                     MASKS
        

    ##########################################################################################
    # OPTIONAL : Create a scatterplot to visualize pixel colors (HSV space) 
    ##########################################################################################

    def generate_color_scatterplot(self,
                                   sample_size: int = 10000, 
                                   plot_size: Tuple[int,int] = (18,5)):
        if self.img is None:
            raise ValueError("No image loaded. Run load_img() first.")
        
        if isinstance(sample_size, float) or sample_size < 1: 
            raise ValueError(f"Invalid sample_size: {sample_size}. Sample size must be an integer > 0.")

        generate_scatter_plot(img_hsv = self.img_hsv,
                              img_rgb = self.img_rgb,
                              sample_size = sample_size,
                              plot_size = plot_size)
        
        return None
    
    ##########################################################################################
    # Create a binary mask for fruits
    ##########################################################################################
    def generate_fruit_mask(self, plot: bool = True, 
                    plot_size: Tuple[int, int] = (5, 5), 
                    stamp: bool = False,  
                    lower_hsv: Optional[List[int]] = None,
                    upper_hsv: Optional[List[int]] = None, 
                    n_iteration: int = 1,
                    kernel_blur: Optional[int] = None, 
                    kernel_open: Optional[int] = None,
                    kernel_close: Optional[int] = None, 
                    canny_min: Optional[int] = None,
                    canny_max: Optional[int] = None,
                    remove_roi: bool = True, 
                    roi_expansion: int = 10,
                    background_color: Optional[str] = None,
                    fill_holes: bool = False,
                    apply_convex_hull: bool = False,
                    detect_color_checker: bool = False,
                    erosion_px: int = 3) -> None:
        """
        Create a mask for fruit detection and segmentation.
        
        This method generates a binary mask to identify fruits in the image with support
        for stamp inversion, locule detection, and automatic ROI removal.
        
        Args:
            stamp: Set to True if image has inverted colors (black background). Default is False.
            plot: Whether to display the generated mask. Default is False.
            plot_size: Figure size for plotting (width, height). Default is (5, 5).
            remove_roi: Automatically remove label and reference regions from the mask. 
                Default is True.
        
        Returns:
            None: self.mask_fruit with the generated binary mask.
        """

        if self.img is None:
            raise ValueError("No image loaded. Run load_img() first.")
        
        
        
        metadata = self.is_metadata_saved
        if metadata:
            self.parameters.generate_fruit_mask_params = {
                'stamp': stamp,
                'lower_hsv': lower_hsv,
                'upper_hsv': upper_hsv,
                'background_color': background_color,
                'kernel_blur': kernel_blur,
                'kernel_open': kernel_open,
                'kernel_close': kernel_close,
                'canny_min': canny_min,
                'canny_max': canny_max,
                'n_iteration': n_iteration,
                'fill_holes': fill_holes,
                'apply_convex_hull': apply_convex_hull,
                'roi_expansion': roi_expansion,
                'remove_roi': remove_roi,
                'fill_holes': fill_holes,
                'apply_convex_hull': apply_convex_hull,
                'detect_color_checker': detect_color_checker,
                'erosion_px': erosion_px
            }

        if stamp:
            img = cv2.bitwise_not(self.img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        else:
            img = self.img_hsv

        # Create base mask - only calculate once
        self.mask_fruit = create_mask(
            img_hsv = img,
            n_iteration=n_iteration,
            plot=False,
            plot_size=plot_size,
            kernel_blur = kernel_blur,
            kernel_open = kernel_open,
            kernel_close = kernel_close,
            canny_max=canny_max,
            canny_min=canny_min,
            lower_hsv=lower_hsv,
            upper_hsv=upper_hsv,
            background_color = background_color,
            fill_holes=fill_holes,
            apply_convex_hull = apply_convex_hull
        )
        
        # Deleting label and reference squares from mask
        if remove_roi:
            # Create a same size black mask 
            mask_rois = np.zeros_like(self.mask_fruit)
            
            # Draw white rectangles over the label roi
            if hasattr(self, 'label_roi') and self.label_roi:
                for box in self.label_roi:
                    x, y = box['x'], box['y']
                    w, h = box['width'], box['height']
                    
                    # Expand the rectangle
                    x_expanded = max(0, x - roi_expansion)
                    y_expanded = max(0, y - roi_expansion)
                    w_expanded = w + 2 * roi_expansion
                    h_expanded = h + 2 * roi_expansion
                    
                    # Draw it
                    cv2.rectangle(mask_rois, 
                                (x_expanded, y_expanded), 
                                (x_expanded + w_expanded, y_expanded + h_expanded), 
                                255, -1)
            
            # Draw white rectangles over the reference roi
            if hasattr(self, 'ref_roi') and self.ref_roi:
                for roi in self.ref_roi:
                    # Draw a polygon over it 
                    cv2.fillPoly(mask_rois, [roi], 255)


            if detect_color_checker and hasattr(self, 'checker_coords') and self.checker_coords is not None:
                if len(self.checker_coords) == 4:
                    x, y, w, h = self.checker_coords
                    
                    # Expand the rectangle 
                    x_expanded = max(0, x - roi_expansion)
                    y_expanded = max(0, y - roi_expansion)
                    w_expanded = w + 2 * roi_expansion
                    h_expanded = h + 2 * roi_expansion
                    
                    # Ensure coords are not exceeding image bounds
                    img_h, img_w = self.mask_fruit.shape[:2]
                    x_expanded = max(0, min(x_expanded, img_w))
                    y_expanded = max(0, min(y_expanded, img_h))
                    w_expanded = min(w_expanded, img_w - x_expanded)
                    h_expanded = min(h_expanded, img_h - y_expanded)
                    
                    # Draw the rectangle
                    cv2.rectangle(mask_rois,
                                (x_expanded, y_expanded),
                                (x_expanded + w_expanded, y_expanded + h_expanded),
                                255, -1)
                
            # Dilate the roi mask if needed
            if roi_expansion > 0:
                kernel_expand = np.ones((roi_expansion, roi_expansion), np.uint8)
                mask_rois = cv2.dilate(mask_rois, kernel_expand, iterations=1)
            
            
            # Remove label and reference from the original mask
            self.mask_fruit = cv2.bitwise_and(self.mask_fruit, cv2.bitwise_not(mask_rois))

            # Apply erosion 
            if erosion_px > 0:
                kernel       = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (erosion_px * 2 + 1, erosion_px * 2 + 1)
                )
                self.mask_fruit  = cv2.erode(self.mask_fruit .copy(), kernel, iterations=1)

        if plot:
            plt.figure(figsize=plot_size)
            plt.imshow(self.mask_fruit, cmap = 'gray')
            plt.axis('off')
            plt.show()

        return None
    
    ##########################################################################################
    # OPTIONAL: Create locule-fruit contrast
    ##########################################################################################
    def enhance_locule_contrast(self,
                    contrast_method: str = 'gamma',
                    gamma: float = 1.5,
                    gain: float = 5,
                    cutoff: float = 0.5,
                    c: float = 0.5,
                    plot: bool = True,
                    plot_size: Tuple[int, int] = (8, 10),
                    compare_method: bool = False,
                    kernel_blur: int = 1,
                    clip_limit: Optional[int] = None,
                    tile_grid_size: Optional[int] = 12) -> np.ndarray:
        """
        Applies contrast transformation to the L channel of a LAB image.
        
        Args:
            img: Image in BGR format
            contrast_method: Contrast method to apply ('gamma', 'sigmoid', or 'exp')
            gamma: Parameter for gamma_contrast (default: 1.5)
            gain: Parameter for sigmoid_contrast (default: 5)
            cutoff: Parameter for sigmoid_contrast (default: 0.5)
            c: Parameter for exp_transform (default: 0.5)
            plot: If True, displays the result of the selected method
            plot_size: Figure size for plotting (default: (12, 12))
            compare: If True, visually compares all 3 methods (overrides plot)
        
        Returns:
            Transformed L channel (2D numpy array)
        
        Raises:
            TypeError: If input image is not a numpy array
            ValueError: If image format is invalid or contrast_method is unknown
        """
        metadata = self.is_metadata_saved

        if metadata:
            self.parameters.enhance_locule_contrast_params = {
                'contrast_method': contrast_method,
                'gamma': gamma if contrast_method == 'gamma' else None,
                'gain': gain if contrast_method == 'sigmoid' else None,
                'cutoff': cutoff if contrast_method == 'sigmoid' else None,
                'c': c if contrast_method == 'exponential' else None,
                'kernel_blur': kernel_blur,
                'clip_limit': clip_limit,
                'tile_grid_size': tile_grid_size if clip_limit else None
            }

        self.l_transformed = apply_contrast(img = self.img, 
                                       contrast_method = contrast_method,
                                       gamma = gamma,
                                       gain = gain,
                                       cutoff = cutoff, 
                                       c = c,
                                       plot = plot, 
                                       plot_size = plot_size,
                                       compare = compare_method,
                                       kernel_blur = kernel_blur,
                                       clip_limit = clip_limit,
                   tile_grid_size = tile_grid_size)
        return None
    
    ##########################################################################################
    # OPTIONAL: Create fruit + locule mask
    ##########################################################################################
    def generate_locule_mask(self, 
                            thresh_min=120,
                            thresh_max=255,
                            kernel_close=None,
                            kernel_open=None,
                            min_fruit_area=5000,
                            invert_locule=False,
                            plot=True,
                            plot_size=(5, 5)):
        """
        Creates and stores a fused mask containing fruits with internal locules.
        
        Args:
            thresh_min: Minimum threshold value for binarization (default: 120)
            thresh_max: Maximum threshold value for binarization (default: 255)
            kernel_close: Kernel size for closing operation (optional, None = no closing)
            kernel_open: Kernel size for opening operation (optional, None = no opening)
            min_fruit_size: Minimum area to consider a contour as a fruit (default: 5000)
            invert_locules: If True, inverts locules mask before fusion (default: False)
            plot: If True, displays the masks (default: False)
            plot_size: Figure size for plotting (default: (15, 5))
        
        Returns:
            None (stores fused mask in self.mask_locules)
        """

        if self.mask_fruit is None:
            raise ValueError("No mask available. Run generate_fruit_mask() first.")
        
        if self.l_transformed is None:
            raise ValueError("Locule contrast not initialized. Run enhance_locule_contrast() first "
                             "(use contrast_method = 'None' if no transformation is requiered)")
        
        # Validate that required attributes exist
        if self.l_transformed is None:
            raise ValueError(
                "l_transformed is not available. Please call enhance_locule_contrast() first."
            )
        
        if self.mask_fruit is None:
            raise ValueError(
                "Fruit mask is not available. Please call generate_fruit_mask() first."
            )
        
        metadata = self.is_metadata_saved
        if metadata:
            self.parameters.generate_locule_mask_params = {
                'thresh_min': thresh_min,
                'thresh_max': thresh_max,
                'min_fruit_area': min_fruit_area,
                'kernel_close': kernel_close,
                'kernel_open': kernel_open,
                'invert_locule': invert_locule
                }
        
        self.mask_locules = create_mask_locules(
            l_transformed=self.l_transformed,
            fruit_mask=self.mask_fruit,
            thresh_min=thresh_min,
            thresh_max=thresh_max,
            kernel_close=kernel_close,
            kernel_open=kernel_open,
            min_fruit_size=min_fruit_area,
            invert_locules=invert_locule,
            plot=plot, 
            plot_size=plot_size
        )
        
        return None
    
    ##########################################################################################
    # Detect fruits on the mask
    ##########################################################################################

    def detect_fruits(self, 
                    min_fruit_circularity: float = 0.5, 
                    verbose: bool = True, 
                    min_locule_area: int = 50, 
                    min_locule_per_fruit: int = 1,  
                    min_fruit_area: Optional[int] = None,
                    max_fruit_area: Optional[int] = None,
                    rescale_factor: Optional[float] = None,
                    plot: bool = False,
                    plot_size: Tuple[int,int] = (5,5),
                    contour_color: Tuple[int,int,int] = (0,255,0),
                    contour_thickness: int = 2) -> None:
        """
        Detect fruits and their locules in the mask.

        Args:
            min_circularity: Minimum circularity for fruit detection. Default is 0.5.
            output_message: Whether to show detection results. Default is True.
            min_locule_area: Minimum area for locule detection. Default is 50.
            min_locule_per_fruit: Minimum locules per fruit. Default is 1.
            max_circularity: Maximum circularity for filtering. Default is 1.0.

        """
        
        # Validation - if mask exists, mask_locules should also exist
        if self.mask_fruit is None:
            raise ValueError("No mask available. Run generate_fruit_mask() first.")
        
        metadata = self.is_metadata_saved
        if metadata:
            self.parameters.detect_fruits_params = {
                'min_fruit_area': min_fruit_area,
                'max_fruit_area': max_fruit_area,
                'min_fruit_circularity': min_fruit_circularity,
                'min_locule_area': min_locule_area,
                'min_locule_per_fruit': min_locule_per_fruit,
                'rescale_factor': rescale_factor}
            
        if self.mask_locules is not None:
            mask = self.mask_locules
        else:
            mask = self.mask_fruit

        self.contours, self.fruit_locule_map = find_fruits(
            mask, 
            min_fruit_area = min_fruit_area,
            max_fruit_area = max_fruit_area,
            min_circularity = min_fruit_circularity,
            min_locule_area = min_locule_area,
            min_locules_per_fruit = min_locule_per_fruit,
            rescale_factor = rescale_factor
        )
        
        if self.fruit_locule_map is not None:
            n_fruits_detected = len(self.fruit_locule_map)
        else:
            n_fruits_detected = '0'

        if verbose:
            optional_config = {
                "min_fruit_area": min_fruit_area,
                "max_fruit_area": max_fruit_area,
                "rescale_factor": rescale_factor
            }
            print("\n" + "=" * 37)
            print(f'        . ݁₊ ⊹ . ݁ ⟡ ݁ Detected fruits: {n_fruits_detected} ⟡ ݁ . ⊹ ₊ ݁.')
            print("\n > Parameters used:")
            print(f"        - min_fruit_circularity: {min_fruit_circularity}")
            print(f"        - min_locule_area: {min_locule_area}")
            print(f"        - min_locule_per_fruit: {min_locule_per_fruit}")
            
            for parameter, value in optional_config.items():
                if value is not None:
                    print(f"        - {parameter}: {value}")
            
            print("=" * 37)

        if plot:
            img_copy = self.img_rgb.copy()
            for fruit_id in self.fruit_locule_map:
                contour = self.contours[fruit_id]
                cv2.drawContours(img_copy, [contour], -1, contour_color, contour_thickness)  
                    
            base_fontsize = 6
            fontsize = base_fontsize + (plot_size[0] )
            
            plt.figure(figsize=plot_size)
            plt.imshow(img_copy)
            plt.axis('off')
            plt.title(f'Fruits: {len(self.fruit_locule_map)}', fontsize = fontsize)
            plt.show()

        return None
    
    ##########################################################################################
    # OPTIONAL: Create tissue masks for a single fruit
    ##########################################################################################
    def generate_single_fruit_masks(self,
            fruit_id: Optional[int] = None,
            plot_size: Tuple[int, int] = (7, 5),
            overlay: bool = False,
            overlay_legend: bool = False,
            margin: int = 5,
            only_fruit: bool = False
        ) -> Dict[str, np.ndarray]:

        # Validation
        if self.mask_fruit is None:
            raise ValueError("No mask available. Run generate_fruit_mask() first.")
        if self.contours is None:
            raise ValueError("No contours available. Run detect_fruits() first.")
        if self.fruit_locule_map is None:
            raise ValueError("No fruit-locule mapping available. Run detect_fruits() first.")
        
        if not self.fruit_locule_map:
            raise ValueError("No fruits detected. Make sure detect_fruits() found valid fruit " \
            "contours before calling analyze_color().")
        
        if self.mask_locules is not None:
            mask = self.mask_locules
        else:
            mask = self.mask_fruit

        get_single_fruit_masks(img = self.img, 
                               mask = mask,
                               contours = self.contours,
                               fruit_locule_map = self.fruit_locule_map,
                               fruit_id = fruit_id, 
                               plot_size = plot_size,
                               overlay = overlay,
                               margin = margin,
                               renumber = True,
                               overlay_legend = overlay_legend, 
                               plot = True,
                               only_fruit = only_fruit)  
        
    ##########################################################################################
    #                                     MORPHOLOGY 
        
        
    ##########################################################################################
    # Extract morphology measurements from the image
    ##########################################################################################
    def analyze_morphology(self, 
                    ## Plot
                    plot: bool = True, 
                    plot_size: Tuple[int, int] = (10, 10), 
                    font_size: float = 1.5, 
                    font_thickness: int = 2, 
                    font_color: Tuple[int, int, int] = (0, 0, 0),
                    label_position: str = 'top',
                    label_color: Tuple[int, int, int] = (255, 255, 255),

                    
                    # Contour mode
                    contour_mode: str = 'raw', 
                    epsilon: float = 0.001, 
                     
                    # Filter locules
                    min_locule_area: int = 100, 
                    max_locule_area: Optional[int] = None, 
                    
                    # Symmetry
                    angle_shifts: int = 500, 

                    # Pericarp thickness
                    num_rays: int = 90,

                    # Contours style
                    pericarp_int_color: Tuple[int, int, int] = (0, 240, 240),
                    pericarp_int_thickness: int = 2,

                    locule_color: Tuple[int,int,int] = (255, 0, 255),
                    locule_thickness: int = 2,

                    pericarp_ext_thickness: int = 2,
                    pericarp_ext_color: Tuple[int, int, int] = (0, 240, 240),

                    centroid_fruit_color: Tuple[int,int,int] = (255, 255, 51),
                    centroid_fruit_thickness: int = 2,

                    centroid_locule_color: Tuple[int,int,int] = (0, 255, 255),
                    centroid_locule_thickness: int = 2,

                    # Result table
                    display_table: Optional[bool] = True,

                    is_locule: bool = True
                    ) -> ResultsImage:
        """
        Analyze detected fruits using analysis.analyze_fruits_morphology().
        
        """
        # Validation
        if self.mask_fruit is None:
            raise ValueError("No mask available. Run generate_fruit_mask() first.")
        if self.contours is None:
            raise ValueError("No contours available. Run detect_fruits() first.")
        if self.fruit_locule_map is None:
            raise ValueError("No fruit-locule mapping available. Run detect_fruits() first.")
        
        if not self.fruit_locule_map:
            raise ValueError("No fruits detected. Make sure detect_fruits() found valid fruit contours before calling analyze_color().")
        
        if self.label_text is None:
            self.label_text = 'No label detected'

        saved_color_results = getattr(self.results, 'color_results', None)
        
        self.results = analyze_fruits_morphology(
            # Image 
            img=self.img_copy,
            path=self.img_path,
            original_img_clean=self.img,
            contours=self.contours,
            fruit_locule_map=self.fruit_locule_map,

            # Size reference and image metadata
            px_per_cm=self.px_per_cm, 
            img_name=self.img_name,
            label_text=self.label_text,
            
            # Fruit contour
            contour_mode=contour_mode,
            epsilon=epsilon,

            # Filter locules
            min_locule_area=min_locule_area,
            max_locule_area=max_locule_area,

            # Angular symmetry       
            angle_shifts=angle_shifts,
            
            # Pericarp thickness
            num_rays=num_rays,

            # Plot annotated image
            plot=plot,
            plot_size=plot_size,

            # Label
            text_color=font_color,
            label_position=label_position,
            font_scale=font_size,
            font_thickness=font_thickness,
            label_background_color=label_color,

            # Contours color and thickness (width)
            pericarp_ext_color = pericarp_ext_color,
            pericarp_ext_thickness = pericarp_ext_thickness,
            centroid_locules_thickness = centroid_locule_thickness,
            centroid_fruit_thickness = centroid_fruit_thickness,
            pericarp_int_color = pericarp_int_color,
            pericarp_int_thickness = pericarp_int_thickness,
            centroid_locule_color = centroid_locule_color,
            centroid_fruit_color = centroid_fruit_color,
            locule_color = locule_color,
            locule_thickness = locule_thickness,

            # Extra
            is_locule = is_locule
            
        )

        if saved_color_results is not None:
            self.results.color_results = saved_color_results

        metadata = self.is_metadata_saved
        if metadata:
            self.parameters.analyze_morphology_params = {
                'contour_mode': contour_mode,
                'epsilon': epsilon if contour_mode == 'approx' else None,
                'min_locule_area': min_locule_area,
                'max_locule_area': max_locule_area,
                'angle_shifts': angle_shifts,
                'num_rays': num_rays,
                'font_size': font_size,
                'font_thickness': font_thickness,
                'font_color': font_color,
                'label_position': label_position,
                'label_color': label_color, 
                'pericarp_int_color': pericarp_int_color,
                'pericarp_int_thickness': pericarp_int_thickness,
                'locule_color': locule_color,
                'locule_thickness': locule_thickness,
                'pericarp_ext_thickness': pericarp_ext_thickness,
                'pericarp_ext_color': pericarp_ext_color,
                'centroid_fruit_color': centroid_fruit_color,
                'centroid_fruit_thickness': centroid_fruit_thickness,
                'centroid_locule_color': centroid_locule_color,
                'centroid_locule_thickness': centroid_locule_thickness,
                'is_locule': is_locule}

        self.results.morphology_results = pd.DataFrame(self.results.morphology_results)

        self.results.morphology_results = pd.DataFrame(self.results.morphology_results)

        # Reorder results table
        _GROUP_ORDER = [
            # Image information
            ['image_name', 'label', 'fruit_id', 'n_locules', 'unit'],
            # Fruit morphology
            ['fruit_area', 'fruit_perimeter', 'fruit_circularity', 'fruit_solidity', 
            'fruit_convexity', 'fruit_major_axis', 'fruit_minor_axis', 'fruit_box_length', 
            'fruit_box_width', 'fruit_aspect_ratio', 'fruit_compactness', 'fruit_lobedness'],
            # External pericarp
            ['total_outer_pericarp_area'],
            # Internal pericarp
            ['outer_pericarp_mean_thickness', 'outer_pericarp_std_thickness',
            'outer_pericarp_cv_thickness'],
            # Internal areas
            ['total_internal_fruit_area', 'total_internal_pericarp_area', 'total_locules_area'],
            # Locules
            ['locules_mean_area', 'locules_std_area', 'locules_cv_area',
            'locules_mean_circularity', 'locules_std_circularity', 'locules_cv_circularity',
            'locules_angular_symmetry', 'locules_radial_symmetry'],
            # Ratios
            ['outer_pericarp_to', 'internal_pericarp_to', 'locules_to'],
        ]

        cols = self.results.morphology_results.columns.tolist()
        ordered, seen = [], set()

        for group in _GROUP_ORDER:
            for keyword in group:
                matched = [c for c in cols if c.startswith(keyword) and c not in seen]
                ordered.extend(matched)
                seen.update(matched)

        # Columns that are not included in group order are added at the end of the df
        remaining = [c for c in cols if c not in seen]
        self.results.morphology_results = self.results.morphology_results[ordered + remaining]

        if display_table:
            return self.results.morphology_results
        
        
    ##########################################################################################
    #                                     PARAMETERS
        
    ##########################################################################################
    # OPTIONAL: Save all the parameters used in the session
    ##########################################################################################
    def save_parameters(self, output_path: Optional[str] = None) -> None:
        """
        Save parameter data as txt and json
        
        Args:
            output_path: Output directory 
                       If None, save it on the input folder
        """
        if output_path is None:
            output_path = os.path.dirname(self.img_path)
        
        base_name = os.path.splitext(os.path.basename(self.img_path))[0]
        
        # Save as .txt
        txt_path = os.path.join(output_path, f"{base_name}_parameters.txt")
        self.parameters.save_to_file(txt_path)
        
        # Save as .json
        json_path = os.path.join(output_path, f"{base_name}_parameters.json")
        self.parameters.save_to_json(json_path)
        
        print(f"\n> Parameters saved:")
        print(f"  - TXT:  {txt_path}")
        print(f"  - JSON: {json_path}")
        
        return None
        
    ##########################################################################################
    #                                     COLOR ANALYSIS

    ##########################################################################################
    # Extract color measurements for different fruit tissues
    ########################################################################################## 
    def analyze_color(self, 
                        stat: Optional[str] = 'mean',
                        tissue: Optional[str] = 'all',
                        color_space: Optional[str] = 'all',
                        display_table: Optional[bool] = True,
                        plot: bool = False,
                        plot_size: Tuple[int,int] = (10,10),

                        # Annotation
                        font_size: int = 2,
                        font_thickness: int = 2,
                        pericarp_ext_color: Tuple[int,int,int] = (0,255,0),
                        pericarp_ext_thickness: int = 2,
                        locule_thickness: int = 2,
                        locule_color: Tuple[int, int, int] = (255,0,255),
                        label_position: str = 'top',
                        font_color: Tuple[int,int,int] = (0,0,0), 
                        label_color: Tuple[int,int,int] = (255,255,255),
                        label_opacity: float = 0.7,
                        get_color_histogram: bool = False):
        
        # Validation
        if self.mask_fruit is None:
            raise ValueError("No mask available. Run generate_fruit_mask() first.")
        if self.contours is None:
            raise ValueError("No contours available. Run detect_fruits() first.")
        if self.fruit_locule_map is None:
            raise ValueError("No fruit-locule mapping available. Run detect_fruits() first.")
        
        if not self.fruit_locule_map:
            raise ValueError("No fruits detected. Make sure detect_fruits() found valid fruit contours before calling analyze_color().")
        
        if self.label_text is None:
            self.label_text = 'No label detected'
        
        metadata = self.is_metadata_saved
        if metadata:
            self.parameters.analyze_color_params = {
                'stat': stat,
                'tissue': tissue,
                'color_space': color_space,
                'font_size': font_size,
                'font_thickness': font_thickness,
                'pericarp_ext_color': pericarp_ext_color,
                'pericarp_ext_thickness': pericarp_ext_thickness,
                'locule_thickness': locule_thickness,
                'locule_color': locule_color,
                'label_position': label_position,
                'font_color': font_color,
                'label_color': label_color,
                'label_opacity': label_opacity,
                'get_color_histogram': get_color_histogram
            }

        # Always reannotate from clean image
        saved_color_results = getattr(self.results, 'color_results', None)

        self.results = ResultsImage(
            bgr_img = self.img,
            morphology_results=[],  
            image_path=self.img_path
        )

        # Annotate from clean image
        annotate_all_fruits(annotated_img = self.results.annotated_image,
                            contours =  self.contours, 
                            fruit_locule_map = self.fruit_locule_map, 
                            img_shape = self.img_shape,
                            font_scale = font_size,
                            font_thickness = font_thickness,
                            pericarp_ext_color = pericarp_ext_color,
                            pericarp_ext_thickness = pericarp_ext_thickness,
                            locule_thickness = locule_thickness, 
                            locule_color = locule_color,
                            label_position = label_position, 
                            margin = 10, 
                            text_color = font_color, 
                            label_background_color = label_color,
                            label_opacity = label_opacity)

        if saved_color_results is not None:
            self.results.color_results = saved_color_results
        
        if plot:
            plt.figure(figsize = plot_size)
            plt.imshow(self.results.annotated_image)
            plt.axis('off')
            plt.show()
    
        
        # use locule mask if available, otherwise use fruit mask
        if self.mask_locules is not None:
            mask = self.mask_locules
        else:
            mask = self.mask_fruit
        
        if get_color_histogram:
            color_results = get_fruit_color_histograms(img = self.img,
                                                          hsv_img = self.img_hsv,
                                                          label = self.label_text,
                                                          contours = self.contours,
                                                          mask = mask,
                                                          fruit_locule_map = self.fruit_locule_map,
                                                          image_name = self.img_name,
                                                          color_space = color_space,
                                                          renumber = True,
                                                          normalize = False)
            
            self.results.color_results = pd.DataFrame(color_results)
    
        else:
            color_results = analyze_all_fruits_color(img = self.img,
                                    mask = mask,
                                    contours = self.contours,
                                    fruit_locule_map = self.fruit_locule_map,
                                    stat = stat,
                                    tissue = tissue,
                                    renumber = True,
                                    color_space = color_space)
            

            df = (
                pd.concat(
                    {fruit_id: pd.DataFrame(tissues).T for fruit_id, tissues in color_results.items()},
                    names=["fruit_id", "tissue"]
                )
                .reset_index()
            )

            df.insert(0, "image_name", self.img_name)
            df.insert(1, "label", self.label_text)
            self.results.color_results = df

        if display_table:
           return self.results.color_results
        
    
    ##########################################################################################
    # Color correction
    ##########################################################################################

    def detect_color_checker(self, plot: bool = False, plot_size=(5,5), verbose: bool = True, scale_factor: float = 0.5):
            """
            Detect color checker and store its ROI coordinates.
            Args:
                plot: If True, displays the image with detected color checker
                plot_size: Figure size for plotting
                scale_factor: Image scale for faster detection (0.2-0.5)
            """
            # Reduce the image size for faster detection
            h_orig, w_orig = self.img_rgb.shape[:2]
            img_small = cv2.resize(self.img_rgb, 
                                (int(w_orig * scale_factor), int(h_orig * scale_factor)),
                                interpolation=cv2.INTER_AREA)
            
            detector = cv2.mcc.CCheckerDetector.create()
            detector.process(img_small, cv2.mcc.MCC24)  
            checkers = detector.getListColorChecker()
            
            if not checkers:
                if verbose:
                    warnings.warn(
                        "No color checker detected in the image. "
                        "Check that the color checker is visible and try adjusting scale_factor.",
                        UserWarning,
                        stacklevel=2
                    )
                self.checker_roi = None
                self.checker_coords = None
                return None
            
            checker = checkers[0]
            
            # Draw color checker grid on the small image (coordinates are in small image space),
            # then resize that drawn region back to full resolution and paste into img_copy.
            # NOTE: img_small must be RGB since CCheckerDraw works in RGB space.
            img_small_drawn = cv2.mcc.CCheckerDraw_create(checker).draw(img_small.copy())

            # Get box points in small image space before scaling
            box = checker.getBox()
            box_points_small = np.int32(box)

            # Compute tight bounding rect in small image space with a safety margin
            x_s, y_s, w_s, h_s = cv2.boundingRect(box_points_small)
            pad_s = 15  # px padding in small-image space
            x_s1 = max(0, x_s - pad_s)
            y_s1 = max(0, y_s - pad_s)
            x_s2 = min(img_small.shape[1], x_s + w_s + pad_s)
            y_s2 = min(img_small.shape[0], y_s + h_s + pad_s)

            # Corresponding region in full resolution image
            x_f1 = int(x_s1 / scale_factor)
            y_f1 = int(y_s1 / scale_factor)
            x_f2 = min(w_orig, int(x_s2 / scale_factor))
            y_f2 = min(h_orig, int(y_s2 / scale_factor))

            # Crop the drawn checker region from small image, resize to full res patch size
            patch_small = img_small_drawn[y_s1:y_s2, x_s1:x_s2]
            patch_full  = cv2.resize(patch_small,
                                    (x_f2 - x_f1, y_f2 - y_f1),
                                    interpolation=cv2.INTER_LINEAR)

            # Convert RGB patch to BGR to match self.img_copy
            patch_full_bgr = cv2.cvtColor(patch_full, cv2.COLOR_RGB2BGR)

            # Paste the drawn grid patch into img_copy at the correct location
            self.img_copy[y_f1:y_f2, x_f1:x_f2] = patch_full_bgr

            # Scale box_points to fullres for bounding rect / ROI extraction
            box_points = (box_points_small / scale_factor).astype(np.int32)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(box_points)
            
            # Add margin (0.1 = 10%)
            margin_x = int(w * 0.1)
            margin_y = int(h * 0.1)
            x_expanded = max(0, x - margin_x)
            y_expanded = max(0, y - margin_y)
            w_expanded = min(self.img_rgb.shape[1] - x_expanded, w + 2 * margin_x)
            h_expanded = min(self.img_rgb.shape[0] - y_expanded, h + 2 * margin_y)
            
            # Store coordinates as tuple (x, y, w, h) for mask removal
            self.checker_coords = (x_expanded, y_expanded, w_expanded, h_expanded)
            
            # Extract ROI image 
            checker_img = self.img_copy[y_expanded:y_expanded+h_expanded, 
                                        x_expanded:x_expanded+w_expanded]
            
            # Draw rectangle on image copy for visualization
            cv2.rectangle(self.img_copy, 
                        (x_expanded, y_expanded), 
                        (x_expanded + w_expanded, y_expanded + h_expanded), 
                        (0, 255, 0), 3)
            
            # Add label
            cv2.putText(self.img_copy, "Color Checker", 
                        (x_expanded, y_expanded - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            
            if verbose:
                print("\n" + "=" * 55)
                print("★ COLOR CARD:")
                print("=" * 55)
                print(f"> Color checker detected: ")
                print(f"    - Coordinates: x={x_expanded}, y={y_expanded}, w={w_expanded}, h={h_expanded}")
            
            if plot: 
                plt.figure(figsize=plot_size)
                plt.imshow(cv2.cvtColor(checker_img, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()
            
            return None
        

    ##########################################################################################
    #                                     BATCH ANALYSIS

    
    ##########################################################################################
    # PRocess a single file (needed for analyze_folder)
    ##########################################################################################
    def process_single_file(
        self,
        config: Optional[Dict] = None,
        json_path: Optional[str] = None,
        analyze_morphology: bool = True,
        analyze_color: bool = True,
        save_image: bool = False,
        output_path: Optional[str] = None,
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Dict], int, Optional[np.ndarray]]:
        """
        Run the full pipeline on a single already-loaded image.
        """
        
        # Load parameters using json file
        if json_path is not None and os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                params = json.load(f)
        elif config is not None:
            params = config
        else:
            params = {}

        def _get(section: str) -> Dict:
            return params.get(section, {}) or {}

        def _clean(d: Dict) -> Dict:
            """Remove None values to avoid overriding defaults."""
            return {k: v for k, v in d.items() if v is not None}

        error_dict    = None
        n_fruits      = 0
        df_morph      = None
        df_color      = None
        annotated_img = None
        
        # Run every step and cath errors (if any)
        try:
            # 1. setup_measurements
            try:
                self.setup_measurements(verbose=False, **_clean(_get('setup_measurements_params')))
            except Exception as e:
                raise RuntimeError(f"[setup_measurements] {e}")

            # 2. generate_fruit_mask
            try:
                self.generate_fruit_mask(plot=False, **_clean(_get('generate_fruit_mask_params')))
            except Exception as e:
                raise RuntimeError(f"[generate_fruit_mask] {e}")

            # 3. enhance_locule_contrast (optional)
            elc = _get('enhance_locule_contrast_params')
            if elc:
                try:
                    self.enhance_locule_contrast(plot=False, **_clean(elc))
                except Exception as e:
                    raise RuntimeError(f"[enhance_locule_contrast] {e}")

                # 4. generate_locule_mask (only if enhance was run)
                glm = _get('generate_locule_mask_params')
                if glm:
                    try:
                        self.generate_locule_mask(plot=False, **_clean(glm))
                    except Exception as e:
                        raise RuntimeError(f"[generate_locule_mask] {e}")

            # 5. detect_fruits
            try:
                self.detect_fruits(verbose=False, **_clean(_get('detect_fruits_params')))
            except Exception as e:
                raise RuntimeError(f"[detect_fruits] {e}")

            if not self.fruit_locule_map:
                raise RuntimeError("[detect_fruits] {No valid fruits detected in image}")

            n_fruits = len(self.fruit_locule_map)

            # 6. analyze_morphology
            if analyze_morphology:
                try:
                    self.analyze_morphology(
                        plot=False, display_table=False,
                        **_clean(_get('analyze_morphology_params'))
                    )
                    if self.results and self.results.morphology_results is not None:
                        df_morph = self.results.morphology_results \
                            if isinstance(self.results.morphology_results, pd.DataFrame) \
                            else pd.DataFrame(self.results.morphology_results)
                except Exception as e:
                    raise RuntimeError(f"[analyze_morphology] {e}")

            # 7. analyze_color
            if analyze_color:
                try:
                    self.analyze_color(
                        display_table=False,
                        **_clean(_get('analyze_color_params'))
                    )
                    if self.results and self.results.color_results is not None:
                        df_color = self.results.color_results \
                            if isinstance(self.results.color_results, pd.DataFrame) \
                            else pd.DataFrame(self.results.color_results)
                except Exception as e:
                    raise RuntimeError(f"[analyze_color] {e}")

            # 8. Get annotated image
            if self.results is not None:
                annotated_img = cv2.cvtColor(self.results.annotated_image, cv2.COLOR_RGB2BGR)
            else:
                raise RuntimeError("[pipeline] No results generated — annotated image unavailable")

            # 9. Save image if requested
            if save_image and annotated_img is not None:
                out_dir = output_path or os.path.dirname(self.img_path)
                base    = os.path.splitext(os.path.basename(self.img_path))[0]
                out_img = os.path.join(out_dir, f"{base}_annotated.jpg")
                cv2.imwrite(out_img, annotated_img)

        except Exception as e:
            error_dict = {
                'filename': os.path.basename(self.img_path),
                'status':   str(e)
            }

        return df_morph, df_color, error_dict, n_fruits, annotated_img


    ##########################################################################################
    # Process all images in a folder
    ##########################################################################################
    
    def analyze_folder(
        self,
        analyze_morphology: bool = True,
        analyze_color: bool = True,
        json_path: Optional[str] = None,
        config: Optional[Dict] = None,
        output_path: Optional[str] = None,
        num_cores: int = 1,
        verbose: bool = True,

        # setup_measurements
        width_cm: Optional[float] = None,
        length_cm: Optional[float] = None,
        diameter_cm: Optional[float] = None,
        fast_calibration: Optional[bool] = None,
        skip_qr: Optional[bool] = None,
        detect_label: Optional[bool] = None,
        confidence: Optional[float] = None,
        detect_color_checker: Optional[bool] = None,
        scale_factor: Optional[float] = None,

        # generate_fruit_mask
        stamp: Optional[bool] = None,
        lower_hsv: Optional[List[int]] = None,
        upper_hsv: Optional[List[int]] = None,
        n_iteration: Optional[int] = None,
        kernel_blur: Optional[int] = None,
        kernel_open: Optional[int] = None,
        kernel_close: Optional[int] = None,
        canny_min: Optional[int] = None,
        canny_max: Optional[int] = None,
        remove_roi: Optional[bool] = None,
        roi_expansion: Optional[int] = None,
        background_color: Optional[str] = None,
        fill_holes: Optional[bool] = None,
        apply_convex_hull: Optional[bool] = None,

        # enhance_locule_contrast
        contrast_method: Optional[str] = None,
        gamma: Optional[float] = None,
        gain: Optional[float] = None,
        cutoff: Optional[float] = None,
        c: Optional[float] = None,
        kernel_blur_contrast: Optional[int] = None,
        clip_limit: Optional[int] = None,
        tile_grid_size: Optional[int] = None,

        # generate_locule_mask
        thresh_min: Optional[int] = None,
        thresh_max: Optional[int] = None,
        min_fruit_area_locule: Optional[int] = None,
        kernel_close_locule: Optional[int] = None,
        kernel_open_locule: Optional[int] = None,
        invert_locule: Optional[bool] = None,

        # detect_fruits
        min_fruit_area: Optional[int] = None,
        max_fruit_area: Optional[int] = None,
        min_fruit_circularity: Optional[float] = None,
        min_locule_area: Optional[int] = None,
        min_locule_per_fruit: Optional[int] = None,
        rescale_factor: Optional[float] = None,

        # analyze_morphology
        contour_mode: Optional[str] = None,
        epsilon: Optional[float] = None,
        min_locule_area_morph: Optional[int] = None,
        max_locule_area: Optional[int] = None,
        angle_shifts: Optional[int] = None,
        num_rays: Optional[int] = None,
        font_size: Optional[float] = None,
        font_thickness: Optional[int] = None,
        font_color: Optional[Tuple[int,int,int]] = None,
        label_position: Optional[str] = None,
        label_color: Optional[Tuple[int,int,int]] = None,
        pericarp_ext_color: Optional[Tuple[int,int,int]] = None,
        pericarp_ext_thickness: Optional[int] = None,
        centroid_fruit_color: Optional[Tuple[int,int,int]] = None,
        centroid_fruit_thickness: Optional[int] = None,
        pericarp_int_color: Optional[Tuple[int,int,int]] = None,
        pericarp_int_thickness: Optional[int] = None,
        locule_color: Optional[Tuple[int,int,int]] = None,
        locule_thickness: Optional[int] = None,
        centroid_locule_color: Optional[Tuple[int,int,int]] = None,
        centroid_locule_thickness: Optional[int] = None,

        # analyze_color
        stat: Optional[str] = None,
        tissue: Optional[str] = None,
        color_space: Optional[str] = None,
        label_opacity: Optional[float] = None,
        get_color_histogram: Optional[bool] = None,
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Process all images in the folder passed to FruitInternalAnalyzer.
        Parameters can be passed individually (e.g. contour_mode='raw') or
        grouped in a config dict — individual params always take priority.
        """

        # Validate output directory
        if not self.is_directory:
            raise ValueError(
                "analyze_folder() requires a directory path. "
                "Pass a folder to FruitInternalAnalyzer(), not a single file."
            )

        folder_path = self.img_path

        # Validate number of cores (num_cores)
        num_cores_message = None
        if num_cores <= 0:
            num_cores_message = f"    > num_cores: {num_cores} must be at least 1. Using num_cores=1 instead."
            num_cores = 1

        max_cores = mp.cpu_count()
        if num_cores > max_cores:
            num_cores_message = f"    > num_cores: {num_cores} exceeds system cores ({max_cores}). Using {max_cores} instead."
            num_cores = max_cores

        # Check valid images in the folder
        img_paths = sorted([
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if Path(f).suffix.lower() in valid_extensions
        ])

        if not img_paths:
            raise ValueError(f"No valid images found in: {folder_path}. Valid image extensions include: {valid_extensions}")

        # Validate output_path, if doesn't exist, create one
        if output_path is None:
            output_path = os.path.join(folder_path, "Results")
        os.makedirs(output_path, exist_ok=True)

        # Build config: json/dict base → override with individual params ─
        import copy
        config = copy.deepcopy(config) if config else {}

        if json_path is not None and os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                json_cfg = json.load(f) or {}
            config.update(json_cfg)

        # Helper: merge non-None individual params into a config section
        def _apply(section: str, mapping: Dict):
            overrides = {k: v for k, v in mapping.items() if v is not None}
            if overrides:
                config.setdefault(section, {})
                config[section].update(overrides)

        _apply('setup_measurements_params', dict(
            width_cm=width_cm, length_cm=length_cm, diameter_cm=diameter_cm,
            fast_calibration=fast_calibration, skip_qr=skip_qr,
            detect_label=detect_label, confidence=confidence,
            detect_color_checker=detect_color_checker, scale_factor=scale_factor,
        ))
        _apply('generate_fruit_mask_params', dict(
            stamp=stamp, lower_hsv=lower_hsv, upper_hsv=upper_hsv,
            n_iteration=n_iteration, kernel_blur=kernel_blur,
            kernel_open=kernel_open, kernel_close=kernel_close,
            canny_min=canny_min, canny_max=canny_max,
            remove_roi=remove_roi, roi_expansion=roi_expansion,
            background_color=background_color, fill_holes=fill_holes,
            apply_convex_hull=apply_convex_hull,
        ))
        _apply('enhance_locule_contrast_params', dict(
            contrast_method=contrast_method, gamma=gamma, gain=gain,
            cutoff=cutoff, c=c, kernel_blur=kernel_blur_contrast,
            clip_limit=clip_limit, tile_grid_size=tile_grid_size,
        ))
        _apply('generate_locule_mask_params', dict(
            thresh_min=thresh_min, thresh_max=thresh_max,
            min_fruit_area=min_fruit_area_locule,
            kernel_close=kernel_close_locule, kernel_open=kernel_open_locule,
            invert_locule=invert_locule,
        ))
        _apply('detect_fruits_params', dict(
            min_fruit_area=min_fruit_area, max_fruit_area=max_fruit_area,
            min_fruit_circularity=min_fruit_circularity,
            min_locule_area=min_locule_area,
            min_locule_per_fruit=min_locule_per_fruit,
            rescale_factor=rescale_factor,
        ))
        _apply('analyze_morphology_params', dict(
            contour_mode=contour_mode, epsilon=epsilon,
            min_locule_area=min_locule_area_morph, max_locule_area=max_locule_area,
            angle_shifts=angle_shifts, num_rays=num_rays,
            font_size=font_size, font_thickness=font_thickness,
            font_color=font_color, label_position=label_position,
            label_color=label_color, pericarp_ext_color=pericarp_ext_color,
            pericarp_ext_thickness=pericarp_ext_thickness,
            centroid_fruit_color=centroid_fruit_color,
            centroid_fruit_thickness=centroid_fruit_thickness,
            pericarp_int_color=pericarp_int_color,
            pericarp_int_thickness=pericarp_int_thickness,
            locule_color=locule_color, locule_thickness=locule_thickness,
            centroid_locule_color=centroid_locule_color,
            centroid_locule_thickness=centroid_locule_thickness,
        ))
        _apply('analyze_color_params', dict(
            stat=stat, tissue=tissue, color_space=color_space,
            label_opacity=label_opacity,
            get_color_histogram=get_color_histogram,
        ))

        # Sync to self.parameters for session report
        _param_keys = [
            'setup_measurements_params', 'generate_fruit_mask_params',
            'enhance_locule_contrast_params', 'generate_locule_mask_params',
            'detect_fruits_params', 'analyze_morphology_params', 'analyze_color_params',
        ]
        for key in _param_keys:
            if key in config and config[key]:
                setattr(self.parameters, key, config[key])

        # Print header message:
        session_start = datetime.now()
        if verbose:
            print("=" * 60)
            print(" " * 42 + "Traitly running ⋆✧｡٩(ˊᗜˋ )و✧*｡   ")
            print("=" * 60 )
            print(f"    > Input folder: {folder_path}")
            print(f"    > Image(s) detected: {len(img_paths)}")
            print(f"    > analyze_morphology: {analyze_morphology}")
            print(f"    > analyze_color: {analyze_color}")
            if num_cores_message is not None:
                print(num_cores_message)
            else:
                print(f"    > num_cores: {num_cores}\n")
            if json_path is not None:
                print(f"    > Parameters loaded from: {json_path}\n")
            

        # Create lists to save results
        all_morphology : List[pd.DataFrame] = []
        all_color      : List[pd.DataFrame] = []
        errors         : List[Dict]         = []
        total_fruits   = 0
        per_image_times: List[Dict]         = []

        def _run_one(img_path: str):
            t0 = time.time()
            try:
                worker = FruitInternalAnalyzer(img_path)
                worker.load_image(plot=False)
                df_m, df_c, err, n, ann_img = worker.process_single_file(
                    config=config,
                    json_path=None,
                    analyze_morphology=analyze_morphology,
                    analyze_color=analyze_color,
                    save_image=True,
                    output_path=output_path,
                )
                return df_m, df_c, err, n, ann_img, os.path.basename(img_path), time.time() - t0
            except Exception as e:
                return None, None, {
                    'filename': os.path.basename(img_path),
                    'status':   f'Error: {str(e)}'
                }, 0, None, os.path.basename(img_path), time.time() - t0

        # Run parallel analysis if num_cores > 1, else run sequential analysis
        if num_cores == 1:
            for img_path in tqdm(img_paths, desc="Processing images",
                                unit="img", disable=not verbose):
                df_m, df_c, err, n, _, fname, elapsed = _run_one(img_path)

                per_image_times.append({'filename': fname, 'time_s': round(elapsed, 2),
                                        'status': 'error' if err else 'ok', 'fruits': n})

                if err:
                    errors.append(err)
                else:
                    if df_m is not None: all_morphology.append(df_m)
                    if df_c is not None: all_color.append(df_c)
                    total_fruits += n

        else:
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                futures = {
                    executor.submit(
                        _process_image_worker,
                        img_path, config, analyze_morphology, analyze_color
                    ): img_path
                    for img_path in img_paths
                }
                for future in tqdm(as_completed(futures), total=len(futures),
                                desc="Processing images", unit="img",
                                disable=not verbose):
                    result   = future.result()
                    df_m, df_c, err, n, ann_img, fname = result[:6]
                    elapsed  = result[6] if len(result) > 6 else 0.0

                    per_image_times.append({'filename': fname, 'time_s': round(elapsed, 2),
                                            'status': 'error' if err else 'ok', 'fruits': n})

                    if err:
                        errors.append(err)
                    else:
                        if ann_img is not None:
                            base    = os.path.splitext(fname)[0]
                            out_img = os.path.join(output_path, f"{base}_annotated.jpg")
                            cv2.imwrite(out_img, ann_img)
                        if df_m is not None: all_morphology.append(df_m)
                        if df_c is not None: all_color.append(df_c)
                        total_fruits += n

        # Merge all the results in a single df
        df_morphology_all = pd.concat(all_morphology, ignore_index=True) \
            if all_morphology else None
        df_color_all      = pd.concat(all_color,      ignore_index=True) \
            if all_color else None

        # Save csv files
        morph_csv = None
        color_csv = None

        if df_morphology_all is not None:
            morph_csv = os.path.join(output_path, "morphology_results.csv")
            df_morphology_all.to_csv(morph_csv, index=False)

        if df_color_all is not None:
            color_csv = os.path.join(output_path, "color_results.csv")
            df_color_all.to_csv(color_csv, index=False)

        # Save session report
        session_end = datetime.now()
        total_time  = (session_end - session_start).total_seconds()
        avg_time    = total_time / len(img_paths) if img_paths else 0

        def _filter_params(p: Dict) -> Dict:
            return {k: v for k, v in p.items()
                    if 'plot' not in k.lower() and 'color' not in k.lower()}

        session_lines = [
            "=" * 70,
            "SESSION REPORT",
            "=" * 70,
            f"traitly          : v{__version__}",
            f"run date         : {session_start.strftime('%Y-%m-%d %H:%M:%S')}",
            f"folder           : {folder_path}",
            f"images found     : {len(img_paths)}",
            f"images ok        : {len(img_paths) - len(errors)}",
            f"images failed    : {len(errors)}",
            f"total fruits     : {total_fruits}",
            f"num_cores        : {num_cores}",
            f"total time       : {total_time:.1f}s",
            f"avg per image    : {avg_time:.1f}s",
            "",
            "=" * 70,
            "ANALYSIS PARAMETERS",
            "=" * 70,
        ]

        param_sections = {
            'SETUP_MEASUREMENTS':      'setup_measurements_params',
            'GENERATE_FRUIT_MASK':     'generate_fruit_mask_params',
            'ENHANCE_LOCULE_CONTRAST': 'enhance_locule_contrast_params',
            'GENERATE_LOCULE_MASK':    'generate_locule_mask_params',
            'DETECT_FRUITS':           'detect_fruits_params',
            'ANALYZE_MORPHOLOGY':      'analyze_morphology_params',
            'ANALYZE_COLOR':           'analyze_color_params',
        }

        for title, attr in param_sections.items():
            raw      = getattr(self.parameters, attr, {}) or {}
            filtered = _filter_params(raw)
            if filtered:
                session_lines.append(f"\n{title}:")
                for k, v in filtered.items():
                    session_lines.append(f"   - {k}: {v}")

        session_lines += [
            "",
            "=" * 70,
            "DEPENDENCIES",
            "=" * 70,
        ] + [f"   - {pkg:<30} {ver}"
            for pkg, ver in self.parameters.get_package_versions().items()]


        # Create session report
        session_txt = os.path.join(output_path, "session_report.txt")
        with open(session_txt, 'w', encoding='utf-8') as f:
            f.write("\n".join(session_lines))

        # Error report table format
        error_txt = None
        if errors:
            col1_w = max(len("IMAGE"),  max(len(e['filename']) for e in errors)) + 2
            col2_w = max(len("ERROR"),  max(len(e['status'])   for e in errors)) + 2
            sep    = f"+{'-' * col1_w}+{'-' * col2_w}+"
            header = f"| {'IMAGE':<{col1_w-2}} | {'ERROR':<{col2_w-2}} |"

            error_lines = [
                "=" * 70,
                "ERROR REPORT",
                "=" * 70,
                f"run date   : {session_start.strftime('%Y-%m-%d %H:%M:%S')}",
                f"folder     : {folder_path}",
                f"failed     : {len(errors)}/{len(img_paths)} images",
                "",
                sep, header, sep,
            ] + [f"| {e['filename']:<{col1_w-2}} | {e['status']:<{col2_w-2}} |"
                for e in errors] + [sep]

            error_txt = os.path.join(output_path, "error_report.txt")
            with open(error_txt, 'w', encoding='utf-8') as f:
                f.write("\n".join(error_lines))
        
        # Final message
        if verbose:
            total_img_processed = len(img_paths) - len(errors)
            print("\n( ദ്ദി ˙ᗜ˙ ) Finished " + "="*47)
            print("    > Image(s) processed:")
            print(f"        - Successfully: {total_img_processed}/{len(img_paths)} img(s)")
            if errors:
                print(f"        - Errors: {len(errors)}/{len(img_paths)} img(s)")
            print(f"        - Total fruits: {total_fruits}")
            print(f"        - Total time: {total_time:.1f}s  (avg {avg_time:.1f}s/img)")
            print("    > Files saved:")
            print(f"        - {total_img_processed} annotated image(s)")
            if morph_csv:
                print(f"        - {os.path.basename(morph_csv)}")
            if color_csv:
                print(f"        - {os.path.basename(color_csv)}")
            print(f"        - {os.path.basename(session_txt)}")
            if error_txt:
                print(f"        - {os.path.basename(error_txt)}")
            print(f"        - Results folder: {output_path}")


        return None
    
    def plot_image(self, annotated = False, plot_size = (10,10)):

        if self.img is None:
            raise ValueError("No image loaded. Run load_img() first.")
        
        if annotated:
            if self.results is None:
                print('Warning: No annotated image, showing original image instead. Run analyze_color() or analyze_morphology() first.')     
                img =  self.img_rgb 
            else:        
                img = self.results.annotated_image
        else:
            img = self.img_rgb


        plt.figure(figsize = plot_size)
        plt.imshow(img)
        plt.axis('off')
        plt.show

        return None