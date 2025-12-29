# traitly/internal_structure/image_analyzer.py

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
# LOCAL/INTERNAL IMPORTS
# ============================================================================
from .mask import create_mask, find_fruits
from .analysis import analyze_fruits
from ..utils import common_functions as cf
from .annotated_image import AnnotatedImage

##############################
# Ignore warnings from torch #
##############################

warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', message='Using CPU')

class FruitAnalyzer:
    """Class for analyzing fruit images with morphological measurements."""
    
    def __init__(self, image_path: str):
        """
        Initialize the image analyzer.
        
        Args:
            image_path: Path to image file or directory
        """
        ## Verify image path exists
        # Assign the path first
        self.image_path = image_path
        
        # Then verify if it was provided and exists
        if self.image_path is not None:
            if not os.path.exists(self.image_path):
                raise FileNotFoundError(
                    f"The path does not exist: {self.image_path}\n"
                    f"Please verify that the path is correct and accessible."
                )

        # load_img
        self.is_directory = os.path.isdir(image_path)
        self.img = None
        self.img_inverted = None
        self.img_annotated = None
        
        # setup_measurements
        self.ref_roi = None
        self.label_roi = None
        self.label_text = None
        self.label_id = None
        self.img_name = None
        
        # create_mask
        self.mask = None
        self.mask_fruits = None
        self.contours = None
        self.fruit_locus_map = None
        
        # analyze fruits
        self.px_per_cm = None  
        self.results = None

    def read_image(self, plot: bool = False, 
                   plot_size: Tuple[int, int] = (5, 5), 
                   plot_axis: bool = False) -> None:
        """
        Load and optionally display the image.
        
        Args:
            plot: Whether to display the image
            plot_size: Figure size for plotting
            plot_axis: Whether to show axis on plot
        """
        self.img = cf.load_img(self.image_path, plot=plot, 
                              plot_size=plot_size, fig_axis=plot_axis)
        if self.img is None:
            raise ValueError(f"Failed to load image: {self.image_path}")

        return None
        
    def setup_measurements(self, plot: bool = False, verbose: bool = True, 
                        font_size: int = 3, confidence: float = 0.6, 
                        plot_size: Tuple[int, int] = (8, 8),
                        detect_label: bool = False,
                        language_label: List[str] = ['es', 'en'],
                        min_area_label: int = 500,
                        min_canny_label: int = 0, max_canny_label: int = 150, 
                        blur_label: Tuple[int, int] = (11, 11),
                        width_cm: Optional[float] = None,
                        length_cm: Optional[float] = None,
                        diameter_cm: Optional[float] = None, gpu: bool = False
                        ) -> None:
        """
        Extract metadata from image including QR code, label text, and physical calibration.
        
        This method identifies labels and establishes physical measurement calibration. 
        It follows a priority system for label detection and supports multiple calibration methods.
        
        Label Detection Priority:
            1. QR Code detection (highest priority)
            2. OCR on detected label boxes
            3. None if no labels found
        
        Physical Calibration Methods:
            1. Automatic detection via reference circles (preferred)
            2. Manual calibration using width_cm and length_cm parameters
            3. Pixel units if no calibration available
        
        Args:
            plot: Whether to display annotated images at the end. Default is False.
            verbose: Print detailed processing information. Default is True.
            font_size: Font size for annotations on output images. Default is 3.
            confidence: Minimum confidence threshold for reference box detection (0.0 to 1.0). 
                Default is 0.6.
            plot_size: Figure size (width, height) in inches for plotting. Default is (8, 8).
            label: Whether to attempt label/QR code detection. If False, skips all label 
                detection. Default is True.
            language_label: Languages for OCR detection. Default is ['es', 'en'].
            min_area_label: Minimum area in pixels for label box detection. Default is 500.
            min_canny_label: Minimum threshold for Canny edge detection on labels. Default is 0.
            max_canny_label: Maximum threshold for Canny edge detection on labels. Default is 150.
            blur_label: Gaussian blur kernel size for label preprocessing. Default is (11, 11).
            width_cm: Physical width of the image in centimeters for manual calibration. 
                Must be provided together with length_cm. Default is None.
            length_cm: Physical length of the image in centimeters for manual calibration. 
                Must be provided together with width_cm. Default is None.
            diameter_cm: Diameter in centimeters of reference circles for automatic calibration. 
                If None, uses default 2.5 cm. Default is None.
        
        Returns:
            None: Modifies instance attributes:
                - self.label_text: Detected text from QR or label
                - self.label_roi: Coordinates of detected label boxes
                - self.px_per_cm: Pixel density for physical measurements
                - self.img_annotated: Image with annotations
                - self.ref_roi: Coordinates of reference regions
        
        Examples:
            Basic usage with automatic detection:
            >>> analyzer.setup_measurements(verbose=True, plot=True)
            
            With custom reference diameter:
            >>> analyzer.setup_measurements(diameter_cm=3.0, plot=True)
            
            Manual calibration without reference circles:
            >>> analyzer.setup_measurements(width_cm=10.0, length_cm=15.0, plot=True)
            
            Skip label detection:
            >>> analyzer.setup_measurements(label=False, plot=True)
        
        Notes:
            - If no QR code is found, the method attempts OCR on rectangular regions
            - Physical calibration requires either reference circles OR both width_cm and length_cm
            - Without calibration, all measurements will be returned in pixel units
        """
        # Save image name
        self.img_name = cf.detect_img_name(self.image_path)
        img_copy = self.img.copy()

        print("\n" + "="*60)
        print("LABEL DETECTION:")
        print("="*60)

        # Step 1: Try to detect QR code first
        self.label_text, img_copy = cf.detect_qr(img=img_copy)

        if self.label_text is not None and "No QR" not in str(self.label_text):
            print(f"    > QR Code detected: {self.label_text}")
        else:
            self.label_text = None

        # Step 2: Try to detect label boxes with YOLO
        self.label_roi = cf.detect_label_box_yolo(img=img_copy, plot=False, conf = 0.4)

        # Step 3: If YOLO failed, try alternative method
        if self.label_roi is None or len(self.label_roi) == 0:
            #print("    - YOLO label detection failed, trying alternative method...") # For debugging
            self.label_roi = cf.detect_label_box(
                img=img_copy, 
                verbose=False,
                plot=False
            )

        # Step 4: If QR was detected, stop here (no need for OCR)
        if self.label_text is not None:
            pass  # Already have text from QR, skip OCR

        # Step 5: If label_text is None, try OCR
        elif self.label_text is None:
    
            if detect_label and self.label_roi is not None and len(self.label_roi) > 0:
                # label=True and we have boxes: Extract text with OCR
                print("    - No QR code found, attempting label detection...")
                
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                
                try:
                    self.label_text = cf.detect_label_text(
                        img=img_copy, 
                        label_roi=self.label_roi,
                        language=language_label,
                        blur_label=blur_label,
                        verbose=False,
                        gpu = gpu
                    )
                finally:
                    sys.stdout = old_stdout
                
                if self.label_text:
                    print(f"    > Label text detected: {self.label_text}")
                else:
                    print("    - No text found in label boxes")
            
            elif detect_label and (self.label_roi is None or len(self.label_roi) == 0):
                # label=True but no boxes detected
                print("    - No QR code found, attempting label detection...")
                print("    - No label boxes detected")
            
            elif not detect_label:
                # label=False: Skip OCR
                print("> No QR detected.")
                print("> Label text detection: SKIPPED (label=False)")
                self.label_roi = None
            
        # Extract image dimensions
        h, w, _ = self.img.shape
        
        # Detect reference size for physical calibration
        print("\n" + "="*60)
        print("REFERENCE SIZE:")
        print("="*60)
        
        print("Image dimensions:")
        print(f"    - Width:  {w:,} pixels")
        print(f"    - Length: {h:,} pixels")
        print()  # Empty line
        
        # Set diameter default value (but don't print message yet)
        if diameter_cm is None:
            diameter_cm = 2.5
            using_default_diameter = True
        else:
            using_default_diameter = False
        
        # Attempt automatic calibration via reference circles
        self.px_per_cm, self.img_annotated, self.ref_roi = cf.px_cm_density(
            img_copy,
            confidence_threshold=confidence,
            plot=False,  # Always False - plot at the end
            font_size=font_size, 
            verbose=verbose,
            width_cm=width_cm,
            length_cm=length_cm, 
            diameter_cm=diameter_cm,
            return_coordinates=True 
        )
        
        # Validate calibration results and print diameter message after detection
        if self.ref_roi is not None:
            # Automatic calibration successful (reference circles detected)
            print()  # Empty line before diameter message
            if using_default_diameter:
                print('Using default reference diameter: 2.5 cm')
                print('   (To use a different diameter, specify the diameter_cm parameter)')
        else:
            # No reference circles detected - try manual calibration
            if width_cm is not None and length_cm is not None:
                # User provided both physical dimensions
                print('Reference circles not detected. Using provided physical dimensions:')
                print(f"    - Width:  {width_cm} cm")
                print(f"    - Length: {length_cm} cm")
                print(f'\n>>> Calculated px/cm density: {self.px_per_cm:.2f} px/cm')
            
            else:
                # Missing required parameters - cannot calibrate
                missing = []
                if width_cm is None:
                    missing.append('width_cm')
                if length_cm is None:
                    missing.append('length_cm')
                
                print(f'Missing parameters: {", ".join(missing)}')
                print('\n IMPORTANT: Measurements will be returned in >> PIXEL << units')
        
        print("="*60)

        # Display annotated image if requested
        if plot:
            if self.img_annotated is not None:
                plt.figure(figsize=plot_size)
                plt.imshow(cv2.cvtColor(self.img_annotated, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()

    def create_mask(self, n_kernel: int = 5, plot: bool = False, 
                    plot_size: Tuple[int, int] = (5, 5), 
                    stamp: bool = False, plot_axis: bool = False, 
                    n_iteration: int = 1, 
                    canny_min: int = 30, canny_max: int = 100, 
                    lower_hsv: Optional[List[int]] = None,
                    upper_hsv: Optional[List[int]] = None, 
                    locules_filled: bool = False, 
                    min_locule_size: int = 300, n_blur: int = 11, 
                    clip_limit: int = 4, 
                    tile_grid_size: int = 8, remove_roi: bool = True, 
                    roi_expansion: int = 10) -> None:
        """
        Create a mask for fruit detection and segmentation.
        
        This method generates a binary mask to identify fruits in the image with support
        for stamp inversion, locule detection, and automatic ROI removal.
        
        Args:
            stamp: Set to True if image has inverted colors (black background). Default is False.
            plot: Whether to display the generated mask. Default is False.
            plot_size: Figure size for plotting (width, height). Default is (5, 5).
            locules_filled: Enable detection of internal fruit structures (useful for tomatoes, 
                oranges, cucumbers). Default is False.
            remove_roi: Automatically remove label and reference regions from the mask. 
                Default is True.
        
        Advanced Parameters (for fine-tuning):
            n_kernel: Morphological operations kernel size. Default is 5.
            n_iteration: Iterations for morphological operations. Default is 1.
            canny_min: Canny edge detector lower threshold. Default is 30.
            canny_max: Canny edge detector upper threshold. Default is 100.
            lower_hsv: Lower HSV bound [H, S, V] for color filtering.
            upper_hsv: Upper HSV bound [H, S, V] for color filtering.
            min_locule_size: Minimum area for locule detection. Default is 300.
            n_blur: Median blur kernel size for noise reduction. Default is 11.
            clip_limit: CLAHE contrast limiting threshold. Default is 4.
            tile_grid_size: CLAHE grid size for histogram equalization. Default is 8.
            roi_expansion: Pixels to expand ROIs before removal. Default is 10.
            plot_axis: Show axis on plot. Default is False.
        
        Returns:
            None: Modifies self.mask with the generated binary mask.
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
        
        if locules_filled:  # Useful for tomato, orange, cucumber, etc...
            # Use the already calculated mask instead of recalculating
            base_mask = self.mask.copy()
            
            # Fill fruit contours
            contours, _ = cv2.findContours(base_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cv2.drawContours(base_mask, [cnt], -1, 255, -1)
                
            # Convert image to Lab for locule processing
            lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
            l_channel = lab[:,:,0]

            clahe = cv2.createCLAHE(clipLimit=clip_limit, 
                                   tileGridSize=(tile_grid_size, tile_grid_size))
            l_clahe = clahe.apply(l_channel)

            _, locule_mask = cv2.threshold(l_clahe, 0, 255, 
                                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            locule_mask = cv2.medianBlur(locule_mask, n_blur)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n_kernel, n_kernel))
            opened = cv2.morphologyEx(locule_mask, cv2.MORPH_OPEN, kernel)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

            # Detect only internal contours (locules)
            inv_closed = cv2.bitwise_not(closed)
            contours, hierarchy = cv2.findContours(inv_closed, cv2.RETR_TREE, 
                                                  cv2.CHAIN_APPROX_SIMPLE)
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

        # Deleting label and reference squares from mask
        if remove_roi:
            # Create a same size black mask 
            mask_rois = np.zeros_like(self.mask)
            
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
            
            # Dilate the roi mask if needed
            if roi_expansion > 0:
                kernel_expand = np.ones((roi_expansion, roi_expansion), np.uint8)
                mask_rois = cv2.dilate(mask_rois, kernel_expand, iterations=1)
            
            # Remove label and reference from the original mask
            self.mask = cv2.bitwise_and(self.mask, cv2.bitwise_not(mask_rois))

        if plot:
            cf.plot_img(self.mask, metadata=False, plot_size=plot_size, fig_axis=plot_axis)

        return None

    def find_fruits(self, min_circularity: float = 0.5, output_message: bool = True, 
                    min_locule_area: int = 50, min_locule_per_fruit: int = 1, 
                    max_circularity: float = 1.0, min_aspect_ratio: float = 0.3, 
                    max_aspect_ratio: float = 3.0, 
                    contour_filters: Optional[Dict] = None) -> None:
        """
        Detect fruits and their locules in the mask.
        
        Args:
            min_circularity: Minimum circularity for fruit detection. Default is 0.5.
            output_message: Whether to show detection results. Default is True.
            min_locule_area: Minimum area for locule detection. Default is 50.
            min_locule_per_fruit: Minimum locules per fruit. Default is 1.
            max_circularity: Maximum circularity for filtering. Default is 1.0.
            min_aspect_ratio: Minimum aspect ratio for filtering. Default is 0.3.
            max_aspect_ratio: Maximum aspect ratio for filtering. Default is 3.0.
            contour_filters: Additional contour filters. Default is None.
        """
        self.contours, self.fruit_locus_map = find_fruits(
            self.mask, 
            min_circularity=min_circularity,
            min_locule_area=min_locule_area,
            max_circularity=max_circularity,
            min_locules_per_fruit=min_locule_per_fruit,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
            contour_filters=contour_filters
        )
        
        if self.fruit_locus_map is not None:
            n_fruits_detected = len(self.fruit_locus_map)
        else:
            n_fruits_detected = '0'

        if output_message:
            print(f'Total detected objects: {len(self.contours)}')
            print(f'Detected fruits after filtering: {n_fruits_detected}')

        return None

    def analyze_image(self, 
                      plot: bool = True, 
                      plot_size: Tuple[int, int] = (10, 10), 
                      font_scale: float = 1.5, 
                      font_thickness: int = 2, 
                      text_color: Tuple[int, int, int] = (0, 0, 0),  # Black by default
                      label_position: str = 'top',  # Options: 'top', 'bottom', 'left', 'right'
                      label_color: Tuple[int, int, int] = (255, 255, 255),  # White by default
                      use_ellipse: bool = False, 
                      contour_mode: str = 'raw', 
                      stamp: bool = False, 
                      epsilon_factor: float = 0.001, 
                      centroid_fruit: int = 2,
                      centroid_locules: int = 2,
                      padding: int = 15, 
                      line_spacing: int = 15, 
                      min_locule_area: int = 100, 
                      min_distance: int = 0, 
                      max_distance: int = 100,
                      max_locule_area: Optional[int] = None, 
                      merge_locules: bool = False,
                      n_shifts: int = 500, 
                      angle_weight: float = 0.5, 
                      radius_weight: float = 0.5,
                      min_radius_threshold: float = 0.1, 
                      num_rays: int = 360) -> AnnotatedImage:
        """
        Analyze detected fruits using analysis.analyze_fruits().
        
        Args:
            plot: Whether to display the annotated image. Default is True.
            plot_size: Figure size for plotting. Default is (10, 10).
            font_scale: Font scale for annotations. Default is 1.5.
            font_thickness: Font thickness for annotations. Default is 2.
            text_color: Text color in BGR format. Default is (0, 0, 0) - black.
            label_position: Position of labels ('top', 'bottom', 'left', 'right'). 
                Default is 'top'.
            label_color: Background color for labels in BGR format. 
                Default is (255, 255, 255) - white.
            use_ellipse: Use ellipse for pericarp calculation. Default is False.
            contour_mode: Contour mode ('raw', 'hull', 'approx', 'ellipse', 'circle'). 
                Default is 'raw'.
            stamp: Invert image colors. Default is False.
            epsilon_factor: Approximation factor for contours. Default is 0.001.
            centroid_fruit: Radius for fruit centroid marker. Default is 2.
            centroid_locules: Radius for locule centroid marker. Default is 2.
            padding: Padding around text labels. Default is 15.
            line_spacing: Spacing between text lines. Default is 15.
            min_locule_area: Minimum locule area for filtering. Default is 100.
            min_distance: Minimum distance for locule merging. Default is 0.
            max_distance: Maximum distance for locule merging. Default is 100.
            max_locule_area: Maximum locule area for filtering. Default is None.
            merge_locules: Whether to merge close locules. Default is False.
            n_shifts: Number of shifts for angular symmetry. Default is 500.
            angle_weight: Weight for angle in rotational symmetry. Default is 0.5.
            radius_weight: Weight for radius in rotational symmetry. Default is 0.5.
            min_radius_threshold: Minimum radius threshold for symmetry. Default is 0.1.
            num_rays: Number of rays for pericarp thickness. Default is 360.
        
        Returns:
            AnnotatedImage: Object containing results and annotated image
        """
        # Call analyze_fruits with single px_per_cm value
        self.results = analyze_fruits(
            img=self.img_annotated,
            contours=self.contours,
            fruit_locus_map=self.fruit_locus_map,
            px_per_cm=self.px_per_cm, 
            img_name=self.img_name,
            label_text=self.label_text,
            use_ellipse=use_ellipse,
            contour_mode=contour_mode,
            plot=plot,
            text_color=text_color,
            label_position=label_position,
            stamp=stamp,
            plot_size=plot_size,
            font_scale=font_scale,
            font_thickness=font_thickness,
            padding=padding,
            line_spacing=line_spacing,
            min_locule_area=min_locule_area,
            max_locule_area=max_locule_area,
            merge_locules=merge_locules,
            bg_color=label_color,
            epsilon_factor=epsilon_factor,
            min_distance=min_distance,
            max_distance=max_distance,
            path=self.image_path,
            num_shifts=n_shifts,
            angle_weight=angle_weight,
            radius_weight=radius_weight,
            min_radius_threshold=min_radius_threshold,
            num_rays=num_rays,
            centroid_locules=centroid_locules,
            centroid_fruit=centroid_fruit
        )

        return self.results

    def _process_single_file(self, filename: str, config: Dict[str, Any]) -> Tuple[Optional[pd.DataFrame], Dict[str, str], int]:
        """
        Internal method to process a single image file.
        Used for parallel processing.
        
        Args:
            filename: Name of the image file to process
            config: Configuration dictionary with processing parameters
        
        Returns:
            Tuple containing:
                - DataFrame with results (or None if processing failed)
                - Dictionary with processing status
                - Number of fruits detected
        """
        import sys
        from io import StringIO
        
        error_dict = {'filename': filename, 'status': 'Unknown'}
        
        try:
            # Suppress all print statements during batch processing
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            try:
                # Create analyzer instance
                file_path = os.path.join(config['path_input'], filename)
                analyzer = FruitAnalyzer(file_path)
                
                # 1. Read image
                analyzer.read_image()
                
                # 2. Setup measurements (calibration) - verbose=False suppresses output
                analyzer.setup_measurements(
                    confidence=config['confidence_threshold'],
                    diameter_cm=config['diameter_cm'],
                    width_cm=config.get('width_cm'),
                    length_cm=config.get('length_cm'),
                    detect_label=config.get('detect_label', True),
                    verbose=False,
                    plot=False, gpu=False
                )
                
                # 3. Create mask
                analyzer.create_mask(
                    stamp=config['stamp'],
                    n_kernel=config['n_kernel'],
                    **config.get('mask_kwargs', {})
                )
                
                # 4. Find fruits
                analyzer.find_fruits(
                    min_circularity=config['min_circularity'],
                    output_message=False
                )
                
            finally:
                # Restore stdout
                sys.stdout = old_stdout
            
            if not analyzer.contours:
                error_dict['status'] = 'No contours found'
                return None, error_dict, 0
            
            # 5. Analyze image
            results = analyzer.analyze_image(
                plot=False,
                contour_mode=config['contour_mode'],
                stamp=config['stamp'],
                font_scale=config['font_scale'],
                font_thickness=config['font_thickness'],
                text_color=config.get('text_color', (0, 0, 0)),
                label_position=config.get('label_position', 'top'),
                label_color=config.get('label_color', (255, 255, 255)),
                padding=config['padding'],
                line_spacing=config['line_spacing'],
                min_locule_area=config['min_locule_area'],
                max_locule_area=config.get('max_locule_area'),
                merge_locules=config.get('merge_locules', False),
                epsilon_factor=config['epsilon_factor'],
                use_ellipse=config['use_ellipse_fruit'],
                min_distance=config['min_distance'],
                max_distance=config['max_distance'],
                n_shifts=config['n_shifts'],
                angle_weight=config['angle_weight'],
                radius_weight=config['radius_weight'],
                min_radius_threshold=config['min_radius_threshold'],
                num_rays=config.get('num_rays', 360),
                centroid_fruit=config.get('centroid_fruit', 2),
                centroid_locules=config.get('centroid_locules', 2)
            )
            
            # Get results - optimized with getattr
            current_results = getattr(results, 'results', [])
            
            if not current_results:
                error_dict['status'] = 'No valid fruits detected'
                return None, error_dict, 0
            
            # Save annotated image - optimized with getattr and or operator
            try:
                annotated_path = os.path.join(config['output_dir'], f"annotated_{filename}")
                img_to_save = getattr(results, 'rgb_image', None)
                
                if img_to_save is None:
                    img_to_save = getattr(results, 'annotated_img', None)
                    
                if img_to_save is None:
                    raise AttributeError("No image attribute found")
                
                cv2.imwrite(annotated_path, cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))

            except Exception as e:
                error_dict['status'] = f'Error saving image: {str(e)}'
                return None, error_dict, 0
            
            # Create DataFrame
            df = pd.DataFrame(current_results)
            df['source_file'] = filename
            n_fruits = len(current_results)

            error_dict['status'] = 'Successfully processed'
            return df, error_dict, n_fruits
            
        except Exception as e:
            # Restore stdout in case of error
            sys.stdout = old_stdout
            error_dict['status'] = f'Processing error: {str(e)}'
            return None, error_dict, 0


    def analyze_folder(self, 
                    output_dir: Optional[str] = None, 
                    stamp: bool = False, 
                    contour_mode: str = 'raw', 
                    n_kernel: int = 5, 
                    min_circularity: float = 0.3, 
                    font_scale: float = 1.5, 
                    font_thickness: int = 2, 
                    text_color: Tuple[int, int, int] = (0, 0, 0),  # Black
                    label_position: str = 'top',  # 'top', 'bottom', 'left', 'right'
                    label_color: Tuple[int, int, int] = (255, 255, 255),  # White
                    padding: int = 15, 
                    line_spacing: int = 15, 
                    min_locule_area: int = 300, 
                    epsilon_factor: float = 0.005, 
                    use_ellipse_fruit: bool = False, 
                    min_distance: int = 2, 
                    max_distance: int = 30,
                    max_locule_area: Optional[int] = None, 
                    merge_locules: bool = False,
                    n_shifts: int = 100, 
                    angle_weight: float = 0.5, 
                    radius_weight: float = 0.5,
                    min_radius_threshold: float = 0.1,
                    num_rays: int = 180,
                    centroid_fruit: int = 2,
                    centroid_locules: int = 2,
                    n_cores: int = 1,
                    # Calibration parameters (for setup_measurements)
                    confidence_threshold: float = 0.6,
                    diameter_cm: float = 2.5,
                    width_cm: Optional[float] = None,
                    length_cm: Optional[float] = None,
                    detect_label: bool = False, gpu: bool = False,
                    **kwargs) -> None:
        """
        Process all images in a folder with optional parallel processing.
        
        Args:
            output_dir: Output directory for results. Default creates 'Results' subfolder.
            stamp: Invert image colors. Default is False.
            contour_mode: Contour mode ('raw', 'hull', 'approx', 'ellipse', 'circle'). 
                Default is 'raw'.
            n_kernel: Morphological kernel size. Default is 7.
            min_circularity: Minimum circularity for fruit detection. Default is 0.3.
            font_scale: Font scale for annotations. Default is 1.5.
            font_thickness: Font thickness for annotations. Default is 2.
            text_color: Text color in BGR format. Default is (0, 0, 0) - black.
            label_position: Position of labels ('top', 'bottom', 'left', 'right'). 
                Default is 'top'.
            label_color: Background color for labels in BGR. Default is (255, 255, 255) - white.
            padding: Padding around text labels. Default is 15.
            line_spacing: Spacing between text lines. Default is 15.
            min_locule_area: Minimum locule area. Default is 300.
            epsilon_factor: Approximation factor for contours. Default is 0.005.
            use_ellipse_fruit: Use ellipse for pericarp. Default is False.
            min_distance: Minimum distance for locule merging. Default is 2.
            max_distance: Maximum distance for locule merging. Default is 30.
            max_locule_area: Maximum locule area. Default is None.
            merge_locules: Whether to merge close locules. Default is False.
            n_shifts: Number of shifts for angular symmetry. Default is 500.
            angle_weight: Weight for angle in rotational symmetry. Default is 0.5.
            radius_weight: Weight for radius in rotational symmetry. Default is 0.5.
            min_radius_threshold: Minimum radius threshold for symmetry. Default is 0.1.
            num_rays: Number of rays for pericarp thickness. Default is 360.
            centroid_fruit: Radius for fruit centroid marker. Default is 2.
            centroid_locules: Radius for locule centroid marker. Default is 2.
            n_cores: Number of CPU cores to use. Default is 1 (sequential).
            
        Calibration parameters (used in setup_measurements):
            confidence_threshold: Confidence for circle detection. Default is 0.6.
            diameter_cm: Reference circle diameter in cm. Default is 2.5.
            width_cm: Image width in cm for manual calibration. Default is None.
            length_cm: Image length in cm for manual calibration. Default is None.
            detect_label: Whether to detect QR/labels. Default is True.
            
        Additional parameters:
            **kwargs: Additional parameters passed to create_mask()
        
        Returns:
            None: Saves results to CSV files in output directory
        """
        if not self.is_directory:
            raise ValueError("This instance was initialized with a single image. "
                        "Use analyze_image() instead or initialize FruitAnalyzer() with a folder path.")
        
        path_input = self.image_path
        output_dir = os.path.join(path_input, "Results") if output_dir is None else output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Validate n_cores
        max_cores_available = mp.cpu_count()
        if n_cores > max_cores_available:
            print(f"Warning: n_cores={n_cores} exceeds available cores ({max_cores_available})")
            print(f"   Setting n_cores to {max_cores_available}")
            n_cores = max_cores_available
        
        if n_cores < 1:
            n_cores = 1
        
        if n_cores > 4:
            print(f"Warning: Using {n_cores} cores may slow down your PC.")
            user_input = input("   Continue anyway? (y/n): ").lower()
            if user_input != 'y':
                print("   Aborting.")
                return
        
        # Initialize counters
        processed_count = 0
        skipped_no_contours = 0
        skipped_errors = 0
        total_fruits = 0
        all_results = []
        errors_report = []
        
        start_time = time.time()
        from datetime import datetime
        start_datetime = datetime.now()
        process = psutil.Process()
        initial_ram_gb = (process.memory_info().rss / 1024**2) / 1024

        # Get file list
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        file_list = [f for f in os.listdir(path_input) 
                    if os.path.splitext(f)[1].lower() in valid_extensions]
        
        total_images = len(file_list)
        
        if total_images == 0:
            print("No valid images found.")
            return

        # Configuration dictionary (only valid parameters)
        config = {
            'path_input': path_input,
            'output_dir': output_dir,
            'stamp': stamp,
            'contour_mode': contour_mode,
            'n_kernel': n_kernel,
            'min_circularity': min_circularity,
            'font_scale': font_scale,
            'font_thickness': font_thickness,
            'text_color': text_color,
            'label_position': label_position,
            'label_color': label_color,
            'padding': padding,
            'line_spacing': line_spacing,
            'min_locule_area': min_locule_area,
            'epsilon_factor': epsilon_factor,
            'use_ellipse_fruit': use_ellipse_fruit,
            'min_distance': min_distance,
            'max_distance': max_distance,
            'max_locule_area': max_locule_area,
            'merge_locules': merge_locules,
            'n_shifts': n_shifts,
            'angle_weight': angle_weight,
            'radius_weight': radius_weight,
            'min_radius_threshold': min_radius_threshold,
            'num_rays': num_rays,
            'centroid_fruit': centroid_fruit,
            'centroid_locules': centroid_locules,
            # Calibration parameters
            'confidence_threshold': confidence_threshold,
            'diameter_cm': diameter_cm,
            'width_cm': width_cm,
            'length_cm': length_cm,
            'detect_label': detect_label,
            # Mask kwargs
            'mask_kwargs': kwargs
        }

        print("Traitly running ⋆✧｡٩(ˊᗜˋ )و✧*｡   ")
        
        # Process images (sequential or parallel)
        if n_cores == 1:
            print(f"Sequential mode: processing {total_images} image(s) one at a time")
            
            for filename in tqdm(file_list, desc='Processing images', unit='image'):
                df, error_dict, n_fruits = self._process_single_file(filename, config)
                
                if df is not None and not df.empty:
                    all_results.append(df)
                    processed_count += 1
                    total_fruits += n_fruits
                else:
                    if 'No contours' in error_dict['status']:
                        skipped_no_contours += 1
                    else:
                        skipped_errors += 1
                
                errors_report.append(error_dict)
        
        else:
            images_per_core = total_images / n_cores
            print(f"Parallel mode: using {n_cores} core(s)")
            print(f"Total images: {total_images}")
            print(f"Images per core: ~{images_per_core:.1f}")
            
            with ProcessPoolExecutor(max_workers=n_cores) as executor:
                future_to_file = {
                    executor.submit(self._process_single_file, f, config): f 
                    for f in file_list
                }
                
                for future in tqdm(as_completed(future_to_file), 
                                total=total_images, 
                                desc='Processing images', 
                                unit='image'):
                    
                    filename = future_to_file[future]
                    
                    try:
                        df, error_dict, n_fruits = future.result()
                        
                        if df is not None and not df.empty:
                            all_results.append(df)
                            processed_count += 1
                            total_fruits += n_fruits
                        else:
                            if 'No contours' in error_dict['status']:
                                skipped_no_contours += 1
                            else:
                                skipped_errors += 1
                        
                        errors_report.append(error_dict)
                        
                    except Exception as e:
                        print(f"Error processing {filename}: {str(e)}")
                        errors_report.append({
                            'filename': filename,
                            'status': f'Unexpected error: {str(e)}'
                        })
                        skipped_errors += 1

        # Calculate statistics - optimized (single time calculation)
        elapsed_time = time.time() - start_time
        elapsed_min = elapsed_time / 60
        final_ram_gb = (process.memory_info().rss / 1024**2) / 1024
        ram_used_gb = final_ram_gb - initial_ram_gb
        total_imgs = skipped_no_contours + processed_count + skipped_errors
        seg_per_img = elapsed_time / total_imgs if total_imgs > 0 else 0

        # Save results - optimized concatenation
        if all_results:
            final_df = pd.concat(all_results, ignore_index=True, copy=False)
            final_df.to_csv(
                os.path.join(output_dir, "all_results.csv"), 
                index=False
            )
        
        # Save error report - optimized
        if errors_report:
            error_csv_path = os.path.join(output_dir, "error_report.csv")
            pd.DataFrame(errors_report).to_csv(error_csv_path, index=False)
            error_message = f"> Error report saved in: {error_csv_path}"
        else: 
            error_message = "! No errors found"
        
        # Save session report
        # session_report = {
        #     'date': [start_datetime.strftime('%Y-%m-%d')],
        #     'time': [start_datetime.strftime('%H:%M:%S')],
        #     'total_images': [total_imgs],
        #     'processed_images': [processed_count],
        #     'total_fruits': [total_fruits],
        #     'skipped_no_contours': [skipped_no_contours],
        #     'skipped_errors': [skipped_errors],
        #     'processing_time_minutes': [round(elapsed_min, 2)],
        #     'avg_time_per_image_sec': [round(seg_per_img, 2)],
        #     'ram_used_gb': [round(ram_used_gb, 2)],
        #     'n_cores': [n_cores],
        #     'contour_mode': [contour_mode],
        #     'stamp': [stamp],
        #     'min_circularity': [min_circularity],
        #     'min_locule_area': [min_locule_area],
        #     'merge_locules': [merge_locules],
        #     'confidence_threshold': [confidence_threshold],
        #     'diameter_cm': [diameter_cm],
        #     'detect_label': [detect_label],
        #     'use_ellipse_fruit': [use_ellipse_fruit],
        # }
        
        # session_df = pd.DataFrame(session_report)
        # session_csv_path = os.path.join(output_dir, "session_report.csv")
        # session_df.to_csv(session_csv_path, index=False)

        # Save session report - optimized with single join
        separator = "══════════════════════════════════════════════"
        session_report_text = '\n'.join([
            separator,
            "SESSION DETAILS:",
            separator,
            f"date: {start_datetime.strftime('%Y-%m-%d')}",
            f"time: {start_datetime.strftime('%H:%M:%S')}",
            f"folder_path: {path_input}",
            f"total_images_detected: {total_imgs}",
            f"processed_images: {processed_count}",
            f"total_fruits_processed: {total_fruits}",
            f"skipped_no_contours: {skipped_no_contours}",
            f"skipped_errors: {skipped_errors}",
            f"processing_time_minutes: {round(elapsed_min, 2)}",
            f"avg_time_per_image_sec: {round(seg_per_img, 2)}",
            f"ram_used_gb: {round(ram_used_gb, 2)}",
            "",
            separator,
            "ARGUMENTS:",
            separator,
            f"n_cores: {n_cores}",
            f"contour_mode: {contour_mode}",
            f"stamp: {stamp}",
            f"min_circularity: {min_circularity}",
            f"min_locule_area: {min_locule_area}",
            f"merge_locules: {merge_locules}",
            f"confidence_threshold: {confidence_threshold}",
            f"diameter_cm: {diameter_cm}",
            f"detect_label: {detect_label}",
            f"use_ellipse_fruit: {use_ellipse_fruit}"
        ])

        # Save as .txt - optimized single write
        session_txt_path = os.path.join(output_dir, "session_report.txt")
        with open(session_txt_path, 'w', encoding='utf-8') as f:
            f.write(session_report_text + '\n')


        # Print summary
        print()
        print(" ( ദ്ദി ˙ᗜ˙ )   Processing completed !")               
        print("══════════════════════════════════════════════════════════════════════")
        print(f"- Total time: {elapsed_min:.2f} minutes (~{seg_per_img:.2f} sec/image)")
        print(f"-  Total images processed: {total_imgs}") 
        print(f"- Successfully annotated: {processed_count}") 
        #print(f"-  Skipped (no contours): {skipped_no_contours}") 
        print(f"- Failed (errors): {skipped_errors}") 
        print(f"- RAM used: {ram_used_gb:.2f} GB (peak: {final_ram_gb:.2f} GB)")
        print(f"> Output folder: {output_dir}")
        print(error_message)
        
        if n_cores > 1:
            images_per_minute = total_imgs / elapsed_min if elapsed_min > 0 else 0
            print(f"> Cores used: {n_cores}")
            print(f"> Throughput: {images_per_minute:.2f} images/minute")

        return None
    



    ###
