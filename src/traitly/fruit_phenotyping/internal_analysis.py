# traitly/fruit_phenotyping/internal_analysis.py

"""
Internal fruit analysis pipeline for traitly.

Provides :class:`FruitInternalAnalyzer`, the core analyzer class for
whole-fruit and internal morphology, color, and symmetry analysis.
Supports single-image and batch folder processing with optional
multiprocessing via :func:`_process_image_worker`.

The typical analysis pipeline follows this order:

1. :meth:`~FruitInternalAnalyzer.load_image`
2. :meth:`~FruitInternalAnalyzer.setup_measurements`
3. :meth:`~FruitInternalAnalyzer.generate_fruit_mask`
4. :meth:`~FruitInternalAnalyzer.enhance_locule_contrast` *(optional)*
5. :meth:`~FruitInternalAnalyzer.generate_locule_mask` *(optional)*
6. :meth:`~FruitInternalAnalyzer.detect_fruits`
7. :meth:`~FruitInternalAnalyzer.analyze_morphology` and/or :meth:`~FruitInternalAnalyzer.analyze_color`

For batch processing, steps 1–7 are orchestrated automatically by
:meth:`~FruitInternalAnalyzer.analyze_folder` or
:meth:`~FruitInternalAnalyzer.process_single_file`.
"""

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
                   generate_scatter_plot,
                   generate_l_channel_histogram,
                   interactive_mask_editor)

from .fruit_config import analyze_fruits_morphology
from ..utils.basic_functions import (load_img,
                                      detect_img_name)

from .processing import annotate_all_fruits

from ..utils.calibration import px_cm_density
from ..utils.label import (detect_qr,
                        detect_label_box_yolo,
                        detect_label_box,
                        detect_label_text)

from traitly import __version__
from ..utils.constants import valid_extensions
from .results_image import ResultsImage
from .color_analysis import (get_single_fruit_masks, 
                             analyze_all_fruits_color, 
                             get_fruit_color_histograms)
from .analysis_parameters import AnalysisParameters

##########################################################################################
# Ignore warnings from torch 
##########################################################################################

warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', message='Using CPU')

##########################################################################################
# Worker function for parallel processing 
##########################################################################################

def _process_image_worker(
    img_path: str,
    config: Dict,
    analyze_morphology: bool,
    analyze_color: bool,
) -> Tuple:
    """
    Worker function for parallel processing of a single image.

    Instantiates a :class:`FruitInternalAnalyzer`, loads the image, and
    runs the full pipeline via :meth:`~FruitInternalAnalyzer.process_single_file`.
    Designed to be called inside a
    :class:`~concurrent.futures.ProcessPoolExecutor`.

    Parameters
    ----------
    img_path : str
        Absolute path to the image file to process.
    config : dict
        Analysis configuration dictionary forwarded to
        :meth:`~FruitInternalAnalyzer.process_single_file`.
    analyze_morphology : bool
        If True, run morphology analysis.
    analyze_color : bool
        If True, run color analysis.

    Returns
    -------
    tuple
        ``(df_morphology, df_color, error_dict, n_fruits, annotated_img,
        filename, elapsed)``. On failure, ``df_morphology`` and
        ``df_color`` are ``None`` and ``error_dict`` contains the error
        message.
    """

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
    """
    Core analyzer for fruit morphology and color from segmented images.

    Manages the full single-image pipeline: image loading, scale
    calibration, label detection, mask generation, fruit and locule
    detection, morphological feature extraction, and color analysis.
    Results are stored in :attr:`results` as a :class:`ResultsImage`
    and analysis metadata is tracked in :attr:`parameters`.

    For external (whole-fruit only) analysis without locule segmentation,
    use :class:`~traitly.fruit_phenotyping.external_analysis.FruitExternalAnalyzer`.

    Parameters
    ----------
    image_path : str
        Path to an image file or a folder containing images. Raises
        :exc:`FileNotFoundError` if the path does not exist.
    """
    
    def __init__(self, image_path: str) -> None:
        """
        Initialize the analyzer and validate the image path.

        Parameters
        ----------
        image_path : str
            Path to an image file or a directory. Raises
            :exc:`FileNotFoundError` if the path does not exist.
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
        self.alpha = None
        
        # save metadata
        self.parameters = AnalysisParameters() 
        self.is_metadata_saved = True
        self.is_morphology_results = None


    ##########################################################################################
    ## Load and display an image 
    ##########################################################################################
    def load_image(
        self,
        plot: bool = True,
        plot_size: Tuple[int, int] = (5, 5),
        show_axis: bool = False,
        x: Optional[int] = None,
        y: Optional[int] = None,
        w: Optional[int] = None,
        h: Optional[int] = None
    ) -> None:
        """
        Load the image from :attr:`img_path` and prepare internal representations.
        Delegates to :func:`~traitly.utils.basic_functions.load_img` and
        populates :attr:`img`, :attr:`img_copy`, :attr:`img_rgb`,
        :attr:`img_hsv`, :attr:`img_shape`, and :attr:`img_name`.

        Parameters
        ----------
        plot : bool, optional
            If True, display the loaded image. Default is True.
        plot_size : tuple of int, optional
            Figure size ``(width, height)`` for the plot. Default is (5, 5).
        show_axis : bool, optional
            If True, display axis ticks and labels on the plot. Default is False.
        x : int, optional
            Left pixel coordinate of the crop region.
        y : int, optional
            Top pixel coordinate of the crop region.
        w : int, optional
            Width of the crop region in pixels.
        h : int, optional
            Height of the crop region in pixels.

        Raises
        ------
        ValueError
            If no image path was set, the file extension is unsupported, or
            the image failed to load.
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
        
        self.img = load_img(self.img_path, plot = plot, 
                            plot_size = plot_size,
                            show_axis = show_axis, 
                            x = x, y = y, w = w, h = h)

        if self.img is None:
            raise ValueError(f"Failed to load image: {self.img_path}."
                             "The file may be corrupted or not in a supported format.")

        self.img_shape = self.img.shape[:2]
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
        Detect QR code, label ROI, and label text for the loaded image.

        Runs up to three detection steps in order:

        1. QR code detection via :func:`~traitly.utils.basic_functions.detect_qr`
        (skipped if ``skip_qr=True``).
        2. Label ROI detection via
        :func:`~traitly.utils.basic_functions.detect_label_box_yolo` then
        :func:`~traitly.utils.basic_functions.detect_label_box` as fallback
        (skipped if ``skip_label_detection=True``).
        3. OCR via :func:`~traitly.utils.basic_functions.detect_label_text`
        only if a ROI was found and no QR text was detected.

        Populates :attr:`label_text` and :attr:`label_roi`.

        Parameters
        ----------
        verbose : bool, optional
            If True, print detection results. Default is True.
        detect_label : bool, optional
            If False, skip OCR and only attempt ROI detection. Default is
            False.
        language_label : list of str, optional
            OCR language codes forwarded to
            :func:`~traitly.utils.basic_functions.detect_label_text`.
            Default is ``["es", "en"]``.
        blur_label : tuple of int, optional
            Gaussian blur kernel size applied before OCR. Default is
            ``(11, 11)``.
        gpu : bool, optional
            If True, use GPU acceleration for OCR. Default is True.
        skip_qr : bool, optional
            If True, skip QR code detection. Default is False.
        skip_label_detection : bool, optional
            If True, skip both ROI detection and OCR. Default is False.
        """

        if verbose:
            print("\n" + "=" * 55)
            print("★ LABEL DETECTION:")
            print("=" * 55)

        if not detect_label:
            # Try to detect label roi
            self.label_roi = None
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
        Detect the size reference and calculate the pixel-to-centimetre factor.

        When ``fast_calibration=True`` and both ``width_cm`` and ``length_cm``
        are provided, the scale is derived geometrically without YOLO detection.
        Otherwise, delegates to
        :func:`~traitly.utils.basic_functions.px_cm_density` for automatic
        reference detection.

        Populates :attr:`px_per_cm` and :attr:`ref_roi`.

        Parameters
        ----------
        verbose : bool, optional
            If True, print calibration results and warnings. Default is True.
        confidence : float, optional
            Minimum detection confidence forwarded to
            :func:`~traitly.utils.basic_functions.px_cm_density`. Default is
            0.6.
        font_size : int, optional
            Font size for annotations drawn on the image by
            :func:`~traitly.utils.basic_functions.px_cm_density`. Default is 3.
        width_cm : float or None, optional
            Known physical width of the reference object in centimetres.
            Used in both fast and standard calibration modes. Default is None.
        length_cm : float or None, optional
            Known physical length of the reference object in centimetres.
            Required for fast calibration. Default is None.
        diameter_cm : float or None, optional
            Known diameter of the reference object. Defaults to 2.5 cm if
            not provided. Default is None.
        fast_calibration : bool, optional
            If True and ``width_cm`` and ``length_cm`` are set, skip YOLO
            detection and calculate scale using phyisical data; else, if 
            ``width_cm`` and ``length_cm`` are None, return pixel measurements. 
            Default is False.
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

        # create an image copy to work with
        self.img_copy = self.img.copy()

        if fast_calibration:
            if  width_cm and length_cm:
                # Fast method: use physical dimensions
                self.px_per_cm = np.sqrt((w * h) / (width_cm * length_cm))
                self.ref_roi = None
                if verbose:
                    print("> Size reference detection: SKIPPED.")
            else:
                # No calibration available: measurements will be in pixels
                self.px_per_cm = None
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
        scale_factor: float = 0.5,
    ) -> None:
        """
        Detect label text and calculate the pixel-to-centimetre scale factor.

        Orchestrates :meth:`setup_label` and :meth:`setup_calibration` in
        order, and optionally runs :meth:`detect_color_checker`. Populates
        :attr:`label_text`, :attr:`label_roi`, :attr:`ref_roi`, and
        :attr:`px_per_cm`.

        Parameters
        ----------
        plot_reference : bool, optional
            If True, display a cropped view of each detected reference ROI.
            Default is False.
        plot_color_checker : bool, optional
            If True, display the detected color checker region. Default is
            False.
        font_size : int, optional
            Font size forwarded to :meth:`setup_calibration`. Default is 3.
        confidence : float, optional
            Detection confidence forwarded to :meth:`setup_calibration`.
            Default is 0.6.
        detect_label : bool, optional
            If True, run full label detection including OCR in
            :meth:`setup_label`. Default is False.
        verbose : bool, optional
            If True, print progress and results. Default is True.
        plot_size : tuple of int, optional
            Figure size for reference ROI plots. Default is (5, 5).
        language_label : list of str, optional
            OCR language codes forwarded to :meth:`setup_label`. Default is
            ``["es", "en"]``.
        width_cm : float or None, optional
            Known reference width in centimetres, forwarded to
            :meth:`setup_calibration`. Default is None.
        length_cm : float or None, optional
            Known reference length in centimetres, forwarded to
            :meth:`setup_calibration`. Default is None.
        diameter_cm : float or None, optional
            Known reference diameter in centimetres, forwarded to
            :meth:`setup_calibration`. Default is None.
        gpu : bool, optional
            If True, use GPU for OCR in :meth:`setup_label`. Default is False.
        skip_qr : bool, optional
            If True, skip QR detection in :meth:`setup_label`. Default is
            False.
        fast_calibration : bool, optional
            If True and ``width_cm`` and ``length_cm`` are set, skip YOLO
            detection in :meth:`setup_calibration`. Default is False.
        detect_color_checker : bool, optional
            If True, run :meth:`detect_color_checker` after calibration.
            Default is False.
        scale_factor : float, optional
            Downscaling factor for color checker detection in
            :meth:`detect_color_checker`. Must be in [0.1, 1.0]. Default is
            0.5.

        Raises
        ------
        ValueError
            If no image is loaded or ``scale_factor`` is outside [0.1, 1.0].
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
        
        # self.img_copy = cv2.cvtColor(self.img_copy, cv2.COLOR_BGR2RGB)

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
    #   
    #                                   MASKS

    ##########################################################################################
    # OPTIONAL : Create a histogram to visualize pixel intensity for L channel
    ##########################################################################################

    def generate_l_channel_histogram(self,
        otsu_offset: int = 0,
        plot_size: Tuple[int,int] = (9,3)
    ) -> None:
        """
        """

        if self.mask_fruit is None:
            raise ValueError("No mask available. Run generate_fruit_mask() first.")
        
        if self.l_transformed is None:
            raise ValueError("Locule contrast not initialized. Run enhance_locule_contrast() first "
                             "(use contrast_method = 'none' if no transformation is requiered)")
        

        generate_l_channel_histogram(l_transformed = self.l_transformed,
                                     fruit_mask = self.mask_fruit,
                                     otsu_offset= otsu_offset,
                                     plot_size = plot_size)
        

    ##########################################################################################
    # OPTIONAL : Open an interactive mask editor
    ##########################################################################################

    def edit_mask(self, verbose: bool = True) -> None:
        """Manually edit the locule mask if available, otherwise the fruit mask."""

        if self.mask_locules is not None:
            mask = 'mask_locules'
        elif self.mask_fruit is not None:
            mask = 'mask_fruit'
        else:
            raise ValueError("No mask found. Run generate_fruit_mask() and optionally, generate_locule_mask() first.")

        if verbose:  
            controls = [
                ("Left click",       "add polygon point (both panels)"),
                ("Right click drag", "pan"),
                ("W",                "fill polygon WHITE (add region)"),
                ("B",                "fill polygon BLACK (remove region)"),
                ("Enter",            "apply current polygon"),
                ("Z",                "undo last edit"),
                ("C",                "clear current polygon points"),
                ("+ / =",            "zoom in"),
                ("- / _",            "zoom out"),
                ("T",                "toggle overlay opacity (10% steps)"),
                ("Q",                "quit and SAVE changes"),
                ("ESC",              "quit and DISCARD all changes"),
            ]

            from IPython.display import display, HTML

            col_w = max(len(k) for k, _ in controls) + 2

            lines = ["=" * 60, " .✦ ݁˖ Interactive mask editor .✦ ݁˖", "=" * 60, 
                "> Draw polygons to add or remove regions.", f"> Editing: {mask}\n"]
            for key, desc in controls:
                lines.append(f"  {key:<{col_w}}: {desc}")

            display(HTML(f"<pre style='font-family:monospace'>{'<br>'.join(lines)}</pre>"))


        if mask == 'mask_locules':
            self.mask_locules = interactive_mask_editor(self.mask_locules, original_img=self.img)
        else:
            self.mask_fruit = interactive_mask_editor(self.mask_fruit, original_img=self.img)


    ##########################################################################################
    # OPTIONAL : Create a scatterplot to visualize pixel colors (HSV space) 
    ##########################################################################################

    def generate_color_scatterplot(
        self,
        sample_size: int = 10000,
        plot_size: Tuple[int, int] = (18, 5),
    ) -> None:
        """
        Display a scatterplot of pixel colors in HSV space.

        Delegates to :func:`~traitly.fruit_phenotyping.mask.generate_scatter_plot`
        using a random pixel sample. Useful for choosing HSV thresholds before
        calling :meth:`generate_fruit_mask`.

        Parameters
        ----------
        sample_size : int, optional
            Number of pixels to sample for the plot. Must be a positive
            integer. Default is 10000.
        plot_size : tuple of int, optional
            Figure size ``(width, height)`` for the scatterplot. Default is
            (18, 5).

        Raises
        ------
        ValueError
            If no image is loaded or ``sample_size`` is not a positive integer.
        """
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
    def generate_fruit_mask(
        self,
        plot: bool = True,
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
        roi_expansion: int = 0,
        background_color: Optional[str] = None,
        fill_holes: bool = False,
        apply_convex_hull: bool = False,
        erosion_px: int = 0
    ) -> None:

        if self.img_rgb is None:
            raise ValueError("No image loaded. Run load_image() first.")

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
                'erosion_px': erosion_px
            }
        if stamp:
            img = 255 - self.img_rgb
            
        else:
            img = cv2.cvtColor(self.img_rgb, cv2.COLOR_RGB2HSV)

        # Create base mask
        self.mask_fruit = create_mask(
            img_hsv=img,
            n_iteration=n_iteration,
            plot=False,
            plot_size=plot_size,
            kernel_blur=kernel_blur,
            kernel_open=kernel_open,
            kernel_close=kernel_close,
            canny_max=canny_max,
            canny_min=canny_min,
            lower_hsv=lower_hsv,
            upper_hsv=upper_hsv,
            background_color=background_color,
            fill_holes=fill_holes,
            apply_convex_hull=apply_convex_hull
        )

        # Remove label and reference ROIs from mask
        if remove_roi:
            mask_rois = np.zeros_like(self.mask_fruit)

            # Label ROI
            if hasattr(self, 'label_roi') and self.label_roi:
                for box in self.label_roi:
                    x, y = box['x'], box['y']
                    w, h = box['width'], box['height']
                    x_expanded = max(0, x - roi_expansion)
                    y_expanded = max(0, y - roi_expansion)
                    w_expanded = w + 2 * roi_expansion
                    h_expanded = h + 2 * roi_expansion
                    cv2.rectangle(mask_rois,
                                (x_expanded, y_expanded),
                                (x_expanded + w_expanded, y_expanded + h_expanded),
                                255, -1)

            # Reference ROI
            if hasattr(self, 'ref_roi') and self.ref_roi:
                for roi in self.ref_roi:
                    x, y, w, h = cv2.boundingRect(roi)
                    x_expanded = max(0, x - roi_expansion)
                    y_expanded = max(0, y - roi_expansion)
                    w_expanded = w + 2 * roi_expansion
                    h_expanded = h + 2 * roi_expansion
                    cv2.rectangle(mask_rois,
                                (x_expanded, y_expanded),
                                (x_expanded + w_expanded, y_expanded + h_expanded),
                                255, -1)

            # Color checker ROI
            if hasattr(self, 'checker_coords') and self.checker_coords is not None:
                if len(self.checker_coords) == 4:
                    x, y, w, h = self.checker_coords
                    x_expanded = max(0, x - roi_expansion)
                    y_expanded = max(0, y - roi_expansion)
                    w_expanded = w + 2 * roi_expansion
                    h_expanded = h + 2 * roi_expansion
                    img_h, img_w = self.mask_fruit.shape[:2]
                    x_expanded = max(0, min(x_expanded, img_w))
                    y_expanded = max(0, min(y_expanded, img_h))
                    w_expanded = min(w_expanded, img_w - x_expanded)
                    h_expanded = min(h_expanded, img_h - y_expanded)
                    cv2.rectangle(mask_rois,
                                (x_expanded, y_expanded),
                                (x_expanded + w_expanded, y_expanded + h_expanded),
                                255, -1)

            self.mask_fruit = cv2.bitwise_and(self.mask_fruit, cv2.bitwise_not(mask_rois))

        # Apply erosion
        if erosion_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (erosion_px * 2 + 1, erosion_px * 2 + 1)
            )
            self.mask_fruit = cv2.erode(self.mask_fruit.copy(), kernel, iterations=1)


        if plot:
            plt.figure(figsize=plot_size)
            plt.imshow(self.mask_fruit, cmap='gray')
            plt.axis('off')
            plt.show()

        return None
    
    ##########################################################################################
    # OPTIONAL: Create locule-fruit contrast
    ##########################################################################################
    def enhance_locule_contrast(
        self,
        contrast_method: str = 'none',
        gamma: float = 1.5,
        gain: float = 5,
        cutoff: float = 0.5,
        c: float = 0.5,
        plot: bool = True,
        plot_size: Tuple[int, int] = (8, 10),
        compare_method: bool = False,
        kernel_blur: int = 1,
        clip_limit: Optional[int] = None,
        tile_grid_size: Optional[int] = 12,
    ) -> None:
        """
        Apply contrast enhancement to the L channel to improve locule visibility.

        Delegates to :func:`~traitly.fruit_phenotyping.mask.apply_contrast`
        and stores the result in :attr:`l_transformed`. The enhanced L channel
        is used by :meth:`generate_locule_mask` to threshold locule regions.

        Parameters
        ----------
        contrast_method : str, optional
            Enhancement method: ``'gamma'``, ``'sigmoid'``, 
            ``'exponential'``, or ``'none'``. Default is ``'gamma'``.
        gamma : float, optional
            Gamma exponent, used when ``contrast_method='gamma'``. Default is
            1.5.
        gain : float, optional
            Sigmoid gain, used when ``contrast_method='sigmoid'``. Default is
            5.
        cutoff : float, optional
            Sigmoid cutoff, used when ``contrast_method='sigmoid'``. Default
            is 0.5.
        c : float, optional
            Exponential factor, used when ``contrast_method='exponential'``.
            Default is 0.5.
        plot : bool, optional
            If True, display the enhanced L channel. Default is True.
        plot_size : tuple of int, optional
            Figure size for the plot. Default is (8, 10).
        compare_method : bool, optional
            If True, display a side-by-side comparison of all methods via
            :func:`~traitly.fruit_phenotyping.mask.apply_contrast`. Default
            is False.
        kernel_blur : int, optional
            Gaussian blur kernel size applied before contrast enhancement.
            Default is 1.
        clip_limit : int or None, optional
            CLAHE clip limit. If provided, CLAHE is applied after the selected
            contrast method. Default is None.
        tile_grid_size : int or None, optional
            CLAHE tile grid size. Used only when ``clip_limit`` is set.
            Default is 12.
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
    def generate_locule_mask(
        self,
        thresh_min: int = 120,
        kernel_close: Optional[int] = None,
        kernel_open: Optional[int] = None,
        kernel_blur: Optional[int] = None,
        erosion_px: int = 10,
        otsu_offset: Optional[int] = None,
        min_fruit_area: int = 5000,
        min_locule_area: int = 0,
        invert_locule: bool = False,
        plot: bool = True,
        plot_size: Tuple[int, int] = (10, 5),
    ) -> None:
        """
        Generate a fused mask containing fruits with their internal locules.

        Delegates to :func:`~traitly.fruit_phenotyping.mask.create_mask_locules`
        using :attr:`l_transformed` (from :meth:`enhance_locule_contrast`) and
        :attr:`mask_fruit`. Populates :attr:`mask_locules`, which replaces
        :attr:`mask_fruit` as the active mask in downstream steps.

        Parameters
        ----------
        thresh_min : int, optional
            Minimum threshold value for L-channel binarization. Default is
            120.
        thresh_max : int, optional
            Maximum threshold value for L-channel binarization. Default is
            255.
        kernel_close : int or None, optional
            Kernel size for morphological closing applied to the locule mask.
            Default is None.
        kernel_open : int or None, optional
            Kernel size for morphological opening applied to the locule mask.
            Default is None.
        min_fruit_area : int, optional
            Minimum area in pixels to retain a fruit region during mask
            fusion. Default is 5000.
        invert_locule : bool, optional
            If True, invert the locule binary mask before fusion. Useful when
            locules are brighter than the surrounding pericarp. Default is
            False.
        plot : bool, optional
            If True, display the fused mask. Default is True.
        plot_size : tuple of int, optional
            Figure size for the mask plot. Default is (5, 5).

        Raises
        ------
        ValueError
            If :attr:`mask_fruit` or :attr:`l_transformed` is not available.
        """
        # Validation

        if self.mask_fruit is None:
            raise ValueError("No mask available. Run generate_fruit_mask() first.")
        
        if self.l_transformed is None:
            raise ValueError("Locule contrast not initialized. Run enhance_locule_contrast() first "
                             "(use contrast_method = 'none' if no transformation is requiered)")
        
        metadata = self.is_metadata_saved
        if metadata:
            self.parameters.generate_locule_mask_params = {
                'thresh_min': thresh_min,
                'min_fruit_area': min_fruit_area,
                'min_locule_area': min_locule_area,
                'kernel_close': kernel_close,
                'kernel_open': kernel_open,
                'kernel_blur': kernel_blur,
                'erosion_px': erosion_px,
                'otsu_offset': otsu_offset,
                'invert_locule': invert_locule
                }
        
        if otsu_offset is not None:
            use_otsu = True
        else:
            use_otsu = False

        self.mask_locules = create_mask_locules(
            l_transformed=self.l_transformed,
            fruit_mask=self.mask_fruit,
            thresh_min=thresh_min,
            kernel_close=kernel_close,
            kernel_open=kernel_open,
            kernel_blur=kernel_blur,
            erosion_px=erosion_px,
            use_otsu=use_otsu,
            otsu_offset=otsu_offset,
            min_fruit_area=min_fruit_area,
            min_locule_area=min_locule_area,
            invert_locules=invert_locule,
            plot=plot, 
            plot_size=plot_size
        )
        
        return None
    
    ##########################################################################################
    # Detect fruits on the mask
    ##########################################################################################

    def detect_fruits(
        self,
        min_fruit_circularity: float = 0.5,
        verbose: bool = True,
        min_locule_area: int = 50,
        min_locule_per_fruit: int = 1,
        min_fruit_area: int = 5000,
        max_fruit_area: Optional[int] = None,
        rescale_factor: Optional[float] = None,
        plot: bool = False,
        plot_size: Tuple[int, int] = (5, 5),
        contour_color: Tuple[int, int, int] = (0, 255, 0),
        locule_color: Tuple[int, int, int] = (255, 0, 255),
        locule_thickness: int = 2,
        contour_thickness: int = 2,
    ) -> None:
        """
        Detect individual fruits and their locules from the binary mask.

        Delegates to :func:`~traitly.fruit_phenotyping.mask.find_fruits` using
        :attr:`mask_locules` when available, otherwise :attr:`mask_fruit`.
        Populates :attr:`contours` and :attr:`fruit_locule_map`.

        Parameters
        ----------
        min_fruit_circularity : float, optional
            Minimum circularity score in [0, 1] to accept a contour as a
            fruit. Default is 0.5.
        verbose : bool, optional
            If True, print a summary of detected fruits and parameters used.
            Default is True.
        min_locule_area : int, optional
            Minimum contour area in pixels to consider a region as a locule.
            Default is 50.
        min_locule_per_fruit : int, optional
            Minimum number of locules required to accept a fruit. Default is
            1.
        min_fruit_area : int or None, optional
            Minimum contour area in pixels to accept a fruit. If None, no
            lower bound is applied. Default is None.
        max_fruit_area : int or None, optional
            Maximum contour area in pixels to accept a fruit. If None, no
            upper bound is applied. Default is None.
        rescale_factor : float or None, optional
            Factor to rescale contours before detection. Default is None.
        plot : bool, optional
            If True, display detected fruit contours on the image. Default is
            False.
        plot_size : tuple of int, optional
            Figure size for the detection plot. Default is (5, 5).
        contour_color : tuple of int, optional
            BGR color for drawing fruit contours. Default is ``(0, 255, 0)``.
        contour_thickness : int, optional
            Line thickness for contour drawing. Default is 2.

        Raises
        ------
        ValueError
            If :attr:`mask_fruit` is not available.
        """
        
        # Validation: if mask exists, mask_locules should also exist
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
            for fruit_id, locule_ids in self.fruit_locule_map.items():
                # Fruits
                cv2.drawContours(img_copy, [self.contours[fruit_id]], -1, contour_color, contour_thickness)
                # Locules
                if locule_ids:
                    for loc_id in locule_ids:
                        # Locule contour
                        cv2.drawContours(img_copy, [self.contours[loc_id]], -1, locule_color, locule_thickness)

                        # Internal pericarp contour (convex hull)
                        all_locule_points = np.concatenate([self.contours[loc_id] for loc_id in locule_ids])
                        hull = cv2.convexHull(all_locule_points)
                        cv2.drawContours(img_copy, [hull], -1, (93,238,255), 2)

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
    def generate_single_fruit_masks(
        self,
        fruit_id: Optional[int] = None,
        plot_size: Tuple[int, int] = (7, 5),
        overlay: bool = False,
        overlay_legend: bool = False,
        margin: int = 5,
        only_fruit: bool = False, # Needed for FruitExternalAnalysis, keep it False  for FruitInternalAnalysis
        alpha: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate and display tissue masks for a single fruit.

        Delegates to
        :func:`~traitly.fruit_phenotyping.color_analysis.get_single_fruit_masks`
        using :attr:`mask_locules` when available, otherwise :attr:`mask_fruit`.
        The fruit is cropped to its bounding box with an optional pixel margin.

        Parameters
        ----------
        fruit_id : int or None, optional
            Sequential fruit ID to visualize. If None, the first detected
            fruit is used. Default is None.
        plot_size : tuple of int, optional
            Figure size for the tissue mask plot. Default is (7, 5).
        overlay : bool, optional
            If True, overlay all tissue masks on the original image. Default
            is False.
        overlay_legend : bool, optional
            If True, include a legend in the overlay plot. Default is False.
        margin : int, optional
            Pixel margin added around the fruit bounding box crop. Default is
            5.
        only_fruit : bool, optional
            If True, display only the whole-fruit mask without internal tissue
            breakdown. Default is False.

        Raises
        ------
        ValueError
            If :attr:`mask_fruit`, :attr:`contours`, or
            :attr:`fruit_locule_map` is not available, or if no fruits were
            detected.
        """

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

        if alpha is not None:
            self.alpha = alpha
        

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
                               only_fruit = only_fruit,
                               alpha = self.alpha)  
        
    ##########################################################################################
    #                                     MORPHOLOGY 
        
        
    ##########################################################################################
    # Extract morphology measurements from the image
    ##########################################################################################
    def analyze_morphology(
        self,
        plot: bool = True,
        plot_size: Tuple[int, int] = (10, 10),
        font_size: float = 1.5,
        font_thickness: int = 2,
        font_color: Tuple[int, int, int] = (0, 0, 0),
        label_position: str = 'top',
        label_color: Tuple[int, int, int] = (255, 255, 255),
        contour_mode: str = 'raw',
        epsilon: float = 0.001,
        min_locule_area: int = 10,
        max_locule_area: Optional[int] = None,
        angle_shifts: int = 500,
        num_rays: int = 90,
        pericarp_int_color: Tuple[int, int, int] = (0, 240, 240),
        pericarp_int_thickness: int = 2,
        locule_color: Tuple[int, int, int] = (255, 0, 255),
        locule_thickness: int = 2,
        pericarp_ext_thickness: int = 2,
        pericarp_ext_color: Tuple[int, int, int] = (0, 240, 0),
        centroid_fruit_color: Tuple[int, int, int] = (255, 255, 51),
        centroid_fruit_thickness: int = 2,
        centroid_locule_color: Tuple[int, int, int] = (0, 255, 255),
        centroid_locule_thickness: int = 2,
        display_table: Optional[bool] = True,
        is_locule: bool = True,
        alpha: float = None,
    ) -> Optional[pd.DataFrame]:
        """
        Extract morphological metrics from detected fruits.

        Delegates to
        :func:`~traitly.fruit_phenotyping.fruit_config.analyze_fruits_morphology`
        and stores results in :attr:`results`. Column order is normalized
        by a predefined group ordering. Preserves any existing color results
        in :attr:`results` across calls.

        Parameters
        ----------
        plot : bool, optional
            If True, display the annotated result image. Default is True.
        plot_size : tuple of int, optional
            Figure size for the annotated image. Default is (10, 10).
        font_size : float, optional
            Font scale for annotation labels. Default is 1.5.
        font_thickness : int, optional
            Thickness of annotation text in pixels. Default is 2.
        font_color : tuple of int, optional
            BGR color for annotation text. Default is black ``(0, 0, 0)``.
        label_position : str, optional
            Position of fruit ID labels: ``'top'``, ``'bottom'``, ``'left'``,
            or ``'right'``. Default is ``'top'``.
        label_color : tuple of int, optional
            BGR background color for annotation labels. Default is white
            ``(255, 255, 255)``.
        contour_mode : str, optional
            Contour representation mode forwarded to
            :func:`~traitly.fruit_phenotyping.fruit_config.analyze_fruits_morphology`.
            One of ``'raw'``, ``'hull'``, ``'approx'``, ``'ellipse'``,
            ``'circle'``. Default is ``'raw'``.
        epsilon : float, optional
            Approximation factor for contour simplification when
            ``contour_mode='approx'``. Default is 0.001.
        min_locule_area : int, optional
            Minimum locule area in pixels included in analysis. Default is
            100.
        max_locule_area : int or None, optional
            Maximum locule area in pixels. If None, no upper limit is applied.
            Default is None.
        angle_shifts : int, optional
            Number of angle steps for angular symmetry computation forwarded
            to :func:`~traitly.fruit_phenotyping.symmetry.angular_symmetry`.
            Default is 500.
        num_rays : int, optional
            Number of rays for radial pericarp thickness estimation forwarded
            to :func:`~traitly.fruit_phenotyping.processing.calculate_pericarp_thickness_radial`.
            Default is 90.
        pericarp_int_color : tuple of int, optional
            BGR color for the internal pericarp contour overlay. Default is
            ``(0, 240, 240)``.
        pericarp_int_thickness : int, optional
            Line thickness for the internal pericarp contour. Default is 2.
        locule_color : tuple of int, optional
            BGR color for locule contours. Default is ``(255, 0, 255)``.
        locule_thickness : int, optional
            Line thickness for locule contours. Default is 2.
        pericarp_ext_color : tuple of int, optional
            BGR color for the external pericarp contour. Default is
            ``(0, 240, 0)``.
        pericarp_ext_thickness : int, optional
            Line thickness for the external pericarp contour. Default is 2.
        centroid_fruit_color : tuple of int, optional
            BGR color for fruit centroid markers. Default is ``(255, 255, 51)``.
        centroid_fruit_thickness : int, optional
            Radius of fruit centroid markers in pixels. Default is 2.
        centroid_locule_color : tuple of int, optional
            BGR color for locule centroid markers. Default is ``(0, 255, 255)``.
        centroid_locule_thickness : int, optional
            Radius of locule centroid markers in pixels. Default is 2.
        display_table : bool or None, optional
            If True, return the morphology results DataFrame. Default is True.
        is_locule : bool, optional
            If False, skip locule, pericarp, and symmetry metrics. Default is
            True.

        Returns
        -------
        pd.DataFrame or None
            Morphology results DataFrame if ``display_table=True``, otherwise
            ``None``.

        Raises
        ------
        ValueError
            If :attr:`mask_fruit`, :attr:`contours`, or
            :attr:`fruit_locule_map` is not available, or if no fruits were
            detected.
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

        if self.img_copy is None:
            self.img_copy = self.img.copy()
    
        saved_color_results = getattr(self.results, 'color_results', None)
        saved_color_image   = getattr(self.results, 'color_image', None)

        if alpha is not None:
            self.alpha = alpha
        
        # For color results
        self.is_morphology_results = True

        self.results = analyze_fruits_morphology(
            # Image 
            img=self.img_copy,
            path=self.img_path,
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

            # Internal pericarp contour
            alpha = self.alpha,

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

        if saved_color_image is not None:
            self.results.color_image = saved_color_image

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
                'is_locule': is_locule,
                'alpha': self.alpha
                }

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
        Save the analysis parameters used in the current session to disk.

        Writes both a ``.txt`` and a ``.json`` file named after the loaded
        image, via :meth:`~traitly.fruit_phenotyping.analysis_parameters.AnalysisParameters.save_to_file`
        and :meth:`~traitly.fruit_phenotyping.analysis_parameters.AnalysisParameters.save_to_json`.

        Parameters
        ----------
        output_path : str or None, optional
            Directory where parameter files are saved. If None, files are
            saved in the same directory as :attr:`img_path`. Default is None.
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
    def analyze_color(
        self,
        stat: Optional[str] = 'mean',
        tissue: Optional[str] = 'all',
        color_space: Optional[str] = 'all',
        display_table: Optional[bool] = True,
        plot: bool = False,
        plot_size: Tuple[int, int] = (10, 10),
        font_size: int = 2,
        font_thickness: int = 2,
        pericarp_ext_color: Tuple[int, int, int] = (0, 255, 0),
        pericarp_ext_thickness: int = 2,
        pericarp_int_color: Tuple[int, int, int] = (255, 255, 0), 
        pericarp_int_thickness: int = 2,    
        locule_thickness: int = 2,
        locule_color: Tuple[int, int, int] = (255, 0, 255),
        label_position: str = 'top',
        font_color: Tuple[int, int, int] = (0, 0, 0),
        label_color: Tuple[int, int, int] = (255, 255, 255),
        label_opacity: float = 0.7,
        get_color_histogram: bool = False,
        alpha: Optional[float] = None,
        dark_thresh: int = 20
    ) -> Optional[pd.DataFrame]:
        """
        Extract color features from detected fruit tissues.

        When ``get_color_histogram=False``, delegates to
        :func:`~traitly.fruit_phenotyping.color_analysis.analyze_all_fruits_color`
        and returns per-tissue summary statistics. When
        ``get_color_histogram=True``, delegates to
        :func:`~traitly.fruit_phenotyping.color_analysis.get_fruit_color_histograms`
        instead. Uses :attr:`mask_locules` when available, otherwise
        :attr:`mask_fruit`. Stores results in :attr:`results`.

        Parameters
        ----------
        stat : str or None, optional
            Summary statistic for color features: ``'mean'`` or ``'median'``.
            Default is ``'mean'``.
        tissue : str or None, optional
            Tissue region to analyze: ``'all'``, ``'total_pericarp'``,
            ``'outer_pericarp'``, ``'inner_pericarp'``, or ``'locules'``.
            Default is ``'all'``.
        color_space : str or None, optional
            Color spaces to extract: ``'all'`` or a subset of ``'rgb'``,
            ``'lab'``, ``'hsv'``, ``'gray'``. Default is ``'all'``.
        display_table : bool or None, optional
            If True, return the color results DataFrame. Default is True.
        plot : bool, optional
            If True, display the annotated image used for color extraction.
            Default is False.
        plot_size : tuple of int, optional
            Figure size for the annotated image plot. Default is (10, 10).
        font_size : int, optional
            Font scale for annotation labels. Default is 2.
        font_thickness : int, optional
            Thickness of annotation text. Default is 2.
        pericarp_ext_color : tuple of int, optional
            BGR color for external pericarp contour overlays. Default is
            ``(0, 255, 0)``.
        pericarp_ext_thickness : int, optional
            Line thickness for external pericarp contours. Default is 2.
        locule_thickness : int, optional
            Line thickness for locule contours. Default is 2.
        locule_color : tuple of int, optional
            BGR color for locule contours. Default is ``(255, 0, 255)``.
        label_position : str, optional
            Position of fruit ID labels: ``'top'``, ``'bottom'``, ``'left'``,
            or ``'right'``. Default is ``'top'``.
        font_color : tuple of int, optional
            BGR color for annotation text. Default is black ``(0, 0, 0)``.
        label_color : tuple of int, optional
            BGR background color for labels. Default is white
            ``(255, 255, 255)``.
        label_opacity : float, optional
            Opacity of label backgrounds in [0, 1]. Default is 0.7.
        get_color_histogram : bool, optional
            If True, compute pixel-level color histograms instead of summary
            statistics. Default is False.

        Returns
        -------
        pd.DataFrame or None
            Color results DataFrame if ``display_table=True``, otherwise
            ``None``.

        Raises
        ------
        ValueError
            If :attr:`mask_fruit`, :attr:`contours`, or
            :attr:`fruit_locule_map` is not available, or if no fruits were
            detected.
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
        
        if alpha is not None:
            self.alpha = alpha

        if self.img_copy is None:
            self.img_copy = self.img_rgb.copy()

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
                'pericarp_int_color': pericarp_int_color,
                'pericarp_int_thickness': pericarp_int_thickness,
                'locule_thickness': locule_thickness,
                'locule_color': locule_color,
                'label_position': label_position,
                'font_color': font_color,
                'label_color': label_color,
                'label_opacity': label_opacity,
                'get_color_histogram': get_color_histogram,
                'alpha': self.alpha,
                'dark_thresh': dark_thresh
            }
        
        # Always reannotate from clean image
        saved_color_results = getattr(self.results, 'color_results', None)
        
        if self.is_morphology_results is None:
            self.results = ResultsImage(
                bgr_img = self.img_copy,
                morphology_results=[],  
                image_path=self.img_path
            )
        
    
        self.results.color_image = self.img_copy.copy()

        # Annotate independent image for color results
        annotate_all_fruits(annotated_img = self.results.color_image,
                                contours =  self.contours, 
                                fruit_locule_map = self.fruit_locule_map, 
                                img_shape = self.img_shape,
                                font_scale = font_size,
                                font_thickness = font_thickness,
                                pericarp_ext_color = pericarp_ext_color,
                                pericarp_ext_thickness = pericarp_ext_thickness,
                                pericarp_int_color = pericarp_int_color,
                                pericarp_int_thickness = pericarp_int_thickness,
                                locule_thickness = locule_thickness, 
                                locule_color = locule_color,
                                label_position = label_position, 
                                margin = 10, 
                                text_color = font_color, 
                                label_background_color = label_color,
                                label_opacity = label_opacity,
                                alpha=self.alpha
                                )
            

        
        if plot:
            plt.figure(figsize = plot_size)
            plt.imshow(cv2.cvtColor(self.results.color_image, cv2.COLOR_BGR2RGB))
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
                                                          normalize = False,
                                                          dark_threshold = dark_thresh)
            
            self.results.color_results = pd.DataFrame(color_results)
    
        else:
            color_results = analyze_all_fruits_color(img = self.img,
                                    mask = mask,
                                    contours = self.contours,
                                    fruit_locule_map = self.fruit_locule_map,
                                    stat = stat,
                                    tissue = tissue,
                                    renumber = True,
                                    color_space = color_space,
                                    alpha = self.alpha,
                                    dark_threshold = dark_thresh)
            

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

    def detect_color_checker(
        self,
        plot: bool = False,
        plot_size: Tuple[int, int] = (5, 5),
        verbose: bool = True,
        scale_factor: float = 0.5,
    ) -> None:
        """
        Detect a Macbeth color checker card and store its bounding coordinates.

        Uses ``cv2.mcc.CCheckerDetector`` on a downscaled copy of the image
        for speed, then scales detected coordinates back to full resolution.
        Draws the detected grid onto :attr:`img_copy` and stores bounding
        coordinates in :attr:`checker_coords` for ROI removal in
        :meth:`generate_fruit_mask`.

        Parameters
        ----------
        plot : bool, optional
            If True, display the cropped color checker region. Default is
            False.
        plot_size : tuple of int, optional
            Figure size for the color checker plot. Default is (5, 5).
        verbose : bool, optional
            If True, print detection results and coordinates. Default is True.
        scale_factor : float, optional
            Downscaling factor applied before detection for speed. Recommended
            range is [0.2, 0.5]. Default is 0.5.

        Notes
        -----
        If no checker is detected, :attr:`checker_roi` and
        :attr:`checker_coords` are set to ``None`` and a ``UserWarning`` is
        issued. This does not raise an exception so the pipeline can continue.
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
        Run the full analysis pipeline on the already-loaded image.

        Executes pipeline steps in order: :meth:`setup_measurements`,
        :meth:`generate_fruit_mask`, optionally :meth:`enhance_locule_contrast`
        and :meth:`generate_locule_mask`, :meth:`detect_fruits`, and
        optionally :meth:`analyze_morphology` and :meth:`analyze_color`.
        Configuration is resolved from ``json_path`` first, then ``config``,
        then defaults. Each step is wrapped in a try/except so partial
        failures are captured in ``error_dict`` without raising.

        Parameters
        ----------
        config : dict or None, optional
            Analysis configuration dictionary. Ignored if ``json_path`` is
            provided. Default is None.
        json_path : str or None, optional
            Path to a JSON configuration file. Takes precedence over ``config``
            if the file exists. Default is None.
        analyze_morphology : bool, optional
            If True, run :meth:`analyze_morphology`. Default is True.
        analyze_color : bool, optional
            If True, run :meth:`analyze_color`. Default is True.
        save_image : bool, optional
            If True, save the annotated result image to ``output_path``.
            Default is False.
        output_path : str or None, optional
            Directory where the annotated image is saved when
            ``save_image=True``. Defaults to the directory of
            :attr:`img_path`.

        Returns
        -------
        tuple
            ``(df_morphology, df_color, error_dict, n_fruits, annotated_img)``
            where:

            - ``df_morphology`` – morphology results DataFrame or ``None``.
            - ``df_color`` – color results DataFrame or ``None``.
            - ``error_dict`` – dict with ``'filename'`` and ``'status'`` on
            failure, otherwise ``None``.
            - ``n_fruits`` – number of fruits detected (0 on failure).
            - ``annotated_img`` – annotated BGR image or ``None``.
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
                if analyze_morphology:
                    annotated_img = self.results.annotated_image
                else:
                    annotated_img = self.results.color_image

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
        alpha: Optional[int] = None,
        
        # analyze_color
        stat: Optional[str] = None,
        tissue: Optional[str] = None,
        color_space: Optional[str] = None,
        label_opacity: Optional[float] = None,
        get_color_histogram: Optional[bool] = None,
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Process all images in the folder passed to :class:`FruitInternalAnalyzer`.

        Collects valid images from the input folder, builds a unified
        configuration from ``json_path``, ``config``, and individual parameter
        arguments (individual params always take priority), and runs the full
        pipeline on each image via :meth:`process_single_file`. Supports
        parallel execution via :func:`_process_image_worker` when
        ``num_cores > 1``. Saves merged CSV results, a session report, and an
        error report to ``output_path``.

        Parameters
        ----------
        analyze_morphology : bool, optional
            If True, run morphology analysis on each image. Default is True.
        analyze_color : bool, optional
            If True, run color analysis on each image. Default is True.
        json_path : str or None, optional
            Path to a JSON configuration file. Merged with ``config`` if both
            are provided. Default is None.
        config : dict or None, optional
            Base configuration dictionary. Individual parameter arguments
            override matching keys. Default is None.
        output_path : str or None, optional
            Directory where results are saved. Defaults to a ``Results/``
            subfolder inside the input folder.
        num_cores : int, optional
            Number of parallel worker processes. Clamped to available CPUs.
            Default is 1 (sequential).
        verbose : bool, optional
            If True, print progress and summary information. Default is True.
        width_cm : float or None, optional
            Known reference width in centimetres for scale calibration.
        length_cm : float or None, optional
            Known reference length in centimetres for scale calibration.
        diameter_cm : float or None, optional
            Known reference diameter in centimetres for scale calibration.
        fast_calibration : bool or None, optional
            If True, use geometric calibration without YOLO detection.
        skip_qr : bool or None, optional
            If True, skip QR code detection.
        detect_label : bool or None, optional
            If True, run full label detection including OCR.
        confidence : float or None, optional
            Minimum detection confidence for reference objects.
        detect_color_checker : bool or None, optional
            If True, detect and remove a color checker from the mask.
        scale_factor : float or None, optional
            Downscaling factor for color checker detection.
        lower_hsv : list of int or None, optional
            Lower HSV threshold for fruit segmentation.
        upper_hsv : list of int or None, optional
            Upper HSV threshold for fruit segmentation.
        background_color : str or None, optional
            Expected background color hint for segmentation.
        n_iteration : int or None, optional
            Number of morphological iterations for mask refinement.
        kernel_blur : int or None, optional
            Gaussian blur kernel size.
        kernel_open : int or None, optional
            Morphological opening kernel size.
        kernel_close : int or None, optional
            Morphological closing kernel size.
        canny_min : int or None, optional
            Minimum Canny edge threshold.
        canny_max : int or None, optional
            Maximum Canny edge threshold.
        fill_holes : bool or None, optional
            If True, fill holes in the binary mask.
        apply_convex_hull : bool or None, optional
            If True, apply convex hull to each fruit region.
        remove_roi : bool or None, optional
            If True, remove reference and label regions from the mask.
        roi_expansion : int or None, optional
            Pixel margin around ROIs before removal.
        stamp : bool or None, optional
            If True, invert image colors before masking.
        contrast_method : str or None, optional
            Contrast enhancement method for locule visibility.
        gamma : float or None, optional
            Gamma exponent for ``contrast_method='gamma'``.
        gain : float or None, optional
            Sigmoid gain for ``contrast_method='sigmoid'``.
        cutoff : float or None, optional
            Sigmoid cutoff for ``contrast_method='sigmoid'``.
        c : float or None, optional
            Exponential factor for ``contrast_method='exponential'``.
        kernel_blur_contrast : int or None, optional
            Blur kernel applied before contrast enhancement.
        clip_limit : int or None, optional
            CLAHE clip limit for contrast enhancement.
        tile_grid_size : int or None, optional
            CLAHE tile grid size.
        thresh_min : int or None, optional
            Minimum threshold for locule mask binarization.
        thresh_max : int or None, optional
            Maximum threshold for locule mask binarization.
        min_fruit_area_locule : int or None, optional
            Minimum fruit area used during locule mask fusion.
        kernel_close_locule : int or None, optional
            Closing kernel for locule mask.
        kernel_open_locule : int or None, optional
            Opening kernel for locule mask.
        invert_locule : bool or None, optional
            If True, invert the locule mask before fusion.
        min_fruit_area : int or None, optional
            Minimum contour area to accept as a fruit.
        max_fruit_area : int or None, optional
            Maximum contour area to accept as a fruit.
        min_fruit_circularity : float or None, optional
            Minimum circularity to accept as a fruit.
        min_locule_area : int or None, optional
            Minimum locule area for fruit detection.
        min_locule_per_fruit : int or None, optional
            Minimum locules required per fruit.
        rescale_factor : float or None, optional
            Contour rescaling factor before detection.
        contour_mode : str or None, optional
            Contour mode for morphology analysis.
        epsilon : float or None, optional
            Approximation factor for contour simplification.
        min_locule_area_morph : int or None, optional
            Minimum locule area for morphology analysis.
        max_locule_area : int or None, optional
            Maximum locule area for morphology analysis.
        angle_shifts : int or None, optional
            Number of angle steps for symmetry computation.
        num_rays : int or None, optional
            Number of rays for pericarp thickness estimation.
        stat : str or None, optional
            Color summary statistic: ``'mean'`` or ``'median'``.
        tissue : str or None, optional
            Tissue region for color analysis.
        color_space : str or None, optional
            Color spaces to extract.
        label_opacity : float or None, optional
            Opacity of annotation label backgrounds.
        get_color_histogram : bool or None, optional
            If True, compute pixel-level color histograms.

        Raises
        ------
        ValueError
            If the instance was not initialized with a directory path, or if
            no valid images are found in the folder.
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
            alpha=alpha
        ))
        _apply('analyze_color_params', dict(
            stat=stat, tissue=tissue, color_space=color_space,
            label_opacity=label_opacity,
            get_color_histogram=get_color_histogram,
            alpha=alpha,
            pericarp_int_color=pericarp_int_color,
            pericarp_int_thickness=pericarp_int_thickness
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
            print(" Traitly running ⋆✧｡٩(ˊᗜˋ )و✧*｡   ")
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
        
        if json_path is not None:
            json_report = json_path
        else:
            json_report = 'No JSON file provided'

        session_lines = [
            "=" * 70,
            "SESSION REPORT",
            "=" * 70,
            f"traitly              : v{__version__}",
            f"run date             : {session_start.strftime('%Y-%m-%d %H:%M:%S')}",
            f"image folder         : {folder_path}",
            f"results folder       : {output_path}",
            f"images found         : {len(img_paths)}",
            f"images ok            : {len(img_paths) - len(errors)}",
            f"images failed        : {len(errors)}",
            f"total fruits         : {total_fruits}",
            f"analyze_morphology   : {analyze_morphology}",
            f"analyze_color        : {analyze_color}",
            f"JSON path            : {json_report}",
            f"num_cores            : {num_cores}",
            f"total time           : {total_time:.1f}s",
            f"avg per image        : {avg_time:.1f}s",
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
            if len(errors) == len(img_paths):
                print("\n( ദ്ദി ༎ຶ‿༎ຶ ) Task failed successfully " + "="*37)
                print(f"    > Image(s) processed:")
                print(f"        - Errors: {len(errors)}/{len(img_paths)} img(s)")
                print(f"    > For more details, check error_report.txt saved in: {output_path}")

            else:
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
    
    def plot_image(
        self,
        annotated: bool = False,
        plot_size: Tuple[int, int] = (10, 10),
    ) -> None:
        """
        Display the original or annotated image.

        Parameters
        ----------
        annotated : bool, optional
            If True, display the annotated result image from :attr:`results`.
            Falls back to the original image with a warning if no results are
            available. Default is False.
        plot_size : tuple of int, optional
            Figure size ``(width, height)`` for the plot. Default is (10, 10).

        Raises
        ------
        ValueError
            If no image has been loaded.
        """    
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
