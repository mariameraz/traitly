# traitly/fruit_phenotyping/internal_analysis.py

"""
Internal fruit analysis pipeline for traitly.

Provides :class:`FruitInternalAnalyzer`, the core analyzer class for
whole-fruit and internal morphology, color, and symmetry analysis.
Supports single-image and batch folder processing with optional
multiprocessing via :func:`_process_internal_image_worker`.
"""

# ============================================================================
# STANDARD LIBRARY
# ============================================================================
import os
import sys
import time
import warnings
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import copy

# ============================================================================
# THIRD-PARTY LIBRARIES
# ============================================================================
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================================
# INTERNAL IMPORTS
# ============================================================================
from traitly.fruit_phenotyping.mask import (
    apply_contrast,
    create_mask,
    create_mask_locules,
    find_fruits,
    generate_l_channel_histogram,
    generate_scatter_plot,
    interactive_mask_editor,
)

from traitly.fruit_phenotyping.processing import annotate_all_fruits, get_internal_pericarp_contour

from traitly.utils.basic_functions import detect_img_name, load_img

from traitly.utils.manage_params import _get_params, _clean_params

from traitly.utils.calibration import px_cm_density

from traitly.utils.label import (
    detect_label_box,
    detect_label_box_yolo,
    detect_label_text,
    detect_qr,
)

from traitly.utils.batch import (
    _setup_batch,
    _print_batch_header,
    _run_fruit_batch_loop,
    _save_fruit_batch_results,
    _config_from_json,
)

from traitly.utils.validation import (
    _validate_path_exists,
    _validate_color_image,
    _validate_img_suffix,
)

from traitly.utils import _save_parameters

from traitly.fruit_phenotyping.analysis_parameters import FruitAnalyzerParameters

from traitly.fruit_phenotyping.results_image import ResultsImage

from traitly.fruit_phenotyping.color_analysis import (
    analyze_all_fruits_color,
    get_fruit_color_histograms,
    get_single_fruit_masks,
)
from traitly.fruit_phenotyping.fruit_config import analyze_fruits_morphology

from traitly.color_correction.color_analysis import _detect_color_checker

##########################################################################################
# Ignore warnings from torch
##########################################################################################
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", message="Using CPU")

##########################################################################################
# Worker function for parallel processing
##########################################################################################

def _process_internal_image_worker(
    img_path: str,
    config: Dict,
    analyze_morphology: bool,
    analyze_color: bool,
    output_path: str = None,
) -> Tuple:
    t0 = time.time()
    try:
        analyzer = FruitInternalAnalyzer(img_path)
        analyzer.load_image(plot=False)
        df_morphology, df_color, error_dict, n_fruits, annotated_img = (
            analyzer._process_single_file(
                config = config,
                json_path = None,
                analyze_morphology = analyze_morphology,
                analyze_color = analyze_color,
                save_image = True,
                output_path = output_path,
            )
        )
        elapsed = time.time() - t0
        filename = os.path.basename(img_path)
        return (
            df_morphology,
            df_color,
            error_dict,
            n_fruits,
            annotated_img,
            filename,
            elapsed,
        )
    except Exception as e:
        return (
            None,
            None,
            {"filename": os.path.basename(img_path), "status": f"Error: {str(e)}"},
            0,
            None,
            os.path.basename(img_path),
            time.time() - t0,
        )


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
    path : str
        Path to an image file or a folder containing images. Raises
        :exc:`FileNotFoundError` if the path does not exist.
    """

    def __init__(self, path: str) -> None:
        """
        Initialize the analyzer and validate the image path.

        Parameters
        ----------
        path : str
            Path to an image file or a directory. Raises
            :exc:`FileNotFoundError` if the path does not exist.
        """

        # Get absolute path
        self.input_path = os.path.abspath(path)
        ## Verify path exists
        _validate_path_exists(self.input_path)

        ## Class attributes:
        # load_img
        self._is_directory = os.path.isdir(os.path.dirname(path))
        self.img = None
        self.img_name = None
        self._img_copy = None
        self.img_shape = None
        self._img_rgb = None
        self._img_hsv = None

        # setup_measurements
        self._ref_roi = None
        self.px_per_cm = None
        self._label_roi = None
        self.label_text = None

        # create_mask
        self.mask_fruit = None
        self.mask_locules = None
        self.l_transformed = None

        # detect fruits
        self.contours = None
        self.fruit_locule_map = None

        # analyze fruits
        self.results = None
        self._dilation_factor = None

        # detect_color_checker
        self._color_charts = None
        self._checker_coords = None

        # save metadata
        self._parameters = FruitAnalyzerParameters()
        self._is_metadata_saved = True
        self._is_morphology_results = None

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
        h: Optional[int] = None,
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

        if self.input_path is None:
            raise ValueError(
                "No image loaded."
                "Run FruitInternalAnalyzer('path/to/your/image.jpg') first."
            )

        # Check image format
        path = Path(self.input_path)
        _validate_img_suffix(path)

        # Load image
        self.img = load_img(
            self.input_path,
            plot=plot,
            plot_size=plot_size,
            show_axis=show_axis,
            x=x,
            y=y,
            w=w,
            h=h,
        )

        # Check image loaded successfully
        _validate_color_image(self.img)

        # Save some image attributes
        self.img_shape = self.img.shape[:2] # Image shape
        self._img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV) # HSV Image
        self._img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB) # RGB Image
        self.img_name = detect_img_name(self.input_path) # Image name

        self._parameters.input_params = {"input_path": self.input_path} # Save used parameters

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
        skip_label_roi: bool = False,
    ) -> None:

        """
        Detect QR code, label ROI, and label text for the loaded image.

        Runs up to three detection steps in order:

        1. QR code detection via :func:`~traitly.utils.basic_functions.detect_qr`
        (skipped if ``skip_qr=True``).
        2. Label ROI detection via
        :func:`~traitly.utils.basic_functions.detect_label_box_yolo` then
        :func:`~traitly.utils.basic_functions.detect_label_box` as fallback
        (skipped if ``skip_label_roi=True``).
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
        skip_label_roi : bool, optional
            If True, skip both ROI detection and OCR. Default is False.
        """

        if verbose:
            print("\n" + "=" * 55)
            print("★ LABEL DETECTION:")
            print("=" * 55)

        # 1. early return if no label detection required
        if not detect_label:
            self._label_roi = None
            self.label_text = "No label detected"
            if verbose:
                print("> Label detection: SKIPPED (detect_label=False)")
            return None

        # 2. If there is detected text, reuse it
        if self.label_text is not None and self.label_text != "No label detected":
            if verbose:
                print(f"> Label text: {self.label_text}")
            return None

        # 3. QR
        if not skip_qr:
            qr_start = time.time()
            self.label_text = detect_qr(img=self.img)
            if verbose and self.label_text:
                print(f"> QR Code detected: {self.label_text} ({time.time() - qr_start:.2f}s)")
        else:
            if verbose:
                print("> QR detection: SKIPPED")

        # 4. ROI + OCR (only if the qr did not detect the text)
        if not skip_label_roi:
            self._label_roi = detect_label_box_yolo(img=self.img, plot=False, conf=0.4)
            if not self._label_roi:
                self._label_roi = detect_label_box(img=self.img, verbose=False, plot=False)

        if self._label_roi and not self.label_text:
            ocr_start = time.time()
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                self.label_text = detect_label_text(
                    img=self.img,
                    label_roi=self._label_roi,
                    language=language_label,
                    blur_label=blur_label,
                    verbose=False,
                    gpu=gpu,
                )
            finally:
                sys.stdout = old_stdout

                if verbose and self.label_text:
                    print(f"> Label text detected: {self.label_text}   (OCR: {time.time() - ocr_start:.2f}s)")

        elif skip_label_roi:
            self._label_roi = None

        # 5. final result
        if not self.label_text:
            self.label_text = "No label detected"

        if verbose and self.label_text == "No label detected":
            print("> No label detected.")

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
        skip_yolo: bool = False,
    ) -> None:
        """
        Detect the size reference and calculate the pixel-to-centimetre factor.

        When ``skip_yolo=True`` and both ``width_cm`` and ``length_cm``
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
        skip_yolo : bool, optional
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
        self._img_copy = self.img.copy()

        if (width_cm is not None) != (length_cm is not None):
            raise ValueError("Error: both width_cm and length_cm are required")

        if width_cm and length_cm:
            self.px_per_cm = np.sqrt((w * h) / (width_cm * length_cm))
            self._ref_roi = None
            if verbose:
                print("> Size reference detection: SKIPPED (skip_yolo=True).")
        else:
            # No calibration available: measurements will be in pixels
            self.px_per_cm = None
            self._ref_roi = None
            if skip_yolo:
                if verbose:
                    print("> Size reference detection: SKIPPED (skip_yolo=True).")
            else:
                self.px_per_cm, self._img_copy, self._ref_roi = px_cm_density(
                    self._img_copy,
                    confidence_threshold=confidence,
                    plot=False,
                    font_size=font_size,
                    verbose=verbose,
                    width_cm=width_cm,
                    length_cm=length_cm,
                    diameter_cm=diameter_cm,
                    return_coordinates=True,
                )

        if self._ref_roi is not None:
            if verbose and using_default_diameter:
                print("\nNote: Default reference diameter (2.5 cm) applied.")
                print("        Specify diameter_cm to override this value.")
        else:
            if width_cm is not None and length_cm is not None:
                if verbose:
                    print("\n> Using provided physical dimensions:")
                    print(f"    - width_cm:  {width_cm} cm")
                    print(f"    - length_cm: {length_cm} cm")
                    print(
                        f"\n        . ݁₊ ⊹ . ݁ ⟡ ݁ px/cm density: {self.px_per_cm:.2f} ⟡ ݁ . ⊹ ₊ ݁."
                    )

            else:
                if verbose:
                    print("\n> Using provided physical dimensions:")
                    print(f"    - width_cm:  {width_cm}")
                    print(f"    - length_cm: {width_cm}")
                    print(
                        "\n        . ݁₊ ⊹ . ݁ ⟡ ݁ Measurements will be returned in PIXEL units ⟡ ݁ . ⊹ ₊ ݁."
                    )

        return None

    ##########################################################################################
    # Wrapper: label + size reference detection
    ##########################################################################################
    def setup_measurements(
        self,
        plot: bool = False,
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
        skip_yolo: bool = False,
    ) -> None:
        """
        Detect label text and calculate the pixel-to-centimetre scale factor.

        Orchestrates :meth:`setup_label` and :meth:`setup_calibration` in
        order. Populates :attr:`label_text`, :attr:`label_roi`, :attr:`ref_roi`, and
        :attr:`px_per_cm`.

        Parameters
        ----------
        plot: bool, optional
            If True, display a cropped view of each detected reference ROI.
            Default is False.
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
        skip_yolo : bool, optional
            If True and ``width_cm`` and ``length_cm`` are set, skip YOLO
            detection in :meth:`setup_calibration`. Default is False.

        """
        if self.img is None:
            raise ValueError("No image loaded. Run load_img() first.")
        metadata = self._is_metadata_saved

        if metadata:
            self._parameters.setup_measurements_params = {
                "width_cm": width_cm,
                "length_cm": length_cm,
                "diameter_cm": diameter_cm,
                "skip_yolo": skip_yolo,
                "skip_qr": skip_qr,
                "detect_label": detect_label,
                "language_label": language_label,
                "confidence": confidence,
                "gpu": gpu,
                "font_size": font_size,
            }

        # 1) label
        self.setup_label(
            detect_label=detect_label,
            verbose=verbose,
            language_label=language_label,
            gpu=gpu,
            skip_qr=skip_qr,
        )

        # 2) calibration
        self.setup_calibration(
            verbose=verbose,
            confidence=confidence,
            font_size=font_size,
            width_cm=width_cm,
            length_cm=length_cm,
            diameter_cm=diameter_cm,
            skip_yolo=skip_yolo,
        )


        # Plot
        if plot and self._ref_roi:
            h_img, w_img = self.img.shape[:2]
            margin = 5  # px

            # Determine orientation and plot size
            is_portrait = h_img > w_img
            n = len(self._ref_roi)

            if is_portrait:
                nrows, ncols = n, 1
                figsize = (plot_size[0], plot_size[1] * n)
            else:
                nrows, ncols = 1, n
                figsize = (plot_size[0], plot_size[1])

            plt.figure(figsize=figsize)

            # Plot all the reference boxes detected
            for i, ref_contour in enumerate(self._ref_roi, 1):
                x, y, w, h = cv2.boundingRect(ref_contour)
                # Add the margin
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(w_img, x + w + margin)
                y2 = min(h_img, y + h + margin)

                roi_ref_img = self._img_copy[y1:y2, x1:x2]

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

    def generate_l_channel_histogram(
        self, otsu_offset: int = 0, plot_size: Tuple[int, int] = (9, 3)
    ) -> None:
        """ """

        if self.mask_fruit is None:
            raise ValueError("No mask available. Run generate_fruit_mask() first.")

        if self.l_transformed is None:
            raise ValueError(
                "Locule contrast not initialized. Run enhance_locule_contrast() first "
                "(use contrast_method = 'none' if no transformation is requiered)"
            )

        generate_l_channel_histogram(
            l_transformed=self.l_transformed,
            fruit_mask=self.mask_fruit,
            otsu_offset=otsu_offset,
            plot_size=plot_size,
        )

    ##########################################################################################
    # OPTIONAL : Open an interactive mask editor
    ##########################################################################################

    def edit_mask(self, verbose: bool = True) -> None:
        """Manually edit the locule mask if available, otherwise the fruit mask."""

        if self.mask_locules is not None:
            self.mask_locules = interactive_mask_editor(
                self.mask_locules,
                original_img=self.img,
                verbose = verbose
            )
        elif self.mask_fruit is not None:
            self.mask_fruit = interactive_mask_editor(
                self.mask_fruit,
                original_img=self.img,
                verbose = verbose
            )
        else:
            raise ValueError(
                "No mask found. Run generate_fruit_mask() and optionally, generate_locule_mask() first."
            )


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
            raise ValueError(
                f"Invalid sample_size: {sample_size}. Sample size must be an integer > 0."
            )

        generate_scatter_plot(
            img_hsv=self._img_hsv,
            img_rgb=self._img_rgb,
            sample_size=sample_size,
            plot_size=plot_size,
        )

        return None

    ##########################################################################################
    # Create a binary mask for fruits
    ##########################################################################################
    def generate_fruit_mask(
        self,
        plot: bool = True,
        plot_size: Tuple[int, int] = (5, 5),
        stamp: bool = False,
        lower_hsv: Optional[Tuple[int,int,int]] = None,
        upper_hsv: Optional[Tuple[int,int,int]] = None,
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
        erosion_px: int = 0,
    ) -> None:

        if self._img_rgb is None:
            raise ValueError("No image loaded. Run load_image() first.")

        metadata = self._is_metadata_saved
        if metadata:
            self._parameters.generate_fruit_mask_params = {
                "stamp": stamp,
                "lower_hsv": lower_hsv,
                "upper_hsv": upper_hsv,
                "background_color": background_color,
                "kernel_blur": kernel_blur,
                "kernel_open": kernel_open,
                "kernel_close": kernel_close,
                "canny_min": canny_min,
                "canny_max": canny_max,
                "n_iteration": n_iteration,
                "fill_holes": fill_holes,
                "apply_convex_hull": apply_convex_hull,
                "roi_expansion": roi_expansion,
                "remove_roi": remove_roi,
                "erosion_px": erosion_px,
            }
        if stamp:
            img = 255 - self._img_rgb

        else:
            img = cv2.cvtColor(self._img_rgb, cv2.COLOR_RGB2HSV)

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
            apply_convex_hull=apply_convex_hull,
        )

        # Remove label and reference ROIs from mask
        if remove_roi:
            mask_rois = np.zeros_like(self.mask_fruit)

            # Label ROI
            if hasattr(self, "_label_roi") and self._label_roi:
                for box in self._label_roi:
                    x, y = box["x"], box["y"]
                    w, h = box["width"], box["height"]
                    x_expanded = max(0, x - roi_expansion)
                    y_expanded = max(0, y - roi_expansion)
                    w_expanded = w + 2 * roi_expansion
                    h_expanded = h + 2 * roi_expansion
                    cv2.rectangle(
                        mask_rois,
                        (x_expanded, y_expanded),
                        (x_expanded + w_expanded, y_expanded + h_expanded),
                        255,
                        -1,
                    )

            # Reference ROI
            if hasattr(self, "_ref_roi") and self._ref_roi:
                for roi in self._ref_roi:
                    x, y, w, h = cv2.boundingRect(roi)
                    x_expanded = max(0, x - roi_expansion)
                    y_expanded = max(0, y - roi_expansion)
                    w_expanded = w + 2 * roi_expansion
                    h_expanded = h + 2 * roi_expansion
                    cv2.rectangle(
                        mask_rois,
                        (x_expanded, y_expanded),
                        (x_expanded + w_expanded, y_expanded + h_expanded),
                        255,
                        -1,
                    )

            # Color checker ROI
            if hasattr(self, "_checker_coords") and self._checker_coords is not None:
                x = self._checker_coords['x1']
                y = self._checker_coords['y1']
                w = self._checker_coords['x2'] - self._checker_coords['x1']
                h = self._checker_coords['y2'] - self._checker_coords['y1']
                x_expanded = max(0, x - roi_expansion)
                y_expanded = max(0, y - roi_expansion)
                w_expanded = w + 2 * roi_expansion
                h_expanded = h + 2 * roi_expansion
                cv2.rectangle(
                    mask_rois,
                    (x_expanded, y_expanded),
                    (x_expanded + w_expanded, y_expanded + h_expanded),
                    255,
                    -1,
                )

            self.mask_fruit = cv2.bitwise_and(
                self.mask_fruit, cv2.bitwise_not(mask_rois)
            )

        # Apply erosion
        if erosion_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (erosion_px * 2 + 1, erosion_px * 2 + 1)
            )
            self.mask_fruit = cv2.erode(self.mask_fruit.copy(), kernel, iterations=1)

        if plot:
            plt.figure(figsize=plot_size)
            plt.imshow(self.mask_fruit, cmap="gray")
            plt.axis("off")
            plt.show()

        return None

    ##########################################################################################
    # OPTIONAL: Create locule-fruit contrast
    ##########################################################################################
    def enhance_locule_contrast(
        self,
        contrast_method: str = "none",
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
        metadata = self._is_metadata_saved

        if metadata:
            self._parameters.enhance_locule_contrast_params = {
                "contrast_method": contrast_method,
                "gamma": gamma if contrast_method == "gamma" else None,
                "gain": gain if contrast_method == "sigmoid" else None,
                "cutoff": cutoff if contrast_method == "sigmoid" else None,
                "c": c if contrast_method == "exponential" else None,
                "kernel_blur": kernel_blur,
                "clip_limit": clip_limit,
                "tile_grid_size": tile_grid_size if clip_limit else None,
            }

        self.l_transformed = apply_contrast(
            img=self.img,
            contrast_method=contrast_method,
            gamma=gamma,
            gain=gain,
            cutoff=cutoff,
            c=c,
            plot=plot,
            plot_size=plot_size,
            compare=compare_method,
            kernel_blur=kernel_blur,
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size,
        )
        return None

    ##########################################################################################
    # OPTIONAL: Create fruit + locule mask
    ##########################################################################################
    def generate_locule_mask(
        self,
        thresh_min: int = 120,
        thresh_max: Optional[int] = None,
        kernel_close: Optional[int] = None,
        kernel_open: Optional[int] = None,
        kernel_blur: Optional[int] = None,
        erosion_px: int = 10,
        otsu_offset: Optional[int] = None,
        min_fruit_area: int = 1000,
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
            fusion. Default is 1000.
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
            raise ValueError(
                "Locule contrast not initialized. Run enhance_locule_contrast() first "
                "(use contrast_method = 'none' if no transformation is requiered)"
            )

        metadata = self._is_metadata_saved
        if metadata:
            self._parameters.generate_locule_mask_params = {
                "thresh_min": thresh_min,
                "min_fruit_area": min_fruit_area,
                "min_locule_area": min_locule_area,
                "kernel_close": kernel_close,
                "kernel_open": kernel_open,
                "kernel_blur": kernel_blur,
                "erosion_px": erosion_px,
                "otsu_offset": otsu_offset,
                "invert_locule": invert_locule,
            }

        if otsu_offset is not None:
            use_otsu = True
        else:
            use_otsu = False

        self.mask_locules = create_mask_locules(
            l_transformed=self.l_transformed,
            fruit_mask=self.mask_fruit,
            thresh_min=thresh_min,
            thresh_max=thresh_max,
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
            plot_size=plot_size,
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
        plot: bool = False,
        plot_size: Tuple[int, int] = (5, 5),
        contour_color: Tuple[int, int, int] = (0, 255, 0),
        locule_color: Tuple[int, int, int] = (255, 0, 255),
        locule_thickness: int = 2,
        contour_thickness: int = 2,
        pericarp_int_color: Tuple[int, int, int] = (93, 238, 255),
        pericarp_int_thickness: int = 2,
        dilation_factor: Optional[float] = None,
        rescale_factor: Optional[float] = None,
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
        plot : bool, optional
            If True, display detected fruit contours on the image. Default is
            False.
        plot_size : tuple of int, optional
            Figure size for the detection plot. Default is (5, 5).
        contour_color : tuple of int, optional
            BGR color for drawing fruit contours. Default is ``(0, 255, 0)``.
        contour_thickness : int, optional
            Line thickness for contour drawing. Default is 2.
        rescale_factor : float or None, optional
            Factor in (0, 1] to downscale the mask before detection. Area
            thresholds are adjusted automatically. If ``None`` or 1, no
            rescaling is applied. Default is ``None``.

        Raises
        ------
        ValueError
            If :attr:`mask_fruit` is not available.
        """

        # Validation: if mask exists, mask_locules should also exist
        if self.mask_fruit is None:
            raise ValueError("No mask available. Run generate_fruit_mask() first.")

        metadata = self._is_metadata_saved
        if metadata:
            self._parameters.detect_fruits_params = {
                "min_fruit_area": min_fruit_area,
                "max_fruit_area": max_fruit_area,
                "min_fruit_circularity": min_fruit_circularity,
                "min_locule_area": min_locule_area,
                "min_locule_per_fruit": min_locule_per_fruit,
                "rescale_factor":rescale_factor
            }

        if self.mask_locules is not None:
            mask = self.mask_locules
        else:
            mask = self.mask_fruit

        self.contours, self.fruit_locule_map = find_fruits(
            mask,
            min_fruit_area=min_fruit_area,
            max_fruit_area=max_fruit_area,
            min_circularity=min_fruit_circularity,
            min_locule_area=min_locule_area,
            min_locules_per_fruit=min_locule_per_fruit,
            rescale_factor=rescale_factor
        )

        if self.fruit_locule_map is not None:
            n_fruits_detected = len(self.fruit_locule_map)
        else:
            n_fruits_detected = "0"

        if verbose:
            optional_config = {
                "min_fruit_area": min_fruit_area,
                "max_fruit_area": max_fruit_area,
            }
            print("\n" + "=" * 37)
            print(
                f"        . ݁₊ ⊹ . ݁ ⟡ ݁ Detected fruits: {n_fruits_detected} ⟡ ݁ . ⊹ ₊ ݁."
            )
            print("\n > Parameters used:")
            print(f"        - min_fruit_circularity: {min_fruit_circularity}")
            print(f"        - min_locule_area: {min_locule_area}")
            print(f"        - min_locule_per_fruit: {min_locule_per_fruit}")

            for parameter, value in optional_config.items():
                if value is not None:
                    print(f"        - {parameter}: {value}")

            print("=" * 37)

        if plot:
            if dilation_factor is not None:
                self._dilation_factor = dilation_factor

            img_copy = self._img_rgb.copy()
            for fruit_id, locule_ids in self.fruit_locule_map.items():
                # Fruits
                cv2.drawContours(
                    img_copy,
                    [self.contours[fruit_id]],
                    -1,
                    contour_color,
                    contour_thickness,
                )
                # Locules
                if locule_ids:
                    for loc_id in locule_ids:
                        # Locule contour
                        cv2.drawContours(
                            img_copy,
                            [self.contours[loc_id]],
                            -1,
                            locule_color,
                            locule_thickness,
                        )

                    internal_contour = get_internal_pericarp_contour(
                        locules=locule_ids,
                        contours=self.contours,
                        fruit_id=fruit_id,
                        img_shape=self.img_shape,
                        dilation_factor=dilation_factor,
                    )

                    if internal_contour is not None and len(internal_contour) > 0:
                        cv2.drawContours(
                            img_copy,
                            [internal_contour],
                            -1,
                            pericarp_int_color,
                            pericarp_int_thickness,
                        )

            base_fontsize = 6
            fontsize = base_fontsize + (plot_size[0])

            plt.figure(figsize=plot_size)
            plt.imshow(img_copy)
            plt.axis("off")
            plt.title(f"Fruits: {len(self.fruit_locule_map)}", fontsize=fontsize)
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
        only_fruit: bool = False,  # Needed for FruitExternalAnalysis, keep it False  for FruitInternalAnalysis
        dilation_factor: Optional[float] = None,
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
            raise ValueError(
                "No fruit-locule mapping available. Run detect_fruits() first."
            )

        if not self.fruit_locule_map:
            raise ValueError(
                "No fruits detected. Make sure detect_fruits() found valid fruit "
                "contours before calling analyze_color()."
            )

        if self.mask_locules is not None:
            mask = self.mask_locules
        else:
            mask = self.mask_fruit

        if dilation_factor is not None:
            self._dilation_factor = dilation_factor

        get_single_fruit_masks(
            img=self.img,
            mask=mask,
            contours=self.contours,
            fruit_locule_map=self.fruit_locule_map,
            fruit_id=fruit_id,
            plot_size=plot_size,
            overlay=overlay,
            margin=margin,
            renumber=True,
            overlay_legend=overlay_legend,
            plot=True,
            only_fruit=only_fruit,
            dilation_factor=self._dilation_factor,
        )

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
        label_position: str = "top",
        label_color: Tuple[int, int, int] = (255, 255, 255),
        contour_mode: str = "raw",
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
        dilation_factor: Optional[float] = None,
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
            raise ValueError(
                "No fruit-locule mapping available. Run detect_fruits() first."
            )

        if not self.fruit_locule_map:
            raise ValueError(
                "No fruits detected. Make sure detect_fruits() found valid fruit contours before calling analyze_color()."
            )

        if self.label_text is None:
            self.label_text = "No label detected"

        if self._img_copy is None:
            self._img_copy = self.img.copy()

        saved_color_results = getattr(self.results, "color_results", None)
        saved_color_image = getattr(self.results, "color_image", None)

        if dilation_factor is not None:
            self._dilation_factor = dilation_factor

        # For color results
        self._is_morphology_results = True

        self.results = analyze_fruits_morphology(
            # Image
            img=self._img_copy,
            path=self.input_path,
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
            dilation_factor=self._dilation_factor,
            img_shape=self.img_shape,
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
            pericarp_ext_color=pericarp_ext_color,
            pericarp_ext_thickness=pericarp_ext_thickness,
            centroid_locule_thickness=centroid_locule_thickness,
            centroid_fruit_thickness=centroid_fruit_thickness,
            pericarp_int_color=pericarp_int_color,
            pericarp_int_thickness=pericarp_int_thickness,
            centroid_locule_color=centroid_locule_color,
            centroid_fruit_color=centroid_fruit_color,
            locule_color=locule_color,
            locule_thickness=locule_thickness,
            # Extra
            is_locule=is_locule,
        )

        if saved_color_results is not None:
            self.results.color_results = saved_color_results

        if saved_color_image is not None:
            self.results.color_image = saved_color_image

        metadata = self._is_metadata_saved
        if metadata:
            self._parameters.analyze_morphology_params = {
                "contour_mode": contour_mode,
                "epsilon": epsilon if contour_mode == "approx" else None,
                "min_locule_area": min_locule_area,
                "max_locule_area": max_locule_area,
                "angle_shifts": angle_shifts,
                "num_rays": num_rays,
                "font_size": font_size,
                "font_thickness": font_thickness,
                "font_color": font_color,
                "label_position": label_position,
                "label_color": label_color,
                "pericarp_int_color": pericarp_int_color,
                "pericarp_int_thickness": pericarp_int_thickness,
                "locule_color": locule_color,
                "locule_thickness": locule_thickness,
                "pericarp_ext_thickness": pericarp_ext_thickness,
                "pericarp_ext_color": pericarp_ext_color,
                "centroid_fruit_color": centroid_fruit_color,
                "centroid_fruit_thickness": centroid_fruit_thickness,
                "centroid_locule_color": centroid_locule_color,
                "centroid_locule_thickness": centroid_locule_thickness,
                "is_locule": is_locule,
                "dilation_factor": self._dilation_factor,
            }

        self.results.morphology_results = pd.DataFrame(self.results.morphology_results)

        # Reorder results table
        _GROUP_ORDER = [
            # Image information
            ["image_name", "label", "fruit_id", "n_locules", "unit"],
            # Fruit morphology
            [
                "fruit_area",
                "fruit_perimeter",
                "fruit_circularity",
                "fruit_solidity",
                "fruit_convexity",
                "fruit_major_axis",
                "fruit_minor_axis",
                "fruit_box_length",
                "fruit_box_width",
                "fruit_aspect_ratio",
                "fruit_compactness",
                "fruit_lobedness",
            ],
            # External pericarp
            ["total_outer_pericarp_area"],
            # Internal pericarp
            [
                "outer_pericarp_mean_thickness",
                "outer_pericarp_std_thickness",
                "outer_pericarp_cv_thickness",
            ],
            # Internal areas
            [
                "total_internal_fruit_area",
                "total_internal_pericarp_area",
                "total_locules_area",
            ],
            # Locules
            [
                "locules_mean_area",
                "locules_std_area",
                "locules_cv_area",
                "locules_mean_circularity",
                "locules_std_circularity",
                "locules_cv_circularity",
                "locules_angular_symmetry",
                "locules_radial_symmetry",
            ],
            # Ratios
            ["outer_pericarp_to", "internal_pericarp_to", "locules_to"],
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
        self.results.morphology_results = self.results.morphology_results[
            ordered + remaining
        ]

        if display_table:
            return self.results.morphology_results

    ##########################################################################################
    #                                     PARAMETERS

    ##########################################################################################
    # OPTIONAL: Save all the parameters used in the session
    ##########################################################################################
    def save_parameters(self, output_path=None):
        _save_parameters(self.input_path, self._parameters, output_path)

    ##########################################################################################
    #                                     COLOR ANALYSIS

    ##########################################################################################
    # Extract color measurements for different fruit tissues
    ##########################################################################################
    def analyze_color(
        self,
        stat: Optional[str] = "mean",
        tissue: Optional[str] = "all",
        color_space: Optional[str] = "all",
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
        label_position: str = "top",
        font_color: Tuple[int, int, int] = (0, 0, 0),
        label_color: Tuple[int, int, int] = (255, 255, 255),
        label_opacity: float = 0.7,
        get_color_histogram: bool = False,
        dark_thresh: int = 20,
        dilation_factor: Optional[float] = None,
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
            raise ValueError(
                "No fruit-locule mapping available. Run detect_fruits() first."
            )

        if not self.fruit_locule_map:
            raise ValueError(
                "No fruits detected. Make sure detect_fruits() found valid fruit contours before calling analyze_color()."
            )

        if self.label_text is None:
            self.label_text = "No label detected"

        if dilation_factor is not None:
            self._dilation_factor = dilation_factor

        if self._img_copy is None:
            self._img_copy = self._img_rgb.copy()

        metadata = self._is_metadata_saved
        if metadata:
            self._parameters.analyze_color_params = {
                "stat": stat,
                "tissue": tissue,
                "color_space": color_space,
                "font_size": font_size,
                "font_thickness": font_thickness,
                "pericarp_ext_color": pericarp_ext_color,
                "pericarp_ext_thickness": pericarp_ext_thickness,
                "pericarp_int_color": pericarp_int_color,
                "pericarp_int_thickness": pericarp_int_thickness,
                "locule_thickness": locule_thickness,
                "locule_color": locule_color,
                "label_position": label_position,
                "font_color": font_color,
                "label_color": label_color,
                "label_opacity": label_opacity,
                "get_color_histogram": get_color_histogram,
                "dilation_factor": self._dilation_factor,
                "dark_thresh": dark_thresh,
            }

        # Initialize ResultsImage if
        if self._is_morphology_results is None:
            self.results = ResultsImage(
                res_img=self._img_copy, morphology_results=[], path=self.input_path
            )

        self.results.color_image = self._img_copy.copy()

        # Annotate independent image for color results
        annotate_all_fruits(
            annotated_img=self.results.color_image,
            contours=self.contours,
            fruit_locule_map=self.fruit_locule_map,
            font_scale=font_size,
            font_thickness=font_thickness,
            pericarp_ext_color=pericarp_ext_color,
            pericarp_ext_thickness=pericarp_ext_thickness,
            pericarp_int_color=pericarp_int_color,
            pericarp_int_thickness=pericarp_int_thickness,
            locule_thickness=locule_thickness,
            locule_color=locule_color,
            label_position=label_position,
            margin=10,
            text_color=font_color,
            label_background_color=label_color,
            label_opacity=label_opacity,
            dilation_factor=self._dilation_factor,
        )

        if plot:
            plt.figure(figsize=plot_size)
            plt.imshow(cv2.cvtColor(self.results.color_image, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.show()

        # use locule mask if available, otherwise use fruit mask
        if self.mask_locules is not None:
            mask = self.mask_locules
        else:
            mask = self.mask_fruit

        if get_color_histogram:
            color_results = get_fruit_color_histograms(
                img=self.img,
                hsv_img=self._img_hsv,
                label=self.label_text,
                contours=self.contours,
                mask=mask,
                fruit_locule_map=self.fruit_locule_map,
                image_name=self.img_name,
                color_space=color_space,
                renumber=True,
                normalize=False,
                dark_threshold=dark_thresh,
            )

            self.results.color_results = pd.DataFrame(color_results)

        else:
            color_results = analyze_all_fruits_color(
                img=self.img,
                mask=mask,
                contours=self.contours,
                fruit_locule_map=self.fruit_locule_map,
                stat=stat,
                tissue=tissue,
                renumber=True,
                color_space=color_space,
                dilation_factor=self._dilation_factor,
                dark_threshold=dark_thresh,
            )

            df = pd.concat(
                {
                    fruit_id: pd.DataFrame(tissues).T
                    for fruit_id, tissues in color_results.items()
                },
                names=["fruit_id", "tissue"],
            ).reset_index()

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

        Notes
        -----
        If no checker is detected, :attr:`checker_roi` and
        :attr:`checker_coords` are set to ``None`` and a ``UserWarning`` is
        issued. This does not raise an exception so the pipeline can continue.
        """

        self._checker_coords, self._color_charts, self.img_copy = _detect_color_checker(
                                                                    self.img,
                                                                    plot = plot,
                                                                    plot_size = plot_size,
                                                                    verbose = verbose,
                                                                )


        return None

    ##########################################################################################
    #                                     BATCH ANALYSIS

    ##########################################################################################
    # PRocess a single file (needed for analyze_folder)
    ##########################################################################################
    def _process_single_file(
        self,
        config: Optional[Dict] = None,
        json_path: Optional[str] = None,
        analyze_morphology: bool = True,
        analyze_color: bool = True,
        save_image: bool = False,
        output_path: Optional[str] = None,
    ) -> Tuple[
        Optional[pd.DataFrame],
        Optional[pd.DataFrame],
        Optional[Dict],
        int,
        Optional[np.ndarray],
    ]:
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

        config = config or {}

        error_dict = None
        n_fruits = 0
        df_morph = None
        df_color = None
        annotated_img = None

        # Run every step and cath errors (if any)
        try:
            # 1. setup_measurements
            try:
                self.setup_measurements(
                    verbose=False, **_clean_params(_get_params(config, "setup_measurements_params"))
                )
            except Exception as e:
                raise RuntimeError(f"[setup_measurements] {e}")

            # 2. generate_fruit_mask
            try:
                self.generate_fruit_mask(
                    plot=False, **_clean_params(_get_params(config, "generate_fruit_mask_params"))
                )
            except Exception as e:
                raise RuntimeError(f"[generate_fruit_mask] {e}")

            # 3. enhance_locule_contrast (optional)
            elc = _get_params(config, "enhance_locule_contrast_params")
            if elc:
                try:
                    self.enhance_locule_contrast(plot=False, **_clean_params(elc))
                except Exception as e:
                    raise RuntimeError(f"[enhance_locule_contrast] {e}")

                # 4. generate_locule_mask (only if enhance was run)
                glm = _get_params(config, "generate_locule_mask_params")
                if glm:
                    try:
                        self.generate_locule_mask(plot=False, **_clean_params(glm))
                    except Exception as e:
                        raise RuntimeError(f"[generate_locule_mask] {e}")

            # 5. detect_fruits
            try:
                self.detect_fruits(
                    verbose=False, **_clean_params(_get_params(config, "detect_fruits_params"))
                )
            except Exception as e:
                raise RuntimeError(f"[detect_fruits] {e}")

            if not self.fruit_locule_map:
                raise RuntimeError(
                    "[detect_fruits] {No valid fruits detected in image}"
                )

            n_fruits = len(self.fruit_locule_map)

            # 6. analyze_morphology
            if analyze_morphology:
                try:
                    self.analyze_morphology(
                        plot=False,
                        display_table=False,
                        **_clean_params(_get_params(config, "analyze_morphology_params")),
                    )
                    if self.results and self.results.morphology_results is not None:
                        df_morph = (
                            self.results.morphology_results
                            if isinstance(self.results.morphology_results, pd.DataFrame)
                            else pd.DataFrame(self.results.morphology_results)
                        )
                except Exception as e:
                    raise RuntimeError(f"[analyze_morphology] {e}")

            # 7. analyze_color
            if analyze_color:
                try:
                    self.analyze_color(
                        display_table=False, **_clean_params(_get_params(config, "analyze_color_params"))
                    )
                    if self.results and self.results.color_results is not None:
                        df_color = (
                            self.results.color_results
                            if isinstance(self.results.color_results, pd.DataFrame)
                            else pd.DataFrame(self.results.color_results)
                        )
                except Exception as e:
                    raise RuntimeError(f"[analyze_color] {e}")

            # 8. Get annotated image
            if self.results is not None:
                if analyze_morphology:
                    annotated_img = self.results.morphology_image
                else:
                    annotated_img = self.results.color_image

            # 9. Save image if requested
            if save_image and self.results is not None:
                self.results.save_img(
                    output_path=output_path,
                    output_message=False,
                )

        except Exception as e:
            error_dict = {"filename": os.path.basename(self.input_path), "status": str(e)}

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
        skip_yolo: Optional[bool] = None,
        skip_qr: Optional[bool] = None,
        detect_label: Optional[bool] = None,
        confidence: Optional[float] = None,
        # generate_fruit_mask
        stamp: Optional[bool] = None,
        lower_hsv: Optional[Tuple[int,int,int]] = None,
        upper_hsv: Optional[Tuple[int,int,int]] = None,
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
        erosion_px: Optional[int] = None,
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
        # analyze_morphology
        contour_mode: Optional[str] = None,
        epsilon: Optional[float] = None,
        min_locule_area_morph: Optional[int] = None,
        max_locule_area: Optional[int] = None,
        angle_shifts: Optional[int] = None,
        num_rays: Optional[int] = None,
        font_size: Optional[float] = None,
        font_thickness: Optional[int] = None,
        font_color: Optional[Tuple[int, int, int]] = None,
        label_position: Optional[str] = None,
        label_color: Optional[Tuple[int, int, int]] = None,
        pericarp_ext_color: Optional[Tuple[int, int, int]] = None,
        pericarp_ext_thickness: Optional[int] = None,
        centroid_fruit_color: Optional[Tuple[int, int, int]] = None,
        centroid_fruit_thickness: Optional[int] = None,
        pericarp_int_color: Optional[Tuple[int, int, int]] = None,
        pericarp_int_thickness: Optional[int] = None,
        locule_color: Optional[Tuple[int, int, int]] = None,
        locule_thickness: Optional[int] = None,
        centroid_locule_color: Optional[Tuple[int, int, int]] = None,
        centroid_locule_thickness: Optional[int] = None,
        dilation_factor: Optional[float] = None,
        # analyze_color
        stat: Optional[str] = None,
        tissue: Optional[str] = None,
        color_space: Optional[str] = None,
        label_opacity: Optional[float] = None,
        get_color_histogram: Optional[bool] = None,
    ) -> None:

        # 1. Valdate analyze conditions first
        if not analyze_color and not analyze_morphology:
            raise ValueError(
                "analyze_color=False and analyze_morphology=False.\n"
                "analyze_folder() requires that at least one of them is True."
            )

        folder_path, output_path, num_cores, num_cores_message, img_paths = _setup_batch(
            is_directory=self._is_directory,
            input_path=self.input_path,
            output_path=output_path,
            num_cores=num_cores,
        )

        # 2. create config for internal analysis
        config = copy.deepcopy(config) if config else {}

        _config_from_json(json_path, config)

        def _apply(section: str, mapping: Dict):
            overrides = {k: v for k, v in mapping.items() if v is not None}
            if overrides:
                config.setdefault(section, {})
                config[section].update(overrides)

        _apply("setup_measurements_params", dict(
            width_cm=width_cm, length_cm=length_cm, diameter_cm=diameter_cm,
            skip_yolo=skip_yolo, skip_qr=skip_qr, detect_label=detect_label,
            confidence=confidence,
        ))
        _apply("generate_fruit_mask_params", dict(
            stamp=stamp, lower_hsv=lower_hsv, upper_hsv=upper_hsv,
            n_iteration=n_iteration, kernel_blur=kernel_blur, kernel_open=kernel_open,
            kernel_close=kernel_close, canny_min=canny_min, canny_max=canny_max,
            remove_roi=remove_roi, roi_expansion=roi_expansion,
            background_color=background_color, fill_holes=fill_holes,
            apply_convex_hull=apply_convex_hull, erosion_px=erosion_px,
        ))
        _apply("enhance_locule_contrast_params", dict(
            contrast_method=contrast_method, gamma=gamma, gain=gain,
            cutoff=cutoff, c=c, kernel_blur=kernel_blur_contrast,
            clip_limit=clip_limit, tile_grid_size=tile_grid_size,
        ))
        _apply("generate_locule_mask_params", dict(
            thresh_min=thresh_min, thresh_max=thresh_max,
            min_fruit_area=min_fruit_area_locule, kernel_close=kernel_close_locule,
            kernel_open=kernel_open_locule, invert_locule=invert_locule,
        ))
        _apply("detect_fruits_params", dict(
            min_fruit_area=min_fruit_area, max_fruit_area=max_fruit_area,
            min_fruit_circularity=min_fruit_circularity,
            min_locule_area=min_locule_area, min_locule_per_fruit=min_locule_per_fruit,
        ))
        _apply("analyze_morphology_params", dict(
            contour_mode=contour_mode, epsilon=epsilon,
            min_locule_area=min_locule_area_morph, max_locule_area=max_locule_area,
            angle_shifts=angle_shifts, num_rays=num_rays,
            font_size=font_size, font_thickness=font_thickness, font_color=font_color,
            label_position=label_position, label_color=label_color,
            pericarp_ext_color=pericarp_ext_color, pericarp_ext_thickness=pericarp_ext_thickness,
            centroid_fruit_color=centroid_fruit_color, centroid_fruit_thickness=centroid_fruit_thickness,
            pericarp_int_color=pericarp_int_color, pericarp_int_thickness=pericarp_int_thickness,
            locule_color=locule_color, locule_thickness=locule_thickness,
            centroid_locule_color=centroid_locule_color, centroid_locule_thickness=centroid_locule_thickness,
            dilation_factor=dilation_factor,
        ))
        _apply("analyze_color_params", dict(
            stat=stat, tissue=tissue, color_space=color_space,
            label_opacity=label_opacity, get_color_histogram=get_color_histogram,
            dilation_factor=dilation_factor,
            pericarp_int_color=pericarp_int_color, pericarp_int_thickness=pericarp_int_thickness,
        ))

        # 3. Syncronize config to self._parameters for the session report
        for key in (
            "setup_measurements_params", "generate_fruit_mask_params",
            "enhance_locule_contrast_params", "generate_locule_mask_params",
            "detect_fruits_params", "analyze_morphology_params", "analyze_color_params",
        ):
            if key in config and config[key]:
                setattr(self._parameters, key, config[key])

        # 4. Verbose header
        session_start = datetime.now()

        _print_batch_header(
            folder_path = folder_path,
            img_paths = img_paths,
            num_cores = num_cores,
            num_cores_message = num_cores_message,
            verbose = verbose,
            json_path = json_path,
            extra_lines = [
                f"    > analyze_morphology: {analyze_morphology}",
                f"    > analyze_color: {analyze_color}",
            ],
        )

        # 5. Process loop:
        all_morphology, all_color, errors, total_fruits, _ = _run_fruit_batch_loop(
            img_paths=img_paths,
            worker_fn=_process_internal_image_worker,
            num_cores=num_cores,
            config=config,
            analyze_morphology=analyze_morphology,
            analyze_color=analyze_color,
            output_path=output_path,
            verbose=verbose,
        )

        # 6. Save results
        _save_fruit_batch_results(
            all_morphology=all_morphology,
            all_color=all_color,
            errors=errors,
            output_path=output_path,
            folder_path=folder_path,
            img_paths=img_paths,
            total_fruits=total_fruits,
            num_cores=num_cores,
            analyze_morphology=analyze_morphology,
            analyze_color=analyze_color,
            json_path=json_path,
            session_start=session_start,
            parameters=self._parameters,
            param_sections={
                "SETUP_MEASUREMENTS": "setup_measurements_params",
                "GENERATE_FRUIT_MASK": "generate_fruit_mask_params",
                "ENHANCE_LOCULE_CONTRAST": "enhance_locule_contrast_params",
                "GENERATE_LOCULE_MASK": "generate_locule_mask_params",
                "DETECT_FRUITS": "detect_fruits_params",
                "ANALYZE_MORPHOLOGY": "analyze_morphology_params",
                "ANALYZE_COLOR": "analyze_color_params",
            },
            verbose=verbose,
        )

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
                print(
                    "Warning: No annotated image, showing original image instead. Run analyze_color() or analyze_morphology() first."
                )
                img = self._img_rgb
            else:
                img = self.results.morphology_image
        else:
            img = self._img_rgb

        plt.figure(figsize=plot_size)
        plt.imshow(img)
        plt.axis("off")
        plt.show

        return None
