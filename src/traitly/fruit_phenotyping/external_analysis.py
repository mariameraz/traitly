# traitly/fruit_phenotyping/external_analysis.py
"""
External fruit analysis pipeline for traitly.

Provides the :class:`FruitExternalAnalyzer` class, which extends
:class:`~traitly.fruit_phenotyping.internal_analysis.FruitInternalAnalyzer`
for analyzing whole-fruit morphology and color without locule or internal
pericarp segmentation. Includes support for single-image and batch folder
processing with optional multiprocessing.

The typical analysis pipeline follows this order:

1. :meth:`~FruitExternalAnalyzer.load_image`
2. :meth:`~FruitExternalAnalyzer.setup_measurements`
3. :meth:`~FruitExternalAnalyzer.generate_fruit_mask`
6. :meth:`~FruitExternalAnalyzer.detect_fruits`
7. :meth:`~FruitExternalAnalyzer.analyze_morphology` and/or :meth:`~FruitExternalAnalyzer.analyze_color`

For batch processing, steps 1–7 are orchestrated automatically by
:meth:`~FruitExternalAnalyzer.analyze_folder` or
:meth:`~FruitExternalAnalyzer.process_single_file`.
"""

# ============================================================================
# STANDARD LIBRARY
# ============================================================================
import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import copy
import json
from datetime import datetime
from pathlib import Path
import time as t
# ============================================================================
# THIRD-PARTY LIBRARIES
# ============================================================================
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
# ============================================================================
# INTERNAL IMPORTS
# ============================================================================
from traitly.fruit_phenotyping.internal_analysis import FruitInternalAnalyzer
from traitly import __version__
from traitly.utils.constants import valid_extensions as _valid_ext

##########################################################################################
# Global worker for parallel processing 
##########################################################################################

def _process_external_worker(img_path: str,
                              config: Dict,
                              analyze_morphology: bool,
                              analyze_color: bool):
    """
    Worker function for parallel processing of a single image.

    Instantiates a :class:`FruitExternalAnalyzer`, loads the image, and
    runs the full analysis pipeline. Designed to be called inside a
    :class:`~concurrent.futures.ProcessPoolExecutor`.

    Parameters
    ----------
    img_path : str
        Absolute path to the image file to process.
    config : Dict
        Analysis configuration dictionary passed to
        :meth:`FruitExternalAnalyzer.process_single_file`.
    analyze_morphology : bool
        If True, run morphology analysis.
    analyze_color : bool
        If True, run color analysis.

    Returns
    -------
    tuple
        A tuple of ``(df_morphology, df_color, error_dict, n_fruits,
        annotated_img, filename, elapsed)``. On failure, ``df_morphology``
        and ``df_color`` are None and ``error_dict`` contains the error message.
    """
    t0 = time.time()
    try:
        analyzer = FruitExternalAnalyzer(img_path)
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

class FruitExternalAnalyzer(FruitInternalAnalyzer):
    """
    Fruit analyzer for external (whole-fruit) morphology and color analysis.

    Extends :class:`~traitly.fruit_phenotyping.internal_analysis.FruitInternalAnalyzer`
    to analyze fruits without internal segmentation (no locules, no internal
    pericarp). Internal-only parameters are automatically stripped from any
    configuration passed to the pipeline.

    Parameters
    ----------
    img_path : str
        Path to an image file or a folder containing images.
    """
    def __init__(self, img_path: str):
        super().__init__(img_path)
    
        self.external_features = None

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
    ):
        """
        Set up scale calibration and [reference size, label text, color checker] detection.

        Delegates to the parent implementation. See
        :meth:`~traitly.fruit_phenotyping.internal_analysis.FruitInternalAnalyzer.setup_measurements`
        for full parameter documentation.
        """
        super().setup_measurements(
            plot_reference = plot_reference,
            plot_color_checker = plot_color_checker,
            font_size = font_size,
            confidence = confidence,
            detect_label = detect_label,
            verbose = verbose,
            plot_size = plot_size,
            language_label = language_label,
            width_cm = width_cm,
            length_cm = length_cm,
            diameter_cm = diameter_cm,
            gpu = gpu,
            skip_qr = skip_qr,
            fast_calibration = fast_calibration,
            detect_color_checker = detect_color_checker,
            scale_factor = scale_factor

        )

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
        roi_expansion: int = 5,
        background_color: Optional[str] = None,
        fill_holes: bool = False,
        apply_convex_hull: bool = False,
        erosion_px: int = 0
    ) -> None:
        """
        Generate a binary mask segmenting fruits from the background.

        Delegates to the parent implementation. See
        :meth:`~traitly.fruit_phenotyping.internal_analysis.FruitInternalAnalyzer.generate_fruit_mask`
        for full parameter documentation.
        """

        if lower_hsv is None or upper_hsv is None:
            if background_color is None:
                background_color = 'blue'  # Default to blue if no HSV or background color provided
       
        super().generate_fruit_mask(
            plot=plot,
            plot_size=plot_size,
            stamp=stamp,
            lower_hsv=lower_hsv,
            upper_hsv=upper_hsv,
            n_iteration=n_iteration,
            kernel_blur=kernel_blur,
            kernel_open=kernel_open,
            kernel_close=kernel_close,
            canny_min=canny_min,
            canny_max=canny_max,
            remove_roi=remove_roi,
            roi_expansion=roi_expansion,
            background_color=background_color,
            fill_holes=fill_holes,
            apply_convex_hull = apply_convex_hull,
            erosion_px = erosion_px 
        )

    def detect_fruits(
        self,
        min_fruit_area: int = 500,
        max_fruit_area: Optional[int] = None,
        min_fruit_circularity: float = 0.5,
        rescale_factor: Optional[float] = None,
        verbose: bool = True,
        plot: bool = False,
        plot_size: Tuple[int, int] = (5, 5),
        contour_color: Tuple[int, int, int] = (0, 255, 0),
        contour_thickness: int = 2
    ) -> None:
        """
        Detect individual fruits from the binary mask.

        Delegates to the parent implementation with ``min_locule_per_fruit``
        fixed to 0. Prints a summary of detected fruits when ``verbose=True``.

        Parameters
        ----------
        min_fruit_area : int, optional
            Minimum contour area in pixels to be considered a fruit.
            Default is 500.
        max_fruit_area : int or None, optional
            Maximum contour area in pixels. If None, no upper limit is applied.
        min_fruit_circularity : float, optional
            Minimum circularity score in [0, 1] to filter non-fruit contours.
            Default is 0.5.
        rescale_factor : float or None, optional
            Factor to rescale contours before detection. If None, no rescaling
            is applied.
        verbose : bool, optional
            If True, print a summary of detected fruits and parameters used.
            Default is True.
        plot : bool, optional
            If True, display detected fruit contours on the image. Default is False.
        plot_size : Tuple[int, int], optional
            Figure size for the detection plot. Default is (5, 5).
        contour_color : Tuple[int, int, int], optional
            BGR color for drawing fruit contours. Default is green (0, 255, 0).
        contour_thickness : int, optional
            Line thickness for contour drawing. Default is 2.
        """
        
        super().detect_fruits(
            min_fruit_area=min_fruit_area,
            max_fruit_area=max_fruit_area,
            min_fruit_circularity=min_fruit_circularity,
            min_locule_per_fruit= 0,
            rescale_factor=rescale_factor,
            verbose=False,
            plot = plot,
            plot_size = plot_size,
            contour_color = contour_color,
            contour_thickness = contour_thickness

        )

        if self.fruit_locule_map is not None:
            n_fruits_detected = len(self.fruit_locule_map)
        else:
            n_fruits_detected = '0'
        
        if verbose:
            optional_config = {
                "max_fruit_area": max_fruit_area,
                "rescale_factor": rescale_factor
            }
            print("\n" + "=" * 37)
            print(f'        . ݁₊ ⊹ . ݁ ⟡ ݁ Detected fruits: {n_fruits_detected} ⟡ ݁ . ⊹ ₊ ݁.')
            print("\n > Parameters used:")
            print(f"        - min_fruit_circularity: {min_fruit_circularity}")
            print(f"        - min_fruit_area: {min_fruit_area}")

        for parameter, value in optional_config.items():
            if value is not None:
                print(f"        - {parameter}: {value}")
            
            print("=" * 37)

        return None
    
        
    def generate_single_fruit_masks(
        self,
        fruit_id: Optional[int] = None,
        plot_size: Optional[Tuple[int, int]] = (7, 5),
        margin: Optional[int] = 5
    ) -> Dict[str, np.ndarray]:
        """
        Generate a mask for a single fruit cropped to its bounding box.

        Delegates to the parent implementation with ``only_fruit=True``,
        displaying only the whole fruit mask.

        Parameters
        ----------
        fruit_id : int or None, optional
            ID of the fruit to analyze. If None, the first available fruit
            is selected automatically.
        plot_size : Tuple[int, int], optional
            Figure size for the visualization. Default is (7, 5).
        margin : int or None, optional
            Pixel margin to add around the fruit bounding box. Default is 5.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of tissue masks. See
            :func:`~traitly.fruit_phenotyping.color_analysis.get_single_fruit_masks`
            for key definitions.
        """
        
        super().generate_single_fruit_masks(
                               fruit_id = fruit_id, 
                               plot_size = plot_size,
                               overlay = False,
                               margin = margin,
                               only_fruit = True) 
        

    def analyze_morphology(
        self,
        # Contour
        contour_mode: str = 'raw',
        epsilon: float = 0.001,
        # Output
        display_table: bool = True,
        # Plot
        plot: bool = True,
        plot_size: Tuple[int, int] = (10, 10),
        # Annotation
        font_size: float = 1.5,
        font_thickness: int = 2,
        font_color: Tuple[int, int, int] = (0, 0, 0),
        label_position: str = 'top',
        label_color: Tuple[int, int, int] = (255, 255, 255),
        pericarp_ext_color: Tuple[int, int, int] = (0, 240, 240),
        pericarp_ext_thickness: int = 2,
    ):
        """
        Analyze fruit morphology and optionally display the annotated image.

        Delegates to
        :meth:`~traitly.fruit_phenotyping.internal_analysis.FruitInternalAnalyzer.analyze_morphology`
        and removes internal-only columns (locule, pericarp, internal, symmetry)
        from the results, since external analysis operates on whole fruits only.

        Parameters
        ----------
        contour_mode : str, optional
            Contour representation mode. Default is ``'raw'``.
        epsilon : float, optional
            Approximation factor for contour simplification. Default is 0.001.
        display_table : bool or None, optional
            If True, return the morphology results DataFrame. Default is True.
        plot : bool, optional
            If True, display the annotated result image. Default is True.
        plot_size : Tuple[int, int], optional
            Figure size for the result image. Default is (10, 10).
        font_size : float, optional
            Font scale for annotation labels. Default is 1.5.
        font_thickness : int, optional
            Thickness of annotation text. Default is 2.
        font_color : Tuple[int, int, int], optional
            BGR color for annotation text. Default is black (0, 0, 0).
        label_position : str, optional
            Position of fruit ID labels: ``'top'`` or ``'center'``.
            Default is ``'top'``.
        label_color : Tuple[int, int, int], optional
            BGR background color for labels. Default is white (255, 255, 255).
        pericarp_ext_thickness : int, optional
            Line thickness for external pericarp contour. Default is 2.
        pericarp_ext_color : Tuple[int, int, int], optional
            BGR color for external pericarp contour. Default is (0, 240, 240).

       

        Returns
        -------
        pd.DataFrame or None
            Morphology results with internal-only columns removed, or None
            if ``display_table`` is False.
        """
        super().analyze_morphology(
        plot=False,
        contour_mode=contour_mode,
        epsilon=epsilon,
        angle_shifts=0,
        num_rays=0,
        font_size=font_size,
        font_thickness=font_thickness,
        font_color=font_color,
        label_position=label_position,
        label_color=label_color,
        pericarp_ext_color=pericarp_ext_color,
        pericarp_ext_thickness=pericarp_ext_thickness,
        # Fixed internally — not exposed to external users
        min_locule_area=100,
        max_locule_area=None,
        pericarp_int_color=(0, 240, 240),
        pericarp_int_thickness=2,
        locule_color=(255, 0, 255),
        locule_thickness=2,
        centroid_fruit_color=(255, 255, 51),
        centroid_fruit_thickness=2,
        centroid_locule_color=(0, 255, 255),
        centroid_locule_thickness=2,
        display_table=True,
        is_locule=False,
        plot_size=plot_size,
    )

        keywords = ('locule', 'pericarp', 'internal', 'symmetry')

        cols_to_drop = [
            col for col in self.results.morphology_results.columns
            if any(kw in col for kw in keywords)
        ]

        self.results.morphology_results = self.results.morphology_results.drop(
            columns=cols_to_drop, errors='ignore'
        )

        # Plot from the correctly annotated results image for external analysis (no locules, and pericarp regions)
        # (super internal plot is suppressed because it shows img_copy without fruit annotations)
        if plot:
            plt.figure(figsize=plot_size)
            plt.imshow(self.results.annotated_image)
            plt.axis('off')
            plt.show()

        if display_table:
            return self.results.morphology_results
        

    def analyze_color(
        self,
        # Color extraction and metrics
        stat: str = 'mean',
        color_space: str = 'all',
        get_color_histogram: bool = False,
        # Output
        display_table: bool = True,
        # Plot
        plot: bool = False,
        plot_size: Tuple[int, int] = (10, 10),
        # Annotation
        font_size: int = 2,
        font_thickness: int = 2,
        font_color: Tuple[int, int, int] = (0, 0, 0),
        label_position: str = 'top',
        label_color: Tuple[int, int, int] = (255, 255, 255),
        label_opacity: float = 0.7,
        pericarp_ext_color: Tuple[int, int, int] = (0, 255, 0),
        pericarp_ext_thickness: int = 2

    ):
        """
        Extract color features from the total pericarp of each fruit.

        Delegates to
            :meth:`~traitly.fruit_phenotyping.internal_analysis.FruitInternalAnalyzer.analyze_color`
            with ``tissue`` fixed to ``'total_pericarp'`` and removes the
            ``'tissue'`` column from results, since external analysis does not
            segment internal tissues.

        Parameters
        ----------
        stat : str or None, optional
            Summary statistic: ``'mean'`` or ``'median'``. Default is ``'mean'``.
        color_space : str or None, optional
            Color spaces to extract. Either ``'all'`` or a comma-separated
            subset of ``'rgb'``, ``'lab'``, ``'hsv'``, ``'gray'``.
            Default is ``'all'``.
        get_color_histogram : bool, optional
            If True, also compute pixel-level color histograms. Default is False.
        display_table : bool or None, optional
            If True, return the color results DataFrame. Default is True.
        plot : bool, optional
            If True, display the annotated result image. Default is False.
        plot_size : Tuple[int, int], optional
            Figure size for the result image. Default is (10, 10).
        font_size : int, optional
            Font scale for annotation labels. Default is 2.
        font_thickness : int, optional
            Thickness of annotation text. Default is 2.
        font_color : Tuple[int, int, int], optional
            BGR color for annotation text. Default is black (0, 0, 0).
        label_position : str, optional
            Position of fruit ID labels: ``'top'`` or ``'center'``.
            Default is ``'top'``.
        label_color : Tuple[int, int, int], optional
            BGR background color for labels. Default is white (255, 255, 255).
        label_opacity : float, optional
            Opacity of label backgrounds in [0, 1]. Default is 0.7.
        pericarp_ext_color : Tuple[int, int, int], optional
            BGR color for the external pericarp contour overlay. Default is (0, 255, 0).
        pericarp_ext_thickness : int, optional
            Line thickness for the external pericarp contour. Default is 2.
       

        Returns
        -------
        pd.DataFrame or None
            Color results without the ``'tissue'`` column, or None if
            ``display_table`` is False.
        """
        
        super().analyze_color(
            stat = stat, 
            tissue = 'total_pericarp',
            color_space = color_space,
            display_table = display_table,
            plot = plot, 
            plot_size = plot_size,
            font_size = font_size,
            font_thickness = font_thickness,
            pericarp_ext_thickness = pericarp_ext_thickness,
            pericarp_ext_color = pericarp_ext_color,
            label_position = label_position,
            font_color = font_color,
            label_color = label_color,
            label_opacity = label_opacity,
            get_color_histogram = get_color_histogram
        )

        self.results.color_results = self.results.color_results.drop(columns = 'tissue', errors='ignore')

        if display_table:
            return self.results.color_results


    ##########################################################################################
    # Config sanitizer, strips all internal-only keys before calling the parent pipeline
    ##########################################################################################

    # Entire config sections that don't exist in the external pipeline
    _INTERNAL_ONLY_SECTIONS = (
        'enhance_locule_contrast_params',
        'generate_locule_mask_params',
    )

    # Individual keys inside sections that only exist in FruitInternalAnalyzer
    _INTERNAL_ONLY_PARAMS = {
        'detect_fruits_params': (
            'min_locule_area',
            'min_locule_per_fruit',
        ),
        'analyze_morphology_params': (
            'angle_shifts',
            'num_rays',
            'min_locule_area',
            'max_locule_area',
            'pericarp_int_color',
            'pericarp_int_thickness',
            'locule_color',
            'locule_thickness',
            'centroid_locule_color',
            'centroid_locule_thickness',
            'centroid_fruit_color', 
            'centroid_fruit_thickness', 
            'is_locule',
            'alpha'
        ),
        'analyze_color_params': (
            'tissue',
            'locule_color',
            'locule_thickness',
            'pericarp_int_color',
            'pericarp_int_thickness',
            'alpha'
        ),
    }

    @classmethod
    def _sanitize_config(cls, config: dict) -> dict:
        """
        Return a deep copy of ``config`` with all internal-only keys removed.

        Strips pipeline sections and parameters that only exist in
        :class:`~traitly.fruit_phenotyping.internal_analysis.FruitInternalAnalyzer`
        and are not compatible with the external analysis pipeline, such as
        locule segmentation and internal pericarp parameters.

        Parameters
        ----------
        config : dict
            Raw configuration dictionary, containing possibly internal-only
            sections or parameter keys.

        Returns
        -------
        dict
            Deep copy of ``config`` with internal-only sections and keys removed,
        safe to pass to :meth:`process_single_file` or :meth:`analyze_folder`.
        """
        cfg = copy.deepcopy(config)
        for section in cls._INTERNAL_ONLY_SECTIONS:
            cfg.pop(section, None)
        for section, keys in cls._INTERNAL_ONLY_PARAMS.items():
            if section in cfg and cfg[section]:
                for k in keys:
                    cfg[section].pop(k, None)
        return cfg

    ##########################################################################################
    # Override process_single_file to clean config before calling the parent
    ##########################################################################################

    def process_single_file(
        self,
        config=None,
        json_path=None,
        analyze_morphology=True,
        analyze_color=True,
        save_image=False,
        output_path=None
    ):
        """
        Run the full analysis pipeline on the loaded image.

        Resolves the configuration from ``json_path`` or ``config``,
        cleans internal-only parameters, and delegates to the parent
        implementation.

        Parameters
        ----------
        config : dict or None, optional
            Analysis configuration dictionary. Ignored if ``json_path`` is
            provided and valid. Default is None.
        json_path : str or None, optional
            Path to a JSON configuration file. Takes precedence over ``config``
            if the file exists. Default is None.
        analyze_morphology : bool, optional
            If True, run morphology analysis. Default is True.
        analyze_color : bool, optional
            If True, run color analysis. Default is True.
        save_image : bool, optional
            If True, save the annotated result image to ``output_path``.
            Default is False.
        output_path : str or None, optional
            Directory where output files are saved. Required if
            ``save_image=True``.

        Returns
        -------
        tuple
            Results tuple as returned by the parent
            :meth:`~traitly.fruit_phenotyping.internal_analysis.FruitInternalAnalyzer.process_single_file`.
        """
        # Resolve config the same way the parent does (json -> dict -> {})
        if json_path is not None and os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                resolved = json.load(f) or {}
        elif config is not None:
            resolved = config
        else:
            resolved = {}

        return super().process_single_file(
            config=self._sanitize_config(resolved),
            json_path=None,
            analyze_morphology=analyze_morphology,
            analyze_color=analyze_color,
            save_image=save_image,
            output_path=output_path,
        )

    ##########################################################################################
    # Process all images in a folder
    ##########################################################################################

    def analyze_folder(
        self,
        # Pipeline control
        analyze_morphology: bool = True,
        analyze_color: bool = True,
        # Configuration
        json_path=None,
        config=None,
        output_path=None,
        num_cores: int = 1,
        verbose: bool = True,
        # setup_measurements
        width_cm=None,
        length_cm=None,
        diameter_cm=None,
        fast_calibration=None,
        skip_qr=None,
        detect_label=None,
        confidence=None,
        detect_color_checker=None,
        scale_factor=None,
        # generate_fruit_mask
        lower_hsv=None,
        upper_hsv=None,
        background_color=None,
        n_iteration=None,
        kernel_blur=None,
        kernel_open=None,
        kernel_close=None,
        canny_min=None,
        canny_max=None,
        fill_holes=None,
        apply_convex_hull=None,
        remove_roi=None,
        roi_expansion=None,
        stamp=None,
        # detect_fruits
        min_fruit_area=None,
        max_fruit_area=None,
        min_fruit_circularity=None,
        rescale_factor=None,
        # analyze_morphology
        contour_mode=None,
        epsilon=None,
        angle_shifts=None,
        num_rays=None,
        # analyze_color
        stat=None,
        color_space=None,
        label_opacity=None,
        get_color_histogram=None,
        # plot
        pericarp_ext_color=None,
        pericarp_ext_thickness=None,
        label_position=None,
        label_color=None,
        font_size=None,
        font_thickness=None,
        font_color=None

    ):
        """
        Process all images in the folder passed to :class:`FruitExternalAnalyzer`.

        Collects images from the input folder, builds a configuration from
        the provided parameters, sanitizes internal-only keys, and runs the
        analysis pipeline on each image. Supports parallel processing via
        ``num_cores``. Saves merged CSV results, a session report, and an
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
            Number of parallel worker processes. Clamped to the number of
            available CPUs. Default is 1 (sequential).
        verbose : bool, optional
            If True, print progress and summary information. Default is True.
        width_cm : float or None, optional
            Known reference width in centimeters for scale calibration.
        length_cm : float or None, optional
            Known reference length in centimeters for scale calibration.
        diameter_cm : float or None, optional
            Known reference diameter in centimeters for scale calibration.
        fast_calibration : bool or None, optional
            If True, use a faster but less precise calibration method.
        skip_qr : bool or None, optional
            If True, skip QR code detection during calibration.
        detect_label : bool or None, optional
            If True, attempt to detect and read sample labels.
        confidence : float or None, optional
            Minimum detection confidence for reference objects.
        detect_color_checker : bool or None, optional
            If True, detect a color checker for color correction.
        scale_factor : float or None, optional
            Downscaling factor applied during reference detection.
        lower_hsv : List[int] or None, optional
            Lower HSV threshold for fruit segmentation.
        upper_hsv : List[int] or None, optional
            Upper HSV threshold for fruit segmentation.
        background_color : str or None, optional
            Expected background color used to guide segmentation.
        n_iteration : int or None, optional
            Number of morphological iterations for mask refinement.
        kernel_blur : int or None, optional
            Kernel size for Gaussian blur pre-processing.
        kernel_open : int or None, optional
            Kernel size for morphological opening.
        kernel_close : int or None, optional
            Kernel size for morphological closing.
        canny_min : int or None, optional
            Minimum threshold for Canny edge detection.
        canny_max : int or None, optional
            Maximum threshold for Canny edge detection.
        fill_holes : bool or None, optional
            If True, fill holes in the binary fruit mask.
        apply_convex_hull : bool or None, optional
            If True, apply convex hull to each fruit contour.
        remove_roi : bool or None, optional
            If True, remove the reference object region from the mask.
        roi_expansion : int or None, optional
            Pixel expansion applied around the reference ROI before removal.
        stamp : bool or None, optional
            If True, stamp the scale reference onto the image.
        min_fruit_area : int or None, optional
            Minimum contour area in pixels to be considered a fruit.
        max_fruit_area : int or None, optional
            Maximum contour area in pixels for fruit filtering.
        min_fruit_circularity : float or None, optional
            Minimum circularity score in [0, 1] to filter non-fruit contours.
        rescale_factor : float or None, optional
            Factor to rescale contours before detection.
        contour_mode : str or None, optional
            Contour representation mode for morphology analysis.
        epsilon : float or None, optional
            Approximation factor for contour simplification.
        angle_shifts : int or None, optional
            Number of angle steps for symmetry computation.
        num_rays : int or None, optional
            Number of rays used to estimate pericarp thickness.
        stat : str or None, optional
            Color summary statistic: ``'mean'`` or ``'median'``.
        color_space : str or None, optional
            Color spaces to extract during color analysis.
        get_color_histogram : bool or None, optional
            If True, compute pixel-level color histograms per fruit.
        label_opacity : float or None, optional
            Opacity of annotation label backgrounds in [0, 1].


        Raises
        ------
        ValueError
            If the instance was not initialized with a directory path, or if
            no valid images are found in the folder.
        """

        if not self.is_directory:
            raise ValueError(
                "analyze_folder() requires a directory path. "
                "Pass a folder to FruitExternalAnalyzer(), not a single file."
            )


        folder_path = self.img_path

        # Validate cores
        num_cores_message = None
        if num_cores <= 0:
            num_cores_message = f"    > num_cores={num_cores} must be ≥ 1. Using 1."
            num_cores = 1
        max_cores = mp.cpu_count()
        if num_cores > max_cores:
            num_cores_message = f"    > num_cores={num_cores} exceeds {max_cores}. Using {max_cores}."
            num_cores = max_cores

        # Collect images
        img_paths = sorted([
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if Path(f).suffix.lower() in _valid_ext
        ])
        if not img_paths:
            raise ValueError(f"No valid images found in: {folder_path}")

        if output_path is None:
            output_path = os.path.join(folder_path, "Results")
        os.makedirs(output_path, exist_ok=True)

        # Build config from json/dict + individual params
        cfg = copy.deepcopy(config) if config else {}
        if json_path is not None and os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                cfg.update(json.load(f) or {})

        def _apply(section, mapping):
            overrides = {k: v for k, v in mapping.items() if v is not None}
            if overrides:
                cfg.setdefault(section, {})
                cfg[section].update(overrides)

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
            canny_min=canny_min, canny_max=canny_max, remove_roi=remove_roi,
            roi_expansion=roi_expansion, background_color=background_color,
            fill_holes=fill_holes, apply_convex_hull=apply_convex_hull,
        ))
        _apply('detect_fruits_params', dict(
            min_fruit_area=min_fruit_area, max_fruit_area=max_fruit_area,
            min_fruit_circularity=min_fruit_circularity, rescale_factor=rescale_factor,
        ))
        _apply('analyze_morphology_params', dict(
            contour_mode=contour_mode, epsilon=epsilon,
            angle_shifts=angle_shifts, num_rays=num_rays,
            font_size=font_size, font_thickness=font_thickness,
            font_color=font_color, label_position=label_position,
            label_color=label_color, pericarp_ext_color=pericarp_ext_color,
            pericarp_ext_thickness=pericarp_ext_thickness
        ))
        _apply('analyze_color_params', dict(
            stat=stat, color_space=color_space,
            label_opacity=label_opacity,
            get_color_histogram=get_color_histogram,
        ))


        # Sync to self.parameters for session report
        for key in ('setup_measurements_params', 'generate_fruit_mask_params',
                    'detect_fruits_params', 'analyze_morphology_params', 'analyze_color_params'):
            value = cfg.get(key)
            if isinstance(value, dict) and value:  
                setattr(self.parameters, key, value)

        # clean before distributing to workers
        cfg = self._sanitize_config(cfg)

        session_start = datetime.now()
        if verbose:
            print("=" * 60)
            print(" Traitly running ⋆✧｡٩(ˊᗜˋ )و✧*｡   ")
            print("=" * 60)
            print(f"    > Input folder: {folder_path}")
            print(f"    > Image(s) detected: {len(img_paths)}")
            print(f"    > analyze_morphology: {analyze_morphology}")
            print(f"    > analyze_color: {analyze_color}")
            print(num_cores_message if num_cores_message else f"    > num_cores: {num_cores}")
            if json_path is not None:
                print(f"    > Parameters loaded from: {json_path}\n")

        all_morphology, all_color, errors = [], [], []
        total_fruits = 0

        def _run_one(img_path):
            t0 = t.time()
            try:
                worker = FruitExternalAnalyzer(img_path)
                worker.load_image(plot=False)
                df_m, df_c, err, n, ann_img = worker.process_single_file(
                    config=cfg, json_path=None,
                    analyze_morphology=analyze_morphology,
                    analyze_color=analyze_color,
                    save_image=True, output_path=output_path,
                )
                return df_m, df_c, err, n, ann_img, os.path.basename(img_path), t.time() - t0
            except Exception as e:
                return None, None, {
                    'filename': os.path.basename(img_path), 'status': f'Error: {str(e)}'
                }, 0, None, os.path.basename(img_path), t.time() - t0

        if num_cores == 1:
            for img_path in tqdm(img_paths, desc="Processing images", unit="img", disable=not verbose):
                df_m, df_c, err, n, _, fname, _ = _run_one(img_path)
                if err:
                    errors.append(err)
                else:
                    if df_m is not None: all_morphology.append(df_m)
                    if df_c is not None: all_color.append(df_c)
                    total_fruits += n
        else:
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                futures = {
                    executor.submit(_process_external_worker,
                                    p, cfg, analyze_morphology, analyze_color): p
                    for p in img_paths
                }
                for future in tqdm(as_completed(futures), total=len(futures),
                                    desc="Processing images", unit="img", disable=not verbose):
                    result = future.result()
                    df_m, df_c, err, n, ann_img, fname = result[:6]
                    if err:
                        errors.append(err)
                    else:
                        if ann_img is not None:
                            out_img = os.path.join(output_path,
                                                    f"{os.path.splitext(fname)[0]}_annotated.jpg")
                            cv2.imwrite(out_img, ann_img)
                        if df_m is not None: all_morphology.append(df_m)
                        if df_c is not None: all_color.append(df_c)
                        total_fruits += n

        # Merge and save CSVs
        df_morph_all = pd.concat(all_morphology, ignore_index=True) if all_morphology else None
        df_color_all = pd.concat(all_color,      ignore_index=True) if all_color      else None

        morph_csv = color_csv = None
        if df_morph_all is not None:
            morph_csv = os.path.join(output_path, "morphology_results.csv")
            df_morph_all.to_csv(morph_csv, index=False)
        if df_color_all is not None:
            color_csv = os.path.join(output_path, "color_results.csv")
            df_color_all.to_csv(color_csv, index=False)

        # Session report
        session_end = datetime.now()
        total_time = (session_end - session_start).total_seconds()
        avg_time   = total_time / len(img_paths) if img_paths else 0

        def _filter_params(p):
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

        for title, attr in (
            ('SETUP_MEASUREMENTS',  'setup_measurements_params'),
            ('GENERATE_FRUIT_MASK', 'generate_fruit_mask_params'),
            ('DETECT_FRUITS',       'detect_fruits_params'),
            ('ANALYZE_MORPHOLOGY',  'analyze_morphology_params'),
            ('ANALYZE_COLOR',       'analyze_color_params'),
        ):
            raw = getattr(self.parameters, attr, {}) or {}
            filtered = _filter_params(raw)
            if filtered:
                session_lines.append(f"\n{title}:")
                for k, v in filtered.items():
                    session_lines.append(f"   - {k}: {v}")

        session_lines += [
            "", "=" * 70, "DEPENDENCIES", "=" * 70,
        ] + [f"   - {pkg:<30} {ver}"
             for pkg, ver in self.parameters.get_package_versions().items()]

        session_txt = os.path.join(output_path, "session_report.txt")
        with open(session_txt, 'w', encoding='utf-8') as f:
            f.write("\n".join(session_lines))

        # Error report
        error_txt = None
        if errors:
            col1_w = max(len("IMAGE"),  max(len(e['filename']) for e in errors)) + 2
            col2_w = max(len("ERROR"),  max(len(e['status'])   for e in errors)) + 2
            sep    = f"+{'-' * col1_w}+{'-' * col2_w}+"
            header = f"| {'IMAGE':<{col1_w-2}} | {'ERROR':<{col2_w-2}} |"
            error_lines = [
                "=" * 70, "ERROR REPORT", "=" * 70,
                f"run date   : {session_start.strftime('%Y-%m-%d %H:%M:%S')}",
                f"folder     : {folder_path}",
                f"failed     : {len(errors)}/{len(img_paths)} images",
                "", sep, header, sep,
            ] + [f"| {e['filename']:<{col1_w-2}} | {e['status']:<{col2_w-2}} |"
                 for e in errors] + [sep]
            error_txt = os.path.join(output_path, "error_report.txt")
            with open(error_txt, 'w', encoding='utf-8') as f:
                f.write("\n".join(error_lines))

        if verbose:
            n_ok = len(img_paths) - len(errors)
            print("\n( ദ്ദി ˙ᗜ˙ ) Finished " + "="*47)
            print(f"        - Successfully: {n_ok}/{len(img_paths)} img(s)")
            if errors:
                print(f"        - Errors: {len(errors)}/{len(img_paths)} img(s)")
            print(f"        - Total fruits: {total_fruits}")
            print(f"        - Total time: {total_time:.1f}s  (avg {avg_time:.1f}s/img)")
            print("    > Files saved:")
            print(f"        - {n_ok} annotated image(s)")
            if morph_csv:  print(f"        - {os.path.basename(morph_csv)}")
            if color_csv:  print(f"        - {os.path.basename(color_csv)}")
            print(f"        - {os.path.basename(session_txt)}")
            if error_txt:  print(f"        - {os.path.basename(error_txt)}")
            print(f"        - Results folder: {output_path}")

        return None