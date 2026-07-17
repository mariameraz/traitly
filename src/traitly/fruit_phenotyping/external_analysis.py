# traitly/fruit_phenotyping/external_analysis.py
"""
External fruit analysis pipeline for traitly.

Provides the :class:`FruitExternalAnalyzer` class, which extends
:class:`~traitly.fruit_phenotyping.internal_analysis.FruitInternalAnalyzer`
for analyzing whole-fruit morphology and color without locule or internal
pericarp segmentation. Includes support for single-image and batch folder
processing with optional multiprocessing.

"""

# ============================================================================
# STANDARD LIBRARY
# ============================================================================
import copy
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
#import logging
# ============================================================================
# THIRD-PARTY LIBRARIES
# ============================================================================
import cv2
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# INTERNAL IMPORTS
# ============================================================================
from traitly.fruit_phenotyping.internal_analysis import FruitInternalAnalyzer

from traitly.utils.batch import (
    _setup_batch,
    _print_batch_header,
    _run_fruit_batch_loop,
    _save_fruit_batch_results,
    _config_from_json,
)

#############
# Get logs  #
#############

# logger = logging.getLogger(__name__)

##########################################################################################
# Worker function for parallel processing
##########################################################################################

def _process_external_image_worker(
    img_path: str,
    config: Dict,
    analyze_morphology: bool,
    analyze_color: bool,
    output_path: str = None,
) -> Tuple:

    # 1. Starts counting time processing
    t0 = time.perf_counter()

    # 2. Run the individual analysis
    try:
        analyzer = FruitExternalAnalyzer(img_path)
        analyzer.load_image(plot = False)
        df_morphology, df_color, error_dict, n_fruits, annotated_img = (
            analyzer._process_single_file(
                config = config,
                json_path = None,
                analyze_morphology = analyze_morphology,
                analyze_color = analyze_color,
                save_image = False,
                output_path = output_path
            )
        )

        # 3. Get processing total time
        elapsed = time.perf_counter() - t0

        # 4. Save image file name
        filename = os.path.basename(img_path)

        # 5. Return all the results for an analyzed image
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
            time.perf_counter() - t0,
        )


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
    path : str
        Path to an image file or a folder containing images.
    """

    def __init__(self, path: str):
        super().__init__(path)

        self.external_features = None

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
    ):
        """
        Set up scale calibration and [reference size, label text, color checker] detection.

        Delegates to the parent implementation. See
        :meth:`~traitly.fruit_phenotyping.internal_analysis.FruitInternalAnalyzer.setup_measurements`
        for full parameter documentation.
        """
        super().setup_measurements(
            plot=plot,
            font_size=font_size,
            confidence=confidence,
            detect_label=detect_label,
            verbose=verbose,
            plot_size=plot_size,
            language_label=language_label,
            width_cm=width_cm,
            length_cm=length_cm,
            diameter_cm=diameter_cm,
            gpu=gpu,
            skip_qr=skip_qr,
            skip_yolo=skip_yolo,
        )

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
        roi_expansion: int = 5,
        background_color: Optional[str] = None,
        fill_holes: bool = False,
        apply_convex_hull: bool = False,
        erosion_px: int = 0,
    ) -> None:
        """
        Generate a binary mask segmenting fruits from the background.

        Delegates to the parent implementation. See
        :meth:`~traitly.fruit_phenotyping.internal_analysis.FruitInternalAnalyzer.generate_fruit_mask`
        for full parameter documentation.
        """

        if lower_hsv is None or upper_hsv is None:
            if background_color is None:
                background_color = (
                    "blue"  # Default to blue if no HSV or background color provided
                )

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
            apply_convex_hull=apply_convex_hull,
            erosion_px=erosion_px,
        )

    def detect_fruits(
        self,
        min_fruit_area: int = 1000,
        max_fruit_area: Optional[int] = None,
        min_fruit_circularity: float = 0.5,
        verbose: bool = True,
        plot: bool = False,
        plot_size: Tuple[int, int] = (5, 5),
        contour_color: Tuple[int, int, int] = (0, 255, 0),
        contour_thickness: int = 2,
        rescale_factor: Optional[float] = None,
    ) -> None:
        """
        Detect individual fruits from the binary mask.

        Delegates to the parent implementation with ``min_locule_per_fruit``
        fixed to 0. Prints a summary of detected fruits when ``verbose=True``.

        Parameters
        ----------
        min_fruit_area : int, optional
            Minimum contour area in pixels to be considered a fruit.
            Default is 1000.
        max_fruit_area : int or None, optional
            Maximum contour area in pixels. If None, no upper limit is applied.
        min_fruit_circularity : float, optional
            Minimum circularity score in [0, 1] to filter non-fruit contours.
            Default is 0.5.
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
            min_locule_per_fruit=0,
            verbose=False,
            plot=plot,
            plot_size=plot_size,
            contour_color=contour_color,
            contour_thickness=contour_thickness,
            rescale_factor = rescale_factor,
        )

        if self.fruit_locule_map is not None:
            n_fruits_detected = len(self.fruit_locule_map)
        else:
            n_fruits_detected = "0"

        if verbose:
            optional_config = {
                "max_fruit_area": max_fruit_area,
            }
            print("\n" + "=" * 37)
            print(
                f"        . ݁₊ ⊹ . ݁ ⟡ ݁ Detected fruits: {n_fruits_detected} ⟡ ݁ . ⊹ ₊ ݁."
            )
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
        margin: Optional[int] = 5,
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
            fruit_id=fruit_id,
            plot_size=plot_size,
            overlay=False,
            margin=margin,
            only_fruit=True,
        )

    def analyze_morphology(
        self,
        # Contour
        contour_mode: str = "raw",
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
        label_position: str = "top",
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

        keywords = ("locule", "pericarp", "internal", "symmetry")

        cols_to_drop = [
            col
            for col in self.results.morphology_results.columns
            if any(kw in col for kw in keywords)
        ]

        self.results.morphology_results = self.results.morphology_results.drop(
            columns=cols_to_drop, errors="ignore"
        )

        if plot:
            plt.figure(figsize=plot_size)
            plt.imshow(cv2.cvtColor(self.results.morphology_image, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.show()

        if display_table:
            return self.results.morphology_results

    def analyze_color(
        self,
        # Color extraction and metrics
        stat: str = "mean",
        color_space: str = "all",
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
        label_position: str = "top",
        label_color: Tuple[int, int, int] = (255, 255, 255),
        label_opacity: float = 0.7,
        pericarp_ext_color: Tuple[int, int, int] = (0, 255, 0),
        pericarp_ext_thickness: int = 2,
        dark_thresh: int = 15,
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
            stat=stat,
            tissue="total_pericarp",
            color_space=color_space,
            display_table=display_table,
            plot=plot,
            plot_size=plot_size,
            font_size=font_size,
            font_thickness=font_thickness,
            pericarp_ext_thickness=pericarp_ext_thickness,
            pericarp_ext_color=pericarp_ext_color,
            label_position=label_position,
            font_color=font_color,
            label_color=label_color,
            label_opacity=label_opacity,
            get_color_histogram=get_color_histogram,
            dark_thresh=dark_thresh,
        )

        self.results.color_results = self.results.color_results.drop(
            columns="tissue", errors="ignore"
        )

        if display_table:
            return self.results.color_results

    ##########################################################################################
    # Config sanitizer, strips all internal-only keys before calling the parent pipeline
    ##########################################################################################

    # Entire config sections that don't exist in the external pipeline
    _INTERNAL_ONLY_SECTIONS = (
        "enhance_locule_contrast_params",
        "generate_locule_mask_params",
    )

    # Individual keys inside sections that only exist in FruitInternalAnalyzer
    _INTERNAL_ONLY_PARAMS = {
        "detect_fruits_params": (
            "min_locule_area",
            "min_locule_per_fruit",
            "locule_thickness",
            "locule_color",
            "pericarp_int_color",
            "pericarp_int_thickness",
            "dilation_factor",
        ),
        "analyze_morphology_params": (
            "angle_shifts",
            "num_rays",
            "min_locule_area",
            "max_locule_area",
            "pericarp_int_color",
            "pericarp_int_thickness",
            "locule_color",
            "locule_thickness",
            "centroid_locule_color",
            "centroid_locule_thickness",
            "centroid_fruit_color",
            "centroid_fruit_thickness",
            "is_locule",
            "dilation_factor",
        ),
        "analyze_color_params": (
            "tissue",
            "locule_color",
            "locule_thickness",
            "pericarp_int_color",
            "pericarp_int_thickness",
            "dilation_factor",
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
        safe to pass to :meth:`_process_single_file` or :meth:`analyze_folder`.
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
    #                                     BATCH ANALYSIS

    ##########################################################################################
    # Process all images in a folder
    ##########################################################################################

    def analyze_folder(
        self,
        # Pipeline control
        analyze_morphology: bool = True,
        analyze_color: bool = True,
        # Configuration
        num_cores: int = 1,
        verbose: bool = True,
        json_path: Optional[str] = None,
        config: Optional[dict] = None,
        output_path: Optional[str] = None,
        # setup_measurements
        width_cm: Optional[float] = None,
        length_cm: Optional[float] = None,
        diameter_cm: Optional[float] = None,
        skip_yolo: Optional[bool] = None,
        skip_qr: Optional[bool] = None,
        detect_label: Optional[bool] = None,
        confidence: Optional[float] = None,
        # generate_fruit_mask
        lower_hsv: Optional[Tuple[int, int, int]] = None,
        upper_hsv: Optional[Tuple[int, int, int]] = None,
        background_color: Optional[str] = None,
        n_iteration: Optional[int]=None,
        kernel_blur: Optional[int]=None,
        kernel_open: Optional[int]=None,
        kernel_close: Optional[int]=None,
        canny_min: Optional[int]=None,
        canny_max: Optional[int]=None,
        fill_holes: Optional[bool] = None,
        apply_convex_hull: Optional[bool] = None,
        remove_roi: Optional[bool] = None,
        roi_expansion: Optional[int] = None,
        stamp: Optional[bool] = None,
        erosion_px: Optional[int] = None,
        # detect_fruits
        min_fruit_area: Optional[int] = None,
        max_fruit_area: Optional[int]=None,
        min_fruit_circularity: Optional[float]=None,
        rescale_factor: Optional[float] = None,
        # analyze_morphology
        contour_mode: Optional[str]=None,
        epsilon: Optional[float]=None,
        angle_shifts: Optional[int]=None,
        num_rays: Optional[int]=None,
        # analyze_color
        stat: Optional[str]=None,
        color_space: Optional[str]=None,
        label_opacity: Optional[float]=None,
        get_color_histogram: Optional[bool]=None,
        # plot
        pericarp_ext_color: Optional[Tuple[int, int, int]]=None,
        pericarp_ext_thickness: Optional[int] = None,
        label_position: Optional[str] = None,
        label_color: Optional[Tuple[int, int, int]]=None,
        font_size: Optional[int] = None,
        font_thickness: Optional[int]=None,
        font_color: Optional[Tuple[int, int, int]]=None,
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
        skip_yolo : bool or None, optional
            If True, use a faster but less precise calibration method.
        skip_qr : bool or None, optional
            If True, skip QR code detection during calibration.
        detect_label : bool or None, optional
            If True, attempt to detect and read sample labels.
        confidence : float or None, optional
            Minimum detection confidence for reference objects.
        lower_hsv : Tuple[int,int,int] or None, optional
            Lower HSV threshold for fruit segmentation.
        upper_hsv : Tuple[int,int,int] or None, optional
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

        _apply(
            "setup_measurements_params",
            dict(
                width_cm=width_cm, length_cm=length_cm, diameter_cm=diameter_cm,
                skip_yolo=skip_yolo, skip_qr=skip_qr,
                detect_label=detect_label, confidence=confidence,
            ),
        )
        _apply(
            "generate_fruit_mask_params",
            dict(
                stamp=stamp, lower_hsv=lower_hsv, upper_hsv=upper_hsv,
                n_iteration=n_iteration, kernel_blur=kernel_blur,
                kernel_open=kernel_open, kernel_close=kernel_close,
                canny_min=canny_min, canny_max=canny_max,
                remove_roi=remove_roi, roi_expansion=roi_expansion,
                background_color=background_color, fill_holes=fill_holes,
                apply_convex_hull=apply_convex_hull, erosion_px = erosion_px
            ),
        )
        _apply(
            "detect_fruits_params",
            dict(
                min_fruit_area=min_fruit_area,
                max_fruit_area=max_fruit_area,
                min_fruit_circularity=min_fruit_circularity,
                rescale_factor = rescale_factor,
            ),
        )
        _apply(
            "analyze_morphology_params",
            dict(
                contour_mode=contour_mode, epsilon=epsilon,
                angle_shifts=angle_shifts, num_rays=num_rays,
                font_size=font_size, font_thickness=font_thickness,
                font_color=font_color, label_position=label_position,
                label_color=label_color, pericarp_ext_color=pericarp_ext_color,
                pericarp_ext_thickness=pericarp_ext_thickness,
            ),
        )
        _apply(
            "analyze_color_params",
            dict(
                stat=stat,
                color_space=color_space,
                label_opacity=label_opacity,
                get_color_histogram=get_color_histogram,
            ),
        )

        # 3. Sanitize parameters
        config = self._sanitize_config(config)

        # 4. Sync to self._parameters for session report
        for key in (
            "setup_measurements_params",
            "generate_fruit_mask_params",
            "detect_fruits_params",
            "analyze_morphology_params",
            "analyze_color_params",
        ):
            value = config.get(key)
            if isinstance(value, dict) and value:
                setattr(self._parameters, key, value)


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
            worker_fn=_process_external_image_worker,
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
                "DETECT_FRUITS": "detect_fruits_params",
                "ANALYZE_MORPHOLOGY": "analyze_morphology_params",
                "ANALYZE_COLOR": "analyze_color_params",
            },
            verbose=verbose,
        )

        return None
