# traitly/fruit_phenotyping/analysis.py
"""
Core morphological analysis functions for fruit phenotyping pipelines.

Provides the :class:`FruitConfig` dataclass and functions to analyze
fruit contours, extract morphological features, process locules, compute
pericarp metrics, and annotate result images. Designed to be called from
higher-level analyzer classes.
"""

# ============================================================================
# STANDARD LIBRARY
# ============================================================================
from typing import List, Dict, Tuple, Optional, Any

# ===========================================================================
# THIRD-PARTY LIBRARIES
# ===========================================================================
import cv2
import numpy as np
import pandas as pd
from dataclasses import dataclass

# ============================================================================
# INTERNAL IMPORTS
# ===========================================================================
from .geometry import (
    calculate_axes, 
    rotate_box, 
    get_fruit_morphology
    )
from .symmetry import (
    angular_symmetry, 
    radial_symmetry, 
    get_unique_locule_counts, 
    precompute_ideal_angles
    )
from .processing import (
    get_internal_pericarp_area,
    calculate_fruit_centroids,
    precalculate_locules_data,
    get_fruit_contour,
    get_internal_pericarp_contour,
    calculate_pericarp_thickness_radial
)
from .mask import merge_locules_func
from .results_image import ResultsImage

@dataclass
class FruitConfig:
    """
    Configuration parameters for fruit morphological analysis.

    Groups all tunable settings for contour processing, locule filtering,
    symmetry computation, pericarp analysis, and visualization into a
    single dataclass passed through the analysis pipeline.

    Attributes
    ----------
    contour_mode : str
        Contour representation mode. Default is ``'raw'``.
    epsilon : float
        Approximation factor for contour simplification when ``contour_mode = 'approx'``. Default is 0.002.
    min_locule_area : int
        Minimum locule area in pixels to be included in analysis.
        Default is 300.
    max_locule_area : int or None
        Maximum locule area in pixels. If None, no upper limit is applied.
    merge_locules : bool
        If True, nearby locules are merged before analysis. Default is False.
    min_locule_distance : int
        Minimum pixel distance between locules for merging. Default is 1.
    max_locule_distance : int
        Maximum pixel distance between locules for merging. Default is 10.
    angle_shifts : int
        Number of angle steps for angular symmetry calculation. Default is 500.
    num_rays : int
        Number of rays used to estimate pericarp thickness. Default is 180.
    plot : bool
        If True, display the annotated result image. Default is True.
    plot_size : Tuple[int, int]
        Figure size for the result image. Default is (20, 10).
    font_scale : int
        Font scale for annotation labels. Default is 1.
    font_thickness : int
        Thickness of annotation text. Default is 2.
    text_color : Tuple[int, int, int]
        BGR color for annotation text. Default is black (0, 0, 0).
    label_background_color : Tuple[int, int, int]
        BGR background color for annotation labels.
        Default is white (255, 255, 255).
    padding : int
        Pixel padding around annotation label backgrounds. Default is 15.
    line_spacing : int
        Pixel spacing between lines in multi-line annotations. Default is 15.
    label_position : str
        Position of fruit ID labels relative to the fruit bounding box.
        One of ``'top'``, ``'bottom'``, ``'left'``, ``'right'``.
        Default is ``'top'``.
    pericarp_ext_color : Tuple[int, int, int]
        BGR color for the external pericarp contour. Default is (0, 255, 0).
    pericarp_ext_thickness : int
        Line thickness for the external pericarp contour. Default is 2.
    locule_color : Tuple[int, int, int]
        BGR color for locule contours. Default is (255, 0, 255).
    locule_thickness : int
        Line thickness for locule contours. Default is 2.
    centroid_fruit_color : Tuple[int, int, int]
        BGR color for fruit centroid markers. Default is (255, 255, 51).
    centroid_fruit_thickness : int
        Radius of fruit centroid markers in pixels. Default is 3.
    centroid_locule_color : Tuple[int, int, int]
        BGR color for locule centroid markers. Default is (0, 255, 255).
    centroid_locule_thickness : int
        Radius of locule centroid markers in pixels. Default is 3.

    """

    # Contour settings
    contour_mode: str = 'raw'
    epsilon: float = 0.002
    
    # Locule settings
    min_locule_area: int = 300
    max_locule_area: Optional[int] = None
    merge_locules: bool = False
    min_locule_distance: int = 1
    max_locule_distance: int = 10
    
    # Symmetry settings
    angle_shifts: int = 500
    
    # Pericarp settings
    num_rays: int = 180
    
    # Visualization settings
    plot: bool = True
    plot_size: Tuple[int, int] = (20, 10)
    font_scale: int = 1
    font_thickness: int = 2
    text_color: Tuple[int, int, int] = (0, 0, 0)
    label_background_color: Tuple[int, int, int] = (255, 255, 255)
    padding: int = 15
    line_spacing: int = 15
    centroid_fruit_thickness: int = 3
    centroid_fruit_color: Tuple[int,int,int] = (255, 255, 51)
    centroid_locule_thickness: int = 3
    centroid_locule_color: Tuple[int,int,int] = (0, 255, 255)
    label_position: str = 'top'
    pericarp_ext_color: Tuple[int,int,int] = (0, 255, 0)
    pericarp_ext_thickness: int = 2
    locule_color: Tuple[int,int,int] = (255, 0, 255)
    locule_thickness: int = 2

##########################
# Morphological analysis #
##########################

def analyze_fruits_morphology(
    img: np.ndarray,
    contours: List[np.ndarray],
    fruit_locule_map: Dict[int, List[int]],
    px_per_cm: Optional[float],
    img_name: str,
    label_text: str,
    label_id: Optional[int] = None,
    path: Optional[str] = None,
    original_img_clean: Optional[np.ndarray] = None,
    is_locule: bool = True,
    epsilon: float = 0.002,
    pericarp_int_color: Tuple[int, int, int] = (0, 240, 240),
    pericarp_int_thickness: int = 2,
    **kwargs
) -> ResultsImage:
    """
    Analyze fruit contours and extract morphological features.

    Iterates over every fruit in ``fruit_locule_map``, delegates per-fruit
    analysis to :func:`_analyze_single_fruit_morphology`, and accumulates
    results into a :class:`ResultsImage`. Symmetry precomputation via
    :func:`precompute_ideal_angles` and centroid precomputation via
    :func:`calculate_fruit_centroids` are performed once before the loop
    to avoid redundant work.

    Parameters
    ----------
    img : np.ndarray
        Input image in BGR format.
    contours : list of np.ndarray
        All detected contours from the segmentation step.
    fruit_locule_map : dict of {int : list of int}
        Mapping from fruit contour index to list of locule contour indices.
    px_per_cm : float or None
        Pixel-to-centimeter conversion factor. Pass ``None`` to work in pixel units.
    img_name : str
        Filename or identifier of the source image, stored in results.
    label_text : str
        Treatment or label identifier stored in results.
    label_id : int or None, optional
        Contour index of the calibration/label region to exclude from analysis.
        Default is ``None``.
    path : str or None, optional
        Filesystem path to the original image, stored in the returned
        :class:`ResultsImage`. Default is ``None``.
    original_img_clean : np.ndarray or None, optional
        Clean copy of the original image used for annotation. If ``None``,
        ``img`` is used. Default is ``None``.
    is_locule : bool, optional
        Whether to analyze internal locule structure. If ``False``, locule,
        pericarp, and symmetry metrics are set to ``NaN``. Default is ``True``.
    epsilon : float, optional
        Approximation factor for contour simplification when
        ``contour_mode='approx'``. Default is 0.002.
    pericarp_int_color : tuple of int, optional
        BGR color for the internal pericarp contour overlay. Default is
        ``(0, 240, 240)``.
    pericarp_int_thickness : int, optional
        Line thickness for the internal pericarp contour overlay. Default is 2.
    **kwargs
        Additional keyword arguments forwarded to :class:`FruitConfig`.

    Returns
    -------
    ResultsImage
        Object containing the annotated image and a list of per-fruit
        morphological result dictionaries.
    """
    config_dict = {k: v for k, v in kwargs.items() 
                   if k in FruitConfig.__dataclass_fields__}
    config_dict['epsilon'] = epsilon  
    
    
    config = FruitConfig(**config_dict)
    
    has_calibration = px_per_cm is not None and px_per_cm > 0
    unit = 'cm' if has_calibration else 'px'
    
    original_img = original_img_clean.copy() if original_img_clean is not None else img.copy()
    
    # Precalculate angular symmetry data
    unique_counts = get_unique_locule_counts(fruit_locule_map)
    precomputed_ideals = precompute_ideal_angles(unique_counts, 
                                                 angle_shifts=config.angle_shifts)
    
    # Precalculate fruit centroids data
    fruit_centroids = calculate_fruit_centroids(contours)

    # If present, filter label contour
    items = fruit_locule_map.items()
    if label_id is not None:
        items = ((fid, locs) for fid, locs in items if fid != label_id)

    # Create empty list to save results
    results = []
    # color_results = []

    # For each fruit and its locules:
    sequential_id = 1

    for fruit_id, locules in items:
        try:

            result = _analyze_single_fruit_morphology(
                fruit_id=fruit_id,
                locules=locules,
                contours=contours,
                fruit_centroids=fruit_centroids,
                annotated_img=original_img,
                px_per_cm=px_per_cm,
                img_name=img_name,
                label_text=label_text,
                sequential_id=sequential_id,
                img_shape=img.shape[:2],
                config=config,
                precomputed_ideals=precomputed_ideals, 
                unit = unit,
                pericarp_int_color = pericarp_int_color,
                pericarp_int_thickness = pericarp_int_thickness,
                is_locule = is_locule
            )
            
            if result is not None:
                if isinstance(result, tuple):
                    morphology_result, color_result = result
                    results.append(morphology_result)
                    
                else:
                    results.append(result)

                sequential_id += 1
                
        except Exception as e:
            print(f"Error processing fruit {fruit_id}: {e}")
            continue
    
    if config.plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=config.plot_size)
        plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
   
    return ResultsImage(
        original_img, 
        results, 
        image_path=path
    )
#######################################
# Single fruit morphological analysis #
#######################################

def _analyze_single_fruit_morphology(
    fruit_id: int,
    locules: List[int],
    contours: List[np.ndarray],
    fruit_centroids: List[Optional[Tuple[float, float]]],
    annotated_img: np.ndarray,
    px_per_cm: Optional[float],
    img_name: str,
    label_text: str,
    sequential_id: int,
    img_shape: Tuple[int, int],
    config: FruitConfig,
    precomputed_ideals: Optional[Dict] = None,
    unit: str = 'px',
    is_locule: bool = True,
    pericarp_int_color: Tuple[int, int, int] = (0, 240, 240),
    pericarp_int_thickness: int = 2,
) -> Optional[Dict[str, Any]]:
    """
    Analyze a single fruit and return its morphological metrics.

    Orchestrates the full per-fruit pipeline by calling, in order:
    :func:`_prepare_fruit_data`, :func:`_calculate_fruit_metrics`,
    :func:`_process_locules`, :func:`_calculate_pericarp_metrics`,
    :func:`_calculate_symmetry_metrics`, :func:`_calculate_derived_metrics`,
    :func:`_annotate_fruit`, and :func:`_format_results`.

    Parameters
    ----------
    fruit_id : int
        Index into ``contours`` identifying the fruit contour.
    locules : list of int
        Indices into ``contours`` for locules belonging to this fruit.
    contours : list of np.ndarray
        Full list of all detected contours.
    fruit_centroids : list of tuple of float or None
        Precomputed ``(cx, cy)`` centroids for every contour. Entry is
        ``None`` if the centroid could not be computed.
    annotated_img : np.ndarray
        BGR image that is modified in-place with overlays and annotations.
    px_per_cm : float or None
        Pixel-to-centimeter conversion factor. ``None`` means pixel units.
    img_name : str
        Image filename stored in the result dictionary.
    label_text : str
        Treatment or label identifier stored in the result dictionary.
    sequential_id : int
        One-based display ID assigned to this fruit in annotation order.
    img_shape : tuple of int
        ``(height, width)`` of the image, used for boundary checks.
    config : FruitConfig
        Configuration dataclass with all tunable analysis parameters.
    precomputed_ideals : dict or None, optional
        Precomputed ideal angle arrays keyed by locule count, used to
        accelerate angular symmetry calculation. Default is ``None``.
    unit : str, optional
        Active measurement unit, either ``'cm'`` or ``'px'``.
        Default is ``'px'``.
    is_locule : bool, optional
        Whether to compute locule, pericarp and symmetry metrics.
        Default is ``True``.
    pericarp_int_color : tuple of int, optional
        BGR color for the internal pericarp contour overlay. Default is
        ``(0, 240, 240)``.
    pericarp_int_thickness : int, optional
        Line thickness for the internal pericarp contour overlay. Default is 2.

    Returns
    -------
    dict of {str : Any} or None
        Flat dictionary of all morphological metrics for the fruit as
        returned by :func:`_format_results`, or ``None`` if the fruit
        centroid could not be determined in :func:`_prepare_fruit_data`.
    """
    # 1. Prepare fruit data

    fruit_data = _prepare_fruit_data(
        fruit_id, contours, fruit_centroids,
        annotated_img, config
    )
    if fruit_data is None:
        return None


    # 2. Calculate fruit metrics
    fruit_metrics = _calculate_fruit_metrics(
        fruit_data['contour'],
        contours[fruit_id],
        px_per_cm,
        unit,
        config,
        annotated_img,
    )

    unit_suffix = 'cm2' if unit == 'cm' else 'px2'
    if is_locule:
        # 3. Process locules

        locule_metrics = _process_locules(
            locules,
            contours,
            fruit_data['centroid'],
            annotated_img,
            px_per_cm,
            config,
            unit
        )

        # 4. Calculate pericarp metrics
        pericarp_metrics = _calculate_pericarp_metrics(
            locule_metrics['filtered_ids'],
            contours,
            fruit_data['contour'],
            fruit_data['centroid'],
            annotated_img,
            img_shape,
            px_per_cm,
            config,
            unit,
            pericarp_int_color,
            pericarp_int_thickness
        )
        

        # 5. Calculate symmetry
        symmetry_metrics = _calculate_symmetry_metrics(
            locule_metrics['data'],
            config,
            precomputed_ideals
        )
        
    else:
        # When no locules, assign NaN defaults 
        locule_metrics = {
            'data':                          [],
            'filtered_ids':                  [],
            'count':                         0,
            f'locules_mean_area_{unit_suffix}':  np.nan,
            f'locules_std_area_{unit_suffix}':   np.nan,
            f'locules_total_area_{unit_suffix}': 0.0,
            'locules_cv_area':               np.nan,
            'locules_mean_circularity':      np.nan,
            'locules_std_circularity':       np.nan,
            'locules_cv_circularity':        np.nan,
        }
        pericarp_metrics = {
            f'total_internal_fruit_area_{unit_suffix}': np.nan,
        }
        symmetry_metrics = {
            'locules_angular_symmetry': np.nan,
            'locules_radial_symmetry':  np.nan,
        }

    # 6. Calculate derived metrics (always runs)
    derived_metrics = _calculate_derived_metrics(
        fruit_metrics,
        pericarp_metrics,
        locule_metrics,
        unit
    )

    # 7. Annotate image
    _annotate_fruit(
        fruit_data['contour'],
        sequential_id,
        locule_metrics['count'],
        annotated_img,
        img_shape,
        config
    )

    return _format_results(
        img_name=img_name,
        label_text=label_text,
        sequential_id=sequential_id,
        fruit_metrics=fruit_metrics,
        locule_metrics=locule_metrics,
        pericarp_metrics=pericarp_metrics,
        symmetry_metrics=symmetry_metrics,
        derived_metrics=derived_metrics,
        unit=unit
    )

def _prepare_fruit_data(
    fruit_id: int,
    contours: List[np.ndarray],
    fruit_centroids: List[Optional[Tuple[float, float]]],
    annotated_img: np.ndarray,
    config: FruitConfig,
) -> Optional[Dict[str, Any]]:
    """
    Extract and prepare contour and centroid data for a single fruit.

    Draws the external pericarp contour and fruit centroid marker onto
    ``annotated_img`` in-place.

    Retrieves the fruit contour via :func:`get_fruit_contour` and draws
    it onto ``annotated_img`` using settings from ``config``. Also draws
    the fruit centroid marker. Returns ``None`` early if the centroid is
    unavailable, signalling that this fruit should be skipped.

    Parameters
    ----------
    fruit_id : int
        Index into ``contours`` identifying the fruit.
    contours : list of np.ndarray
        Full list of all detected contours.
    fruit_centroids : list of tuple of float or None
        Precomputed ``(cx, cy)`` centroids. Entry is ``None`` if the
        centroid could not be computed for that contour.
    annotated_img : np.ndarray
        BGR image modified in-place with the fruit contour and centroid.
    config : FruitConfig
        Configuration with color and thickness settings.

    Returns
    -------
    dict or None
        Dictionary with keys:

        - ``'contour'`` – processed fruit contour (np.ndarray).
        - ``'centroid'`` – ``(cx, cy)`` tuple of float.

        Returns ``None`` if ``fruit_centroids[fruit_id]`` is ``None``.
    """

    fruit_contour = get_fruit_contour(
        fruit_id=fruit_id,
        contours=contours,
        contour_mode=config.contour_mode,
        epsilon = config.epsilon 
    )
    
    cv2.drawContours(annotated_img, [fruit_contour], -1, config.pericarp_ext_color, 
                     config.pericarp_ext_thickness)
    
    fruit_centroid = fruit_centroids[fruit_id]
    if fruit_centroid is None:
        return None
    
    cx, cy = map(int, fruit_centroid)
    cv2.circle(annotated_img, (cx, cy), config.centroid_fruit_thickness, config.centroid_fruit_color, -1)
    
    return {
        'contour': fruit_contour,
        'centroid': fruit_centroid
    }

def _calculate_fruit_metrics(
    fruit_contour: np.ndarray,
    original_contour: np.ndarray,
    px_per_cm: Optional[float],
    unit: str,
    config: FruitConfig,
    annotated_img: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate morphological metrics for a fruit contour.

    Delegates to :func:`get_fruit_morphology` for area, perimeter and
    circularity, :func:`calculate_axes` for the ellipse-fitted major and
    minor axes, and :func:`rotate_box` for the minimum bounding box.
    Axis lines and the bounding box are drawn onto ``annotated_img``
    in-place.

    Parameters
    ----------
    fruit_contour : np.ndarray
        Processed contour (raw or approximated) forwarded to
        :func:`calculate_axes` and :func:`rotate_box`.
    original_contour : np.ndarray
        Unprocessed contour forwarded to :func:`get_fruit_morphology` to
        avoid distorting area and perimeter calculations by approximation.
    px_per_cm : float or None
        Pixel-to-centimeter conversion factor forwarded to
        :func:`get_fruit_morphology`, :func:`calculate_axes`, and
        :func:`rotate_box`. ``None`` means pixel units.
    unit : str
        Active measurement unit, either ``'cm'`` or ``'px'``. Controls
        which values are selected from the dual-unit outputs of the
        underlying functions.
    config : FruitConfig
        Configuration forwarded to :func:`get_fruit_morphology` for
        :attr:`~FruitConfig.contour_mode` and :attr:`~FruitConfig.epsilon`.
    annotated_img : np.ndarray
        BGR image modified in-place by :func:`calculate_axes` (axis lines)
        and :func:`rotate_box` (bounding box).

    Returns
    -------
    dict of {str : float}
        Morphological metrics keyed by the active unit suffix, including
        ``fruit_area_{unit}2``, ``fruit_major_axis_{unit}``,
        ``fruit_minor_axis_{unit}``, ``fruit_box_length_{unit}``,
        ``fruit_box_width_{unit}``, and ``fruit_aspect_ratio``.
    """
    
    # Get morphology (returns both _cm and _px keys)
    morphology = get_fruit_morphology(
        contour=original_contour,
        px_per_cm=px_per_cm,
        contour_mode=config.contour_mode,
        epsilon = config.epsilon
    )
    
    # Calculate axes (returns: major_cm, minor_cm, major_px, minor_px)
    major_cm, minor_cm, major_px, minor_px = calculate_axes(
        fruit_contour,
        px_per_cm=px_per_cm,
        img=annotated_img,
        draw_axes=True
    )
    
    # Calculate rotated box (returns: len_cm, wid_cm, len_px, wid_px)
    box_len_cm, box_wid_cm, box_len_px, box_wid_px = rotate_box(
        fruit_contour,
        px_per_cm=px_per_cm,
        img=annotated_img,
        draw_box=True
    )
    
    # Select values based on unit
    if unit == 'cm':
        major_val = major_cm
        minor_val = minor_cm
        box_len_val = box_len_cm
        box_wid_val = box_wid_cm
    else:
        major_val = major_px
        minor_val = minor_px
        box_len_val = box_len_px
        box_wid_val = box_wid_px
    
    # Calculate aspect ratio
    aspect_ratio = float(box_wid_val / box_len_val) if box_len_val > 0 else np.nan
    
    # Filter morphology to only include the active unit
    filtered_metrics = {
    k: v for k, v in morphology.items() 
    if (k.endswith(f'_{unit}') or k.endswith(f'_{unit}2')) or 
       not (k.endswith('_cm') or k.endswith('_cm2') or k.endswith('_px') or k.endswith('_px2'))
    }   
    
    return {
        **filtered_metrics,
        f'fruit_major_axis_{unit}': major_val,
        f'fruit_minor_axis_{unit}': minor_val,
        f'fruit_box_length_{unit}': box_len_val,
        f'fruit_box_width_{unit}': box_wid_val,
        'fruit_aspect_ratio': aspect_ratio
    }

def _process_locules(
    locules: List[int],
    contours: List[np.ndarray],
    fruit_centroid: Tuple[float, float],
    annotated_img: np.ndarray,
    px_per_cm: Optional[float],
    config: FruitConfig,
    unit: str,
) -> Dict[str, Any]:
    """
    Filter, optionally merge, and characterize locules for a single fruit.

    Computes per-locule data via :func:`precalculate_locules_data`, applies
    area filtering using :attr:`~FruitConfig.min_locule_area` and
    :attr:`~FruitConfig.max_locule_area`, and optionally merges nearby
    locules via :func:`merge_locules_func`. Draws locule contours and
    centroid markers onto ``annotated_img`` in-place. Area and circularity
    statistics are delegated to :func:`_calculate_locule_statistics`.

    Parameters
    ----------
    locules : list of int
        Indices into ``contours`` for candidate locule contours.
    contours : list of np.ndarray
        Full list of all detected contours.
    fruit_centroid : tuple of float
        ``(cx, cy)`` centroid of the parent fruit, forwarded to
        :func:`precalculate_locules_data` to compute radial distances.
    annotated_img : np.ndarray
        BGR image modified in-place with locule contours
        (:attr:`~FruitConfig.locule_color`,
        :attr:`~FruitConfig.locule_thickness`) and centroid markers
        (:attr:`~FruitConfig.centroid_locule_color`,
        :attr:`~FruitConfig.centroid_locule_thickness`).
    px_per_cm : float or None
        Pixel-to-centimeter conversion factor forwarded to
        :func:`_calculate_locule_statistics`. ``None`` means pixel units.
    config : FruitConfig
        Configuration with area thresholds, merge settings, and
        visualization parameters.
    unit : str
        Active measurement unit, either ``'cm'`` or ``'px'``, forwarded
        to :func:`_calculate_locule_statistics`.

    Returns
    -------
    dict of {str : Any}
        Dictionary with keys:

        - ``'data'`` – list of per-locule dicts from
          :func:`precalculate_locules_data`.
        - ``'filtered_ids'`` – list of contour indices that passed area
          filtering.
        - ``'count'`` – final locule count (after optional merging via
          :func:`merge_locules_func`).
        - Area and circularity statistics from
          :func:`_calculate_locule_statistics`, suffixed by the active unit.
    """

    # Precalculate locule data
    locules_data = precalculate_locules_data(contours, locules, fruit_centroid)
    
    # Filter by area
    if config.max_locule_area is None:
        filtered_data = [d for d in locules_data if d['area'] >= config.min_locule_area]
    else:
        filtered_data = [d for d in locules_data 
                        if config.min_locule_area <= d['area'] <= config.max_locule_area]
    
    filtered_ids = [d['contour_id'] for d in filtered_data]
    
    # Merge or draw contours
    if config.merge_locules:
        merged_contours = merge_locules_func(
            locules_indices=filtered_ids,
            contours=contours,
            max_distance=config.max_locule_distance,
            min_distance=config.min_locule_distance
        ) or []
        
        # Draw only merge contours
        for contour in merged_contours:
            if len(contour) > 0:
                cv2.drawContours(annotated_img, [contour], -1, config.locule_color, 
                                 config.locule_thickness)
        
        # Update the new locule number
        locule_count = len(merged_contours)
    else:
        # Draw the original contours
        for locule_id in filtered_ids:
            contour = contours[locule_id]
            if len(contour) > 0:
                cv2.drawContours(annotated_img, [contour], -1, config.locule_color, 
                                 config.locule_thickness)
        
        # Use the original locule number
        locule_count = len(filtered_data)
    
    # Draw centroids 
    for loc_data in filtered_data:
        cx, cy = loc_data['centroid']
        cv2.circle(annotated_img, (int(cx), int(cy)), 
                  config.centroid_locule_thickness, config.centroid_locule_color, -1)
    
    # Calculate statistics in correct unit
    stats = _calculate_locule_statistics(filtered_data, px_per_cm, unit)

    return {
        'data': filtered_data,
        'filtered_ids': filtered_ids,
        'count': locule_count, 
        **stats
    }

def _calculate_locule_statistics(
    locules_data: List[Dict[str, Any]],
    px_per_cm: Optional[float],
    unit: str,
) -> Dict[str, float]:
    """
    Calculate area and circularity statistics for a set of locules.

    Parameters
    ----------
    locules_data : list of dict
        Per-locule data dicts as returned by :func:`precalculate_locules_data`,
        each containing at minimum ``'area'`` (float, pixels²) and
        ``'circularity'`` (float) keys.
    px_per_cm : float or None
        Pixel-to-centimeter conversion factor used to convert pixel² areas
        to cm² when ``unit='cm'``. ``None`` means no conversion is applied.
    unit : str
        Active measurement unit, either ``'cm'`` or ``'px'``. Controls
        the suffix of returned area keys and whether unit conversion is
        applied.

    Returns
    -------
    dict of {str : float}
        Statistics dictionary with keys:

        - ``locules_mean_area_{unit_suffix}``
        - ``locules_std_area_{unit_suffix}``
        - ``locules_total_area_{unit_suffix}``
        - ``locules_cv_area``
        - ``locules_mean_circularity``
        - ``locules_std_circularity``
        - ``locules_cv_circularity``

        All values are ``NaN`` if ``locules_data`` is empty.
    """
    # Use cm2 for areas when unit is cm, px when unit is px
    unit_suffix = 'cm2' if unit == 'cm' else 'px2'

    if not locules_data:
        return {
            f'locules_mean_area_{unit_suffix}': np.nan,
            f'locules_std_area_{unit_suffix}': np.nan,
            f'locules_total_area_{unit_suffix}': 0.0,
            'locules_cv_area': np.nan,
            'locules_mean_circularity': np.nan,
            'locules_std_circularity': np.nan,
            'locules_cv_circularity': np.nan
        }
    
    
    # Reuse areas array for circularity calculation
    areas = np.array([d['area'] for d in locules_data]) # Get ALL the areas for a single fruit

    # Convert to cm2 if unit is cm
    if unit == 'cm' and px_per_cm is not None and px_per_cm > 0:
        inv = 1.0 / (px_per_cm * px_per_cm)
        areas = areas * inv

    circularities = np.array([d['circularity'] for d in locules_data]) 
    
    # Calculate area statistics
    mean_area = float(areas.mean())
    std_area = float(areas.std())
    cv_area = float(std_area / mean_area * 100) if mean_area > 0 else np.nan
    
    # Calculate circularity statistics
    mean_circ = float(circularities.mean())
    std_circ = float(circularities.std())
    cv_circ = float(std_circ / mean_circ * 100) if mean_circ > 0 else np.nan
    
    return {
        f'locules_mean_area_{unit_suffix}': mean_area,
        f'locules_std_area_{unit_suffix}': std_area,
        f'locules_total_area_{unit_suffix}': float(areas.sum()),
        'locules_cv_area': cv_area,
        'locules_mean_circularity': mean_circ,
        'locules_std_circularity': std_circ,
        'locules_cv_circularity': cv_circ
    }


def _calculate_pericarp_metrics(
    filtered_locule_ids: List[int],
    contours: List[np.ndarray],
    fruit_contour: np.ndarray,
    fruit_centroid: Tuple[float, float],
    annotated_img: np.ndarray,
    img_shape: Tuple[int, int],
    px_per_cm: Optional[float],
    config: FruitConfig,
    unit: str,
    pericarp_int_color: Tuple[int, int, int] = (0, 240, 240),
    pericarp_int_thickness: int = 2,
) -> Dict[str, float]:
    """
    Calculate pericarp area and thickness metrics for a single fruit.

    Computes the total internal fruit area via :func:`get_internal_pericarp_area`,
    retrieves the internal contour via :func:`get_internal_pericarp_contour`,
    and estimates radial pericarp thickness via
    :func:`calculate_pericarp_thickness_radial`. The internal pericarp
    contour is drawn onto ``annotated_img`` in-place by
    :func:`get_internal_pericarp_area`.

    Parameters
    ----------
    filtered_locule_ids : list of int
        Contour indices of locules that passed area filtering, forwarded
        to :func:`get_internal_pericarp_area` and
        :func:`get_internal_pericarp_contour`.
    contours : list of np.ndarray
        Full list of all detected contours, forwarded to
        :func:`get_internal_pericarp_area` and
        :func:`get_internal_pericarp_contour`.
    fruit_contour : np.ndarray
        Processed outer fruit contour forwarded to
        :func:`calculate_pericarp_thickness_radial` as the outer boundary.
    fruit_centroid : tuple of float
        ``(cx, cy)`` centroid of the fruit used as the ray origin by
        :func:`calculate_pericarp_thickness_radial`.
    annotated_img : np.ndarray
        BGR image modified in-place by :func:`get_internal_pericarp_area`
        with the internal pericarp contour overlay.
    img_shape : tuple of int
        ``(height, width)`` of the image, forwarded to
        :func:`calculate_pericarp_thickness_radial`.
    px_per_cm : float or None
        Pixel-to-centimeter conversion factor forwarded to
        :func:`get_internal_pericarp_area` and
        :func:`calculate_pericarp_thickness_radial`. ``None`` means pixel
        units.
    config : FruitConfig
        Configuration forwarded to :func:`calculate_pericarp_thickness_radial`
        for :attr:`~FruitConfig.num_rays`.
    unit : str
        Active measurement unit, either ``'cm'`` or ``'px'``. Controls
        which area value is selected and which thickness keys are retained.
    pericarp_int_color : tuple of int, optional
        BGR color for the internal pericarp contour overlay, forwarded to
        :func:`get_internal_pericarp_area`. Default is ``(0, 240, 240)``.
    pericarp_int_thickness : int, optional
        Line thickness for the internal pericarp contour overlay, forwarded
        to :func:`get_internal_pericarp_area`. Default is 2.

    Returns
    -------
    dict of {str : float}
        Dictionary with ``total_internal_fruit_area_{unit_suffix}`` and
        radial thickness statistics from
        :func:`calculate_pericarp_thickness_radial`, all keyed by the
        active unit suffix.
    """
    
    # Internal pericarp area (returns both cm2 and px)
    inner_area_cm2, inner_area_px2 = get_internal_pericarp_area(
        locules=filtered_locule_ids,
        contours=contours,
        px_per_cm=px_per_cm,
        img=annotated_img,
        draw_inner_pericarp=True, 
        contour_color = pericarp_int_color,
        contour_thickness = pericarp_int_thickness
    )
    
    # Get internal contour
    inner_contour = get_internal_pericarp_contour(
        locules=filtered_locule_ids,
        contours=contours
    )
    
    # Calculate thickness (returns dict with _cm or _px keys)
    thickness_stats = calculate_pericarp_thickness_radial(
        outer_contour=fruit_contour,
        inner_contour=inner_contour,
        fruit_centroid=fruit_centroid,
        img_shape=img_shape,
        num_rays=config.num_rays,
        px_per_cm=px_per_cm
    )
    
    # Select correct area values based on unit
    # For internal (inner) area
    inner_area = inner_area_cm2 if unit == 'cm' else inner_area_px2
    
    # Filter thickness stats to only include the active unit
    thickness_filtered = {
        k: v for k, v in thickness_stats.items()
        if unit in k or 'cv_' in k
    }
    
    # Use cm2 for area when unit is cm, px when unit is px
    unit_suffix = 'cm2' if unit == 'cm' else 'px2'

    return {
       f'total_internal_fruit_area_{unit_suffix}': inner_area,
        **thickness_filtered
    }

def _calculate_symmetry_metrics(
    locules_data: List[Dict[str, Any]],
    config: FruitConfig,
    precomputed_ideals: Optional[Dict[int, np.ndarray]] = None,
) -> Dict[str, float]:
    """
    Calculate angular and radial symmetry scores for a fruit's locules.

    Delegates angular symmetry to :func:`angular_symmetry` and radial
    symmetry to :func:`radial_symmetry`. Returns ``NaN`` for both scores
    if fewer than two locules are present.

    Parameters
    ----------
    locules_data : list of dict
        Per-locule data dicts as returned by :func:`precalculate_locules_data`.
        Must contain at minimum the angle and distance information consumed
        by :func:`angular_symmetry` and :func:`radial_symmetry`.
    config : FruitConfig
        Configuration forwarded to :func:`angular_symmetry` for
        :attr:`~FruitConfig.angle_shifts`.
    precomputed_ideals : dict of {int : np.ndarray} or None, optional
        Precomputed ideal angle arrays keyed by locule count, forwarded to
        :func:`angular_symmetry` to avoid redundant computation across
        fruits. Default is ``None``.

    Returns
    -------
    dict of {str : float}
        Dictionary with keys:

        - ``'locules_angular_symmetry'`` – score in [0, 1] from
          :func:`angular_symmetry`, or ``NaN`` if fewer than 2 locules.
        - ``'locules_radial_symmetry'`` – score in [0, 1] from
          :func:`radial_symmetry`, or ``NaN`` if fewer than 2 locules.
    """
    
    if not locules_data or len(locules_data) < 2:
        return {
            'locules_angular_symmetry': np.nan,
            'locules_radial_symmetry': np.nan
        }
    
    angular_sym = angular_symmetry(
        locules_data, 
        angle_shifts=config.angle_shifts,
        precomputed_ideals=precomputed_ideals 
    )
    
    radial_sym = radial_symmetry(locules_data)

    return {
        'locules_angular_symmetry': angular_sym,
        'locules_radial_symmetry': radial_sym
        
    }


def _calculate_derived_metrics(
    fruit_metrics: Dict[str, float],
    pericarp_metrics: Dict[str, float],
    locule_metrics: Dict[str, Any],
    unit: str,
) -> Dict[str, float]:
    """
    Calculate derived area ratios and compactness from primary metrics.

    All inputs are expected to share the same unit system (``'cm'`` or
    ``'px'``), controlled by ``unit``. Values are retrieved directly from
    the dictionaries returned by :func:`_calculate_fruit_metrics`,
    :func:`_calculate_pericarp_metrics`, and :func:`_process_locules`.

    Parameters
    ----------
    fruit_metrics : dict of {str : float}
        Morphological metrics as returned by :func:`_calculate_fruit_metrics`.
    pericarp_metrics : dict of {str : float}
        Pericarp metrics as returned by :func:`_calculate_pericarp_metrics`.
    locule_metrics : dict of {str : Any}
        Locule metrics as returned by :func:`_process_locules`.
    unit : str
        Active measurement unit, either ``'cm'`` or ``'px'``. Controls
        which keys are retrieved from input dicts and which suffixes appear
        in output keys.

    Returns
    -------
    dict of {str : float}
        Derived metrics including:

        - ``'fruit_compactness'`` – fruit area divided by bounding box area.
        - ``total_outer_pericarp_area_{unit_suffix}``
        - ``total_internal_pericarp_area_{unit_suffix}``
        - ``total_locules_area_{unit_suffix}``
        - ``'outer_pericarp_to_fruit_ratio'``
        - ``'internal_pericarp_to_fruit_ratio'``
        - ``'locules_to_fruit_ratio'``
        - ``'locules_to_total_internal_ratio'``
        - ``'internal_pericarp_to_total_internal_ratio'``

    Warnings
    --------
    Key lookup for ``box_length`` and ``box_width`` currently uses
    ``f'box_length_{unit}'`` instead of ``f'fruit_box_length_{unit}'``,
    which will always return 0 and cause ``fruit_compactness`` to be
    ``NaN``.
    """
    
    unit_suffix = 'cm2' if unit == 'cm' else 'px2'

    # Reuse fruit, pericarp and locule metrics
    fruit_area = fruit_metrics.get(f'fruit_area_{unit_suffix}', 0)
    inner_area = pericarp_metrics.get(f'total_internal_fruit_area_{unit_suffix}', 0)
    total_locules_area = locule_metrics.get(f'locules_total_area_{unit_suffix}', 0)
    box_len = fruit_metrics.get(f'fruit_box_length_{unit}', 0)
    box_wid = fruit_metrics.get(f'fruit_box_width_{unit}', 0)
    
    # Calculate derived metrics
    compactness = fruit_area / (box_len * box_wid) if (box_len > 0 and box_wid > 0) else np.nan

    outer_pericarp_area = fruit_area - inner_area
    internal_pericarp_area = inner_area - total_locules_area
    
    result = {
    'fruit_compactness': compactness,

    # Absolute metrics
    f'total_outer_pericarp_area_{unit_suffix}': outer_pericarp_area,
    f'total_internal_pericarp_area_{unit_suffix}': internal_pericarp_area,
    f'total_locules_area_{unit_suffix}': total_locules_area,
    
    # Ratios y percentages
    'outer_pericarp_to_fruit_ratio': outer_pericarp_area / fruit_area if fruit_area > 0 else np.nan,
    'internal_pericarp_to_fruit_ratio': internal_pericarp_area / fruit_area if fruit_area > 0 else np.nan,
    'locules_to_fruit_ratio': total_locules_area / fruit_area if fruit_area > 0 else np.nan,
    'locules_to_total_internal_ratio': total_locules_area / inner_area if inner_area > 0 else np.nan,
    'internal_pericarp_to_total_internal_ratio': internal_pericarp_area / inner_area if inner_area > 0 else np.nan
    
    }

    return result


def _annotate_fruit(
    fruit_contour: np.ndarray,
    sequential_id: int,
    n_locules: int,
    annotated_img: np.ndarray,
    img_shape: Tuple[int, int],
    config: FruitConfig,
) -> None:
    """
    Draw a text label annotation onto the fruit in the annotated image.

    Computes label placement from ``cv2.boundingRect`` applied to
    ``fruit_contour``, then draws a semi-transparent background rectangle
    and multi-line text using ``cv2.putText``. Position is controlled by
    :attr:`~FruitConfig.label_position` and clamped to stay within image
    boundaries.

    Parameters
    ----------
    fruit_contour : np.ndarray
        Processed fruit contour used by ``cv2.boundingRect`` to derive
        the anchor point for label placement.
    sequential_id : int
        One-based display ID shown in the annotation label.
    n_locules : int
        Number of locules shown in the label. If 0, only the fruit ID
        is shown.
    annotated_img : np.ndarray
        BGR image modified in-place with the background rectangle
        (:attr:`~FruitConfig.label_background_color`) and text
        (:attr:`~FruitConfig.text_color`, :attr:`~FruitConfig.font_scale`,
        :attr:`~FruitConfig.font_thickness`).
    img_shape : tuple of int
        ``(height, width)`` of the image, used to clamp the label
        position within bounds.
    config : FruitConfig
        Configuration with :attr:`~FruitConfig.label_position`,
        :attr:`~FruitConfig.padding`, :attr:`~FruitConfig.line_spacing`,
        and all font and color settings.

    Returns
    -------
    None
    """
    
    x, y, w, h = cv2.boundingRect(fruit_contour) # Not rotated

    if n_locules == 0:
            text = f"id {sequential_id}"
    else:
            text = f"id {sequential_id}: \n{n_locules} loc"
    
    # Calculate text dimensions
    font = cv2.FONT_HERSHEY_SIMPLEX
    (size_w, size_h), _ = cv2.getTextSize("Test", font, config.font_scale, config.font_thickness)
    
    single_line_height = size_h
    num_lines = text.count('\n') + 1
    total_height = (single_line_height * num_lines) + (15 * (num_lines - 1))

    
    # Calculate max text width
    text_width = max([
        cv2.getTextSize(line, font, config.font_scale, config.font_thickness)[0][0]
        for line in text.split('\n')
    ])
    
    # Calculate position based on label_position
    img_height, img_width = img_shape
    
    if config.label_position == 'top':
        text_x = max(10, x)
        text_y = max(total_height + 15, y - 15)
    elif config.label_position == 'bottom':
        text_x = max(10, x)
        text_y = min(img_height - 15, y + h + total_height + 15)
    elif config.label_position == 'left':
        text_x = max(10, x - text_width - config.padding * 2 - 15)
        text_y = max(total_height + 15, y + h // 2)
    elif config.label_position == 'right':
        text_x = min(img_width - text_width - config.padding * 2 - 10, x + w + 15)
        text_y = max(total_height + 15, y + h // 2)
    else:
        text_x = max(10, x)
        text_y = max(total_height + 15, y - 15)
    
    # Ensure text stays within bounds
    text_x = max(config.padding, min(text_x, img_width - text_width - config.padding * 2))
    text_y = max(total_height + config.padding, min(text_y, img_height - config.padding))
    
    # Draw background
    text_bg_layer = annotated_img.copy()
    cv2.rectangle(
        text_bg_layer,
        (text_x - config.padding, text_y - total_height - config.padding),
        (text_x + text_width + config.padding, text_y + config.padding),
        config.label_background_color, -1
    )
    cv2.addWeighted(
        text_bg_layer,              
        0.5,
        annotated_img,              
        0.5,
        0,
        annotated_img              
    )

    # Draw text
    for i, line in enumerate(text.split('\n')):
        y_offset = text_y - (total_height - single_line_height) + \
                   (i * (single_line_height + config.line_spacing))
        cv2.putText(
            annotated_img, line, (text_x, y_offset),
            font, config.font_scale, config.text_color, 
            config.font_thickness, cv2.LINE_AA
        )

    return None



def _format_results(
    img_name: str,
    label_text: str,
    sequential_id: int,
    fruit_metrics: Dict[str, float],
    locule_metrics: Dict[str, Any],
    pericarp_metrics: Dict[str, float],
    symmetry_metrics: Dict[str, float],
    derived_metrics: Dict[str, float],
    unit: str,
) -> Dict[str, Any]:
    """
    Assemble all per-fruit metrics into a flat result dictionary.

    Merges outputs from :func:`_calculate_fruit_metrics`,
    :func:`_process_locules`, :func:`_calculate_pericarp_metrics`,
    :func:`_calculate_symmetry_metrics`, and
    :func:`_calculate_derived_metrics` into a single flat dictionary
    suitable for appending to a results list or converting to a
    ``pd.DataFrame`` row. Internal keys ``'filtered_ids'`` and ``'data'``
    from ``locule_metrics`` are excluded from the output.

    Parameters
    ----------
    img_name : str
        Image filename stored under the ``'image_name'`` key.
    label_text : str
        Treatment or label identifier stored under the ``'label'`` key.
    sequential_id : int
        One-based display ID stored under the ``'fruit_id'`` key.
    fruit_metrics : dict of {str : float}
        Morphological metrics as returned by :func:`_calculate_fruit_metrics`.
    locule_metrics : dict of {str : Any}
        Locule metrics as returned by :func:`_process_locules`. The
        ``'filtered_ids'`` and ``'data'`` keys are excluded from the output.
    pericarp_metrics : dict of {str : float}
        Pericarp metrics as returned by :func:`_calculate_pericarp_metrics`.
    symmetry_metrics : dict of {str : float}
        Symmetry scores as returned by :func:`_calculate_symmetry_metrics`.
    derived_metrics : dict of {str : float}
        Derived ratios and areas as returned by
        :func:`_calculate_derived_metrics`.
    unit : str
        Active measurement unit (``'cm'`` or ``'px'``), stored under the
        ``'unit'`` key.

    Returns
    -------
    dict of {str : Any}
        Flat dictionary combining identification fields and all metric
        groups, ready for downstream aggregation.
    """
    
    # Morphology results
    morphology_results = {
        # Identification
        'image_name': img_name,
        'label': label_text,
        'fruit_id': sequential_id,
        'n_locules': locule_metrics['count'],
        'unit': unit,
        
        # Metrics filtered to single unit (cm or px)
        **fruit_metrics,
        **{k: v for k, v in locule_metrics.items() if k not in ['filtered_ids', 'data', 'count']},
        **pericarp_metrics,
        **symmetry_metrics,
        **derived_metrics
    }

    return morphology_results
