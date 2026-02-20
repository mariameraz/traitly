# traitly/fruit_phenotyping/analysis.py

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
    """Configuration for fruit analysis."""
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
    angle_weight: float = 0.5
    radius_weight: float = 0.5
    
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

    # Color
    extract_color: bool = False
    color_stat: str = 'mean'

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
    pericarp_int_color: Tuple[int, int, int] = (0, 240, 240),
    pericarp_int_thickness: int = 2, 
    epsilon: float = 0.002,
    is_locule: bool = True,
    **kwargs
) -> ResultsImage:
    """
    Analyze fruit contours and extract morphological features.
    
    Args:
        img: Input image (BGR)
        contours: All detected contours
        fruit_locule_map: Mapping of fruit IDs to locule indices
        px_per_cm: Pixel-to-cm conversion factor (None for pixel units)
        img_name: Image filename
        label_text: Label or treatment identifier
        label_id: Contour ID to exclude (label region)
        path: Original image path
        **kwargs: Additional configuration (see FruitConfig)
    
    Returns:
        ResultsImage with results and annotated image
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

def _analyze_single_fruit_morphology(
    fruit_id: int,
    locules: List[int],
    contours: List[np.ndarray],
    fruit_centroids: List[Tuple[int, int]],
    annotated_img: np.ndarray,
    px_per_cm: Optional[float],
    img_name: str,
    label_text: str,
    sequential_id: int,
    img_shape: Tuple[int, int],
    config: FruitConfig,
    precomputed_ideals: Optional[Dict] = None,
    unit: str = 'px',
    pericarp_int_color: Tuple[int, int, int] = (0, 240, 240),
    pericarp_int_thickness: int = 2,
    is_locule: bool = True
) -> Optional[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]:
    """Analyze a single fruit and return its metrics."""

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
        annotated_img,
        px_per_cm,
        config,
        unit,
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
    fruit_centroids: List[Tuple[int, int]],
    annotated_img: np.ndarray,
    config: FruitConfig,
) -> Optional[Dict[str, Any]]:
    """Extract and prepare fruit contour and centroid data."""
    
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
    annotated_img: np.ndarray,
    px_per_cm: Optional[float],
    config: FruitConfig,
    unit: str, 

) -> Dict[str, float]:
    """Calculate all fruit morphological metrics in single unit."""
    
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
    fruit_centroid: Tuple[int, int],
    annotated_img: np.ndarray,
    px_per_cm: Optional[float],
    config: FruitConfig,
    unit: str
) -> Dict[str, Any]:
    """Process and filter locules, returning metrics in single unit."""
    
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
    locules_data: List[Dict],
    px_per_cm: Optional[float],
    unit: str
) -> Dict[str, float]:
    """
    Calculate area and circularity statistics for locules in single unit.
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
    fruit_centroid: Tuple[int, int],
    annotated_img: np.ndarray,
    img_shape: Tuple[int, int],
    px_per_cm: Optional[float],
    config: FruitConfig,
    unit: str,
    pericarp_int_color: Tuple[int, int, int] = (0, 240, 240),
    pericarp_int_thickness: int = 2,
) -> Dict[str, float]:
    """Calculate pericarp area and thickness metrics in single unit."""
    
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
    locules_data: List[Dict],
    config: FruitConfig,
    precomputed_ideals: Optional[Dict] = None  
) -> Dict[str, float]:
    
    """Calculate angular, radial, and rotational symmetry (unitless)."""
    
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
    """Calculate derived metrics (ratios, percentages) in single unit."""
    
    unit_suffix = 'cm2' if unit == 'cm' else 'px2'

    # Reuse fruit, pericarp and locule metrics
    fruit_area = fruit_metrics.get(f'fruit_area_{unit_suffix}', 0)
    inner_area = pericarp_metrics.get(f'total_internal_fruit_area_{unit_suffix}', 0)
    total_locules_area = locule_metrics.get(f'locules_total_area_{unit_suffix}', 0)
    box_len = fruit_metrics.get(f'box_length_{unit}', 0)
    box_wid = fruit_metrics.get(f'box_width_{unit}', 0)
    
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
    """Draw text annotation on the fruit based on specified position."""
    
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


def _format_results(
    img_name: str,
    label_text: str,
    sequential_id: int,
    fruit_metrics: Dict[str, float],
    locule_metrics: Dict[str, Any],
    pericarp_metrics: Dict[str, float],
    symmetry_metrics: Dict[str, float],
    derived_metrics: Dict[str, float],
    unit: str = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Format all metrics into final result dictionary."""
    
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
