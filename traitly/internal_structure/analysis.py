# traitly/internal_structure/analysis.py

# ============================================================================
# STANDARD LIBRARY
# ============================================================================
from typing import List, Dict, Tuple, Optional, Any

# ============================================================================
# THIRD-PARTY LIBRARIES
# ===========================================================================
import cv2
import numpy as np
import pandas as pd
from dataclasses import dataclass

# ============================================================================
# LOCAL/INTERNAL IMPORTS
# ===========================================================================
from .geometry import calculate_axes, rotate_box, get_fruit_morphology
from .symmetry import angular_symmetry, radial_symmetry, rotational_symmetry
from .processing import (
    get_inner_pericarp_area,
    calculate_fruit_centroids,
    precalculate_locules_data,
    get_fruit_contour,
    get_inner_pericarp_contour,
    calculate_pericarp_thickness_radial
)
from .mask import merge_locules_func
from .annotated_image import AnnotatedImage


@dataclass
class FruitConfig:
    """Configuration for fruit analysis."""
    # Contour settings
    contour_mode: str = 'raw'
    epsilon_factor: float = 0.001
    use_ellipse: bool = False
    
    # Locule settings
    min_locule_area: int = 300
    max_locule_area: Optional[int] = None
    merge_locules: bool = False
    min_distance: int = 1
    max_distance: int = 10
    
    # Symmetry settings
    num_shifts: int = 500
    angle_weight: float = 0.5
    radius_weight: float = 0.5
    min_radius_threshold: float = 0.1
    
    # Pericarp settings
    num_rays: int = 360
    
    # Visualization settings
    stamp: bool = False
    plot: bool = True
    plot_size: Tuple[int, int] = (20, 10)
    font_scale: int = 1
    font_thickness: int = 2
    text_color: Tuple[int, int, int] = (0, 0, 0)
    bg_color: Tuple[int, int, int] = (255, 255, 255)
    padding: int = 15
    line_spacing: int = 15
    centroid_fruit: int = 3
    centroid_locules: int = 3
    label_position: str = 'top'


def analyze_fruits(
    img: np.ndarray,
    contours: List[np.ndarray],
    fruit_locus_map: Dict[int, List[int]],
    px_per_cm: Optional[float],
    img_name: str,
    label_text: str,
    label_id: Optional[int] = None,
    path: Optional[str] = None,
    **kwargs
) -> AnnotatedImage:
    """
    Analyze fruit contours and extract morphological features.
    
    Args:
        img: Input image (BGR)
        contours: All detected contours
        fruit_locus_map: Mapping of fruit IDs to locule indices
        px_per_cm: Pixel-to-cm conversion factor (None for pixel units)
        img_name: Image filename
        label_text: Label or treatment identifier
        label_id: Contour ID to exclude (label region)
        path: Original image path
        **kwargs: Additional configuration (see FruitConfig)
    
    Returns:
        AnnotatedImage with results and annotated image
    """
    config = FruitConfig(**{k: v for k, v in kwargs.items() 
                           if k in FruitConfig.__dataclass_fields__})
    
    annotated_img = cv2.bitwise_not(img.copy()) if config.stamp else img.copy()
    fruit_centroids = calculate_fruit_centroids(contours)
    
    results = []
    sequential_id = 1
    
    for fruit_id, locules in fruit_locus_map.items():
        if fruit_id == label_id:
            continue
        
        try:
            result = _analyze_single_fruit(
                fruit_id=fruit_id,
                locules=locules,
                contours=contours,
                fruit_centroids=fruit_centroids,
                annotated_img=annotated_img,
                px_per_cm=px_per_cm,
                img_name=img_name,
                label_text=label_text,
                sequential_id=sequential_id,
                img_shape=img.shape[:2],
                config=config
            )
            
            if result is not None:
                results.append(result)
                sequential_id += 1
                
        except Exception as e:
            print(f"Error processing fruit {fruit_id}: {e}")
            continue
    
    if config.plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=config.plot_size)
        plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    
    return AnnotatedImage(annotated_img, results, image_path=path)


def _analyze_single_fruit(
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
    config: FruitConfig
) -> Optional[Dict[str, Any]]:
    """Analyze a single fruit and return its metrics."""
    
    # Determine unit once
    has_calibration = px_per_cm is not None and px_per_cm > 0
    unit = 'cm' if has_calibration else 'px'
    
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
        fruit_data['centroid'],
        contours[fruit_id],
        annotated_img,
        px_per_cm,
        config,
        unit
    )
    
    # 3. Process locules
    locule_metrics = _process_locules(
        locules, contours, fruit_data['centroid'],
        annotated_img, px_per_cm, config, unit
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
        unit
    )
    
    # 5. Calculate symmetry
    symmetry_metrics = _calculate_symmetry_metrics(
        locule_metrics['data'], config
    )
    
    # 6. Calculate derived metrics
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
    
    # 8. Format final results
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
    config: FruitConfig
) -> Optional[Dict[str, Any]]:
    """Extract and prepare fruit contour and centroid data."""
    
    fruit_contour = get_fruit_contour(
        fruit_id=fruit_id,
        contours=contours,
        contour_mode=config.contour_mode,
        epsilon_factor=config.epsilon_factor
    )
    
    cv2.drawContours(annotated_img, [fruit_contour], -1, (0, 255, 0), 2)
    
    fruit_centroid = fruit_centroids[fruit_id]
    if fruit_centroid is None:
        return None
    
    cx, cy = map(int, fruit_centroid)
    cv2.circle(annotated_img, (cx, cy), config.centroid_fruit, (255, 255, 51), -1)
    
    return {
        'contour': fruit_contour,
        'centroid': fruit_centroid
    }


def _calculate_fruit_metrics(
    fruit_contour: np.ndarray,
    fruit_centroid: Tuple[int, int],
    original_contour: np.ndarray,
    annotated_img: np.ndarray,
    px_per_cm: Optional[float],
    config: FruitConfig,
    unit: str
) -> Dict[str, float]:
    """Calculate all fruit morphological metrics in single unit."""
    
    # Get morphology (returns both _cm and _px keys)
    morphology = get_fruit_morphology(
        contour=original_contour,
        px_per_cm=px_per_cm,
        contour_mode=config.contour_mode,
        epsilon_factor=config.epsilon_factor
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
        if k.endswith(f'_{unit}') or not (k.endswith('_cm') or k.endswith('_px'))
    }
    
    return {
        **filtered_metrics,
        f'major_axis_{unit}': major_val,
        f'minor_axis_{unit}': minor_val,
        f'box_length_{unit}': box_len_val,
        f'box_width_{unit}': box_wid_val,
        'aspect_ratio': aspect_ratio
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
    
    # Filter by area (using pixel values initially)
    if config.max_locule_area is None:
        filtered_data = [d for d in locules_data if d['area'] >= config.min_locule_area]
    else:
        filtered_data = [d for d in locules_data 
                        if config.min_locule_area <= d['area'] <= config.max_locule_area]
    
    filtered_ids = [d['contour_id'] for d in filtered_data]
    
    # Merge or draw locules
    if config.merge_locules:
        merged_contours = merge_locules_func(
            locules_indices=filtered_ids,
            contours=contours,
            max_distance=config.max_distance,
            min_distance=config.min_distance
        ) or []
        
        for contour in merged_contours:
            if len(contour) > 0:
                cv2.drawContours(annotated_img, [contour], -1, (255, 0, 255), 2)
    else:
        for locule_id in filtered_ids:
            contour = contours[locule_id]
            if len(contour) > 0:
                cv2.drawContours(annotated_img, [contour], -1, (255, 0, 255), 2)
    
    # Draw centroids
    for loc_data in filtered_data:
        cx, cy = loc_data['centroid']
        cv2.circle(annotated_img, (int(cx), int(cy)), 
                  config.centroid_locules, (0, 255, 255), -1)
    
    # Calculate statistics in correct unit
    stats = _calculate_locule_statistics(filtered_data, px_per_cm, unit)
    
    return {
        'data': filtered_data,
        'filtered_ids': filtered_ids,
        'count': len(filtered_data),
        **stats
    }


def _calculate_locule_statistics(
    locules_data: List[Dict],
    px_per_cm: Optional[float],
    unit: str
) -> Dict[str, float]:
    """
    Calculate area and circularity statistics for locules in single unit.
    OPTIMIZED: Reuses computed arrays instead of recalculating.
    """
    # Use cm2 for areas when unit is cm, px when unit is px
    unit_suffix = 'cm2' if unit == 'cm' else 'px'
    
    if not locules_data:
        return {
            f'mean_area_{unit_suffix}': np.nan,
            f'std_area_{unit_suffix}': np.nan,
            f'total_area_{unit_suffix}': 0.0,
            'cv_area': np.nan,
            'mean_circularity': np.nan,
            'std_circularity': np.nan,
            'cv_circularity': np.nan
        }
    
    # OPTIMIZATION: Extract arrays once instead of multiple list comprehensions
    areas = np.array([d['area'] for d in locules_data])
    perimeters = np.array([d['perimeter'] for d in locules_data])
    
    # Convert to cm² if unit is cm
    if unit == 'cm' and px_per_cm is not None and px_per_cm > 0:
        areas = areas / (px_per_cm ** 2)
    
    # OPTIMIZED: Reuse 'areas' array for circularity calculation
    # Note: Use original pixel areas for circularity (dimensionless metric)
    areas_px = np.array([d['area'] for d in locules_data])
    circularities = (4 * np.pi * areas_px) / (perimeters**2 + 1e-6)
    
    # Calculate area statistics
    mean_area = float(areas.mean())
    std_area = float(areas.std())
    cv_area = float(std_area / mean_area * 100) if mean_area > 0 else np.nan
    
    # Calculate circularity statistics
    mean_circ = float(circularities.mean())
    std_circ = float(circularities.std())
    cv_circ = float(std_circ / mean_circ * 100) if mean_circ > 0 else np.nan
    
    return {
        f'mean_area_{unit_suffix}': mean_area,
        f'std_area_{unit_suffix}': std_area,
        f'total_area_{unit_suffix}': float(areas.sum()),
        'cv_area': cv_area,
        'mean_circularity': mean_circ,
        'std_circularity': std_circ,
        'cv_circularity': cv_circ
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
    unit: str
) -> Dict[str, float]:
    """Calculate pericarp area and thickness metrics in single unit."""
    
    # Inner pericarp area (returns both cm2 and px)
    inner_area_cm2, inner_area_px = get_inner_pericarp_area(
        locules=filtered_locule_ids,
        contours=contours,
        px_per_cm=px_per_cm,
        img=annotated_img,
        draw_inner_pericarp=True,
        use_ellipse=config.use_ellipse,
        epsilon=config.epsilon_factor
    )
    
    # Get inner contour
    inner_contour = get_inner_pericarp_contour(
        locules=filtered_locule_ids,
        contours=contours,
        use_ellipse=config.use_ellipse,
        epsilon=config.epsilon_factor
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
    
    # Select correct values based on unit
    inner_area = inner_area_cm2 if unit == 'cm' else inner_area_px
    
    # Filter thickness stats to only include the active unit
    thickness_filtered = {
        k: v for k, v in thickness_stats.items()
        if unit in k or 'cv_' in k
    }
    
    # Use cm2 for area when unit is cm, px when unit is px
    unit_suffix = 'cm2' if unit == 'cm' else 'px'
    
    return {
        f'inner_pericarp_area_{unit_suffix}': inner_area,
        **thickness_filtered
    }


def _calculate_symmetry_metrics(
    locules_data: List[Dict],
    config: FruitConfig
) -> Dict[str, float]:
    """Calculate angular, radial, and rotational symmetry (unitless)."""
    
    if not locules_data or len(locules_data) < 2:
        return {
            'angular_symmetry': np.nan,
            'radial_symmetry': np.nan,
            'rotational_symmetry': np.nan
        }
    
    return {
        'angular_symmetry': angular_symmetry(locules_data, num_shifts=config.num_shifts),
        'radial_symmetry': radial_symmetry(locules_data),
        'rotational_symmetry': rotational_symmetry(
            locules_data,
            angle_error=None,
            angle_weight=config.angle_weight,
            radius_weight=config.radius_weight,
            min_radius_threshold=config.min_radius_threshold
        )
    }


def _calculate_derived_metrics(
    fruit_metrics: Dict[str, float],
    pericarp_metrics: Dict[str, float],
    locule_metrics: Dict[str, Any],
    unit: str
) -> Dict[str, float]:
    """Calculate derived metrics (ratios, percentages) in single unit."""
    
    # Get values using correct unit suffix (cm2 for areas in cm, px for pixels)
    unit_suffix = 'cm2' if unit == 'cm' else 'px'
    
    fruit_area = fruit_metrics.get(f'fruit_area_{unit_suffix}', 0)
    inner_area = pericarp_metrics.get(f'inner_pericarp_area_{unit_suffix}', 0)
    total_locule_area = locule_metrics.get(f'total_area_{unit_suffix}', 0)
    box_len = fruit_metrics.get(f'box_length_{unit}', 0)
    box_wid = fruit_metrics.get(f'box_width_{unit}', 0)
    
    # Calculate derived metrics
    outer_pericarp = fruit_area - inner_area if fruit_area > inner_area else 0
    compactness = fruit_area / (box_len * box_wid) if (box_len > 0 and box_wid > 0) else np.nan
    inner_ratio = inner_area / fruit_area if fruit_area > 0 else 0
    locule_pct = (total_locule_area / fruit_area) * 100 if fruit_area > 0 else 0
    packing_eff = (total_locule_area / inner_area) * 100 if inner_area > 0 else 0
    
    return {
        'compactness_index': compactness,
        f'outer_pericarp_area_{unit_suffix}': outer_pericarp,
        'inner_area_ratio': inner_ratio,
        'locule_area_percentage': locule_pct,
        'locule_packing_efficiency': packing_eff
    }


def _annotate_fruit(
    fruit_contour: np.ndarray,
    sequential_id: int,
    n_locules: int,
    annotated_img: np.ndarray,
    img_shape: Tuple[int, int],
    config: FruitConfig
) -> None:
    """Draw text annotation on the fruit based on specified position."""
    
    x, y, w, h = cv2.boundingRect(fruit_contour)
    text = f"id {sequential_id}: \n{n_locules} loc"
    
    # Calculate text dimensions
    font = cv2.FONT_HERSHEY_SIMPLEX
    (size_w, size_h), _ = cv2.getTextSize("Test", font, config.font_scale, config.font_thickness)
    
    single_line_height = size_h
    total_height = (single_line_height * 2) + config.line_spacing
    
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
        config.bg_color, -1
    )
    cv2.addWeighted(text_bg_layer, 0.7, annotated_img, 0.3, 0, annotated_img)
    
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
    unit: str
) -> Dict[str, Any]:
    """Format all metrics into final result dictionary."""
    
    return {
        # Identification
        'image_name': img_name,
        'label': label_text,
        'fruit_id': sequential_id,
        'n_locules': locule_metrics['count'],
        'unit': unit,
        
        # All metrics (already filtered to single unit)
        **fruit_metrics,
        **{k: v for k, v in locule_metrics.items() if k not in ['filtered_ids', 'data', 'count']},
        **pericarp_metrics,
        **symmetry_metrics,
        **derived_metrics
    }