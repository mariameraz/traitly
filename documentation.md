# Trait Table – Traitly

## Table of Contents

1. [**Introduction**](#introduction)
2. [**Trait Tables**](#trait-tables)
   - 2.1 [Identification](#1-identification)
   - 2.2 [Fruit Morphology](#2-fruit-morphology)
   - 2.3 [Locule Statistics](#3-locule-statistics)
   - 2.4 [Pericarp Metrics](#4-pericarp-metrics)
   - 2.5 [Symmetry Metrics](#5-symmetry-metrics)
   - 2.6 [Derived Metrics](#6-derived-metrics)
   - 2.7 [Color Metrics](#7-color-metrics-optional---when-extract_colortrue)
     * RGB Channels
     * HSV Channels
     * Lab Channels
     * Grayscale & Derived Indices
     * Variability Metrics
3. [**Notes**](#notes)
   - 3.1 [Unit conversion](#units)
   - 3.2 [Symmetry Interpretation](#symmetry-interpretation)
   - 3.3 [Typical Ranges](#typical-ranges)
   - 3.4 [Key Formulas](#key-formulas)

---

## Introduction

**Traitly** is a comprehensive fruit phenotyping tool that extracts morphological, structural, and color features from fruit images. This document catalogs all the traits (features) that Traitly can measure and export.

### What does Traitly measure?

Traitly analyzes fruits by detecting their contours and internal structures (locules/seeds) to compute:

- **Basic morphology**: Size, shape, and geometric properties
- **Internal structure**: Number, size, and distribution of locules
- **Pericarp characteristics**: Thickness and uniformity of fruit walls
- **Symmetry**: Angular and radial distribution patterns
- **Color properties**: Multi-channel color analysis (optional)

For more details, see: [**Trait Tables**](#trait-tables)

### Output formats


Traitly generates the following outputs for each processed image:

**CSV Files:**
- **`*_results.csv`**: Morphological and structural traits (area, perimeter, symmetry, etc.)
- **`*_color_results.csv`**: Color traits (RGB, HSV, Lab channels) — only when `extract_color=True`

**Session Reports:**
- **`session_report.txt`**: _For batch analysis only_. Processing summary with date, time, total images processed, fruits detected, processing time, RAM usage, and all parameters used
- **`error_report.csv`**: Status log for each image (successfully processed, errors, or skipped files)
- 
**Annotated Images:**
- **`*_annotated.jpg`**: Visual output with detected contours and unique fruit IDs
  - Green contours: Fruit boundaries
  - Magenta contours: Locules
  - Yellow contours: Internal cavity (locules + internal flesh)
  - Yellow dots: Centroids
  - Labels: Fruit ID and locule count (e.g., "id 1: 4 loc")

Each fruit receives a unique sequential ID (`fruit_id`) that links the visual annotation to its corresponding row in the CSV file, enabling easy cross-referencing between images and measurements.

---


## Trait Tables

About the table:

- Each trait includes a **description**, **formula**, and **expected range**
- Traits ending in `_cm` / `_cm2` require physical calibration (reference circles)
- Traits ending in `_px` are in pixel units (no calibration needed)
- Unitless metrics (ratios, percentages) work regardless of calibration

### 1. IDENTIFICATION

| Trait | Description | Formula | Type/Range |
|-------------|------------------------------------------------|---------|-----------|
| `image_name` | Name of the processed image file | N/A | `str` |
| `label` | Detected label or treatment (QR/OCR) | N/A | `str` |
| `fruit_id` | Unique sequential ID of the fruit in the image | N/A | `int` ≥ 1 |
| `n_locules` | Total number of locules detected in the fruit | N/A | `int` ≥ 0 |
| `unit` | Measurement unit used (`'cm'` or `'px'`) | N/A | `str` |

---

### 2. FRUIT MORPHOLOGY

| Trait | Description | Formula | Type/Range |
|-------------------------------------------|--------------------------------------------------------------|---------------------------------------------------|-------------|
| `fruit_area_cm2` / `fruit_area_px` | Total fruit area | `cv2.contourArea(contour)` | `float` > 0 |
| `fruit_perimeter_cm` / `fruit_perimeter_px` | Fruit contour perimeter | `cv2.arcLength(contour, True)` | `float` > 0 |
| `fruit_circularity` | Measure of how circular the fruit is (1 = perfect circle) | `(4π × area) / perimeter²` | `float` 0–1 |
| `fruit_solidity` | Proportion of fruit area relative to its convex hull | `area / convex_area` | `float` 0–1 |
| `fruit_compactness` | Measure of contour compactness | `perimeter² / area` | `float` ≥ 4π |
| `fruit_convex_hull_area_cm2` / `fruit_convex_hull_area_px` | Area of the fruit convex hull | `cv2.contourArea(convexHull)` | `float` > 0 |
| `major_axis_cm` / `major_axis_px` | Length of the major axis (maximum distance between points) | `max(euclidean_distances)` | `float` > 0 |
| `minor_axis_cm` / `minor_axis_px` | Length of the minor axis (width perpendicular to major axis) | `max(perpendicular_projections) - min(projections)` | `float` > 0 |
| `box_length_cm` / `box_length_px` | Length of the minimum rotated bounding box (longer side) | `max(width, height)` from `minAreaRect` | `float` > 0 |
| `box_width_cm` / `box_width_px` | Width of the minimum rotated bounding box (shorter side) | `min(width, height)` from `minAreaRect` | `float` > 0 |
| `aspect_ratio` | Ratio between box width and length | `box_width / box_length` | `float` 0–1 |

---

### 3. LOCULE STATISTICS

| Trait | Description | Formula | Type/Range |
|-----------------------------------|-------------------------------------------------------|----------------------------------|---------------|
| `mean_area_cm2` / `mean_area_px` | Mean locule area | `mean(locule_areas)` | `float` > 0 |
| `std_area_cm2` / `std_area_px` | Standard deviation of locule area | `std(locule_areas)` | `float` ≥ 0 |
| `total_area_cm2` / `total_area_px` | Total combined locule area | `sum(locule_areas)` | `float` > 0 |
| `cv_area` | Coefficient of variation of locule area (homogeneity) | `(std / mean) × 100` | `float` ≥ 0 (%) |
| `mean_circularity` | Mean circularity of locules | `mean((4π × area) / perimeter²)` | `float` 0–1 |
| `std_circularity` | Standard deviation of locule circularity | `std(circularities)` | `float` ≥ 0 |
| `cv_circularity` | Coefficient of variation of circularity | `(std / mean) × 100` | `float` ≥ 0 (%) |

---

### 4. PERICARP METRICS

| Trait | Description | Formula | Type/Range |
|------------------------------------------------------|--------------------------------------------------------|------------------------------------------------------|---------------|
| `inner_pericarp_area_cm2` / `inner_pericarp_area_px` | Inner pericarp area (convex hull enclosing all locules) | Area of the hull enclosing all locules | `float` > 0 |
| `mean_thickness_cm` / `mean_thickness_px` | Mean pericarp thickness | `mean(radial_distances)` from outer to inner contour | `float` > 0 |
| `median_thickness_cm` / `median_thickness_px` | Median pericarp thickness | `median(radial_distances)` | `float` > 0 |
| `std_thickness_cm` / `std_thickness_px` | Standard deviation of pericarp thickness | `std(radial_distances)` | `float` ≥ 0 |
| `min_thickness_cm` / `min_thickness_px` | Minimum pericarp thickness | `min(radial_distances)` | `float` > 0 |
| `max_thickness_cm` / `max_thickness_px` | Maximum pericarp thickness | `max(radial_distances)` | `float` > 0 |
| `cv_thickness` | Coefficient of variation of thickness (uniformity) | `(std / mean) × 100` | `float` ≥ 0 (%) |
| `lobedness_cm` / `lobedness_px` | Fruit surface irregularity (std of outer radii) | `std(outer_radial_distances)` | `float` ≥ 0 |

---

### 5. SYMMETRY METRICS

| Trait | Description | Formula | Type/Range |
|----------------------|----------------------------------------------------------------|-------------------------------------------------------------------|-------------|
| `angular_symmetry` | Angular symmetry of locule distribution (lower = more symmetric) | Mean angular error after optimal alignment with ideal distribution | `float` ≥ 0 |
| `radial_symmetry` | Radial symmetry of locule distances from center (0 = perfect) | `std(radii) / mean(radii)` (CV of radial distances) | `float` ≥ 0 |
| `rotational_symmetry` | Combined rotational symmetry (angular + radial, 0 = perfect) | Weighted, normalized combination of angular and radial errors | `float` 0–1 |

---

### 6. DERIVED METRICS

| Trait | Description | Formula | Type/Range |
|------------------------------------------------------|----------------------------------------------------------------|--------------------------------------------------|-----------------|
| `compactness_index` | Fruit compactness index relative to bounding box | `fruit_area / (box_length × box_width)` | `float` 0–1 |
| `outer_pericarp_area_cm2` / `outer_pericarp_area_px` | Outer pericarp area (between outer contour and inner region) | `fruit_area - inner_pericarp_area` | `float` ≥ 0 |
| `internal_flesh_area_cm2` / `internal_flesh_area_px` | Internal flesh area (inner pericarp minus locules) | `inner_pericarp_area - total_locule_area` | `float` ≥ 0 |
| `internal_cavity_area_cm2` / `internal_cavity_area_px` | Total internal cavity area (same as inner pericarp) | `inner_pericarp_area` | `float` ≥ 0 |
| `locule_cavity_area_cm2` / `locule_cavity_area_px` | Total locule cavity area | `total_locule_area` | `float` ≥ 0 |
| `outer_pericarp_ratio` | Proportion of outer pericarp relative to total fruit | `(fruit_area - inner_area) / fruit_area` | `float` 0–1 |
| `internal_cavity_ratio` | Proportion of internal cavity relative to total fruit | `inner_area / fruit_area` | `float` 0–1 |
| `locule_to_fruit_ratio` | Proportion of locules relative to total fruit | `total_locule_area / fruit_area` | `float` 0–1 |
| `locule_to_cavity_ratio` | Proportion of locules relative to internal cavity | `total_locule_area / inner_area` | `float` 0–1 |
| `flesh_to_cavity_ratio` | Proportion of flesh relative to internal cavity | `(inner_area - total_locule_area) / inner_area` | `float` 0–1 |
| `locule_to_fruit_percentage` | Percentage of total area occupied by locules | `(total_locule_area / fruit_area) × 100` | `float` 0–100 (%) |
| `locule_to_cavity_percentage` | Packing efficiency of locules within inner region | `(total_locule_area / inner_area) × 100` | `float` 0–100 (%) |

---

### 7. COLOR METRICS (Optional - when `extract_color=True`)

**NOTE:** Color metrics are extracted for three regions: **whole_fruit**, **outer_pericarp**, and **inner_pericarp**. Each region has its own set of color features with the region name as prefix (e.g., `wholefruit_R_mean`, `outerpericarp_L_mean`, `innerpericarp_H_mean`).

| Trait Pattern | Description | Formula/Method | Type/Range |
|--------------------------------|----------------------------------------------------------|-----------------------------------|----------------|
| `{region}_R_{stat}` | Red channel statistic | `mean` or `median` of R values | `float` 0–255 |
| `{region}_G_{stat}` | Green channel statistic | `mean` or `median` of G values | `float` 0–255 |
| `{region}_B_{stat}` | Blue channel statistic | `mean` or `median` of B values | `float` 0–255 |
| `{region}_H_{stat}` | Hue statistic (0–360°) | `mean` or `median` of H values | `float` 0–360 |
| `{region}_S_{stat}` | Saturation statistic (0–100%) | `mean` or `median` of S values | `float` 0–100 |
| `{region}_V_{stat}` | Value/Brightness statistic (0–100%) | `mean` or `median` of V values | `float` 0–100 |
| `{region}_L_{stat}` | Lightness statistic (0–100) | `mean` or `median` of L* values | `float` 0–100 |
| `{region}_a_{stat}` | Green-Red axis statistic (-128 to +127) | `mean` or `median` of a* values | `float` -128–127 |
| `{region}_b_{stat}` | Blue-Yellow axis statistic (-128 to +127) | `mean` or `median` of b* values | `float` -128–127 |
| `{region}_Gray_{stat}` | Grayscale intensity statistic | `mean` or `median` of gray values | `float` 0–255 |
| `{region}_hue_circular_{stat}` | Circular mean of hue (0–360°) | Circular statistics on hue | `float` 0–360 |
| `{region}_hue_homogeneity` | Hue concentration (1 = uniform, 0 = dispersed) | Resultant vector length | `float` 0–1 |
| `{region}_a_L_ratio_{stat}` | Red-green index normalized by lightness | `a* / (L* + ε)` | `float` |
| `{region}_r_g_ratio_{stat}` | Red/Green ratio | `R / (G + ε)` | `float` > 0 |
| `{region}_r_b_ratio_{stat}` | Red/Blue ratio | `R / (B + ε)` | `float` > 0 |
| `{region}_r_ratio_{stat}` | Red normalized by green+blue | `R / (G + B + ε)` | `float` > 0 |
| `{region}_R_std` | Standard deviation of red channel | `std(R)` | `float` ≥ 0 |
| `{region}_R_cv` | Coefficient of variation of red channel | `std(R) / mean(R)` | `float` ≥ 0 |
| *(same pattern for G, B, H, S, V, L, a, b, Gray, a_L_ratio)* | Standard deviation and CV for all channels | | |

**Regions:**
- `wholefruit_*`: Metrics for the entire fruit area
- `outerpericarp_*`: Metrics for outer pericarp only (fruit - inner pericarp)
- `innerpericarp_*`: Metrics for inner pericarp only (excluding locules when `locules_filled=False`)

**`{stat}` values:**
- `mean`: Arithmetic mean (default)
- `median`: Median value (when `color_stat='median'`)

--- 

## Notes

### Unit conversion
* **cm² / cm**: When size reference or physical dimensions are available (`px_per_cm`)
* **px**: When no size reference or physical dimensions are available (pixel units)
* **Unitless metrics**: Ratios, percentages, indices (independent of px_per_cm)

### Symmetry Interpretation
* **Lower values**: Higher symmetry
* **Higher values**: Lower symmetry (irregular distribution)
* **Angular symmetry**: Mean angular error in radians (0 = perfect)
* **Radial symmetry**: CV of radii (0 = perfect)
* **Rotational symmetry**: Normalized 0–1 (0 = perfect)

### Typical Ranges

These reference values can guide you in setting **filtering parameters** during fruit processing (e.g., `min_circularity`, `min_aspect_ratio`) to exclude artifacts or damaged fruits:

#### Shape Metrics
* **`fruit_circularity`**: 0.60–0.95
  - High values (0.85–0.95): Round fruits like cranberries, cherry tomatoes, oranges
  - Medium values (0.70–0.85): Standard tomatoes, apples, peaches
  - Low values (0.60–0.70): Elongated fruits like plum tomatoes, peppers
  - *Filter recommendation*: Use `min_circularity=0.5` to exclude irregular fragments

* **`fruit_solidity`**: 0.90–0.99
  - High values (0.95–0.99): Smooth, convex fruits without indentations
  - Lower values (0.85–0.95): Fruits with slight concavities or lobes
  - *Filter recommendation*: Use `min_solidity=0.85` to remove severely damaged specimens

* **`aspect_ratio`**: 0.60–1.00
  - Near 1.0: Perfectly round fruits
  - 0.80–0.95: Slightly flattened tomatoes
  - 0.60–0.80: Elongated varieties
  - *Filter recommendation*: Set `min_aspect_ratio=0.3` and `max_aspect_ratio=3.0` to exclude extreme shapes

#### Compactness
* **`compactness_index`**: 0.70–0.85
  - High values (0.80–0.85): Tightly packed, round tomatoes
  - Medium values (0.70–0.80): Standard commercial varieties
  - *Interpretation*: Higher = fruit fills its bounding box more efficiently

#### Internal Structure
* **`internal_cavity_ratio`**: 0.30–0.60
  - High values (0.50–0.60): Large internal cavity (thin pericarp)
  - Medium values (0.40–0.50): Balanced pericarp thickness
  - Low values (0.30–0.40): Thick pericarp, smaller cavity
  - *Depends on*: Variety and maturity stage

* **`locule_to_cavity_percentage`**: 60–90%
  - High values (80–90%): Dense locule packing, well-developed seeds
  - Medium values (70–80%): Standard fruit development
  - Low values (60–70%): Sparse locules or air pockets
  - *Interpretation*: Indicates seed/gel filling efficiency within the cavity

#### Symmetry (Lower = More Symmetric)
* **`angular_symmetry`**: 0.10–0.50 radians
  - Excellent (0.10–0.20): Very evenly distributed locules
  - Good (0.20–0.35): Slight irregularities
  - Poor (>0.40): Highly asymmetric distribution

* **`radial_symmetry`**: 0.10–0.30 (CV)
  - Excellent (0.10–0.15): Uniform distance from center
  - Good (0.15–0.25): Moderate variation
  - Poor (>0.30): Irregular radial placement

* **`rotational_symmetry`**: 0.10–0.40
  - Excellent (0.10–0.20): Near-perfect rotational balance
  - Good (0.20–0.30): Acceptable commercial quality
  - Poor (>0.35): Noticeable asymmetry

---

#### Quick Filtering Guide

**For high-quality fruit selection:**

### Key Formulas

**Circularity**:
```
C = (4π × A) / P²
```

**Solidity**:
```
S = A / A_convex
```

**Coefficient of Variation (CV)**:
```
CV = (σ / μ) × 100
```

**Aspect Ratio**:
```
AR = box_width / box_length
```

