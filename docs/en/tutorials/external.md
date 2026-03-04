# External Analysis: Class and Methods

Module for external fruit analysis: whole-fruit morphology and color without internal tissue segmentation (available in `traitly.fruit_phenotyping.external_analysis`)

---

## Overview

This module provides the `FruitExternalAnalyzer` class. It supports single-image analysis and batch processing from a folder.

### Typical analysis pipeline

```
1. load_image()
2. setup_measurements()
3. generate_fruit_mask()
4. detect_fruits()
5. analyze_morphology() and/or analyze_color()
```

For batch processing, steps 1–5 are orchestrated automatically by `analyze_folder()`.

<br>

---

## Class `FruitExternalAnalyzer`

Analyzer for whole-fruit morphology and color from segmented images, **without locule or inner pericarp segmentation**.

```python
from traitly.fruit_phenotyping import FruitExternalAnalyzer

analyzer = FruitExternalAnalyzer("path/to/image.jpg")
```

> 💡 For internal analysis with locule and pericarp segmentation, use `FruitInternalAnalyzer`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `image_path` | `str` | Path to an image file or directory |

> ⚠️ Raises `FileNotFoundError` if the path does not exist.

<br>

---

### Main attributes

| Attribute | Type | Description |
|----------|------|-------------|
| `img_path` | `str` | Path to the image or folder |
| `img` | `ndarray` | Image loaded in BGR format |
| `img_rgb` | `ndarray` | Image in RGB format |
| `img_hsv` | `ndarray` | Image in HSV color space |
| `mask_fruit` | `ndarray` | Binary fruit mask |
| `contours` | `list` | Detected fruit contours |
| `fruit_locule_map` | `dict` | Fruit mapping. Keeps the same name as in `FruitInternalAnalyzer` for compatibility, but in external analysis each fruit is mapped to an empty list of locules |
| `px_per_cm` | `float` | Pixel density per centimeter |
| `label_text` | `str` | Detected label text |
| `results` | `ResultsImage` | Analysis results |
| `parameters` | `AnalysisParameters` | Session parameters and metadata |

<br>

---

## Methods

<br>

### `load_image(plot, plot_size)`

Loads the image and prepares the internal representations (BGR, RGB, HSV).

```python
analyzer.load_image(plot=True, plot_size=(5, 5))
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `plot` | `bool` | `True` | Displays the loaded image |
| `plot_size` | `tuple` | `(5, 5)` | Figure size |

> ⚠️ Raises `ValueError` if there is no image, the extension is not valid, or loading fails.

<br>

---

### `setup_measurements(...)`

Detects the label and calculates the pixel/cm scale factor.

Notes:
- When `detect_label=True`, the label is detected in order: QR first, and if not found, falls back to OCR. To skip QR detection and go directly to OCR, set `skip_qr=True`.
- When `fast_calibration=False` (default), the size reference is first detected with YOLO and, if it fails, falls back to the image physical measurements provided (`width_cm`, `length_cm`). If no reference is found and `width_cm` and `length_cm` are `None`, results are expressed in pixels.
- For size reference detection, it is assumed that the reference circles are black and the reference background is white.
- When the size reference is used for calibration, the px/cm factor is calculated from the average diameter of all detected circles. By default, circles whose diameter deviates more than 2 standard deviations from the mean are discarded to avoid bias in the scale estimation.
- When `detect_color_checker=True`, the color card is detected using OpenCV's MCC module (`cv2.mcc`), compatible with standard 24-color cards (Macbeth style). Detection is performed on a downscaled version of the image according to `scale_factor`, which speeds up the process but may affect the precision of the detected area for each color patch. You can inspect the detection in detail with `plot_color_checker=True`.

```python
# Using physical measurements and detecting label
analyzer.setup_measurements(
    width_cm=29.7,
    length_cm=21.0,
    detect_label=True
)

# Using size reference and detecting label with OCR only
analyzer.setup_measurements(
    diameter_cm=1.7,
    detect_label=True,
    skip_qr=True
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `detect_label` | `bool` | `False` | If `True`, activates label detection (QR -> OCR) |
| `width_cm` | `float` | `None` | Known image width in cm |
| `length_cm` | `float` | `None` | Known image length in cm |
| `diameter_cm` | `float` | `None` | Known reference circle diameter in cm; if not provided, defaults to 2.5 cm |
| `fast_calibration` | `bool` | `False` | If `True`, skips YOLO and calibrates using `width_cm` and `length_cm`; if not provided, results are expressed in pixels |
| `confidence` | `float` | `0.6` | Minimum confidence for YOLO reference detection |
| `skip_qr` | `bool` | `False` | If `True`, skips QR detection and attempts OCR directly |
| `gpu` | `bool` | `False` | If `True`, uses GPU for OCR; NVIDIA only. Falls back to CPU if it fails |
| `detect_color_checker` | `bool` | `False` | If `True`, detects a color card (24 colors, Macbeth style) after calibration |
| `scale_factor` | `float` | `0.5` | Image downscaling factor for color card detection; must be between 0.1 and 1.0 |
| `language_label` | `list` | `["es", "en"]` | Languages for OCR |
| `font_size` | `int` | `3` | Font size for annotations on reference circles |
| `plot_reference` | `bool` | `False` | If `True`, displays a cropped view of the detected and annotated size reference |
| `plot_color_checker` | `bool` | `False` | If `True`, displays a cropped view of the detected and annotated color card |
| `plot_size` | `tuple` | `(5, 5)` | Figure size for plots |
| `verbose` | `bool` | `True` | If `True`, prints results to console |

<br>

---

### `generate_color_scatterplot(sample_size, plot_size)`

Displays a scatterplot of pixel colors from the **full image** (fruits, background, references, etc.) in HSV space. Useful for selecting appropriate thresholds before creating the mask (`lower_hsv` and `upper_hsv` parameters in `generate_fruit_mask`).

```python
analyzer.generate_color_scatterplot(sample_size=10000)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sample_size` | `int` | `10000` | Number of pixels to sample for the plot |
| `plot_size` | `tuple` | `(18, 5)` | Figure size |

> ⚠️ Raises `ValueError` if no image is loaded or `sample_size` is not a positive integer.

<br>

---

### `generate_fruit_mask(...)`

Generates a binary mask by segmenting the image background in HSV space and detecting everything that does not correspond to the background (fruits, size reference, label, etc.).

By default, a **blue background** is assumed (`background_color='blue'`), which is removed automatically. In the resulting mask, the background is represented in black (0) and fruits in white (1).

If regions corresponding to the size reference, color card, or label are detected in `setup_measurements`, these areas are masked to black in the final mask. However, residual contours may remain, which can be discarded during contour filtering in `detect_fruits`. If these regions are not previously detected, they will appear as white in the mask, as they are classified as non-background.

```python
# Using custom HSV ranges
analyzer.generate_fruit_mask(
    lower_hsv=[20, 30, 30],
    upper_hsv=[80, 255, 255]
)

# Using predefined ranges
analyzer.generate_fruit_mask(background_color='white')
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lower_hsv` | `list[int]` | `None` | Lower HSV threshold `[H, S, V]` to select the background color; if `None`, automatic thresholding is applied |
| `upper_hsv` | `list[int]` | `None` | Upper HSV threshold `[H, S, V]` to select the background color; if `None`, automatic thresholding is applied |
| `background_color` | `str` | `'blue'` | Predefined options: `'black'`, `'white'`, `'blue'`. Used to define default HSV thresholds for the background |
| `n_iteration` | `int` | `1` | Number of iterations for morphological operations (only applies if `kernel_open` and/or `kernel_close` are defined) |
| `kernel_blur` | `int` | `None` | Gaussian blur kernel size |
| `kernel_open` | `int` | `None` | Morphological opening kernel size |
| `kernel_close` | `int` | `None` | Morphological closing kernel size |
| `canny_min` | `int` | `None` | Minimum Canny threshold |
| `canny_max` | `int` | `None` | Maximum Canny threshold |
| `remove_roi` | `bool` | `True` | If `True`, removes label, reference, and color card regions from the mask |
| `roi_expansion` | `int` | `10` | Pixel margin around ROIs before removing them |
| `fill_holes` | `bool` | `False` | If `True`, fills closed holes in the binary mask |
| `apply_convex_hull` | `bool` | `False` | If `True`, applies convex hull to external fruit contours |
| `erosion_px` | `int` | `3` | Radius in pixels of the elliptical erosion applied to the final mask |
| `stamp` | `bool` | `False` | If `True`, inverts the image colors before masking; assumes a white original background |
| `plot` | `bool` | `True` | Displays the generated mask |
| `plot_size` | `tuple` | `(5, 5)` | Figure size |

> ⚠️ Raises `ValueError` if no image is loaded.

<br>

---

### `detect_fruits(...)`

Detects individual fruits from the binary mask generated by `generate_fruit_mask()`.

Detection is based on contours and morphological criteria of **size** and **shape** (area and circularity), allowing unwanted objects to be filtered out. Unlike `FruitInternalAnalyzer`, no locule detection or mapping is performed.

As a result, two main structures are generated:

* `self.contours`: list of detected fruit contours.
* `self.fruit_locule_map`: dictionary that maps each fruit to an empty list of locules, maintaining consistency with the rest of the pipeline.

Notes:
- When working with very large images, `rescale_factor` can be used to temporarily reduce the scale during contour detection. Once detection is complete, contours are automatically rescaled back to the original image size. This can improve computational performance, although it may affect detection precision for images with very small fruits or low quality.
- Before continuing with the analysis, you can quickly inspect the detected contours with `plot=True`.

```python
analyzer.detect_fruits(
    min_fruit_circularity=0.5,
    min_fruit_area=500
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_fruit_circularity` | `float` | `0.5` | Minimum circularity `[0, 1]` to accept a contour as a fruit |
| `min_fruit_area` | `int` | `500` | Minimum fruit area (px) |
| `max_fruit_area` | `int` | `None` | Maximum fruit area (px); if `None`, no upper limit is applied |
| `rescale_factor` | `float` | `None` | Factor for rescaling contours before detection |
| `verbose` | `bool` | `True` | Prints a detection summary and parameters used |
| `plot` | `bool` | `False` | Displays detected fruit contours on the image |
| `plot_size` | `tuple[int, int]` | `(5, 5)` | Figure size (only if `plot=True`) |
| `contour_color` | `tuple[int, int, int]` | `(0, 255, 0)` | BGR color for drawing detected contours (only if `plot=True`) |
| `contour_thickness` | `int` | `2` | Line thickness for drawing contours (only if `plot=True`) |

> ⚠️ **Requires** that a mask exists (`generate_fruit_mask()`).
Raises `ValueError` if no mask is available.

<br>

---

### `analyze_morphology(...)`

Extracts morphological metrics from detected fruits at the **whole-fruit level**, without including locule, inner pericarp, or symmetry metrics.

Results are stored in `self.results` as a `ResultsImage` instance. This class contains:

* `self.results.morphology_results`: `pd.DataFrame` with the morphological metrics for each fruit.
* `self.results.annotated_img`: annotated image for visual inspection.

Additionally, `self.results` includes methods to save results:

* `self.results.save_all()` saves the annotated image and the CSV file.
* `self.results.save_csv()` saves only the CSV.
* `self.results.save_img()` saves only the image.

By default, files are saved in the same folder as the input image, using the original filename as the base. The output directory and an alternative base name can be specified via `output_dir='PATH/'` and `base_name='new_name'`. For more details, refer to the `ResultsImage` class documentation.

The annotated image displays a **unique ID for each fruit** and highlights the following elements:

* **fruit contour** (cyan),
* ***bounding box* rectangle**,
* **major axis** and **minor axis**.

**Note:**
- For fruits with very irregular edges, it may be useful to try different `contour_mode` values (e.g., `'hull'` or `'approx'`) to smooth the contour. Depending on the mode (except `'raw'`), some traits may be fixed by construction; for example, if `'circle'` is used, fruit circularity will be `1` (perfect circle) for all fruits.

```python
analyzer.analyze_morphology(
    contour_mode="hull",
    label_position="bottom",
    label_color=(255, 255, 0)
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `contour_mode` | `str` | `'raw'` | Contour mode used for metrics: `'raw'`, `'hull'`, `'approx'`, `'ellipse'`, `'circle'` |
| `epsilon` | `float` | `0.001` | Approximation factor (only if `contour_mode='approx'`) |
| `display_table` | `bool` | `True` | If `True`, returns the results `DataFrame` |
| `plot` | `bool` | `True` | If `True`, displays the annotated image |
| `plot_size` | `tuple[int, int]` | `(10, 10)` | Figure size (only if `plot=True`) |
| `font_size` | `float` | `1.5` | Text size in the annotation |
| `font_thickness` | `int` | `2` | Text thickness in the annotation |
| `font_color` | `tuple[int,int,int]` | `(0, 0, 0)` | Text color (BGR) |
| `label_position` | `str` | `'top'` | Label position (`'top'`, `'bottom'`, `'left'`, `'right'`) |
| `label_color` | `tuple[int,int,int]` | `(255, 255, 255)` | Label background color (BGR) |
| `pericarp_ext_color` | `tuple[int,int,int]` | `(0, 240, 240)` | Fruit contour color (BGR) |
| `pericarp_ext_thickness` | `int` | `2` | Fruit contour thickness |

> ⚠️ **Requires** that a mask exists (`generate_fruit_mask()`) and that `detect_fruits()` has been executed. Raises `ValueError` if any of these are missing or if no valid fruits were detected.

<br>

---

### `analyze_color(...)`

Extracts color features from the **total pericarp** of detected fruits using the original image and the mask generated in the pipeline. Unlike `FruitInternalAnalyzer`, internal tissues are not segmented; color is extracted only over the complete fruit region.

Color extraction always uses the original contours in `'raw'` mode, regardless of the `contour_mode` selected in `analyze_morphology()`. This ensures that the color extraction area faithfully corresponds to the segmented region in the mask, without being affected by geometric contour simplifications.

Results are stored in `self.results` as a `ResultsImage` instance. This class contains:

* `self.results.color_results`: `pd.DataFrame` with color metrics for each fruit.
* `self.results.annotated_img`: annotated image for visual inspection.

Additionally, `self.results` includes methods to save results:

* `self.results.save_all()` saves the annotated image and the CSV file.
* `self.results.save_csv()` saves only the CSV.
* `self.results.save_img()` saves only the image.

By default, files are saved in the same folder as the input image. The output directory and base name can be specified via `output_dir='PATH/'` and `base_name='new_name'`.

**Notes:**
* `analyze_color()` is **independent** of `analyze_morphology()`. If only `analyze_color()` is executed, a basic annotated image is generated with the **fruit ID** and the **fruit contour** in green.
* If `analyze_morphology()` was previously executed, when saving results, the morphology annotated image is **reused**, as it contains more complete annotations.
* By default, the function calculates a summary statistic (`'mean'` or `'median'`) per channel. Alternatively, pixel-level color histograms can be calculated by activating `get_color_histogram=True`, which returns complete distributions per channel instead of a single summary value.

```python
df = analyzer.analyze_color(
    stat='median',
    color_space='hsv, lab',
    plot=False
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stat` | `str` | `'mean'` | Statistic: `'mean'` or `'median'` (ignored if `get_color_histogram=True`) |
| `color_space` | `str` | `'all'` | Spaces: `'all'`, `'rgb'`, `'lab'`, `'hsv'`, `'gray'` |
| `display_table` | `bool` | `True` | If `True`, returns the results `DataFrame` |
| `plot` | `bool` | `False` | If `True`, displays the annotated image used for color extraction |
| `plot_size` | `tuple[int, int]` | `(10, 10)` | Figure size (only if `plot=True`) |
| `font_size` | `int` | `2` | Text size in the annotation |
| `font_thickness` | `int` | `2` | Text thickness in the annotation |
| `font_color` | `tuple[int,int,int]` | `(0, 0, 0)` | Text color (BGR) |
| `label_position` | `str` | `'top'` | Label position (`'top'`, `'bottom'`, `'left'`, `'right'`) |
| `label_color` | `tuple[int,int,int]` | `(255, 255, 255)` | Label background color (BGR) |
| `pericarp_ext_color` | `tuple[int,int,int]` | `(0, 255, 0)` | Fruit contour color (BGR) |
| `pericarp_ext_thickness` | `int` | `2` | Fruit contour thickness |
| `label_opacity` | `float` | `0.7` | Label background opacity `[0, 1]` |
| `get_color_histogram` | `bool` | `False` | If `True`, returns pixel-level histograms instead of summary statistics |

> ⚠️ **Requires** that a mask exists (`generate_fruit_mask()`) and that `detect_fruits()` has been executed. Raises `ValueError` if any of these are missing or if no valid fruits were detected.

<br>

---

## `generate_single_fruit_masks(...)` *(optional)*

Generates and visualizes the whole-fruit mask for a specific fruit, useful for inspecting segmentation results in detail before running `analyze_color()`.

The fruit is cropped to its *bounding box* with an optional margin. The `fruit_id` parameter corresponds to the fruit identifier in the annotated image or results table, as it appears in the outputs generated by `analyze_morphology()` or `analyze_color()`.

```python
analyzer.generate_single_fruit_masks(fruit_id=3)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fruit_id` | `int` | `None` | ID of the fruit to visualize; if `None`, uses the first detected fruit |
| `plot_size` | `tuple[int, int]` | `(7, 5)` | Figure size |
| `margin` | `int` | `5` | Margin (px) around the fruit crop |

> ⚠️ **Requires** that a mask exists and that `detect_fruits()` has been executed. Raises `ValueError` if there is no mask, no contours, if a `fruit_id` that does not exist in the image is requested, or if no fruits were detected.

<br>

---

### `save_parameters(...)`

Exports the **analysis parameters from the current session** in `.txt` and `.json` format, ready for inspection, reuse, and reproducibility.

The parameters stored in `self.parameters` are exported using the loaded image name as the base, automatically generating two files:

* `<image_name>_parameters.txt` — human-readable version for inspection.
* `<image_name>_parameters.json` — structured version for programmatic use.

Both are saved by default in the same folder as the input image, or in the directory specified by `output_path`. They are especially useful for:

* reusing configurations in batch analysis with `analyze_folder`,
* running reproducible analyses from the terminal with **Traitly**,
* archiving and sharing analysis pipelines.

#### Notes
* Only the parameters of the functions executed during the session are exported (segmentation, detection, morphology, color).
* Returns no value; prints the file paths to the console.

```python
analyzer.save_parameters()
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_path` | `str` | `None` | Output directory. If `None`, the same directory as the input image is used. |

<br>

---

### `plot_image(...)`

Displays the original or **annotated results image** according to the value of `annotated`, reusing images already stored in memory without reloading or regenerating them.

```python
analyzer.plot_image(annotated=True)
```

* When `annotated=False`, the loaded **original image** is displayed.
* When `annotated=True`, the **annotated image** generated during `analyze_morphology()` or `analyze_color()` is displayed.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `annotated` | `bool` | `True` | If `True`, displays the annotated image; if `False`, displays the original image |
| `plot_size` | `tuple[int, int]` | `(10, 10)` | Figure size |

⚠️ **Raises `ValueError`** if `annotated=True` and neither `analyze_morphology()` nor `analyze_color()` has been previously executed, or if no annotated image is available in `self.results`.

<br>

---

### `analyze_folder(...)`

Processes in batch all images in the folder passed to `FruitExternalAnalyzer`, running the full pipeline (steps 1–5) on each image sequentially or in parallel (when `num_cores > 1`). By default, both morphological and color analyses are executed; each can be independently disabled with `analyze_morphology=False` or `analyze_color=False`.

For each analyzed image, an **annotated image** is generated with the identifiers and visual annotations of the analysis. Results from all images are consolidated into a single CSV file per analysis type:

* `morphology_results.csv`: morphological metrics for all detected fruits.
* `color_results.csv`: color metrics for all detected fruits.

Additionally, a `session_report.txt` is always generated with a session summary (images processed, fruits detected, timing, parameters used, and dependencies). If any image fails during processing, an `error_report.txt` is also generated detailing what occurred in each case.

All files are saved in the directory specified by `output_path`, or in a `Results/` subfolder inside the input folder if not specified.

> 💡 This function accepts all pipeline parameters (steps 1–5) individually. However, for greater convenience and reproducibility, it is recommended to explore and standardize parameters on a representative image using `save_parameters()`, then pass the generated `.json` file via `json_path`.

```python
# Using individual parameters
analyzer.analyze_folder(
    lower_hsv=[0, 0, 0],
    upper_hsv=[180, 80, 80],
    min_fruit_area=500,
    analyze_color=True
)

# Using a saved parameters file
analyzer.analyze_folder(json_path="image_parameters.json")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `analyze_morphology` | `bool` | `True` | If `True`, runs morphological analysis on each image |
| `analyze_color` | `bool` | `True` | If `True`, runs color analysis on each image |
| `json_path` | `str` | `None` | Path to a `.json` parameters file generated by `save_parameters()` |
| `config` | `dict` | `None` | Base configuration as a dictionary; individual parameters take priority |
| `output_path` | `str` | `None` | Output directory. If `None`, a `Results/` subfolder is created inside the input folder |
| `num_cores` | `int` | `1` | Number of parallel processes. Automatically capped to available cores |
| `verbose` | `bool` | `True` | If `True`, prints progress and session summary |
| `width_cm` | `float` | `None` | Known image width in cm -> `setup_measurements` |
| `length_cm` | `float` | `None` | Known image length in cm -> `setup_measurements` |
| `diameter_cm` | `float` | `None` | Known reference diameter in cm -> `setup_measurements` |
| `fast_calibration` | `bool` | `None` | If `True`, skips YOLO and calibrates with physical dimensions -> `setup_measurements` |
| `skip_qr` | `bool` | `None` | If `True`, skips QR detection -> `setup_measurements` |
| `detect_label` | `bool` | `None` | If `True`, activates label detection with OCR -> `setup_measurements` |
| `confidence` | `float` | `None` | Minimum YOLO detection confidence -> `setup_measurements` |
| `detect_color_checker` | `bool` | `None` | If `True`, detects and removes color card -> `setup_measurements` |
| `scale_factor` | `float` | `None` | Downscaling factor for color card detection -> `setup_measurements` |
| `lower_hsv` | `list[int]` | `None` | Lower HSV threshold for segmentation -> `generate_fruit_mask` |
| `upper_hsv` | `list[int]` | `None` | Upper HSV threshold for segmentation -> `generate_fruit_mask` |
| `background_color` | `str` | `None` | Predefined background color -> `generate_fruit_mask` |
| `n_iteration` | `int` | `None` | Morphological operation iterations -> `generate_fruit_mask` |
| `kernel_blur` | `int` | `None` | Gaussian blur kernel size -> `generate_fruit_mask` |
| `kernel_open` | `int` | `None` | Morphological opening kernel size -> `generate_fruit_mask` |
| `kernel_close` | `int` | `None` | Morphological closing kernel size -> `generate_fruit_mask` |
| `canny_min` | `int` | `None` | Minimum Canny threshold -> `generate_fruit_mask` |
| `canny_max` | `int` | `None` | Maximum Canny threshold -> `generate_fruit_mask` |
| `fill_holes` | `bool` | `None` | If `True`, fills holes in the mask -> `generate_fruit_mask` |
| `apply_convex_hull` | `bool` | `None` | If `True`, applies convex hull to each fruit -> `generate_fruit_mask` |
| `remove_roi` | `bool` | `None` | If `True`, removes reference and label regions -> `generate_fruit_mask` |
| `roi_expansion` | `int` | `None` | Pixel margin around ROIs -> `generate_fruit_mask` |
| `stamp` | `bool` | `None` | If `True`, inverts colors before masking -> `generate_fruit_mask` |
| `min_fruit_area` | `int` | `None` | Minimum area to accept a contour as a fruit -> `detect_fruits` |
| `max_fruit_area` | `int` | `None` | Maximum area to accept a contour as a fruit -> `detect_fruits` |
| `min_fruit_circularity` | `float` | `None` | Minimum circularity to accept a fruit -> `detect_fruits` |
| `rescale_factor` | `float` | `None` | Contour rescaling factor -> `detect_fruits` |
| `contour_mode` | `str` | `None` | Contour mode for morphological metrics -> `analyze_morphology` |
| `epsilon` | `float` | `None` | Contour approximation factor -> `analyze_morphology` |
| `stat` | `str` | `None` | Color statistic: `'mean'` or `'median'` -> `analyze_color` |
| `color_space` | `str` | `None` | Color spaces to extract -> `analyze_color` |
| `get_color_histogram` | `bool` | `None` | If `True`, computes pixel-level histograms -> `analyze_color` |

> ⚠️ **Requires** that `FruitExternalAnalyzer` was initialized with a folder path, not a file path. Raises `ValueError` if the path is not a directory or if no valid images are found in the folder.