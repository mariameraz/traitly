<div class="animate" markdown>

# Internal Analysis: Class and Methods

This section covers everything you need to work with `FruitInternalAnalyzer`, the main class for analyzing images of fruit cross-sections. Each method is explained along with its parameters and how to fit it into your workflow.

---

## 1. Main Class

`FruitInternalAnalyzer` is the primary tool for analyzing the internal morphology, color, and symmetry of fruits from cross-section images. You can use it to process a single image or an entire folder with hundreds of images.

```python
from traitly.fruit_phenotyping import FruitInternalAnalyzer

# To analyze a single image
analyzer = FruitInternalAnalyzer(image_path = "path/to/my/image.jpg")

# To analyze multiple images in a folder
analyzer = FruitInternalAnalyzer(image_path = "path/to/my/folder/with/images/")
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `image_path` | `str` | Path to the image or folder you want to analyze |


!!! tip "Recommendation"
    When you have a folder of images to process, we suggest:
    
    1. **Start with a representative image** to fine-tune the parameters
    2. Work through the methods step by step until you get good results
    3. Save the best configuration with `save_parameters()`
    4. Use `analyze_folder(json_path="your_file.json")` to run the full batch automatically with those same parameters
    
    For hands-on examples of this workflow, check out the [Tutorials](../tutorials/quickstart.md).



<br>

</div>

---

## 2. How the Analysis Is Organized

When working with `FruitInternalAnalyzer`, the analysis follows this logical order:

```python
from traitly.fruit_phenotyping import FruitInternalAnalyzer

# Analyze a single image
analyzer = FruitInternalAnalyzer('path/to/my/image.jpg')

analyzer.load_image()                          # Load the image
analyzer.setup_measurements()                  # Set up calibration and labels
analyzer.generate_fruit_mask()                 # Separate fruits from the background
analyzer.enhance_locule_contrast()             # (Optional) Enhance locule contrast
analyzer.generate_l_channel_histogram()        # (Optional) Visualize L channel distribution to choose threshold
analyzer.generate_locule_mask()                # (Optional) Segment locules
analyzer.edit_mask()                           # (Optional) Manually correct the active mask
analyzer.detect_fruits()                       # Identify individual fruits
analyzer.analyze_morphology()                  # Get morphological measurements
analyzer.analyze_color()                       # (Optional) Get color measurements

## 3. Save results
analyzer.results.save_all()               # Save all results (CSV and annotated image)
analyzer.save_parameters()                # (Optional) Save the parameters used in the session

```

If you're working with batches of images, you don't need to run each step individually — `analyze_folder()` handles everything automatically:

```python
# Analyze multiple images
analyzer = FruitInternalAnalyzer('path/to/my/folder')              # Initialize with your folder path
analyzer.analyze_folder(json_path = 'path/to/my/parameters.json')  # Run the analysis, optionally using saved parameters

```


<br>

---

## 4. What You Can Get from the Analyzer

After running the methods, the analyzer stores results in attributes you can inspect:

| Attribute | Contents |
|----------|--------------|
| `img_path` | Path of the image being analyzed |
| `img_name` | Image name |
| `img_shape` | Image size |
| `img`, `img_rgb`, `img_hsv` | The image in different color formats |
| `mask_fruit` | Mask where fruits appear white and the background is black |
| `mask_locules` | Mask where locules appear black and the rest of the fruit is white (if `generate_locule_mask()` was run) |
| `contours` | List of contours for all detected fruits |
| `fruit_locule_map` | Map linking each fruit to the contour indices of its corresponding locules, grouped by fruit |
| `px_per_cm` | Pixel-to-centimeter conversion factor (if calibrated) |
| `label_text` | Detected label text (if label detection was used) |
| `results` | All analysis results (tables + annotated image) |
| `parameters` | Parameters used in the current session |

<br>

---

## 5. Methods:

!!! example ""
    All the methods includes default values for the parameters, so you can start simple and adjust as needed.



### `load_image`

Loads the image and prepares its internal representations (BGR, RGB, HSV).
Optionally, a region of interest can be cropped using `x`, `y`, `w`, `h`.

```python
analyzer.load_image(plot=True, plot_size=(5, 5))
analyzer.load_image(plot=True, show_axis=True, x=1500, y=0, w=2600, h=2700)
```

<br>

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `plot` | `bool` | `True` | Displays the loaded image |
| `plot_size` | `tuple` | `(5, 5)` | Figure size |
| `show_axis` | `bool` | `False` | Shows axis ticks on the plot |
| `x` | `int` | `None` | Left coordinate of the crop |
| `y` | `int` | `None` | Top coordinate of the crop |
| `w` | `int` | `None` | Crop width in pixels |
| `h` | `int` | `None` | Crop height in pixels |

<br>

---


### `setup_measurements`

Handles label detection and size reference detection, then calculates the pixel/cm scale factor.

??? note "Notes"

    - When `detect_label=True`, labels are detected in this order: QR first, falling back to OCR if no QR code is found. To skip QR detection and go straight to OCR, set `skip_qr=True`.

    - When `fast_calibration=False` (default), the size reference is first detected with YOLO; if that fails, it falls back to the physical dimensions provided (`width_cm`, `length_cm`). If no reference is found and both `width_cm` and `length_cm` are `None`, results are expressed in pixels.
    
    - For size reference detection, the reference circles are assumed to be black on a white background.

    - When using the size reference for calibration, the px/cm factor is derived from the average diameter of all detected circles. By default, circles whose diameter deviates more than 2 standard deviations from the mean are excluded to prevent bias in the scale estimate.


```python
# Using physical dimensions and detecting label
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

<br>

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `detect_label` | `bool` | `False` | If `True`, enables label detection (QR -> OCR) |
| `width_cm` | `float` | `None` | Known image width in cm |
| `length_cm` | `float` | `None` | Known image length in cm |
| `diameter_cm` | `float` | `None` | Known diameter of the reference circle in cm; defaults to 2.5 cm if not provided |
| `fast_calibration` | `bool` | `False` | If `True`, skips YOLO and calibrates using `width_cm` and `length_cm`; results are in pixels if neither is provided |
| `confidence` | `float` | `0.6` | Minimum confidence for YOLO reference detection |
| `skip_qr` | `bool` | `False` | If `True`, skips QR detection and attempts OCR directly |
| `gpu` | `bool` | `False` | If `True`, uses GPU for OCR; NVIDIA only. Falls back to CPU on failure |
| `detect_color_checker` | `bool` | `False` | If `True`, detects a color checker (24-color Macbeth-style card) after calibration |
| `scale_factor` | `float` | `0.5` | Image downscaling factor for color checker detection; must be between 0.1 and 1.0, where 1.0 uses the full image size and 0.1 applies a 90% reduction |
| `language_label` | `list` | `["es", "en"]` | Languages for OCR |
| `font_size` | `int` | `3` | Font size for annotations on reference circles |
| `plot_reference` | `bool` | `False` | If `True`, displays a cropped and annotated view of the detected size reference |
| `plot_color_checker` | `bool` | `False` | If `True`, displays a cropped and annotated view of the detected color card |
| `plot_size` | `tuple` | `(5, 5)` | Figure size for plots |
| `verbose` | `bool` | `True` | If `True`, prints results to the console |


<br>

---

### `generate_color_scatterplot`

*Optional*

Displays a scatterplot of pixel colors from the **full image** (fruits, background, references, etc.) in HSV space. Useful for picking appropriate thresholds before building the mask (`lower_hsv` and `upper_hsv` in `generate_fruit_mask()`).

```python
analyzer.generate_color_scatterplot(sample_size=10000)
```
<br>

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sample_size` | `int` | `10000` | Number of pixels to sample for the plot |
| `plot_size` | `tuple` | `(18, 5)` | Figure size |


<br>

---

### `generate_fruit_mask`

Generates a binary mask by segmenting the background in HSV space and keeping everything that isn't background (fruits, size reference, label, etc.).

By default, a black background is assumed and removed automatically. In the resulting mask, the background is represented in black (0) and fruits in white (1). In fruits with hollow locules (e.g., pepper or cranberry), the internal regions corresponding to the locules may appear as black, since they contain no fruit tissue.

If regions corresponding to the size reference or label are detected in `setup_measurements()`, those areas are masked out (set to black) in the final mask. Residual contours may still remain and can be filtered out later during contour filtering. If those regions were not detected beforehand, they will appear as white in the mask, since they are classified as non-background.

```python
# Using custom HSV ranges
analyzer.generate_fruit_mask(
    lower_hsv=[20, 30, 30],
    upper_hsv=[80, 255, 255]
)

# Using preset ranges
analyzer.generate_fruit_mask(background_color = 'white')

```

<br>


| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lower_hsv` | `list[int]` | `None` | Lower HSV threshold `[H, S, V]` for background color selection; automatic thresholding is applied if `None` |
| `upper_hsv` | `list[int]` | `None` | Upper HSV threshold `[H, S, V]` for background color selection; automatic thresholding is applied if `None` |
| `background_color` | `str` | `None` | Preset options: `'black'` (default), `'white'`, `'blue'`. Used to set predefined HSV thresholds for the background |
| `n_iteration` | `int` | `1` | Number of iterations for morphological operations (only applies if `kernel_open` and/or `kernel_close` are set) |
| `kernel_blur` | `int` | `None` | Gaussian blur kernel size |
| `kernel_open` | `int` | `None` | Morphological opening kernel size |
| `kernel_close` | `int` | `None` | Morphological closing kernel size |
| `canny_min` | `int` | `None` | Minimum Canny threshold |
| `canny_max` | `int` | `None` | Maximum Canny threshold |
| `remove_roi` | `bool` | `True` | If `True`, removes label, reference, and color checker regions from the mask |
| `roi_expansion` | `int` | `10` | Pixel margin around ROIs before removing them |
| `fill_holes` | `bool` | `False` | If `True`, fills closed holes in the binary mask |
| `apply_convex_hull` | `bool` | `False` | If `True`, applies convex hull only to the outer fruit contours; does not apply to locules or other internal regions |
| `erosion_px` | `int` | `0` | Elliptical erosion radius in pixels applied to the final mask |
| `stamp` | `bool` | `False` | If `True`, inverts image colors before masking; assumes a white original background |
| `plot` | `bool` | `True` | Displays the generated mask |
| `plot_size` | `tuple` | `(5, 5)` | Figure size |



<br>

---

### `enhance_locule_contrast`


*Optional*

Applies contrast enhancement to the L channel (Lab) to increase the separation between pericarp (fruit) and locules, making threshold-based segmentation in grayscale easier within `generate_locule_mask()`.

This is especially useful when locules are not hollow (e.g., tomato or orange) and therefore do not appear as black (0) in the binary mask produced by `generate_fruit_mask()`.

??? note "Note" 
    Once you've picked a method using `compare_method=True`, you need to run the function again with `contrast_method='...'` to continue the pipeline with that method.

```python
# Compare all contrast methods
analyzer.enhance_locule_contrast(compare_method=True)

# Apply gamma contrast
analyzer.enhance_locule_contrast(
    contrast_method='gamma',
    gamma=1.5,
    plot=True
)
```

<br>

| Parameter         | Type              | Default   | Description                                                                                                    |
| ----------------- | ----------------- | --------- | -------------------------------------------------------------------------------------------------------------- |
| `contrast_method` | `str`             | `'none'` | Enhancement method: `'gamma'`, `'sigmoid'`, `'exponential'`, or `'none'` (no transformation)                   |
| `gamma`           | `float`           | `1.5`     | Gamma exponent (only if `contrast_method='gamma'`)                                                              |
| `gain`            | `float`           | `5`       | Sigmoid gain (only if `contrast_method='sigmoid'`)                                                              |
| `cutoff`          | `float`           | `0.5`     | Sigmoid cutoff (only if `contrast_method='sigmoid'`)                                                            |
| `c`               | `float`           | `0.5`     | Exponential factor (only if `contrast_method='exp'`)                                                    |
| `kernel_blur`     | `int`             | `1`       | Gaussian blur kernel size applied before enhancement                                                            |
| `clip_limit`      | `int`             | `None`    | Applies CLAHE after the selected method                                                                         |
| `tile_grid_size`  | `int`             | `12`      | CLAHE grid size (only if `clip_limit` is set)                                                                   |
| `compare_method`  | `bool`            | `False`   | If `True`, shows a side-by-side comparison of all available methods                                             |
| `plot`            | `bool`            | `True`    | Shows the enhanced L channel when `contrast_method=...` or the method comparison when `compare_method=True`     |
| `plot_size`       | `tuple[int, int]` | `(8, 10)` | Figure size                                                                                                     |

<br>

---

### `generate_locule_mask`

*Optional*

Generates a binary locule mask by thresholding the previously enhanced L channel (Lab), then merges it with the fruit mask from `generate_fruit_mask()`.

The method segments locule tissue from the rest of the fruit by thresholding the L channel. By default, it uses **Otsu's method** (`use_otsu=True`) to automatically find the optimal threshold — useful when processing batches with variable illumination. You can also set the threshold manually with `thresh_min`. Within this range, darker regions are interpreted as locules or internal tissues, while brighter regions correspond to the pericarp.

In fruits where the opposite is true (e.g., dragon fruit), where the pericarp is darker than the locular space, set `invert_locule=True`. This will internally invert the locule mask after the threshold is applied.

The merged output produces a final mask where fruits are represented in white (1) and locules or internal tissues in black (0), maintaining consistency with the rest of the pipeline's segmentation scheme.

??? note "Choosing a threshold"
    Before running this method, you can visualize the L channel pixel distribution with `generate_l_channel_histogram()`. This plot shows how pixel intensities are distributed within the fruit and where Otsu's threshold falls, making it easier to decide whether to use `use_otsu=True` or a manual `thresh_min`, and whether an `otsu_offset` is needed to fine-tune the split.

```python
# Using Otsu's automatic threshold (default)
analyzer.generate_locule_mask(plot=True)

# Fine-tuning Otsu with an offset
analyzer.generate_locule_mask(use_otsu=True, otsu_offset=10, plot=True)

# Using a manual threshold
analyzer.generate_locule_mask(use_otsu=False, thresh_min=107, plot=True)

# Inverted locules (pericarp darker than locular space)
analyzer.generate_locule_mask(invert_locule=True, plot=True)
```

<br>


| Parameter          | Type              | Default   | Description                                                                                      |
| ------------------ | ----------------- | --------- | ------------------------------------------------------------------------------------------------ |
| `thresh_min`       | `int`             | `120`     | Manual binarization threshold for the L channel; only used when `use_otsu=False`                 |
| `use_otsu`         | `bool`            | `False`    | If `True`, automatically computes the threshold using Otsu's method, ignoring `thresh_min`       |
| `otsu_offset`      | `int`             | `0`       | Value added to the Otsu threshold; positive values capture more pixels, negative values less     |
| `kernel_close`     | `int`             | `None`    | Kernel size for morphological closing applied to the locule mask                                 |
| `kernel_open`      | `int`             | `None`    | Kernel size for morphological opening applied to the locule mask                                 |
| `kernel_blur`      | `int`             | `None`    | Gaussian blur kernel size applied after morphological operations                                 |
| `erosion_px`       | `int`             | `10`      | Erosion radius (px) applied to the fruit mask before masking locules; removes false border locules |
| `min_fruit_area`   | `int`             | `5000`    | Minimum area (in pixels) to keep a fruit region during merging                                   |
| `min_locule_area`  | `int`             | `0`       | Minimum area (in pixels) to retain a locule blob; removes small noise after morphological operations |
| `invert_locule`    | `bool`            | `False`   | Internally inverts the locule mask after thresholding                                            |
| `plot`             | `bool`            | `True`    | Displays the locule mask and the final merged mask                                               |
| `plot_size`        | `tuple[int, int]` | `(10, 5)` | Figure size                                                                                      |


!!! warning "Important"
    **Requires** that both `generate_fruit_mask()` and `enhance_locule_contrast()` have been run beforehand.

<br>

---

### `generate_l_channel_histogram`

*Optional*

Displays the pixel intensity distribution of the L channel (Lab) within the fruit mask. Useful for choosing the right threshold before calling `generate_locule_mask()`.

The plot shows two panels: the full L channel distribution on the left, and the same distribution split by Otsu's threshold on the right (darker vs. lighter pixels). A grayscale intensity bar is included along the x-axis as a visual reference. If `otsu_offset` is set, the adjusted threshold line is also shown.

```python
# Visualize distribution before choosing a threshold
analyzer.generate_l_channel_histogram()

# With Otsu offset
analyzer.generate_l_channel_histogram(otsu_offset=10)
```

<br>

| Parameter      | Type              | Default    | Description                                                                 |
| -------------- | ----------------- | ---------- | --------------------------------------------------------------------------- |
| `otsu_offset`  | `int`             | `0`        | Offset added to Otsu's threshold; shown as a second line in the right panel |
| `plot_size`    | `tuple[int, int]` | `(9, 3)`   | Figure size                                                                 |

!!! warning "Important"
    **Requires** that both `generate_fruit_mask()` and `enhance_locule_contrast()` have been run beforehand.

<br>

---

### `edit_mask`

*Optional*

Opens an interactive editor to manually correct the active mask — `mask_locules` if available, otherwise `mask_fruit`. Allows drawing polygons to add (white) or remove (black) regions from the mask.

Both panels are shown side by side: the mask on the left and the original image with a semi-transparent mask overlay on the right, so you can visually compare them while editing. Changes are applied only when confirmed with `Enter`, and can be undone with `Z`. When you close the editor with `Q`, changes are saved; with `ESC`, all edits are discarded.

```python
# Open the mask editor
analyzer.edit_mask()

# Without printing the controls guide
analyzer.edit_mask(verbose=False)
```

<br>

| Parameter | Type   | Default | Description                                                            |
| --------- | ------ | ------- | ---------------------------------------------------------------------- |
| `verbose` | `bool` | `True`  | If `True`, prints a controls guide in the notebook before opening the editor |

??? note "Controls"

    | Key | Action |
    |-----|--------|
    | Left click | Add polygon point |
    | Right click drag | Pan the view |
    | `W` | Switch to ADD mode (fill white) |
    | `B` | Switch to REMOVE mode (fill black) |
    | `Enter` | Apply current polygon |
    | `Z` | Undo last applied polygon |
    | `C` | Clear current polygon points |
    | `+` / `=` | Zoom in |
    | `-` / `_` | Zoom out |
    | `T` | Toggle mask overlay opacity on the original image (10% steps) |
    | `Q` | Quit and **save** changes |
    | `ESC` | Quit and **discard** all changes |

!!! warning "Important"
    **Requires** that at least `generate_fruit_mask()` has been run. Requires running outside of a pure browser environment — needs a desktop display (e.g., running locally or via a remote desktop).

<br>

---

### `detect_fruits`

Detects individual fruits and their locules from a binary mask (from `generate_fruit_mask()` or `generate_locule_mask()` if the latter was created).

Detection is based on contours and morphological criteria — **size** and **shape** (area and circularity) — which lets you filter out unwanted objects.

Two main structures are produced:

* `analyzer.contours`: list of detected contours (includes fruit contours and, if applicable, internal contours such as locules).
* `analyzer.fruit_locule_map`: dictionary mapping each fruit to the contour indices of its corresponding locules, **grouped by fruit**.


??? note "Notes"
    - For very large images, `rescale_factor` can be used to temporarily downscale the image during contour detection. Once detection is complete, contours are automatically rescaled back to the original image size. This can improve computational performance, though it may affect detection accuracy for very small fruits or low-quality images.

    - Before moving on, you can quickly check the detected fruit contours with `plot=True`.

```python
analyzer.detect_fruits(
    min_fruit_circularity=0.5,
    min_fruit_area=500
)
```

<br>


| Parameter               | Type                   |       Default | Description                                                         |
| ----------------------- | ---------------------- | ------------: | ------------------------------------------------------------------- |
| `min_fruit_circularity` | `float`                |         `0.5` | Minimum circularity `[0, 1]` to accept a contour as a fruit        |
| `min_locule_area`       | `int`                  |          `50` | Minimum area (px) to consider a contour as a locule                |
| `min_locule_per_fruit`  | `int`                  |           `1` | Minimum number of locules required to accept a fruit               |
| `min_fruit_area`        | `int`                  |        `None` | Minimum fruit area (px); no lower limit if `None`                  |
| `max_fruit_area`        | `int`                  |        `None` | Maximum fruit area (px); no upper limit if `None`                  |
| `rescale_factor`        | `float`                |        `None` | Factor for rescaling contours before detection                      |
| `verbose`               | `bool`                 |        `True` | Prints a detection summary with the parameters used                |
| `plot`                  | `bool`                 |       `False` | Shows detected fruit contours overlaid on the image                |
| `plot_size`             | `tuple[int, int]`      |      `(5, 5)` | Figure size (only if `plot=True`)                                  |
| `contour_color`         | `tuple[int, int, int]` | `(0, 255, 0)` | BGR color for drawing detected fruit contours (only if `plot=True`) |
| `contour_thickness`     | `int`                  |           `2` | Line thickness for drawing detected fruit contours (only if `plot=True`) |


!!! warning "Important"
    **Requires** that a mask exists (`generate_fruit_mask()` at minimum).

<br>

---


### `analyze_morphology`

Extracts morphological metrics from detected fruits along with associated locule and pericarp measurements.

Results are stored in `analyzer.results` as a `ResultsImage` instance (`traitly.fruit_phenotyping.results_image`). This class contains:

* `analyzer.results.morphology_results`: `pd.DataFrame` with morphological metrics for each fruit.
* `analyzer.results.annotated_img`: annotated image for visual inspection.

`analyzer.results` also includes methods for saving results:

```python
analyzer.results.save_all() # Saves the annotated image and the CSV file
analyzer.results.save_csv() # Saves only the CSV
analyzer.results.save_img() # Saves only the image
```
By default, files are saved to the same folder as the input image, using the original filename as a base. The output directory and an alternative base name can be specified with `output_dir='PATH/'` and `base_name='new_name'`, respectively. For more details and additional parameters, see the `ResultsImage` class documentation under **API Reference**.

The annotated image includes a **unique ID for each fruit**, its **locule count**, and highlights the following elements:

* **outer pericarp** contour (green),
* **inner pericarp** contour (yellow),
* **locules** (pink),
* **locule centroid** (yellow),
* **fruit centroid** (blue),
* ***bounding box* rectangle**,
* **major axis** (blue) and **minor axis** (green).

!!! tip ""
    For a detailed description of the calculated traits and visual annotations, see the [Results](results/overview.md) section.

??? note "Notes on contour modes"

    For stamp analysis or fruits with very irregular edges, it may help to try different `contour_mode` values to smooth the contour:

    - **`'raw'`** (default): Uses the original contour without any modifications. Most accurate, but also most sensitive to edge irregularities.
    - **`'hull'`**: Computes the convex hull enclosing the fruit, filling in indentations or dents. Useful when edge irregularities are not part of the fruit's natural morphology (e.g., mechanical damage or shadows) and you want to recover the expected convex shape.
    - **`'approx'`**: Simplifies the contour by reducing the number of vertices, smoothing out small irregularities while preserving the overall shape.
    - **`'ellipse'`**: Fits an ellipse to the fruit contour. Best suited for oval-shaped fruits or when only length and width matter.
    - **`'circle'`**: Fits a circle to the contour. Useful for spherical fruits or when only the equivalent diameter is needed.

    <br>

    Depending on the mode (except `'raw'`), some traits may be fixed by construction. For example:

    - With `'circle'`, fruit circularity will be `1` (perfect circle) for all fruits.
    - With `'ellipse'`, certain shape metrics will be derived from the fitted ellipse rather than the actual contour.


    <div style="text-align: center;">
        <img src="../../assets/images/contours.png" alt="contours" width="800">
        <p><em>Examples of available contours with `contour_mode`</em></p>
    </div>

??? note "Notes on radial rays"
    The `num_rays` parameter controls the number of radial rays cast from the fruit centroid outward. These rays are used to calculate `outer_pericarp_mean_thickness` and `fruit_lobedness`. The angular spacing between rays is `360 / num_rays`.

    Higher values give better resolution for complex or irregular shapes, but also increase computation time. For most fruits, values between 45 and 90 are sufficient. Increase if the fruit has a highly irregular contour or deep lobes. 
    
    !!! tip ""
        For more details on how these traits are calculated, see the [Measurements](results/measurements.md#pericarp-thickness-and-lobedness) section.

    <div style="text-align: center;">
        <img src="../../assets/images/num_rays.png" alt="num_rays" width="400">
        <p><em>Effect of <code>num_rays</code> on ray density. Higher values capture more detail along the fruit contour.</em></p>
    </div>

??? note "Notes on angle_shifts"
    `angle_shifts` controls how many rotational offsets are tested when computing `locules_angular_symmetry`. The algorithm works by comparing the observed locule angles against an ideal equally-spaced arrangement, trying `angle_shifts` different rotations of that ideal to find the best match. A higher value tests more rotations and gives a more accurate alignment, at the cost of computation time.

    The default value of 500 is sufficient for most fruits. Very low values (e.g., below 50) may produce slightly inaccurate results for fruits where the locules are close to but not exactly at ideal positions. 

    !!! tip ""
        For more details on how angular symmetry is calculated, see the [Measurements](results/measurements.md#symmetry-interpretation) section.

```python
analyzer.analyze_morphology(
    contour_mode="hull",
    label_position="bottom",
    label_color=(255,255,0)
)
```

<br>

| Parameter                   | Type                 | Default           | Description                                                                                      |
| --------------------------- | -------------------- | ----------------- | ------------------------------------------------------------------------------------------------ |
| `contour_mode`              | `str`                | `'raw'`           | Contour mode for metrics: `'raw'`, `'hull'`, `'approx'`, `'ellipse'`, `'circle'`                 |
| `epsilon`                   | `float`              | `0.001`           | Approximation factor (only if `contour_mode='approx'`)                                           |
| `angle_shifts`              | `int`                | `500`             | Angular steps used for symmetry metrics                                                          |
| `num_rays`                  | `int`                | `90`              | Number of rays used for pericarp thickness estimation                                            |
| `display_table`             | `bool`               | `True`            | If `True`, returns the `DataFrame` with results                                                  |
| `plot`                      | `bool`               | `True`            | If `True`, displays the annotated image                                                          |
| `plot_size`                 | `tuple[int, int]`    | `(10, 10)`        | Figure size (only if `plot=True`)                                                                |
| `font_size`                 | `float`              | `1.5`             | Text size in the annotation                                                                      |
| `font_thickness`            | `int`                | `2`               | Text thickness in the annotation                                                                 |
| `font_color`                | `tuple[int,int,int]` | `(0, 0, 0)`       | Text color (BGR)                                                                                 |
| `label_position`            | `str`                | `'top'`           | Label position (`'top'`, `'bottom'`, `'left'`, `'right'`)                                        |
| `label_color`               | `tuple[int,int,int]` | `(255, 255, 255)` | Label background color (BGR)                                                                     |
| `pericarp_ext_color`        | `tuple[int,int,int]` | `(0, 240, 0)`     | Outer pericarp contour color (BGR)                                                               |
| `pericarp_ext_thickness`    | `int`                | `2`               | Outer pericarp contour thickness                                                                 |
| `pericarp_int_color`        | `tuple[int,int,int]` | `(0, 240, 240)`   | Inner pericarp contour color (BGR)                                                               |
| `pericarp_int_thickness`    | `int`                | `2`               | Inner pericarp contour thickness                                                                 |
| `locule_color`              | `tuple[int,int,int]` | `(255, 0, 255)`   | Locule contour color (BGR)                                                                       |
| `locule_thickness`          | `int`                | `2`               | Locule contour thickness                                                                         |
| `centroid_fruit_color`      | `tuple[int,int,int]` | `(255, 255, 51)`  | Fruit centroid marker color (BGR)                                                                |
| `centroid_fruit_thickness`  | `int`                | `2`               | Fruit centroid marker size                                                                       |
| `centroid_locule_color`     | `tuple[int,int,int]` | `(0, 255, 255)`   | Locule centroid marker color (BGR)                                                               |
| `centroid_locule_thickness` | `int`                | `2`               | Locule centroid marker size                                                                      |
| `alpha`                     | `float`              | `None`            | Alpha parameter for the concave hull of the inner pericarp contour. Smaller values produce a tighter fit to the actual fruit shape; if `None`, the convex hull is used |


!!! warning "Important"
    **Requires** that a mask exists (`generate_fruit_mask()` at minimum) and that `detect_fruits()` has been run.

<br>

---
### `analyze_color`

Extracts color features from fruit tissue in detected fruits, using the original image and the masks generated throughout the pipeline.

Color extraction always uses the original contours in `'raw'` mode, regardless of the `contour_mode` selected in `analyze_morphology()`. This ensures that the color extraction area accurately reflects the segmented region in the mask, without being affected by geometric simplifications of the contour.

Results are stored in `analyzer.results` as a `ResultsImage` instance (`traitly.fruit_phenotyping.results_image`). This class contains:

* `analyzer.results.color_results`: `pd.DataFrame` with color metrics for each fruit/tissue.
* `analyzer.results.annotated_img`: annotated image for visual inspection of IDs and contours during color extraction.

`analyzer.results` also includes methods for saving results:

```python
analyzer.results.save_all() # Saves the annotated image and the CSV file.
analyzer.results.save_csv() # Saves only the CSV.
analyzer.results.save_img() # Saves only the image.
```

By default, files are saved to the same folder as the input image, using the original filename as a base. The output directory and an alternative base name can be specified with `output_dir='PATH/'` and `base_name='new_name'`, respectively. For more details and additional parameters, see the `ResultsImage` class documentation.

This function extracts color from different fruit tissues: **total pericarp**, **outer pericarp**, **inner pericarp**, and **locules**. To visually inspect how these tissues are segmented, use `generate_single_fruit_masks`. If you don't need all tissues, a specific one can be selected with `tissue='...'`.

<div style="text-align: center;">
    <img src="../../assets/images/internal_tissues.png" alt="Setup with black box" width="900">
    <p><em>Example of tissues from which color is extracted in cranberry slices</em></p>
</div>

!!! tip ""
    For more details about the color extraction, see the [Measurements](results/measurements.md#tissue-regions-and-color-extraction) and [Results](results/overview.md) sections.

??? note "Notes"
    * `analyze_color()` is **independent** of `analyze_morphology()`. Running only `analyze_color()` generates a basic annotated image with the **fruit ID**, its **locule count**, the **fruit contour** (outer pericarp) in green, and the **locule contours** in pink.
    * If `analyze_morphology()` was run beforehand, saving results (e.g., with `save_all()`) will **reuse** the morphology annotated image, since it contains more complete annotations.
    * If `analyze_color()` is run first (without saving) and `analyze_morphology()` is run afterward, saving results will use the annotated image from `analyze_morphology()`.
    * Color extraction always uses the original contours in `'raw'` mode, regardless of the `contour_mode` selected in `analyze_morphology()`. This ensures that the color extraction area accurately reflects the segmented region in the mask, without being affected by geometric simplifications of the contour.
    * By default, the function computes a summary statistic (`'mean'` or `'median'`) per channel and tissue. Alternatively, you can compute per-pixel color histograms by setting `get_color_histogram=True`, which returns full channel distributions instead of a single summary value.


```python
analyzer.analyze_color(
    stat='median',
    tissue='outer_pericarp, locules',
    color_space='hsv, lab',
    plot=False
)
```

<br>

| Parameter                | Type                 |           Default | Description                                                                              |
| ------------------------ | -------------------- | ----------------: | ---------------------------------------------------------------------------------------- |
| `stat`                   | `str`                |          `'mean'` | Statistic: `'mean'` or `'median'` (ignored if `get_color_histogram=True`)                |
| `tissue`                 | `str`                |           `'all'` | Tissue: `'all'`, `'total_pericarp'`, `'outer_pericarp'`, `'internal_pericarp'`, `'locules'` |
| `color_space`            | `str`                |           `'all'` | Color spaces: `'all'`, `'rgb'`, `'lab'`, `'hsv'`, `'gray'`                               |
| `display_table`          | `bool`               |            `True` | If `True`, returns the `DataFrame` with results                                          |
| `plot`                   | `bool`               |           `False` | If `True`, displays the annotated image used for color extraction                        |
| `plot_size`              | `tuple[int, int]`    |        `(10, 10)` | Figure size (only if `plot=True`)                                                        |
| `font_size`              | `int`                |               `2` | Text size in the annotation                                                              |
| `font_thickness`         | `int`                |               `2` | Text thickness in the annotation                                                         |
| `font_color`             | `tuple[int,int,int]` |       `(0, 0, 0)` | Text color (BGR)                                                                         |
| `label_position`         | `str`                |           `'top'` | Label position (`'top'`, `'bottom'`, `'left'`, `'right'`)                                |
| `label_color`            | `tuple[int,int,int]` | `(255, 255, 255)` | Label background color (BGR)                                                             |
| `pericarp_ext_color`     | `tuple[int,int,int]` |     `(0, 255, 0)` | Outer pericarp contour color (BGR)                                                       |
| `pericarp_ext_thickness` | `int`                |               `2` | Outer pericarp contour thickness                                                         |
| `locule_color`           | `tuple[int,int,int]` |   `(255, 0, 255)` | Locule contour color (BGR)                                                               |
| `locule_thickness`       | `int`                |               `2` | Locule contour thickness (BGR)                                                           |
| `pericarp_int_color`     | `tuple[int,int,int]` |   `(255, 255, 0)` | Inner pericarp contour color (BGR)                                                       |
| `pericarp_int_thickness` | `int`                |               `2` | Inner pericarp contour thickness                                                         |
| `label_opacity`          | `float`              |             `0.7` | Label background opacity `[0, 1]`                                                        |
| `get_color_histogram`    | `bool`               |           `False` | If `True`, returns per-pixel histograms instead of summary statistics                    |
| `alpha`                  | `float`              |            `None` | Alpha parameter for the concave hull of the inner pericarp contour. Smaller values produce a tighter fit to the actual fruit shape; if `None`, the convex hull is used |


!!! warning "Important"
    **Requires** that a mask exists (`generate_fruit_mask()` or `generate_locule_mask()`) and that `detect_fruits()` has been run.

<br>

---

### `generate_single_fruit_masks`


*Optional*

Generates and displays tissue masks for a specific fruit, useful for taking a close look at segmentation results.

Lets you see how the different fruit tissues (total pericarp, outer pericarp, inner pericarp, and locules) are segmented from the masks generated earlier in the pipeline. Also helpful for deciding which tissues to include in a subsequent `analyze_color()` call or other analysis steps.

Uses `mask_locules` if available; otherwise falls back to `mask_fruit`. The fruit is cropped to its *bounding box* with an optional margin.

The `fruit_id` parameter corresponds to the fruit identifier in the annotated image or results table, as it appears in the outputs from `analyze_morphology()` or `analyze_color()`.

```python
# Show overlaid masks for fruit 10
analyzer.generate_single_fruit_masks(fruit_id=10, overlay=True)
```
<br>

| Parameter        | Type              |  Default | Description                                                                    |
| ---------------- | ----------------- | -------: | ------------------------------------------------------------------------------ |
| `fruit_id`       | `int`             |   `None` | ID of the fruit to visualize; uses the first detected fruit if `None`          |
| `plot_size`      | `tuple[int, int]` | `(7, 5)` | Figure size                                                                    |
| `overlay`        | `bool`            |  `False` | Overlays masks on top of the original image                                    |
| `overlay_legend` | `bool`            |  `False` | Includes a legend in the overlay (only if `overlay=True`)                      |
| `margin`         | `int`             |      `5` | Margin (px) around the fruit crop                                              |


!!! warning "Important"
    **Requires** that a mask exists and that `detect_fruits()` has been run.

 
<br>

---

### `save_parameters`

*Optional*

Exports the **analysis parameters from the current session** in both `.txt` and `.json` format, ready for review, reuse, and reproducibility.

The parameters stored in `analyzer.parameters` are exported using the loaded image's name as a base, automatically generating two files:

* `<image_name>_parameters.txt`: human-readable version for inspection.
* `<image_name>_parameters.json`: structured version for programmatic use.

Both are saved by default to the same folder as the input image, or to the directory specified by `output_path`. They are particularly useful for:

* reusing configurations in batch analysis with `analyze_folder()`,
* running reproducible analyses from the command line with Traitly,
* archiving and sharing analysis pipelines.


??? note "Notes"
    * Only parameters from functions that were actually run during the session are exported (mask, segmentation, detection, morphology, color).
    * Returns nothing; prints the paths of the generated files to the console.

```python
analyzer.save_parameters()
```

<br>

| Parameter     | Type  | Default | Description                                                                                             |
| ------------- | ----- | ------- | ------------------------------------------------------------------------------------------------------- |
| `output_path` | `str` | `None`  | Output directory. If `None`, uses the same directory as the input image. |

<br>

---

### `plot_image`

*Optional*

Displays either the original image or the **results-annotated image**, depending on the value of `annotated`, reusing images already in memory without reloading or regenerating them.

* When `annotated=False`, shows the **original loaded image**.
* When `annotated=True`, shows the **annotated image** produced by `analyze_morphology()` or `analyze_color()`.

The annotated image corresponds to the one stored in `analyzer.results.annotated_img` and includes fruit identifiers and visual annotations generated during the analysis.

```python
analyzer.plot_image(annotated=True)
```

<br>

| Parameter   | Type              | Default    | Description                                                                  |
| ----------- | ----------------- | ---------- | ---------------------------------------------------------------------------- |
| `annotated` | `bool`            | `True`     | If `True`, shows the annotated image; if `False`, shows the original image   |
| `plot_size` | `tuple[int, int]` | `(10, 10)` | Figure size                                                                  |


<br>

---

### `analyze_folder`

Processes in batch all images in the folder specified when initializing `FruitInternalAnalyzer`, running the full pipeline on each image either sequentially (`num_cores=1`) or in parallel (`num_cores` > 1). By default, both morphological and color analysis are run; each can be disabled independently with `analyze_morphology=False` or `analyze_color=False`.

For each image processed, an **annotated image** is generated with identifiers and visual analysis annotations. Results from all images are consolidated into a single CSV file per analysis type:

* `morphology_results.csv`: morphological metrics for all detected fruits.
* `color_results.csv`: color metrics for all detected fruits.

A `session_report.txt` is always generated with a session summary (images processed, fruits detected, timing, parameters used, and dependencies). If any image fails during processing, an `error_report.txt` is also generated detailing what went wrong in each case.

All files are saved to the directory specified by `output_path`. If not provided, files are saved to a `Results/` subfolder inside the input folder.

??? note "Note"
    This function accepts all pipeline step parameters individually. However, for convenience and reproducibility, we recommend fine-tuning parameters on a representative image with `save_parameters()`, then passing the generated `.json` file via `json_path`.

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

<br>


| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `analyze_morphology` | `bool` | `True` | If `True`, runs morphological analysis on each image |
| `analyze_color` | `bool` | `True` | If `True`, runs color analysis on each image |
| `json_path` | `str` | `None` | Path to a `.json` parameters file generated by `save_parameters()` |
| `config` | `dict` | `None` | Base configuration as a dictionary; individual parameters take priority |
| `output_path` | `str` | `None` | Output directory. If `None`, a `Results/` subfolder is created inside the input folder |
| `num_cores` | `int` | `1` | Number of parallel processes. Automatically capped at available cores |
| `verbose` | `bool` | `True` | If `True`, prints progress and session summary |
| `width_cm` | `float` | `None` | Known image width in cm -> `setup_measurements` |
| `length_cm` | `float` | `None` | Known image length in cm -> `setup_measurements` |
| `diameter_cm` | `float` | `None` | Known reference diameter in cm -> `setup_measurements` |
| `fast_calibration` | `bool` | `None` | If `True`, skips YOLO and calibrates with physical dimensions -> `setup_measurements` |
| `skip_qr` | `bool` | `None` | If `True`, skips QR detection -> `setup_measurements` |
| `detect_label` | `bool` | `None` | If `True`, enables label detection with OCR -> `setup_measurements` |
| `confidence` | `float` | `None` | Minimum confidence for YOLO detection -> `setup_measurements` |
| `detect_color_checker` | `bool` | `None` | If `True`, detects and removes color checker -> `setup_measurements` |
| `scale_factor` | `float` | `None` | Downscaling factor for color checker detection -> `setup_measurements` |
| `lower_hsv` | `list[int]` | `None` | Lower HSV threshold for segmentation -> `generate_fruit_mask` |
| `upper_hsv` | `list[int]` | `None` | Upper HSV threshold for segmentation -> `generate_fruit_mask` |
| `background_color` | `str` | `None` | Preset background color -> `generate_fruit_mask` |
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
| `contrast_method` | `str` | `None` | Contrast enhancement method -> `enhance_locule_contrast` |
| `gamma` | `float` | `None` | Gamma exponent -> `enhance_locule_contrast` |
| `gain` | `float` | `None` | Sigmoid gain -> `enhance_locule_contrast` |
| `cutoff` | `float` | `None` | Sigmoid cutoff -> `enhance_locule_contrast` |
| `c` | `float` | `None` | Exponential factor -> `enhance_locule_contrast` |
| `kernel_blur_contrast` | `int` | `None` | Blur before contrast enhancement -> `enhance_locule_contrast` |
| `clip_limit` | `int` | `None` | CLAHE limit -> `enhance_locule_contrast` |
| `tile_grid_size` | `int` | `None` | CLAHE grid size -> `enhance_locule_contrast` |
| `thresh_min` | `int` | `None` | Minimum L channel binarization threshold -> `generate_locule_mask` |
| `thresh_max` | `int` | `None` | Maximum L channel binarization threshold -> `generate_locule_mask` |
| `min_fruit_area_locule` | `int` | `None` | Minimum fruit area during mask merging -> `generate_locule_mask` |
| `kernel_close_locule` | `int` | `None` | Closing kernel for locule mask -> `generate_locule_mask` |
| `kernel_open_locule` | `int` | `None` | Opening kernel for locule mask -> `generate_locule_mask` |
| `invert_locule` | `bool` | `None` | If `True`, inverts the locule mask -> `generate_locule_mask` |
| `min_fruit_area` | `int` | `None` | Minimum area to accept a contour as a fruit -> `detect_fruits` |
| `max_fruit_area` | `int` | `None` | Maximum area to accept a contour as a fruit -> `detect_fruits` |
| `min_fruit_circularity` | `float` | `None` | Minimum circularity to accept a fruit -> `detect_fruits` |
| `min_locule_area` | `int` | `None` | Minimum locule area -> `detect_fruits` |
| `min_locule_per_fruit` | `int` | `None` | Minimum number of locules per fruit -> `detect_fruits` |
| `rescale_factor` | `float` | `None` | Contour rescaling factor -> `detect_fruits` |
| `contour_mode` | `str` | `None` | Contour mode for morphological metrics -> `analyze_morphology` |
| `epsilon` | `float` | `None` | Contour approximation factor -> `analyze_morphology` |
| `min_locule_area_morph` | `int` | `None` | Minimum locule area for morphology -> `analyze_morphology` |
| `max_locule_area` | `int` | `None` | Maximum locule area -> `analyze_morphology` |
| `angle_shifts` | `int` | `None` | Angular steps for symmetry -> `analyze_morphology` |
| `num_rays` | `int` | `None` | Rays for pericarp thickness estimation -> `analyze_morphology` |
| `alpha` | `float` | `None` | Alpha parameter for the inner pericarp concave hull -> `analyze_morphology`, `analyze_color` |
| `stat` | `str` | `None` | Color statistic: `'mean'` or `'median'` -> `analyze_color` |
| `tissue` | `str` | `None` | Tissue to analyze -> `analyze_color` |
| `color_space` | `str` | `None` | Color spaces to extract -> `analyze_color` |
| `label_opacity` | `float` | `None` | Label background opacity `[0, 1]` -> `analyze_color` |
| `pericarp_int_color` | `tuple[int,int,int]` | `None` | Inner pericarp contour color (BGR) -> `analyze_color` |
| `pericarp_int_thickness` | `int` | `None` | Inner pericarp contour thickness -> `analyze_color` |
| `get_color_histogram` | `bool` | `None` | If `True`, computes per-pixel histograms -> `analyze_color` |


!!! warning "Important"
    **Requires** that `FruitInternalAnalyzer()` was initialized with a folder path, not a file path.
