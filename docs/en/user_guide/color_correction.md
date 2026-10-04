<div class="animate" markdown>

# Color Correction: Class and Methods

This section covers everything you need to work with `ColorCorrection`, the class used to correct the colors of your images with a color checker card. Each method is explained along with its parameters and how to fit it into your workflow.

---

## 1. Main Class

`ColorCorrection` adjusts the colors of an image so they match the known reference values of a 24-patch color checker card placed in the same photo. This reduces differences caused by lighting, camera, or settings, making color measurements comparable across images and sessions. You can use it to correct a single image or an entire folder.

```python
from traitly.color_correction import ColorCorrection

# To correct a single image
cc = ColorCorrection(path = "path/to/my/image.jpg")

# To correct multiple images in a folder
cc = ColorCorrection(path = "path/to/my/folder/with/images/")
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | Path to the image or folder that contains the images you want to correct |

!!! info "What you need"
    - Every image must include a **24-patch color checker card** (X-Rite ColorChecker Classic, also known as Macbeth chart), fully visible.
    - Only one card is detected per image.
    - Card detection relies on the `mcc` module of `opencv-contrib-python`, which is not available in the [Intel Mac / older macOS setup](../installation.md). On those systems, **color correction is not supported**.

!!! tip "Recommendation"
    When you have a folder of images to process, we suggest:

    1. **Start with a representative image** to check that the card is detected and the result looks right
    2. Evaluate the correction with `calculate_delta_e_stats()`
    3. Save the configuration with `save_parameters()`
    4. Use `analyze_folder(json_path="your_file.json")` to correct the whole batch with the same parameters

    Once your corrected images are saved, you can analyze them with [`FruitInternalAnalyzer`](internal_class.md) or [`FruitExternalAnalyzer`](external_class.md).

### How the correction works

The card contains 24 color patches whose [colors reference under daylight](https://www.xrite.com/service-support/new_color_specifications_for_colorchecker_sg_and_classic_charts) conditions is available. The class measures the color of each patch in your image, compares it with its reference value, and builds a model that describes how your camera and lighting changed those colors. That model is then applied to **every pixel of the image**.

<br>

</div>

---

## 2. How the Correction Is Organized

When working with `ColorCorrection`, the process follows this logical order:

```python
from traitly.color_correction import ColorCorrection

# Correct a single image
cc = ColorCorrection('path/to/my/image.jpg')

cc.load_image()                 # Load the image
cc.detect_color_checker()       # Find the color card in the image
cc.apply_color_correction()     # Correct the colors of the whole image
cc.calculate_delta_e_stats()    # (Optional) Measure how much the colors improved

## Save results
cc.save_img()                   # Save the corrected image
cc.save_csv()                   # (Optional) Save the color error per patch
cc.save_parameters()            # (Optional) Save the parameters used in the session
```

If you're working with batches of images, you don't need to run each step individually — `analyze_folder()` handles everything automatically:

```python
# Correct multiple images
cc = ColorCorrection('path/to/my/folder')                    # Initialize with your folder path
cc.analyze_folder(json_path = 'path/to/my/parameters.json')  # Run the correction, optionally using saved parameters
```

<br>

---

## 3. What You Can Get from the Corrector

After running the methods, `cc` stores results in attributes you can inspect:

| Attribute | Contents |
|-----------|----------|
| `input_path` | Path of the image (or folder) being processed |
| `original_img` | Original image in BGR color format |
| `corrected_img` | Corrected image in BGR color format (available after `apply_color_correction()`) |

<br>

---

## 4. Methods

!!! example ""
    All the methods include default values for the parameters, so you can start simple and adjust as needed.

### `load_image`

Loads the image you want to correct.

```python
cc.load_image()
cc.load_image(plot=True, plot_size=(8, 8), show_axis=True)
```

<br>

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `plot` | `bool` | `True` | Displays the loaded image |
| `plot_size` | `tuple[int, int]` | `(5, 5)` | Figure size |
| `show_axis` | `bool` | `False` | Shows axis ticks on the plot |

<br>

---

### `detect_color_checker`

Finds the color card in the image and reads the color of each of its 24 patches. These measured colors are what the correction is built from.

??? note "Notes"

    - If the card is not detected, check that it is fully visible, in focus, and free of strong glare or shadows.

    - Only the first card found is used if more than one appears in the image.

```python
# Detect the card and display it
cc.detect_color_checker(plot=True)

# Run silently
cc.detect_color_checker(verbose=False)
```

<br>

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `plot` | `bool` | `False` | If `True`, displays a cropped view of the detected card |
| `plot_size` | `tuple[int, int]` | `(5, 5)` | Figure size (only if `plot=True`) |
| `verbose` | `bool` | `True` | If `True`, prints the detection result and the card coordinates |

!!! warning "Important"
    **Requires** that `load_image()` has been run beforehand.

<br>

---

### `apply_color_correction`

Corrects the colors of the whole image using the color card detected by `detect_color_checker()`. The corrected image is stored in `cc.corrected_img`.

The defaults work well in most cases. If you change `degree` or `num_components`, keep in mind that the model is built from only 24 patches: more complex settings can make the card look perfect while distorting the rest of the image. When adjusting them, always check the result visually and with `calculate_delta_e_stats()`.

??? note "Choosing `degree` and `num_components`"

    - `degree` controls how flexible the correction is. Higher values can fix more complex color shifts, but also increase the risk of over-fitting to the card.

    - `num_components` cannot be larger than the number of terms generated by `degree`: **10** for `degree=2` and **20** for `degree=3`. With the default `num_components=11`, you must use `degree=3` or higher; if you use `degree=2`, lower `num_components` to 10 or less.

```python
# Default correction
cc.apply_color_correction()

# Without plots or console messages
cc.apply_color_correction(plot=False, verbose=False)

# Simpler correction
cc.apply_color_correction(degree=2, num_components=8)
```

<br>

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `degree` | `int` | `3` | Flexibility of the correction; higher values allow more complex corrections |
| `num_components` | `int` | `11` | Number of components used by the model; cannot exceed 10 when `degree=2` or 20 when `degree=3` |
| `max_iterations` | `int` | `1000` | Maximum number of iterations used when fitting the model |
| `plot` | `bool` | `True` | If `True`, shows the original and corrected images side by side |
| `plot_size` | `tuple[int, int]` | `(8, 5)` | Figure size (only if `plot=True`) |
| `verbose` | `bool` | `True` | If `True`, prints progress messages |

!!! warning "Important"
    **Requires** that `load_image()` and `detect_color_checker()` have been run beforehand. Large images can take a few seconds to process.

<br>

---

### `calculate_delta_e_stats`

*Optional*

Measures how close the colors of the card are to their reference values, before and after the correction. The measurement is the **color difference ΔE (CIE 2000)**: lower values mean a better match, and 0 is a perfect match.

The result is calculated for each of the 24 patches and stored for export with `save_csv()`. With `verbose=True`, a summary table with the mean ΔE before and after, and the improvement per patch, is printed to the console.

??? note "Notes"
    - The card must also be detected in the **corrected** image to compute the "after" values. If the correction is very strong and the card can no longer be found, this step will fail.

    - Use it to compare different settings of `apply_color_correction()` on the same image.

```python
cc.calculate_delta_e_stats(verbose=True)
```

<br>

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `verbose` | `bool` | `False` | If `True`, prints the mean ΔE and a per-patch table to the console |

!!! warning "Important"
    **Requires** that `apply_color_correction()` has been run beforehand.

<br>

---

### `save_img`

Saves the corrected image. By default, the file is saved in the same folder as the input image, using the original filename plus the suffix `_corrected`.

```python
cc.save_img()
cc.save_img(output_path='path/to/output/', base_name='my_image', format='png')
```

<br>

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_path` | `str` | `None` | Output directory. If `None`, uses the same directory as the input image |
| `base_name` | `str` | `None` | Base name for the file. If `None`, uses the original filename |
| `format` | `str` | `'png'` | Image format of the saved file |
| `quality` | `int` | `100` | Image quality; relevant for compressed formats such as JPEG |
| `verbose` | `bool` | `True` | If `True`, prints the path of the saved file |

!!! warning "Important"
    **Requires** that `apply_color_correction()` has been run beforehand.

<br>

---

### `save_csv`

*Optional*

Saves the color differences (ΔE) of each patch to a CSV file. By default, the file is saved in the same folder as the input image, using the original filename plus the suffix `_delta_e_stats`.

The file contains one row per patch with the following columns:

| Column | Description |
|--------|-------------|
| `Patch` | Position of the patch on the card (e.g., `A1`) |
| `Color` | Name of the patch (e.g., `dark skin`) |
| `DeltaE_Before` | Color difference with the reference before the correction |
| `DeltaE_After` | Color difference with the reference after the correction |
| `DeltaE_Improvement` | `DeltaE_Before` minus `DeltaE_After`; positive values mean the color got closer to the reference |

```python
cc.save_csv()
cc.save_csv(output_path='path/to/output/', sep=';')
```

<br>

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_path` | `str` | `None` | Output directory. If `None`, uses the same directory as the input image |
| `base_name` | `str` | `None` | Base name for the file. If `None`, uses the original filename |
| `sep` | `str` | `','` | Column separator |
| `verbose` | `bool` | `True` | If `True`, prints a message if there are no results to save |

!!! warning "Important"
    **Requires** that `calculate_delta_e_stats()` has been run beforehand.

<br>

---

### `save_parameters`

*Optional*

Exports the **correction parameters from the current session** in both `.txt` and `.json` format, ready for review, reuse, and reproducibility.

* `<image_name>_parameters.txt`: human-readable version for inspection.
* `<image_name>_parameters.json`: structured version for programmatic use.

Both are saved by default to the same folder as the input image, or to the directory specified by `output_path`. The `.json` file can be reused in batch processing with `analyze_folder(json_path=...)`.

```python
cc.save_parameters()
```

<br>

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_path` | `str` | `None` | Output directory. If `None`, uses the same directory as the input image |

!!! warning "Important"
    **Requires** that `apply_color_correction()` has been run beforehand.

<br>

---

### `analyze_folder`

Corrects in batch all images in the folder specified when initializing `ColorCorrection`, either sequentially (`num_cores=1`) or in parallel (`num_cores` > 1). Each image is corrected using **its own** color card, so every image in the folder must include one.

For each image, a corrected copy is saved with the suffix `_corrected`. When `delta_e=True`, the color differences of all images are consolidated into a single file:

* `delta_e_results.csv`: ΔE per patch, before and after the correction, for all images.

A `session_report.txt` is always generated with a session summary (images processed, timing, and parameters used). If any image fails during processing (for example, because the card was not detected), an `error_report.txt` is also generated detailing what went wrong in each case.

All files are saved to the directory specified by `output_path`. If not provided, files are saved to a `Results/` subfolder inside the input folder.

??? note "Note"
    For convenience and reproducibility, we recommend testing the parameters on a representative image, saving them with `save_parameters()`, and then passing the generated `.json` file via `json_path`.

```python
# Using default parameters
cc.analyze_folder()

# Using individual parameters
cc.analyze_folder(num_cores=4, delta_e=False)

# Using a saved parameters file
cc.analyze_folder(json_path="image_parameters.json")
```

<br>

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `delta_e` | `bool` | `True` | If `True`, calculates the color differences (ΔE) for each image and saves them in `delta_e_results.csv` |
| `json_path` | `str` | `None` | Path to a `.json` parameters file generated by `save_parameters()` |
| `config` | `dict` | `None` | Base configuration as a dictionary; individual parameters take priority |
| `output_path` | `str` | `None` | Output directory. If `None`, a `Results/` subfolder is created inside the input folder |
| `num_cores` | `int` | `1` | Number of parallel processes. Automatically capped at available cores |
| `verbose` | `bool` | `True` | If `True`, prints progress and session summary |
| `degree` | `int` | `None` | Flexibility of the correction -> `apply_color_correction` |
| `num_components` | `int` | `None` | Number of components used by the model -> `apply_color_correction` |
| `max_iterations` | `int` | `None` | Maximum number of iterations for fitting the model -> `apply_color_correction` |

!!! warning "Important"
    **Requires** that `ColorCorrection()` was initialized with a folder path, not a file path.
