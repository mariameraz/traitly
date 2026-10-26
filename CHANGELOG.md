# Changelog

*All notable changes to Traitly are documented here:*

## v0.3.0 - 2026-10-25

Documentation: [https://traitly.readthedocs.io/en/v0.3.0/](https://traitly.readthedocs.io/en/v0.3.0/)

### Changed

- `detect_qr()` now runs internally the WeChat QR detector without the bundled Caffe models. Only its traditional detector is used (no CNN detector), with bicubic resizing instead of the super-resolution model. This avoids model format incompatibilities between OpenCV 5.x (ONNX) and earlier versions (Caffe) (more details: [opencv#29712](https://github.com/opencv/opencv/issues/29712#issuecomment-6406219990)). The traditional detector is sufficient for controlled images of fruits most of the time. The fallback to `cv2.QRCodeDetector` and then to `detectAndDecodeCurved` is unchanged.

- Removed the `min_aspect_ratio` and `max_aspect_ratio` thresholds from `find_fruits`, and the aspect ratio filter that `detect_fruits` applied internally, to prevent very oval contours from being discarded.

- Optimize `angular_symmetry()`, replacing the Hungarian algorithm and the 500 sampled rotations per fruit with an exact search that derives the candidate rotations from the `n` locule deviations. The new method is faster and returns the exact minimum. Expressed on the same scale, results are very similar to those of the previous method, with differences below 0.001.

#### *Breaking:*

- `angular_symmetry()` now returns the mean angular error as a fraction of the ideal spacing (`2π/n` dimensionless) instead of radians. This makes values comparable across fruits with different locule counts. All values change relative to previous versions, so they should not be compared **directly** with results from earlier versions. Values obtained with the old method (version < 0.3.0) can be converted to the new scale with `old_value / (2π / n_locules)`.

- Remove the `angle_shifts` argument from `angular_symmetry()`, since the new method does not need a fixed number of rotations.

### Fixed

- In `FruitInternalAnalyzer`, `detect_color_checker` now draws on `_img_copy` instead of replacing it. Now, if it runs after `setup_measurements()`, the size-reference annotation is kept in the final image used by `analyze_morphology` and `analyze_color`.
- `get_easyocr_reader()` now passes the `gpu` argument to EasyOCR. Previously, `gpu=False` was ignored and EasyOCR chose its own device, so it could silently use a GPU. The default `gpu=False` now forces CPU. With `gpu=True`, the GPU is used, falling back to CPU with a logged warning if none is available.

### Added

- Add `background_color="gray"` to the predefined color thresholds in `generate_fruit_mask()` for both `FruitInternalAnalyzer` and `FruitExternalAnalyzer`
- Allow configurable scalers in color correction by accepting scaler names as strings: `'minmax'`, `'robust'`, `'maxabs'` and '`standard'`

## v0.2.2 - 2026-10-06

Quick patch release.

Documentation: unchanged from [v0.2.0](https://traitly.readthedocs.io/en/v0.2.0/)


### Fixed
- Include bundled YOLO model weights (`size_reference.pt`, `label.pt`) in the wheel and sdist. They were excluded by `*.pt` in `.gitignore`, so PyPI installs could not load them and measurements silently fell back to pixels.
    - Affected versions: `0.2.0` and `0.2.1`. Upgrading to `0.2.2` is recommended.


## v0.2.1 - 2026-10-06

Documentation: unchanged from [v0.2.0](https://traitly.readthedocs.io/en/v0.2.0/)

### Fixed
- `apply_color_correction` raised an error because `PolynomialFeatures` was not imported, which is required for the regression step.


## v0.2.0 - 2026-10-05

Documentation: [https://traitly.readthedocs.io/en/v0.2.0/](https://traitly.readthedocs.io/en/v0.2.0/)

### Fixed

- In `setup_label` from both `FruitInternalAnalyzer` and `FruitExternalAnalyzer`:
    - If a QR code was detected, the label ROI detection step was skipped, and `label_roi = None`
    - Now, label ROI detection runs independently of QR detection
- Fix `outer_pericarp_mean_thickness`: the internal/outer pericarp boundary was previously set at the first 255-valued pixel on the ray, which, since internal pericarp is also 255, always landed at the centroid, therefore measuring the fruit's radius, not the pericarp. Now the boundary uses the last 255-valued pixel, correctly marking the internal-> outer pericarp transition.
- Fix memory leak during batch analysis: input images were previously cached via an LRU cache, keeping every processed image in RAM. This caused memory usage to grow uncontrollably when analyzing large batches. Removed the cache entirely, so each image is now loaded, processed, and released independently per worker.
- Fix `mcc` and `wechat_qrcode` module incompatibilities with OpenCV >= 5.0:
    - `cv2.mcc.CCheckerDetector.process()` moved the chart type argument to a new `setColorChartType()` method; add version detection to call the correct API depending on availability
    - `cv2.mcc.CCheckerDraw` was removed; the draw method moved onto `CCheckerDetector` itself; add fallback accordingly
    - `cv2.wechat_qrcode_WeChatQRCode`'s legacy constructor no longer accepts custom Caffe model paths; add fallback to the new built-in WeChat detector, then to `cv2.QRCodeDetector`
- Remove Otsu binarization before the classic `cv2.QRCodeDetector` fallback, which was causing QR codes to go undetected on labels


### Changed
- Encapsulate attributes only relevant for internal processing in  `FruitExternalAnalyzer` and `FruitInternalAnalyzer` for cleaner user interface. 

#### *Breaking:*
- Drop support for Python 3.9; Python 3.10+ is now required.
- CLI mode selection changed from mutually exclusive flags to subcommands:
  - Old: `traitly --fruit_internal -i PATH` / `traitly --fruit_external -i PATH`
  - New: `traitly fruit_internal -i PATH` / `traitly fruit_external -i PATH`
- Includes a dedicated method (`detect_color_checker()`) on `FruitInternalAnalyzer` and `FruitExternalAnalyzer` to detect color cards on an image. Therefore, `setup_measurements()` no longer accepts `detect_color_checker` and `scale_factor` arguments.

### Added
- New `traitly info` CLI command to print package, system, and dependency metadata
- New `traitly.utils.metadata` module with `get_package_versions()` to retrieve installed versions of all package dependencies and Python version.
- New `traitly.color_correction` module with `ColorCorrection` class for color-correcting images or entire folders using a Macbeth Color Checker (24 patches) card.

---

## v0.1.2 – 2026-05-18

Documentation: [https://traitly.readthedocs.io/en/v0.1.2/](https://traitly.readthedocs.io/en/v0.1.2/)

### Fixed

- Fix verbose output for `edit_mask` in terminal (previously only worked in Jupyter) (reported by @AlvaroGuerrero)
- Add `IPython` dependency required to open interactive windows with `edit_mask` on CLI (reported by @AlvaroGuerrero)
- Fix crash in `annotate_all_fruits` when fruits have no detected locules — `get_internal_pericarp_contour` returned `None` and was passed directly to `cv2.drawContours`
- Fix crash in `detect_color_checker` when `cv2.mcc.CCheckerDetector` is not available
- Fix SSL certificate error when easyocr tries to download models on first use
- Fix hardcoded version on `cli.py`
- Patch in `_load_img_cached` which raises `FileNotFoundError` instead of returning `None` on Windows (ref. upstream bug: [ultralytics#24405](https://github.com/ultralytics/ultralytics/issues/24405))
- Fix rename duplicate image names issue with `pdf_to_img` when multiple PDF pages share the same QR code:
  - Only the first image was renamed 
  - Files are now named `<qr_text>.jpg`, `<qr_text>_1.jpg`, `<qr_text>_2.jpg`, etc.

### Added
- Improve QR detection with two additions:
  - Add `cv2.wechat_qrcode_WeChatQRCode` detector as primary method for more robust detection of small QR codes
  - Add `cv2.detectAndDecodeCurved` as fallback when standard `cv2.detectAndDecode` fails



## v0.1.1 – 2026-05-04

Documentation: [https://traitly.readthedocs.io/en/v0.1.1/](https://traitly.readthedocs.io/en/v0.1.1/)

### Fixed
- Rename `fast_calibration` to `skip_yolo` in JSON example files to match code parameters (reported by @Hector-LM)
- Shiny App:
	- Fix broken image example path in main page
	- Fix sidebar pipeline steps resetting after returning from another tab

### Changed
- Standardize `min_fruit_area` default value to 1000 $px^2$ across all classes (reported by @Hector-LM)
- Show total session time in seconds or minutes depending on duration in batch analysis reports.
- Move `convert_pdf` from `utils` to a dedicated `pdf` module:
  - Old import: `from traitly.utils.convert_pdf import pdf_to_img`
  - New import: `from traitly.pdf import pdf_to_img`
- Rename optional dependency `traitly[all]` to `traitly[app]`
- Shiny App:
	- Optimize morphology and color exports by removing temporary disk writes and using in-memory processing
	- Improve batch export memory usage by writing ZIP files to disk instead of keeping them in RAM
	- Use managed temporary directories for batch and PDF processing with proper cleanup 

### Added
- Add `erosion_px` parameter in `analyze_folder()` for both `FruitInternalAnalyzer` and `FruitExternalAnalyzer` classes

### Docs
- Pin dependency versions


## v0.1.0 — 2026-04-07

Initial release.

Documentation: [https://traitly.readthedocs.io/en/v0.1.0/](https://traitly.readthedocs.io/en/v0.1.0/)

### Features
- Internal fruit, locule, and stamp analysis with `FruitInternalAnalyzer` 
- Whole fruit morphology and color analysis with `FruitExternalAnalyzer`
- Batch processing with optional multiprocessing (`analyze_folder`)
- Pixel-to-centimeter conversion using size references
- QR code, text label, and color checker detection
- Command-line interface (`traitly`)
- Interactive web application (`traitly-app`)

### Measurements
- Morphological traits: area, perimeter, axes, shape indices, pericarp thickness, symmetry
- Color traits: RGB, HSV, Lab, and Grayscale per tissue region

### Outputs
- Annotated images, CSV results, session and error reports, and parameter files
