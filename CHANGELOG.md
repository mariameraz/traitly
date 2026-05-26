# Changelog

*All notable changes to Traitly are documented here:*

## v.0.1.3 – Unrelease

### Fixed

- In `setup_label`:
    - If a QR code was detected, the ROI detection step was skipped, and `label_roi = None`
    - Now, label ROI detection runs independently of QR detection

### Changed

- Encapsulate attributes only relevant for internal processing in  `FruitExternalAnalyzer` and `FruitInternalAnalyzer` for cleaner user interface. 
- Move `detect_color_checker` from `FruitInternalAnalyzer` to dedicated `color_correction` module
- `setup_measurements` no longer accepts `detect_color_checker` and `scale_factor` arguments.
  Use `detect_color_checker()` method instead.

---

## v0.1.2 – 2026-05-18

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

Documentation: https://traitly.readthedocs.io/en/v0.1.2/

----

## v0.1.1 – 2026-05-04

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

Documentation: https://traitly.readthedocs.io/en/v0.1.1/

----

## v0.1.0 — 2026-04-07

Initial release.

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

Documentation: https://traitly.readthedocs.io/en/v0.1.0/
