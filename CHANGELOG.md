# Changelog

## v0.1.1 – 2026-05-04

### Fixed
- Rename `fast_calibration` to `skip_yolo` in JSON example files to match code parameters
- Shiny App:
	- Fix broken image example path in main page
	- Fix sidebar pipeline steps resetting after returning from another tab

### Changed
- Standarize `min_fruit_area` default value to 1000 $px^2$across all classes
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
