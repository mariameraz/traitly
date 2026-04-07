# Changelog

## v0.1.0 — April 7, 2026

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
