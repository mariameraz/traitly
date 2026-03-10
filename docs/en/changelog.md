# Changelog ⊹ ࣪ ˖

All notable changes to Traitly are documented here.

---

## v0.1.0 — February 2026

Initial release.

### Added
- `FruitInternalAnalyzer` — internal fruit analysis with locule segmentation
- `FruitExternalAnalyzer` — whole-fruit analysis without locule segmentation
- Batch folder processing with optional multiprocessing (`analyze_folder`)
- Pixel-to-metric calibration via reference markers
- QR code and text label detection
- Morphology traits: area, perimeter, axes, shape indices, pericarp, symmetry
- Color traits: RGB, HSV, Lab channels per tissue region
- Annotated image output
- Session report and error report for batch processing
- `AnalysisParameters` dataclass for reproducibility tracking
