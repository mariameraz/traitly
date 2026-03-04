# Quickstart

*Get a complete analysis running in minutes.*

**Requirements:** Traitly installed ([Installation guide](../installation.md)) and a scanned cross-section image of a fruit.

---

## Internal analysis

Use `FruitInternalAnalyzer` when your images contain visible locules (internal structure).

```python
from traitly.fruit_phenotyping import FruitInternalAnalyzer

analyzer = FruitInternalAnalyzer("my_image.jpg")
analyzer.load_image()
analyzer.setup_measurements()
analyzer.generate_fruit_mask()
analyzer.detect_fruits()

# Morphology traits → returns a DataFrame
df = analyzer.analyze_morphology()

# Color traits (optional)
df_color = analyzer.analyze_color()
```

---

## External analysis

Use `FruitExternalAnalyzer` for whole-fruit analysis without locule segmentation.

```python
from traitly.fruit_phenotyping import FruitExternalAnalyzer

analyzer = FruitExternalAnalyzer("my_image.jpg")
analyzer.load_image()
analyzer.setup_measurements()
analyzer.generate_fruit_mask()
analyzer.detect_fruits()

df = analyzer.analyze_morphology()
```

---

## Batch processing

Process an entire folder of images automatically.

```python
analyzer.analyze_folder(
    folder_path="my_images/",
    output_path="results/",
    analyze_morphology=True,
    analyze_color=True
)
```

This generates in `results/`:

- `morphology_results.csv`
- `color_results.csv` *(if `analyze_color=True`)*
- `*_annotated.jpg` for each image
- `session_report.txt`
- `error_report.txt` *(if any image failed)*

---

## What's next?

- [Internal Analysis tutorial](internal.md) — detailed walkthrough with all parameters
- [External Analysis tutorial](external.md) — whole-fruit analysis step by step
- [API Reference](../api/internal_analysis.md) — full parameter documentation
- [Trait Table](../traits.md) — what each CSV column means
