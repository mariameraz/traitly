---
hide:
  - navigation
  - toc
---

<div style="text-align: center;" markdown>

# Quick Start: complete cranberry analysis running in minutes.

<p style="color:gray; margin-top: -35px; margin-bottom: 55px;" markdown>**Traitly v0.1.0 – March, 2026**</p>

</div>

**Requirements:** Traitly installed ([Installation Guide](../installation.md)).

!!! tip "Sample images"
    :fontawesome-solid-file-code: The examples in this tutorial use **cranberry** images. If you don't have your own images, you can download the sample images [here](https://github.com/mariameraz/traitly/tree/main/tutorials_data/images). For fruits with a more complex internal structure (e.g., tomato, orange, or cucumber), explore the [Locule Segmentation](segmentate_locules.md) tutorial.

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

# Morphological and color analysis
analyzer.analyze_morphology()
analyzer.analyze_color()

# Save results → returns a CSV and an annotated image
analyzer.results.save_all()

# Optionally, save the parameters used in the session
analyzer.save_parameters()
```

---

## External analysis

Use `FruitExternalAnalyzer` for whole-fruit analysis (without locule or other internal structure segmentation).

```python
from traitly.fruit_phenotyping import FruitExternalAnalyzer

analyzer = FruitExternalAnalyzer("my_image.jpg")
analyzer.load_image()
analyzer.setup_measurements()
analyzer.generate_fruit_mask()
analyzer.detect_fruits()

# Morphological and color analysis
analyzer.analyze_morphology()
analyzer.analyze_color()

# Save results → returns a CSV and an annotated image
analyzer.results.save_all()

# Optionally, save the parameters used in the session
analyzer.save_parameters()
```

---

## Batch processing

Process an entire folder of images automatically.

```python
# Internal analysis
from traitly.fruit_phenotyping import FruitInternalAnalyzer

analyzer = FruitInternalAnalyzer("PATH_FOLDER/")
analyzer.analyze_folder(
    folder_path="my_images/",
    output_path="results/",
    analyze_morphology=True,
    analyze_color=True,
    json_path="path/file.json"  # Optional, useful for defining parameters
)

# External analysis
from traitly.fruit_phenotyping import FruitExternalAnalyzer

analyzer = FruitExternalAnalyzer("PATH_FOLDER/")
analyzer.analyze_folder(
    folder_path="my_images/",
    output_path="results/",
    analyze_morphology=True,
    analyze_color=True,
    json_path="path/file.json"  # Optional, useful for defining parameters
)
```

This generates a folder called `results/` with:

- `morphology_results.csv` *(if `analyze_morphology=True`)*
- `color_results.csv` *(if `analyze_color=True`)*
- `*_annotated.jpg` for each **successfully** analyzed image
- `session_report.txt`
- `error_report.txt` *(only if any image failed)*

---

## What's next?

- [Internal Analysis Guide](../user_guide/internal_class.md) — detailed guide with all available parameters and methods for `FruitInternalAnalyzer`.
- [External Analysis Guide](../user_guide/external_class.md) — detailed guide with all available parameters and methods for `FruitExternalAnalyzer`.
- [External Analysis Tutorial](individual_img_tutorial.md) — analyzing an image step by step.
- [Traits Table](../user_guide/results/measurements.md) — what each column in the CSV means.

<div style="text-align: center;" markdown>

[← Back to Tutorials](overview.md){ .md-button }

</div>
