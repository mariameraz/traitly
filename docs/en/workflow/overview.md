<div class="animate" markdown>

# Traitly Architecture ⊹ ࣪ ˖

Traitly is built around a class structure that clearly separates two types of phenotypic analysis: internal and external fruit analysis. This organization lets you choose the level of detail you need without loading functionality you won't use.

The main module `traitly.fruit_phenotyping` contains two core classes:
```python
# Cross-section analysis (with locules)
from traitly.fruit_phenotyping import FruitInternalAnalyzer

# Whole fruit analysis (external contour only)
from traitly.fruit_phenotyping import FruitExternalAnalyzer
```

### What does each class do?

| Class | Focus | What it detects | Typical application |
|--------|---------|---------------|-------------------|
| `FruitInternalAnalyzer` | **Internal morphology** | Full fruit contour and each individual locule | Cross-sections where the goal is to quantify internal organization (locule count, relative area, symmetry, pericarp thickness, etc.) and the color of different tissues (pericarp and locules) |
| `FruitExternalAnalyzer` | **External morphology** | The outer fruit contour only | Whole fruits for shape, size, and external color studies |

Both classes share the same pipeline logic — image processing, segmentation, contour extraction, and trait calculation — but are optimized for their respective goals:

- **`FruitInternalAnalyzer`** looks for hierarchical relationships: a fruit contour containing multiple locule contours within it.
- **`FruitExternalAnalyzer`** focuses on the full fruit silhouette, ignoring internal structures.

![Analyzer pipelines](../assets/images/workflow.png)
*General workflow for each analysis. **A)** `FruitExternalAnalyzer`: pipeline for whole fruit external appearance analysis. **B)** `FruitInternalAnalyzer`: extended pipeline for fruit **and** locule detection and segmentation in cross-sections.*

---

## How to use Traitly?

Traitly can be used in two ways, depending on your working environment:

**From Python**: recommended for exploring parameters on individual images before processing a full batch, and for analyzing multiple images using `analyze_folder()` with a `.json` parameters file.

**From the terminal (CLI)**: to run the analysis directly from the terminal without writing a Python script - especially useful on servers or computing environments without a graphical interface. See the [CLI](cli.md) section for details.

!!! info ""
    The following sections contain detailed documentation for each class, including the specific traits they extract, usage examples, configurable parameters, and image requirements for each type of analysis.

</div>