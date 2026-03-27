<div class="animate" markdown>

# Traitly Architecture ⊹ ࣪ ˖

Traitly is built around a class structure in Python that clearly separates two types of phenotypic analysis: internal and external fruit analysis. This organization lets you choose the level of detail you need without loading functionality you won't use.

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

Traitly is available in three environments:

**From Python (Jupyter Notebook or script)**: recommended for interactively exploring and adjusting parameters on individual images, and for efficient processing of large image batches.

**From the terminal (CLI)**: to run the analysis directly from the terminal without writing a Python script. This is especially useful on servers or computing environments without a graphical interface. See the [CLI](cli.md) section for details.

**From the Shiny app**: for interactive analysis without writing code. Available locally by running `traitly-app` in the terminal (see the [Installation](../installation.md#optional-dependencies) section for the required dependencies), or online through the [interactive demo](https://huggingface.co/spaces/mariameraz/traitly).


!!! info ""
    The following sections contain detailed documentation for each environment, including usage examples, configurable parameters, and extracted traits. See also the image specifications for [internal analysis](int_image_requirements.md) and [external analysis](ext_image_requirements.md) to ensure the best results.
    
</div>
