<div class="animate" markdown>

# Traitly Architecture ⊹ ࣪ ˖

Traitly is built on a class-based structure that clearly separates two types of phenotyping analyses: internal and external fruit analysis. This organization allows you to choose the level of detail you need without loading functionality you won't use.

The main `traitly.fruit_phenotyping` module contains two core classes:

```python
# Cross-section analysis (with locules)
from traitly.fruit_phenotyping import FruitInternalAnalyzer

# Whole fruit analysis (external contour only)
from traitly.fruit_phenotyping import FruitExternalAnalyzer
```

### What does each class do?

| Class | Focus | What it detects | Typical application |
|--------|---------|---------------|-------------------|
| `FruitInternalAnalyzer` | **Internal morphology** | Contours of the whole fruit and each individual locule | Cross-sections where internal organization needs to be quantified (number of locules, relative area, symmetry, pericarp thickness, etc.) and color of different tissues (pericarp and locules) |
| `FruitExternalAnalyzer` | **Surface morphology** | Only the outer fruit contour | Whole fruits for shape, size, and external color studies |

Both classes share the same pipeline logic —image processing, segmentation, contour extraction, and trait calculation— but are optimized for their respective objectives:

- **`FruitInternalAnalyzer`** looks for hierarchical relationships: a fruit contour containing multiple locule contours inside it.
- **`FruitExternalAnalyzer`** focuses on the complete fruit silhouette, ignoring internal structures.

In the following sections, you'll find detailed documentation for each class, including the specific traits they extract, usage examples, and configurable parameters.

</div>