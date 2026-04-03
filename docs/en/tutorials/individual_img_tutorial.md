---
hide:
  - navigation
  - toc
---

<div style="text-align: center;" markdown>

# External Appearance Analysis — Individual Processing

<p style="color:gray; margin-top: -35px; margin-bottom: 55px;" markdown>**Created by: Héctor López-Moreno; Traitly v0.1.0 – March, 2026**</p>

</div>

In this tutorial, we will demonstrate how to perform external appearance analysis of fruits using `FruitExternalAnalyzer` to extract morphology and color measurements from a single image.

!!! tip "Follow along"
    :fontawesome-solid-file-code: Download the Jupyter notebook and sample images for this tutorial [here](https://github.com/mariameraz/traitly/tree/main/tutorials_data/ext_analysis_ind_img_sample).

!!! info "Method and parameter reference"
    Throughout this tutorial, methods such as `setup_measurements()`, `generate_fruit_mask()`, `detect_fruits()`, `analyze_morphology()`, and `analyze_color()` are used. For a complete description of each method and its available parameters, see the [External Analyzer Class](../user_guide/external_class.md).

First, we load the `FruitExternalAnalyzer` class from Traitly and the image to be analyzed.

```python
from traitly.fruit_phenotyping import FruitExternalAnalyzer


path_img = '~/ext_analysis_sample1.jpg'

pic_test = FruitExternalAnalyzer(path_img)

pic_test.load_image()
```

![png](individual_img_tutorial_en_files/individual_img_tutorial_en_1_0.png)

Then we run `setup_measurements()` to detect the size references in the image (black circles).

```python
pic_test.setup_measurements()
```

    =======================================================
    ★ LABEL DETECTION:
    =======================================================
    
    =======================================================
    ★ REFERENCE SIZE:
    =======================================================
    > Reference size detected:
      - Processing reference box(es) with a confidence threshold >=0.6:
                Ref 1: 452x2534 px, conf: 0.951
                Ref 2: 450x2551 px, conf: 0.943
    
      - Total circles detected: 12
                Range: [310.2, 314.1] px
                Filtered circles: 11/12 (std > 2)
    
            . ݁₊ ⊹ . ݁ ⟡ ݁ px/cm density: 124.9 (diameter_cm: 2.5 cm) ⟡ ݁ . ⊹ ₊ ݁.
    
    Note: Default reference diameter (2.5 cm) applied.
            Specify diameter_cm to override this value.


Now let's generate the fruit masks with `generate_fruit_mask()` using the default background color value `white`, and see which objects in the image were detected as fruits with `detect_fruits()`.

!!! note "About background color"
    If you need to segment fruits on a background color other than the predefined ones – `white`, `blue`, and `black` – you can refer to the [Background Segmentation](background_segmentation.md) tutorial for more details.

As we can see in the plots generated below, most fruits have been effectively segmented. However, we notice that some fruits were not detected by `detect_fruits()` (they don't have the green contour). In these cases, we need to modify some parameters to improve both the masks and the detection; let's address that below.

```python
pic_test.generate_fruit_mask(background_color='white')

pic_test.detect_fruits(
    plot=True, plot_size=(5,5),
    contour_thickness=7,
)
```

![png](individual_img_tutorial_en_files/individual_img_tutorial_en_5_0.png)

![png](individual_img_tutorial_en_files/individual_img_tutorial_en_5_1.png)

    =====================================
            . ݁₊ ⊹ . ݁ ⟡ ݁ Detected fruits: 28 ⟡ ݁ . ⊹ ₊ ݁.
    
     > Parameters used:
            - min_fruit_circularity: 0.5
            - min_fruit_area: 500
    =====================================


In `generate_fruit_mask()`, fruit masks can sometimes show indentations or edge regions that are not correctly segmented. To close these gaps, we can use the `apply_convex_hull=True` parameter, which applies a [convex hull](https://www.geeksforgeeks.org/dsa/convex-hull-algorithm/) around the fruit contour, ensuring a smoother and more closed result. Additionally, `kernel_blur=5` helps better define fruit contours by blurring and simplifying the colors in the image, which facilitates segmentation. Having well-defined contours is fundamental, since all calculations are based on them, and any gap in the contour directly impacts subsequent analyses. Optionally, `erosion_px=3` can be applied to remove some pixels around the contour whose color could be affected by background color reflections. Erosion also helps remove portions of the background that might be included in the mask and that could skew the fruit color estimates.

In `detect_fruits()`, `min_fruit_circularity=0.4` ensures that we capture all fruits by lowering the circularity threshold, since some have a more elongated shape.

```python
pic_test.generate_fruit_mask(background_color='white',
                             apply_convex_hull=True,
                             kernel_blur=5,
                             erosion_px=3)

pic_test.detect_fruits(
    plot=True, plot_size=(5,5),
    contour_thickness=7,
    min_fruit_circularity=0.4
)
```

![png](individual_img_tutorial_en_files/individual_img_tutorial_en_7_0.png)

![png](individual_img_tutorial_en_files/individual_img_tutorial_en_7_1.png)

    =====================================
            . ݁₊ ⊹ . ݁ ⟡ ݁ Detected fruits: 30 ⟡ ݁ . ⊹ ₊ ݁.
    
     > Parameters used:
            - min_fruit_circularity: 0.4
            - min_fruit_area: 500
    =====================================


Now that the fruits have been correctly segmented and detected, we can perform the morphological and color analyses.

```python
pic_test.analyze_morphology(display_table=False,
                            plot=True,
                            plot_size=(5,5))

pic_test.analyze_color(display_table=False,
                       plot=False)
```

![png](individual_img_tutorial_en_files/individual_img_tutorial_en_9_0.png)

When `analyze_morphology()` and `analyze_color()` are run, the results object containing the `save_all()` method is generated, which we can call as follows to save both the CSV files with the results of each analysis and the annotated image. In the annotated image, we can verify that both the fruits and the size references have been correctly detected.

```python
pic_test.results.save_all()
```

    Image saved at: ./ext_analysis_sample1_annotated.jpg
    Morphology CSV saved at: ./ext_analysis_sample1_morphology_results.csv
    Color CSV saved at: ./ext_analysis_sample1_color_results.csv

Optionally, you can save the parameters and session information from your analysis with `save_parameters()` to ensure reproducibility of your future analyses or to use them in batch processing.

```python
pic_test.save_parameters()
```

    > Parameters saved:
      - TXT:  ./ext_analysis_sample1_parameters.txt
      - JSON: ./ext_analysis_sample1_parameters.json


## What's next?

- [How to perform batch processing](batch_tutorial.md) — external appearance analysis of multiple images.
- [Traits Table](../user_guide/results/measurements.md) — what each column in the CSV means.

<div style="text-align: center;" markdown>

[← Back to Tutorials](overview.md){ .md-button }

</div>
