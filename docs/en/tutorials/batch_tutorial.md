---
hide:
  - navigation
  - toc
---

<div style="text-align: center;" markdown>

# Batch Processing

<p style="color:gray; margin-top: -35px; margin-bottom: 55px;" markdown>**Created by: Héctor López-Moreno; Traitly v0.1.0 – March, 2026**</p>

</div>

In this tutorial, we demonstrate how to use `FruitExternalAnalyzer` to run automated batch analysis on multiple images.

!!! note
    Although this tutorial focuses on `FruitExternalAnalyzer`, the same workflow applies to batch processing with `FruitInternalAnalyzer`.

!!! tip "Follow along"
    :fontawesome-solid-file-code: Download the Jupyter notebook, the folder with sample images, and the `.json` file for this tutorial [here](https://github.com/mariameraz/traitly/tree/main/tutorials/ext_analysis_batch_sample).
    
Unlike the [individual image analysis](individual_img_tutorial.md), batch processing is performed by running only `analyze_folder()`, which iterates through all the steps of the individual analysis for each image found in a folder.

There are some default parameters (e.g., background color, minimum fruit size, etc.) that may or may not work depending on the characteristics of your images. For this reason, it is recommended to explore the effectiveness of the default values on individual images and adjust them if necessary before processing the entire folder, as shown in the [external appearance tutorial](individual_img_tutorial.md). It is advisable to test the parameters with a few (~2-3) different images from your folder, especially if there is high variation (different light exposure, contrasting fruit colors or shapes, etc.). Once you are satisfied with the settings, save your session information with `save_parameters()`, which generates a `.json` file with all the necessary information to replicate the analysis in batch processing.

In this tutorial we will run batch analysis on a folder with 7 images, including the one used in the individual analysis tutorial. We will use the `.json` file generated at the end of that tutorial, which contains the optimized parameters for our images. With this file, the analysis is completed in a single step with `analyze_folder()`, as shown below.


First, we load the `FruitExternalAnalyzer()` class from Traitly, and define the path to the folder to analyze and the `.json` file with the predefined parameters we will use.

```python
from traitly.fruit_phenotyping import FruitExternalAnalyzer 

img_folder = '~/ext-analysis_batch_sample'

parameters = '~/ext_analysis_batch_sample/ext_analysis_sample1_parameters.json'
```

Now we initialize the `FruitExternalAnalyzer()` class to load the folder and then analyze the images with `analyze_folder()`.

??? note "About the results folder"
    In this case, we use the `output_path` parameter to specify the directory where the results will be saved. However, this parameter is *optional*. If it's not provided, a folder named `Results` will be automatically created inside the image folder.

```python
folder_test = FruitExternalAnalyzer(img_folder)

folder_test.analyze_folder( 
    json_path = parameters,
    output_path = '~ext-analysis_batch_sample/ext_analysis_results',
    analyze_morphology = True,
    analyze_color = True
    )  
```

    ============================================================
     Traitly running ⋆✧｡٩(ˊᗜˋ )و✧*｡   
    ============================================================
        > Input folder: ~/ext-analysis_batch_sample
        > Image(s) detected: 7
        > analyze_morphology: True
        > analyze_color: True
        > num_cores: 1
        > Parameters loaded from: ~/ext_analysis_ind_img_sample/ext_analysis_sample1_parameters.json
    


    Processing images: 100%|██████████| 7/7 [00:10<00:00,  1.50s/img]

    
    ( ദ്ദി ˙ᗜ˙ ) Finished ===============================================
        > Image(s) processed:
            - Successfully: 7/7 img(s)
            - Total fruits: 195
            - Total time: 10.5s  (avg 1.5s/img)
        > Files saved:
            - 7 annotated image(s)
            - morphology_results.csv
            - color_results.csv
            - session_report.txt
            - Results folder: ~/ext-analysis_batch_sample/ext_analysis_results


Our analysis is complete! As can be seen in the output above, a summary of the most relevant characteristics of the analysis is provided: the analyzed input, the parameters used, the process characteristics, and the obtained output.


## What's next?

- [Result files](../user_guide/results/overview.md) – description of result files created by Traitly
- [Traits Table](../user_guide/results/measurements.md) – what each column in the CSV means.

<div style="text-align: center;" markdown>

[← Back to Tutorials](overview.md){ .md-button style="background-color: black; color: white; border-color: black;" }

</div>
