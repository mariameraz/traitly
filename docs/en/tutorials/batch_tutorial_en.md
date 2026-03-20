# External Appearance Analysis — Batch Processing

*Traitly version used in this tutorial: 0.1.0*

In this tutorial, we will demonstrate how to perform external appearance analysis on a folder containing multiple photos in an automated manner. Unlike individual image analysis, batch processing is performed with `analyze_folder()`, which retains all the adjustable arguments from individual analysis. As shown in the individual analysis, there are some default parameters (e.g., background color) that may or may not work depending on the characteristics of your images. For this reason, it is recommended that before starting the analysis of all photos in the folder, you explore the effectiveness of the default values on individual images and adjust them if necessary, as shown in the [tutorial](https://github.com/mariameraz/traitly/blob/main/docs/en/tutorials/individual_img_tutorial_es.md). It is advisable to test the parameters with a few (~2-3) different images from your folder, especially if there is high variation (different light exposure, contrasting fruit colors or shapes, etc.). Once you are satisfied with the settings, save your session information and the parameters used with `save_parameters()`, which generates a `.json` file with all the necessary information to replicate the analysis in batch processing.

In this tutorial, we will perform batch analysis on a folder with 7 images, including the one from the individual analysis tutorial. We will use the `.json` file generated at the end of that tutorial, which contains the optimized parameters for our images. With this `.json` file, the analysis can be completed in a single step using `analyze_folder()`, as shown below.

!!! tip "Follow along"
    :fontawesome-solid-file-code: Download the Jupyter notebook, folder with sample images, and `.json` file for this tutorial [here]().

First, we load the `FruitExternalAnalyzer()` class from traitly, the image to be analyzed, and the `.json` file with the predefined parameters.


```python
from traitly.fruit_phenotyping import FruitExternalAnalyzer 

img_folder = '~/ext-analysis_batch_sample'

parameters = '~/ext-analysis_batch_sample/ext_analysis_sample1_parameters.json'
```

Now we initialize the `FruitExternalAnalyzer()` class to load the folder and then analyze the images with `analyze_folder()`.


```python
folder_test = FruitExternalAnalyzer(img_folder)

folder_test.analyze_folder( 
    json_path = parameters,
    output_path = '~/ext-analysis_batch_sample/ext_analysis_results',
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
        > Parameters loaded from: ~/ext-analysis_batch_sample/ext_analysis_sample1_parameters.json
    


    Processing images: 100%|██████████| 7/7 [00:09<00:00,  1.42s/img]

    
    ( ദ്ദി ˙ᗜ˙ ) Finished ===============================================
        > Image(s) processed:
            - Successfully: 7/7 img(s)
            - Total fruits: 195
            - Total time: 10.0s  (avg 1.4s/img)
        > Files saved:
            - 7 annotated image(s)
            - morphology_results.csv
            - color_results.csv
            - session_report.txt
            - Results folder: ~/ext-analysis_batch_sample/ext_analysis_results


    


Our analysis is complete! As can be seen in the output above, a summary of the most relevant characteristics of the analysis is provided: the analyzed input, the parameters used, the process characteristics, and the obtained output. The results were saved in the ext_analysis_results folder, located within the folder containing the analyzed images. This folder contains annotated images for each analyzed image, a `.csv` file with color results, another with morphology results, and a `.txt` file with the session report.
