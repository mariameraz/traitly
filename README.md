---
title: Traitly
colorFrom: green
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# Traitly

Disponible en: [![Spanish](https://img.shields.io/badge/Idioma-Espa%C3%B1ol-pink)](README_ES.md)

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-green.svg)](https://github.com/mariameraz/traitly/blob/main/LICENSE) [![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)](https://www.python.org/) [![Version](https://img.shields.io/badge/Version-0.1.0--beta-orange)]() [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18738367.svg)](https://doi.org/10.5281/zenodo.18738367)

***Traitly*** is an open-source Python tool for high-throughput fruit phenotyping that automatically extracts quantitative traits from digital images of whole fruits or fruit slices. It focuses on the phenotyping of internal fruit structures and external morphology, using computer vision–based methods to quantify morphology, anatomy, symmetry, and color traits.

The tool supports both single-image and batch processing workflows, allowing users to analyze large image datasets with only a few lines of code, making it suitable for plant breeding and research.
</br>

> **Note:**  
> A manuscript describing the software and its applications is currently in preparation and is expected to be submitted in **Spring-Summer, 2026**. In the meantime, if you use Traitly in your research, please cite it as:
>
> Torres-Meraz, M. A., & Lopez-Moreno, H. (2026). Traitly: A Python Tool for High-Throughput Fruit Phenotyping. Zenodo. https://doi.org/10.5281/zenodo.18738367

</br>

### What can Traitly do?

Traitly processes fruit images to measure:

* **Fruit morphology**: Area, perimeter, circularity, aspect ratio, and bounding box dimensions
* **Locule anatomy**: Locule number, size distribution, and spatial arrangement
* **Pericarp structure**: Thickness profiles, uniformity (CV), and surface irregularity (lobedness)
* **Color phenotypes**: Multi-channel analysis (RGB, HSV, Lab) across different fruit regions


**👉 For a complete list of extracted traits, see:** 
- [![Documentation_EN](https://img.shields.io/badge/Documentation-English-lightblue)](docs/documentation.md)
- [![Documentation_ES](https://img.shields.io/badge/Documentaci%C3%B3n-Espa%C3%B1ol-red)](docs/documentation_ES.md)

</br>

## Project Status

**Traitly is currently in beta and undergoing testing across different systems and environments.**

The source code is now publicly available. The project's architecture and core logic are established, and early testers are currently evaluating the tool across different systems, workflows, and use cases.

Documentation is a work in progress, and additional details, examples, and clarifications will be added as testing advances.

A web application built with Streamlit is currently under development, aiming to provide a user-friendly interface for running Traitly without writing code.

Updates will be announced through this repository and [LinkedIn](https://www.linkedin.com/in/alemeraz/).
Interested users are encouraged to follow or watch the repository to stay informed.

</br>

## Publications & Presentations

Posters related to Traitly can be found in this folder:

- [Posters](https://drive.google.com/drive/folders/1AvlHWKcDvoE9m9QcmCJ5o-ma9W-LNQMe?usp=share_link) ★ˎˊ˗

These materials provide additional methodological details and related research results.

</br>

## Usage

Below is a basic example of how to use **traitly**.

### Running with Python

#### Internal analysis
```python
from traitly.fruit_phenotyping import FruitInternalAnalyzer

##########################
# Single image analysis  #
##########################
path = 'PATH/my_image.jpg'
analyzer = FruitInternalAnalyzer(path)  # Initialize the FruitInternalAnalyzer class
analyzer.load_image()                   # Read the image
analyzer.setup_measurements()           # Obtain label and reference size information
analyzer.generate_fruit_mask()          # Create a binary mask to segment fruits and locules
analyzer.detect_fruits()                # Filter detected fruits
analyzer.analyze_morphology()           # Run the morphology analysis
analyzer.analyze_color()                # Run the color analysis
analyzer.results.save_all()             # Save the color and morphology .csv files and the annotated image
analyzer.save_parameters()              # Save session parameters as .txt and .json files

###################
# Batch analysis  #
###################
path = 'PATH/my_folder'
json = 'my_parameters.json'
analyzer = FruitInternalAnalyzer(path)          # Initialize the FruitInternalAnalyzer class
analyzer.analyze_folder(json_path = json)       # Run the analysis on all valid images in the folder
# A single CSV file and the corresponding annotated images will be saved.
```

#### External analysis
```python
from traitly.fruit_phenotyping import FruitExternalAnalyzer

##########################
# Single image analysis  #
##########################
path = 'PATH/my_image.jpg'
analyzer = FruitExternalAnalyzer(path)  # Initialize the FruitExternalAnalyzer class
analyzer.load_image()                   # Read the image
analyzer.setup_measurements()           # Obtain label and reference size information
analyzer.generate_fruit_mask()          # Create a binary mask to segment fruits
analyzer.detect_fruits()                # Filter detected fruits
analyzer.analyze_morphology()           # Run the morphology analysis
analyzer.analyze_color(stat='median',
 color_channel='RGB')  # Extract median RGB channel values for each fruit
analyzer.results.save_all()             # Save the color and morphology .csv files and the annotated image
analyzer.save_parameters()              # Save session parameters as .txt and .json files

###################
# Batch analysis  #
###################
path = 'PATH/my_folder'
json = 'my_parameters.json'
analyzer = FruitExternalAnalyzer(path)          # Initialize the FruitExternalAnalyzer class
analyzer.analyze_folder(json_path = json)       # Run the analysis on all valid images in the folder
# A single CSV file and the corresponding annotated images will be saved.
```

### Command-line usage

```bash
# Internal structure analysis (single image or folder)
traitly --fruit_internal -i tests/sample_data/
traitly --fruit_internal -i tests/sample_data/ -o results/ --num_cores 4
traitly --fruit_internal -i tests/sample_data/ --json config.json

# External analysis (single image or folder)
traitly --fruit_external -i tests/sample_data/
traitly --fruit_external -i tests/sample_data/ -o results/ --json config.json --num_cores 4
```

</br>



More detailed examples:
👉 [https://github.com/mariameraz/traitly/tutorials](https://github.com/mariameraz/traitly/blob/main/tutorials)

</br>


## Contact ˖᯽ ݁˖

For inquiries regarding the project or potential collaborations, please reach out to:

* [ma.torresmeraz@gmail.com](mailto:ma.torresmeraz@gmail.com)
* [torresmeraz@wisc.edu](mailto:torresmeraz@wisc.edu)

We are open to collaborations, including developing species-specific pipelines, adding new traits or measurements, and creating tutorials or workflows tailored to specific crops or plant tissues.
