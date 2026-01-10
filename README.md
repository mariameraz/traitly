***Traitly*** is an open-source Python tool for high-throughput fruit phenotyping that automatically extracts quantitative traits from digital images of fruit slices. It focuses on the phenotyping of internal fruit structures, using computer vision–based methods to quantify morphology, anatomy, symmetry, and color traits.

The tool supports both single-image and batch processing workflows, allowing users to analyze large image datasets with only a few lines of code, making it suitable for plant breeding and research.

</br>

> **Note:**  
> A manuscript describing this software is currently under preparation and is expected to be released in **Spring-Summer, 2026**.

</br>

### What can Traitly do?

Traitly processes fruit images to measure:

* **Fruit morphology**: Area, perimeter, circularity, aspect ratio, and bounding box dimensions
* **Locule anatomy**: Locule number, size distribution, and spatial arrangement
* **Pericarp structure**: Thickness profiles, uniformity (CV), and surface irregularity (lobedness)
* **Color phenotypes**: Multi-channel analysis (RGB, HSV, Lab) across different fruit regions


**👉 For a complete list of extracted traits, see:** [Trait Tables](docs/documentation.md)

</br>

## Project Status

**Traitly is in pre-release and under active development.** 
The source code is not yet publicly available.

The current documentation corresponds to a **preliminary version of the manual** and is subject to change.
Additional details, examples, and clarifications will be provided in future updates.

Updates regarding the public release will be announced through this repository and [LinkedIn](https://www.linkedin.com/in/alemeraz/).
Interested users are encouraged to follow or watch the repository to stay informed.

</br>

## Publications & Presentations

Posters related to Traitly can be found in this folder:

- [Posters]([docs/posters/](https://drive.google.com/drive/folders/1AvlHWKcDvoE9m9QcmCJ5o-ma9W-LNQMe?usp=share_link)) ★ˎˊ˗

These materials provide additional methodological details and related research results.

</br>

## Usage

Below is a basic example of how to use **traitly**:

```python
from traitly.internal_structure import FruitAnalyzer

##########################
# Single image analysis #
##########################
path = 'PATH/my_image.jpg'

analyzer = FruitAnalyzer(path)  # Initialize the FruitAnalyzer class

analyzer.read_image()           # Read the image
analyzer.setup_measurements()   # Obtain label and reference size information
analyzer.create_mask()          # Create a binary mask to segment fruits and locules
analyzer.find_fruits()          # Filter detected fruits
analyzer.analyze_image()        # Run the fruit analysis
analyzer.results.save_all()     # Save both the CSV file and the annotated image

###################
# Batch analysis #
###################
path = 'PATH/my_folder'

analyzer = FruitAnalyzer(path)  # Initialize the FruitAnalyzer class
analyzer.analyze_folder()       # Run the analysis on all valid images in the folder.
                                # A single CSV file and the corresponding annotated images will be saved.
```

</br>

More detailed examples:
👉 [https://github.com/mariameraz/traitly/blob/main/traitly-examples.ipynb](https://github.com/mariameraz/traitly/blob/main/docs/traitly-examples.ipynb)

</br>

## Contact ˖᯽ ݁˖

For inquiries regarding the project or potential collaborations, please send a message to ma.torresmeraz@gmail.com or torresmeraz@wisc.edu
