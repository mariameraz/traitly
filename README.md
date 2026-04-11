<div align="center">
<h1>Traitly</h1>

[![PyPI](https://img.shields.io/pypi/v/traitly?logo=pypi&logoColor=white)](https://pypi.org/project/traitly)
[![Python 3.9+](https://github.com/mariameraz/traitly/actions/workflows/python_compatibility.yml/badge.svg)](https://github.com/mariameraz/traitly/actions/workflows/python_compatibility.yml)
[![Testing](https://github.com/mariameraz/traitly/actions/workflows/pytest.yml/badge.svg?branch=main)](https://github.com/mariameraz/traitly/actions/workflows/pytest.yml)
[![Coverage](https://codecov.io/gh/mariameraz/traitly/branch/main/graph/badge.svg)](https://codecov.io/gh/mariameraz/traitly)
[![Documentation Status](https://readthedocs.org/projects/traitly/badge/?version=latest)](https://traitly.readthedocs.io/en/latest/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19447593.svg)](https://doi.org/10.5281/zenodo.19447593)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-green?logo=gnu&logoColor=white)](https://github.com/mariameraz/traitly/blob/main/LICENSE)


<a href="https://traitly.readthedocs.io/en/latest/">Documentation</a> ⋆
<a href="https://traitly.readthedocs.io/en/latest/installation/">Installation</a> ⋆
<a href="https://traitly.readthedocs.io/en/latest/user_guide/overview/">User Guide</a> ⋆
<a href="https://traitly.readthedocs.io/en/latest/tutorials/overview/">Tutorials</a>

</div>

<br>

Página disponible en: [![Spanish](https://img.shields.io/badge/Idioma-Espa%C3%B1ol-pink)](README_ES.md)


**Traitly** is an open-source Python tool for automated, high-throughput fruit phenotyping from digital images. Using computer vision methods, it quantifies morphological, symmetry, and color traits across both internal structures and the external appearance of the fruit.

It supports both single-image analysis and batch processing workflows, making it easy to handle large image datasets with just a few lines of code, which is especially useful in plant breeding programs and research.

</br>

> [!IMPORTANT]
> We are working on a manuscript describing this software and its applications, expected to be submitted in **Spring–Summer 2026**. In the meantime, if you use Traitly in your research, please cite it as:
>
> Torres-Meraz, M. A., Lopez-Moreno, H., & Zalapa, J. (2026). Traitly: A Python Toolkit for High-Throughput Fruit Phenotyping. Zenodo. https://doi.org/10.5281/zenodo.18738366

</br>

### What can Traitly do?

Traitly processes fruit images to measure:

* **Fruit morphology**: Area, perimeter, circularity, dimensions, and aspect ratio
* **Locule anatomy**: Locule number, size distribution, and spatial arrangement
* **Pericarp structure**: Thickness, uniformity (CV), and surface irregularity (lobedness)
* **Color phenotypes**: Multi-channel analysis (RGB, HSV, Lab) across different fruit regions

</br>

## Publications & Presentations

Posters related to Traitly can be found in this folder:

- [Posters](https://drive.google.com/drive/folders/1AvlHWKcDvoE9m9QcmCJ5o-ma9W-LNQMe?usp=share_link) ★ˎˊ˗

These materials provide additional methodological details and results from research derived from our package.

</br>

## Usage

Traitly can be run in different ways:

| Environment                | Status         |
| -------------------------- | -------------- |
| Jupyter Notebook           | ✔ Available    |
| Command line (CLI)         | ✔ Available    |
| Web app (Shiny)            | ✔ Available    |

 ⤷ You can also try our [interactive demo](https://huggingface.co/spaces/mariameraz/traitly) onlineˎˊ˗

---

Below is a basic example of how to use **Traitly**.

### ⋆ Python usage

#### Internal fruit morphology analysis
```python
from traitly.fruit_phenotyping import FruitInternalAnalyzer

# Single image analysis
path = 'PATH/my_image.jpg'
analyzer = FruitInternalAnalyzer(path)  # Initialize the FruitInternalAnalyzer class
analyzer.load_image()                   # Read the image
analyzer.setup_measurements()           # Obtain label and reference size information
analyzer.generate_fruit_mask()          # Create a binary mask to segment fruits and locules
analyzer.detect_fruits()                # Filter detected fruits
analyzer.analyze_morphology()           # Run the morphology analysis
analyzer.analyze_color()                # Run the color analysis
analyzer.results.save_all()             # Save results (color and morphology .csv files and annotated image)
analyzer.save_parameters()              # Save session parameters as .txt and .json files

# Batch analysis
path = 'PATH/my_folder'
json = 'my_parameters.json'
analyzer = FruitInternalAnalyzer(path)
analyzer.analyze_folder(json_path = json)
```

#### External fruit morphology analysis
```python
from traitly.fruit_phenotyping import FruitExternalAnalyzer

# Single image analysis
path = 'PATH/my_image.jpg'
analyzer = FruitExternalAnalyzer(path)       # Initialize the FruitExternalAnalyzer class
analyzer.load_image()                        # Read the image
analyzer.setup_measurements()                # Obtain label and reference size information
analyzer.generate_fruit_mask()               # Create a binary mask to segment fruits
analyzer.detect_fruits()                     # Filter detected fruits
analyzer.analyze_morphology()                # Run the morphology analysis
analyzer.analyze_color(color_channel='RGB')  # Extract mean RGB channel values for each fruit
analyzer.results.save_all()                  # Save results (color and morphology .csv files and annotated image)
analyzer.save_parameters()                   # Save session parameters as .txt and .json files

# Batch analysis
path = 'PATH/my_folder'
json = 'my_parameters.json'
analyzer = FruitExternalAnalyzer(path)
analyzer.analyze_folder(json_path = json)
```

### ⋆ Command-line usage

```bash
# Run the web app
traitly-app

# Internal morphology analysis (single image or folder)
traitly --fruit_internal -i tests/sample_data/
traitly --fruit_internal -i tests/sample_data/ -o results/ --num_cores 4
traitly --fruit_internal -i tests/sample_data/ --json config.json

# External morphology analysis (single image or folder)
traitly --fruit_external -i tests/sample_data/
traitly --fruit_external -i tests/sample_data/ -o results/ --json config.json --num_cores 4
```

</br>

For more detailed examples, [check our tutorials](https://traitly.readthedocs.io/en/latest/tutorials/overview/) ᯓ★

</br>

## Contact

For questions or comments about the project, feel free to reach out to:

* [ma.meraz@proton.me](mailto:ma.meraz@proton.me)
* [torresmeraz@wisc.edu](mailto:torresmeraz@wisc.edu)

We are open to collaborations, including adding new traits, and creating tutorials or workflows for specific crops or plant tissues.

## Contributions

<!-- CONTRIBUTORS-START -->
| Contributor | Role |
|-------------|------|
| [<img src="https://github.com/mariameraz.png" width="44" height="44" valign="middle">&nbsp;María Meraz](https://github.com/mariameraz) | 💻 📆 💬 🚧 🚇 📓 ✅ 🐛 📖 ⚠️ 🤔 🌍 🎨 |
| [<img src="https://github.com/hector-LM.png" width="44" height="44" valign="middle">&nbsp;Héctor López](https://github.com/hector-LM) | 📖 📓 ✅ 🤔 🐛 🔣 🌍 |
| Juan Zalapa | 🔣 |
<!-- CONTRIBUTORS-END -->

## Acknowledgements ♡

We thank the developers of [OpenCV](https://opencv.org/), [Ultralytics](https://github.com/ultralytics/ultralytics), [EasyOCR](https://github.com/JaidedAI/EasyOCR), [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/), and [Shiny](https://shiny.posit.co/py/), as well as all  open-source libraries that made this project possible.
