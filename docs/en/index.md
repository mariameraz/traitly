---
hide:
  - navigation
  - toc
---

<div class="animate" markdown>

# Welcome to Traitly

**Traitly** is a Python library designed to **automate fruit image analysis**, from a single sample to hundreds of fruits in a single run. From standard RGB images, it extracts **color, shape, and size traits** from both internal (cross-section) and external (fruit surface) images.

Traitly is committed to **open and reproducible science**: every analysis automatically generates a session report with all parameters and versions used, ensuring complete traceability of results.

!!! info ""
    You can find our documentation in both **English** and **Spanish** :fontawesome-regular-face-smile-beam:

---



## Get Started ⊹ ࣪ ˖


<div class="grid cards" markdown>

-   :simple-rocket: __**Installation**__

    ---

    Install Traitly and its dependencies.

    [:octicons-arrow-right-24: Start installation](installation.md)

-   :material-star-shooting: __**Quickstart**__

    ---

    Run your first analysis in minutes.

    [:octicons-arrow-right-24: Launch tutorial](tutorials/quickstart.md)

-   :fontawesome-brands-readme: __**Tutorials**__

    ---

    Step-by-step guides for different workflows.

    [:octicons-arrow-right-24: Explore guides](tutorials/overview.md)

-   :material-table-heart: __**Traits Table**__

    ---

    Detailed description of all extracted traits.

    [:octicons-arrow-right-24: View reference](workflow/results/measurements.md)


</div>


---

## What does Traitly analyze?

Traitly works with two main types of fruit images:

### **Internal images (cross-section)**

* Internal morphology
* Number and distribution of locules
* Pericarp thickness
* Symmetry
* Color of internal tissues

### **External images (surface)**

* General fruit shape
* Size
* Surface color

In both cases, traits are extracted from standard RGB images.
Optionally, Traitly can **convert pixels to real metric units** through automatic detection of a size reference marker present in the image.

---

## Methodological approach

The core of Traitly's analysis is based primarily on **classical segmentation and traditional image processing**, complemented by pre-trained models for auxiliary tasks such as label or size reference detection.

This design choice prioritizes **robustness, interpretability, and reproducibility**, and allows the method to be **easily adaptable** beyond fruits. With minimal parameter adjustments, the same approach can be applied to other biological tissues such as **seeds or leaves**, without the need to redefine the pipeline architecture.

---

## Key features

* **Single image or batch processing**:
  Analyze a single image or entire folders in one run.

* **Per-fruit measurements**:
  Each detected fruit receives a unique ID and is measured independently.
  For example, an image with 25 fruits generates 25 rows in the output.

* **Fully automated**:
  Detection, segmentation, calibration, and trait extraction without manual measurements, reducing bias and phenotyping time.

* **Pre-trained models included**:
  Automatic detection of size markers and sample labels, with no additional setup.

* **Color correction**:
  **Macbeth Color Checker** detection to standardize color across experiments.

* **Automatic sample identification**:
  Detection of **QR codes** and **text labels**.

* **PDF support**:
  Direct conversion of scanned PDF files to images.

* **Session reports**
  Automatically saves parameters, dependency versions, and metadata for every run.

---

## Where can you use Traitly?

| Environment                | Status          |
| -------------------------- | --------------- |
| Jupyter Notebook           | :fontawesome-solid-square-check: Available    |
| Command line (CLI)         | :fontawesome-solid-square-check: Available    |
| Web app (Streamlit)        | :fontawesome-solid-hammer: Coming soon |

---

## A growing project and collaborations

Traitly is a **project under active development**, designed to grow alongside the scientific community. Its modular architecture makes it easy to incorporate new ideas without compromising the consistency or reproducibility of the analysis.

Contributions are welcome and appreciated across different areas, including:

* :fontawesome-brands-readme: **Tutorials and documentation**:
  New examples, use cases, or workflows.

* :fontawesome-solid-seedling: **New trait proposals**:
  Ideas for incorporating new morphological, geometric, or color traits based on different experimental needs.

* :fontawesome-solid-globe: **Translation**:
  Expanding the documentation to new languages.

* :fontawesome-solid-heart: **Methodological extensions**:
  Adaptations to other tissues, species, or experimental contexts.

Our goal is for Traitly to grow into a collaborative, flexible, and scientifically robust tool, driven by real research use.

---

## Built on solid foundations

Traitly relies on well-established libraries from the Python scientific ecosystem. Core processing uses **OpenCV (contrib)**, **NumPy**, **SciPy**, **pandas**, and **matplotlib**, all with C/C++ backends that guarantee high performance even in large-scale batch analyses.

This makes Traitly particularly well-suited for **high-throughput phenotyping experiments** in plant breeding and genetics, where analyzing large populations is common.

</div>