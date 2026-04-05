---
hide:
  - navigation
  - toc
---

<div style="text-align: center;" markdown>

# Extracting Images from a PDF

<p style="color:gray; margin-top: -35px; margin-bottom: 55px;" markdown>**Created by: María A. Torres-Meraz; Traitly v0.1.0 – April, 2026**</p>

</div>

!!! tip ""
    **Requirements:** The optional dependencies for PDF manipulation in Traitly must be installed. For more details, see the [Installation Guide](../installation.md#optional-dependencies).

!!! tip "Follow along"
    :fontawesome-solid-file-code: Download the Jupyter notebook and sample images for this tutorial [here](https://github.com/mariameraz/traitly/tree/main/tutorials_data/cranberry_internal_analysis).

In this tutorial, we will learn how to extract images from a PDF using Traitly.

When scanning multiple samples, it may be more convenient to export them all into a single PDF file. In these cases, Traitly's `pdf_to_img` function is especially useful, as it allows you to extract the images from the file to continue with their analysis.

The first step is to import the function from Traitly into our workspace, as shown below:


```python
from traitly.utils.convert_pdf import pdf_to_img
```

Next, we need to specify the location of our files using the `pdf_path` parameter. By default, images will be renamed using the PDF filename as a base, appending the suffix `_page1` and so on. However, if the images contain QR codes, we can set `detect_qr = True`: this will enable automatic QR detection and reading, and the images will be renamed according to their content. Only the first word of the code (no spaces) will be used as the filename.

It is also possible to define the desired output format; by default, images are exported in JPG format.

Once the process is complete, a message will be printed indicating how many files were analyzed and how many images were extracted. If you wish to suppress this message, you can use the `verbose = False` parameter.

Note that the function returns the path of each generated image. If you do not want these values displayed on screen, you can redirect them to a temporary variable such as `temp`.


```python
input_path = "./cranberry_slices.pdf"  # Path to the PDF file

temp = pdf_to_img(pdf_path = path,
                  dpi = 150, 
                  detect_qr = True, 
                  output_format = 'png')
```

    =================================================================
    Extracting images may take a few minutes... ⋆✧｡٩(ˊᗜˋ )و✧*｡
    =================================================================
    > Processing 1 PDF file:
        – Images extracted: 2
        – QR detected: 2/2 img(s)
        – Results folder: /Users/traitly/tutorials_data/pdf_extraction/Images_from_PDF

We can inspect `temp` to see in detail what `pdf_to_img` returns:

```python
print(temp)
```

    ['/Users/traitly/tutorials_data/pdf_extraction/Images_from_PDF/SW-1073.png', '/Users/traitly/tutorials_data/pdf_extraction/Images_from_PDF/DP14-497.png']


It is important to keep in mind that a low DPI can interfere with the correct detection of QR codes, as illustrated in the following example. When extracting images with `dpi=70`, the output message indicates that no QR codes could be detected for the same PDF. This is not a problem: when QR codes are not detected, images are simply renamed based on the PDF filename, as mentioned earlier. For this reason, we recommend adjusting the DPI according to the size of the objects in the image, ensuring they are sharp and legible.


```python
input_path = "./cranberry_slices.pdf"  # Path to the PDF file

temp = pdf_to_img(pdf_path = path,
                  dpi = 70, 
                  detect_qr = True, 
                  output_format = 'png')
```

    =================================================================
    Extracting images may take a few minutes... ⋆✧｡٩(ˊᗜˋ )و✧*｡
    =================================================================
    > Processing 1 PDF file:
        – Images extracted: 2
        – QR detected: 0/2 img(s)
        – Results folder: /Users/traitly/tutorials_data/pdf_extraction/Images_from_PDF


Finally, the function is also capable of processing multiple PDFs contained in a folder. You only need to provide the path to the folder and the function will automatically search for all available PDF files within it.

When working with a large number of files, the extraction process can become slow. To reduce processing time, the task can be parallelized using the `num_cores` argument, which distributes the workload across multiple processor cores. If not specified, the default value is `num_cores = 1`.


```python
input_path = "pdf_extraction/" # Path to the folder containing the PDFs

temp = pdf_to_img(pdf_path = path,
                  dpi = 150, 
                  num_cores = 2,
                  detect_qr = True)
```

    =================================================================
    Extracting images may take a few minutes... ⋆✧｡٩(ˊᗜˋ )و✧*｡
    =================================================================
    > Processing 1 PDF file(s):
        – Images extracted: 2
        - QR detected: 2/2 img(s)
        – num_cores: 2
        – Results folder: /Users/traitly/tutorials_data/pdf_extraction/Images_from_PDF


!!! note ""
    In both cases, whether processing a single file or an entire folder, the extracted images will be automatically saved in a folder called `Images_from_PDF`. If you wish to specify a different location, you can do so using the `output_path` argument.

## What's next?

- [Internal Analysis Guide](../user_guide/internal_class.md) — detailed guide with all available parameters and methods for `FruitInternalAnalyzer`.
- [External Analysis Guide](../user_guide/external_class.md) — detailed guide with all available parameters and methods for `FruitExternalAnalyzer`.
- [External Analysis Tutorial](individual_img_tutorial.md) — analyzing an image step by step.
- [Cranberry Internal Analysis Tutorial](cranberry_internal_analysis.md) — analyzing an image step by step.

<div style="text-align: center;" markdown>

[← Back to Tutorials](overview.md){ .md-button }

</div>
