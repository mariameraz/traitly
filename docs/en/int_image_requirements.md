<div class="animate" markdown>

# Image Specifications

Analysis quality depends directly on image quality. Traitly is designed to be robust, but following these recommendations will ensure the best results.

---

## Image Acquisition

### Recommended Equipment

For consistent results, we recommend using a conventional **flatbed scanner**. This approach gives you control over lighting conditions and keeps the capture distance consistent across samples.

### Sample Preparation

- **Fruit cutting**: Use a sharp knife or blade to obtain clean cross-sections.
- **Blade maintenance**: Replace or sharpen the blade regularly — a dull blade affects cut quality and can introduce morphological measurement bias.
- **Why it matters**: Uneven cuts can skew morphological measurements.
- **Juicy fruits**: For fruits with high juice content, gently blot excess juice with a cloth before placing them on the scanner. You can also wipe the scanner surface with alcohol between scans to remove residue that could be detected as fruit contours.
- **Non-fruit objects**: Keep images as clean as possible — stems, leaves, loose seeds, or debris should be minimized. Although these can be filtered out in later analysis steps, they increase processing time since runtime scales with the number of contours detected per image.

### Background Setup

Traitly assumes a **black background** by default. To achieve this:

1. Place the fruits directly on the scanner surface
2. Cover the scanner with a cardboard box or any other material that blocks external light
3. This ensures a uniform and consistent background across all images

??? tip "Scanner settings"
    Many scanners allow you to adjust parameters such as color profile, white balance correction, and other image settings. For reproducible color measurements, configure the scanner to capture colors as the sensor records them, without applying automatic color corrections or white balance adjustments.

    Regardless of the configuration you choose, it is essential to scan **all images in an experiment under the same conditions**: same scanner, same resolution, and same software settings. Any variation between sessions can introduce inconsistencies in color and morphology measurements.

<br>

<div style="display: flex; gap: 16px; justify-content: center; align-items: flex-start;">
  <figure style="text-align: center; margin: 0;">
    <img src="../../assets/images/scanner_box.jpg" alt="Black box and scanner setup"
         style="height: 600px; width: auto;">
    <figcaption><em>Example of a black box and scanner setup</em></figcaption>
  </figure>
  <figure style="text-align: center; margin: 0;">
    <img src="../../assets/images/slices_image.jpg" alt="Example of a scanned image"
         style="height: 600px; width: auto;">
    <figcaption><em>Example of a scanned image</em></figcaption>
  </figure>
</div>



### Format and Resolution

**Supported formats:**

- `.jpg`, `.jpeg`
- `.png`
- `.tif`, `.tiff`
- `.bmp`

**PDF:**

When scanning, it is more practical to configure the scanner to save all captures as a single multi-page PDF rather than managing individual files. Traitly includes functions to automatically extract each page as a separate image ready for analysis. See [Tutorials](tutorials/quickstart.md) for details.

**Resolution (DPI):**

- There is no strict minimum requirement
- The right resolution depends on the size and complexity of the structures being measured: small fruits or locules require higher resolution than larger ones
- **Key recommendation**: Use the **same resolution (DPI)** for all images in an experiment
- Consistency > absolute resolution

---

## Size References

Traitly offers multiple ways to convert pixels to real metric units.

### Calibration Methods

| Method | When to use | Reproducibility |
|--------|-------------|----------------|
| **Circular reference** | Include a strip of circles of known diameter | :fontawesome-solid-star:{ .icon-yellow }:fontawesome-solid-star:{ .icon-yellow }:fontawesome-solid-star:{ .icon-yellow } Across equipment and experiments |
| **Known dimensions** | You know the scanner's capture area (e.g. 21×29.7 cm) | :fontawesome-solid-star:{ .icon-yellow }:fontawesome-solid-star:{ .icon-yellow } Same scanner and resolution |
| **No calibration** | You only need relative measurements or comparisons within the same image | :fontawesome-solid-star:{ .icon-yellow } Same batch with identical configuration (e.g., DPI and image size) |

!!! tip "Recommendation"
    Use the **strip of black reference circles on a white background** whenever possible. Traitly automatically detects these circles using a YOLO model trained specifically for this purpose.

    When using the template, verify the actual diameter of the printed circles with a ruler. Printers can scale documents during printing, so the final size may differ from the file.

    [:octicons-download-24: Download circular reference template](../../assets/templates/size_reference_template.pdf)

### Why use a circular reference?

Scanners can have small variations between their declared and actual capture dimensions.

The circular reference:

- Corrects for these internal scanner variations
- Provides per-image independent calibration
- Is more accurate than assuming the scanner's declared dimensions
- By deriving the scale from the average diameter of multiple detected circles, the method buffers the effect of small geometric distortions in the scanner
- By deriving the scale from the reference rather than the image dimensions, it allows batch processing of images of different sizes, since the pixel/cm conversion is invariant to image size or cropping

---

## Sample Identification

Traitly can automatically extract sample information by reading QR codes or recognizing text (OCR), storing the information in the results tables. We recommend using **QR codes** whenever possible, as they are faster to detect, more tolerant of poor image quality, and do not depend on font type the way OCR does.

To generate QR codes, any available tool will work. If you need to create multiple labels from a text file (`.txt`, `.csv` or `.tsv`), we recommend **[QRLabel](https://github.com/mariameraz/qrlabel)**, the tool we used to generate the labels in the example image.

### Label Detection

Detection follows this order:

1. Attempts to detect a QR code
2. If no QR is found, applies OCR

!!! warning "OCR is sensitive to image quality and font choice"
    Unlike QR codes, OCR detection accuracy depends directly on image resolution, label contrast, and font type. A poorly designed label or a low-quality image can result in text being detected incorrectly or not at all. If you use OCR, follow the recommendations in the table below.

For optimal OCR detection:

| Recommendation | Good example :octicons-check-circle-fill-24:{ .icon-green } | Bad example :octicons-x-circle-fill-24:{ .icon-red } |
|---------------|--------------------------------------------------------------|-------------------------------------------------------|
| **Light background, dark text** | Black text on white | Gray text on gray background |
| **Clear, sans-serif font** | Arial, Helvetica, Consolas, Roboto, Verdana | Decorative or cursive fonts |
| **Use only digits and uppercase** | `TOM-001` | Mixing `I`, `l`, `1` or `O`, `0`: `TOM-00I`, `T0MAT-l` |
| **Use separators between fields** | `TOM-001`, `MANZ-02-A` | `TOM001`, `MANZA02` |
| **Sufficient font size** | ≥ 14 pt | Very small text reduces detection rate |
| **Sufficient contrast** | High contrast | Low contrast |

**Well-designed label examples:**
```
Good:   TOM-001      CHILE-02       MANZ-123
        TOM-001-A    CHILE-02-REP1

Avoid:  TOM-00I      CHlLE-02       MANZ-I23   <- ambiguous I/l/1
        TOM001       CHILE02        MANZ123    <- no separators
        Tom-001      chile-02       Manz-123   <- mixed upper/lowercase
```

## Best Practices Summary

:octicons-check-circle-fill-24:{ .icon-green } **Do:**

- Use a scanner with a black background (covered box)
- Replace blade regularly for clean cuts
- Blot excess juice with a cloth on high-moisture fruits
- Clean the scanner with alcohol between scans
- Include a circular reference for accurate calibration
- Include QR codes for faster sample identification
- If using OCR, design labels with high contrast and clear fonts
- Keep the same DPI throughout the experiment

:octicons-x-circle-fill-24:{ .icon-red } **Avoid:**

- Variable lighting or reflections
- Non-fruit objects (stems, leaves, loose seeds) and debris in images
- Cutting with dull blades
- Mixing resolutions within the same batch
- Labels with ambiguous characters (`I`/`l`, `O`/`0`)
- Assuming scanner dimensions and reference circle sizes without verifying

</div>