# traitly/utils/convert_pdf.py
"""
PDF-to-image conversion utilities for traitly.

Provides :func:`pdf_to_img` to convert single PDF files or entire
folders of PDFs to images using PyMuPDF (``fitz``). Supports optional
QR code detection for automatic renaming of output files.
"""

# ============================================================================
# STANDARD LIBRARY
# ============================================================================
import os
import re
from typing import List, Optional

# ============================================================================
# THIRD-PARTY LIBRARIES
# ============================================================================
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

if not HAS_PYMUPDF:
    raise RuntimeError(
        "pdf_to_img function requires PyMuPDF. To install, run:\n"
        "pip install traitly[pdf]"
    )

# ============================================================================
# LOCAL LIBRARIES
# ============================================================================
from traitly.utils.label import detect_qr

##############################################################################
# Convert PDF files to images
##############################################################################

def pdf_to_img(
    pdf_path: str,
    dpi: int = 300,
    output_dir: Optional[str] = None,
    n_threads: Optional[int] = None,
    output_message: bool = True,
    qr_label: bool = False,
    output_format: str = 'jpg',
) -> List[str]:
    """
    Convert a PDF file (or all PDFs in a folder) to images, one per page.

    Each page is rendered at ``dpi`` resolution and saved in
    ``output_dir``. When ``qr_label=True``, a QR code is decoded from
    each page image and used to rename the file; pages without a
    detectable QR fall back to the default ``<pdf_name>_page<N>`` naming.

    Parameters
    ----------
    pdf_path : str
        Path to a single ``.pdf`` file or a folder containing PDF files.
        When a folder is provided, all PDFs inside are processed
        recursively with a single summary message.
    dpi : int, optional
        Rendering resolution in dots per inch. Default is 300.
    output_dir : str or None, optional
        Directory to save output images. If ``None``, a subdirectory
        ``images_from_pdf/`` is created next to the input PDF (or inside
        the input folder for batch mode). Default is ``None``.
    n_threads : int or None, optional
        Reserved for future parallel processing. Currently unused.
        Default is ``None``.
    output_message : bool, optional
        If True, print progress and summary messages. Default is True.
    qr_label : bool, optional
        If True, detect QR codes in each page and rename the saved image
        using the decoded text. Falls back to default naming when no QR
        is found. Default is False.
    output_format : str, optional
        Output image format. Supported values: ``'jpg'``, ``'jpeg'``,
        ``'png'``, ``'tiff'``, ``'tif'``, ``'ppm'``, ``'pnm'``,
        ``'pgm'``, ``'pbm'``, ``'pam'``. ``'jpeg'`` is normalized to
        ``'jpg'`` and ``'tif'`` to ``'tiff'``. Default is ``'jpg'``.

    Returns
    -------
    list of str
        Absolute paths to all generated (and possibly renamed) image
        files.

    Raises
    ------
    ValueError
        If ``output_format`` is not supported, the input file is not a
        ``.pdf``, the file is not found, or no PDF files exist in the
        provided folder.
    RuntimeError
        If the PDF conversion process fails unexpectedly.
    """
    valid_formats = [
        'jpg', 'jpeg', 'png', 'tiff', 'tif',
        'ppm', 'pnm', 'pgm', 'pbm', 'pam',
    ]
    output_format = output_format.lower()

    if output_format not in valid_formats:
        raise ValueError(
            f"Invalid output format: '{output_format}'. "
            f"Supported formats are: {', '.join(valid_formats)}"
        )

    # Normalize aliases
    if output_format == 'jpeg':
        output_format = 'jpg'
    elif output_format == 'tif':
        output_format = 'tiff'

    # ── Batch mode: folder of PDFs ─────────────────────────────────────────
    if os.path.isdir(pdf_path):
        pdf_files = [
            os.path.join(pdf_path, f)
            for f in os.listdir(pdf_path)
            if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(pdf_path, f))
        ]

        if not pdf_files:
            raise ValueError(f"No PDF files found in directory: {pdf_path}")

        if output_message:
            print("Extracting images may take a few minutes... ⋆✧｡٩(ˊᗜˋ )و✧*｡")
            print(f"Processing {len(pdf_files)} PDF files...")

        all_saved_paths = []
        for pdf_file in pdf_files:
            paths = pdf_to_img(
                pdf_file, dpi=dpi, output_dir=output_dir,
                n_threads=n_threads, output_message=False,
                qr_label=qr_label, output_format=output_format,
            )
            all_saved_paths.extend(paths)

        if output_message:
            final_dir = output_dir if output_dir else os.path.join(pdf_path, 'images_from_pdf')
            print(f"{len(all_saved_paths)} images saved in: {final_dir}")

        return all_saved_paths

    # ── Single file mode ───────────────────────────────────────────────────
    if not os.path.isfile(pdf_path):
        raise ValueError(f"File not found: {pdf_path}")

    if not pdf_path.lower().endswith('.pdf'):
        raise ValueError("Input file must be a PDF (.pdf extension)")

    pdf_dir  = os.path.dirname(pdf_path)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    if output_dir is None:
        output_dir = os.path.join(pdf_dir, 'images_from_pdf')

    os.makedirs(output_dir, exist_ok=True)

    try:
        if output_message:
            print("Extracting images may take a few minutes... ⋆✧｡٩(ˊᗜˋ )و✧*｡")

        doc  = fitz.open(pdf_path)
        zoom = dpi / 72
        mat  = fitz.Matrix(zoom, zoom)

        saved_paths = []
        used_names  = {}

        for i in range(len(doc)):
            page = doc[i]
            pix  = page.get_pixmap(matrix=mat)

            img_name    = f"{pdf_name}_page{i + 1}.{output_format}"
            output_path = os.path.join(output_dir, img_name)

            pix.save(output_path)

            if qr_label:
                qr_text, _ = detect_qr(img_path=output_path)

                if output_message:
                    print(f"Page {i + 1} - QR detected: {qr_text}")

                if qr_text and qr_text != 'No QR code detected':
                    sanitized_name = _sanitize_filename(qr_text)

                    if output_message:
                        print(f"  Sanitized name: {sanitized_name}")

                    if sanitized_name in used_names:
                        used_names[sanitized_name] += 1
                        final_name = f"{sanitized_name}_{used_names[sanitized_name]}.{output_format}"
                    else:
                        used_names[sanitized_name] = 0
                        final_name = f"{sanitized_name}.{output_format}"

                    new_path = os.path.join(output_dir, final_name)

                    if output_message:
                        print(f"  Renaming: {img_name} -> {final_name}")

                    os.rename(output_path, new_path)
                    output_path = new_path

            saved_paths.append(output_path)

        doc.close()

        if output_message and not qr_label:
            print(f"{len(saved_paths)} images saved in: {output_dir}")

        return saved_paths

    except Exception as e:
        error_msg = f"PDF conversion error: {str(e)}"
        if output_message:
            print(error_msg)
        raise RuntimeError(error_msg) from e


##############################################################################
# Filename sanitization helper
##############################################################################

def _sanitize_filename(text: str, max_length: int = 100) -> str:
    """
    Sanitize arbitrary text so it is safe to use as a filename.

    Replaces characters that are illegal on common filesystems with
    underscores, collapses runs of whitespace or underscores, strips
    leading/trailing underscores, and truncates to ``max_length``.

    Parameters
    ----------
    text : str
        The text to sanitize (e.g. a QR code payload).
    max_length : int, optional
        Maximum character length of the returned string. Default is 100.

    Returns
    -------
    str
        Sanitized filename string without an extension. Returns
        ``'unnamed'`` if the sanitized result would be empty.
    """
    # Replace characters that are invalid in filenames
    sanitized = re.sub(r'[^\w\s-]', '_', text)

    # Collapse multiple whitespace or underscores into one
    sanitized = re.sub(r'[\s_]+', '_', sanitized)

    # Strip leading/trailing underscores
    sanitized = sanitized.strip('_')

    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]

    return sanitized if sanitized else 'unnamed'