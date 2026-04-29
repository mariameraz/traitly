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
from pathlib import Path
from typing import List, Optional

# ============================================================================
# THIRD-PARTY LIBRARIES
# ============================================================================
try:
    import fitz  # PyMuPDF

except ImportError:
    raise RuntimeError(
        'pdf_to_img function requires PyMuPDF. To install, run:\n'
        'pip install "traitly[pdf]"'
    )

# ============================================================================
# LOCAL LIBRARIES
# ============================================================================
from traitly.utils.label import detect_qr as det_qr

##############################################################################
# Convert PDF files to images
##############################################################################


def pdf_to_img(
    pdf_path: str,
    dpi: int = 150,
    output_path: Optional[str] = None,
    num_cores: Optional[int] = None,
    verbose: bool = True,
    detect_qr: bool = False,
    output_format: str = "jpg",
) -> List[str]:
    """
    Convert a PDF file (or all PDFs in a folder) to images, one per page.

    Each page is rendered at ``dpi`` resolution and saved in
    ``output_path``. When ``detect_qr=True``, a QR code is decoded from
    each page image and used to rename the file; pages without a
    detectable QR fall back to the default ``<pdf_name>_page<N>`` naming.

    Parameters
    ----------
    pdf_path : str
        Path to a single ``.pdf`` file or a folder containing PDF files.
    dpi : int, optional
        Rendering resolution in dots per inch. Default is 150.
    output_path : str or None, optional
        Directory to save output images. If ``None``, a subdirectory
        ``Images_from_PDF/`` is created next to the input PDF (or inside
        the input folder for batch mode). Default is ``None``.
    n_cores : int or None, optional
        Reserved for future parallel processing. Currently unused.
        Default is ``None``.
    verbose : bool, optional
        If True, print progress and summary messages. Default is True.
    detect_qr : bool, optional
        If True, detect QR codes in each page and rename the saved image
        using the decoded text. Default is False.
    output_format : str, optional
        Output image format. Supported values: ``'jpg'``, ``'jpeg'``,
        ``'png'``, ``'tiff'``, ``'tif'``, ``'ppm'``, ``'pnm'``,
        ``'pgm'``, ``'pbm'``, ``'pam'``. Default is ``'jpg'``.

    Returns
    -------
    list of str
        Absolute paths to all generated image files.

    Raises
    ------
    ValueError
        If ``output_format`` is not supported, the input is not a valid
        ``.pdf`` file or folder, or no PDFs are found in the folder.
    RuntimeError
        If the PDF conversion process fails unexpectedly.
    """
    valid_formats = [
        "jpg",
        "jpeg",
        "png",
        "tiff",
        "tif",
        "ppm",
        "pnm",
        "pgm",
        "pbm",
        "pam",
    ]  # Valid for PyMuPDF

    output_format = output_format.lower()

    if output_format not in valid_formats:
        raise ValueError(
            f"Invalid output format: '{output_format}'. "
            f"Supported formats are: {', '.join(valid_formats)}"
        )

    # Normalize suffix
    output_format = {"jpeg": "jpg", "tif": "tiff"}.get(output_format, output_format)

    if os.path.isdir(pdf_path):
        return _process_folder(
            folder_path=pdf_path,
            dpi=dpi,
            output_path=output_path,
            verbose=verbose,
            detect_qr=detect_qr,
            output_format=output_format,
            num_cores=num_cores,
        )
    elif os.path.isfile(pdf_path):
        return _process_single(
            pdf_path=pdf_path,
            dpi=dpi,
            output_path=output_path,
            verbose=verbose,
            detect_qr=detect_qr,
            output_format=output_format,
            num_cores=num_cores,
        )
    else:
        raise ValueError(f"Path not found: {pdf_path}")


##############################################################################
# Determine single or batch analysis
##############################################################################


def _process_folder(
    folder_path: str,
    dpi: int,
    output_path: Optional[str],
    verbose: bool,
    detect_qr: bool,
    output_format: str,
    num_cores: Optional[int] = None,
) -> List[str]:
    """Process all PDFs found in a folder."""

    pdf_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".pdf") and os.path.isfile(os.path.join(folder_path, f))
    ]

    if not pdf_files:
        raise ValueError(f"No PDF files found in directory: {folder_path}")

    if verbose:
        print("=" * 65, flush=True)
        print("Extracting images may take a few minutes... ⋆✧｡٩(ˊᗜˋ )و✧*｡", flush=True)
        print("=" * 65, flush=True)
        print(f"> Processing {len(pdf_files)} PDF file(s):", flush=True)

    all_saved_paths = []

    for pdf_file in pdf_files:
        paths = _process_single(
            pdf_file,
            dpi,
            output_path,
            verbose=False,
            detect_qr=detect_qr,
            output_format=output_format,
            num_cores=num_cores,
        )
        all_saved_paths.extend(paths)

    if verbose:
        qr_detected = sum(
            1 for p in all_saved_paths if "_page" not in os.path.basename(p)
        )

        final_dir = (
            output_path if output_path else os.path.join(folder_path, "Images_from_PDF")
        )
        print(f"    – Images extracted: {len(all_saved_paths)}")
        if qr_detected > 0:
            print(f"    – QR detected: {qr_detected}/{len(all_saved_paths)} img(s)")
        if num_cores is not None:
            print(f"    – num_cores: {num_cores}")
        print(f"    – Results folder: {os.path.abspath(final_dir)}")

    return all_saved_paths


def _process_single(
    pdf_path: str,
    dpi: int,
    output_path: Optional[str],
    verbose: bool,
    detect_qr: bool,
    output_format: str,
    num_cores: Optional[int] = None,
) -> List[str]:
    """Process a single PDF file and convert each page to an image."""
    pdf_path = Path(pdf_path)

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Input file must be a PDF (.pdf extension)")

    pdf_dir = os.path.dirname(os.path.abspath(pdf_path))
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    out_dir = (
        os.path.abspath(output_path)
        if output_path
        else os.path.join(pdf_dir, "Images_from_PDF")
    )
    os.makedirs(out_dir, exist_ok=True)

    if verbose:
        print("=" * 65, flush=True)
        print("Extracting images may take a few minutes... ⋆✧｡٩(ˊᗜˋ )و✧*｡", flush=True)
        print("=" * 65, flush=True)
        print("> Processing 1 PDF file:", flush=True)

    try:
        doc = fitz.open(pdf_path)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)

        saved_paths = []
        used_names = {}
        qr_detected = 0

        for i in range(len(doc)):
            page = doc[i]
            pix = page.get_pixmap(matrix=mat)

            img_name = f"{pdf_name}_page{i + 1}.{output_format}"
            img_path = os.path.join(out_dir, img_name)
            pix.save(img_path)

            if detect_qr:
                qr_text, _ = det_qr(img_path=img_path)
                if qr_text and qr_text != "No QR code detected":
                    cleaned_name = _clean_filename(qr_text)

                    if cleaned_name in used_names:
                        used_names[cleaned_name] += 1
                        final_name = (
                            f"{cleaned_name}_{used_names[cleaned_name]}.{output_format}"
                        )
                    else:
                        used_names[cleaned_name] = 0
                        final_name = f"{cleaned_name}.{output_format}"

                    new_path = os.path.join(out_dir, final_name)
                    Path(img_path).replace(new_path)
                    img_path = new_path
                    qr_detected += 1

            saved_paths.append(img_path)

        doc.close()

        if verbose:
            print(f"    – Images extracted: {len(saved_paths)}")
            if detect_qr:
                print(f"    – QR detected: {qr_detected}/{len(saved_paths)} img(s)")
            if num_cores is not None:
                print(f"    – num_cores: {num_cores}")
            print(f"    – Results folder: {out_dir}")

        return saved_paths

    except Exception as e:
        raise RuntimeError(f"PDF conversion error: {str(e)}") from e


##############################################################################
# Filename cleaner helper
##############################################################################


def _clean_filename(text: str, max_length: int = 100) -> str:
    """
    clean arbitrary text so it is safe to use as a filename.
    """
    cleaned = re.sub(r"[^\w\s-]", "_", text)
    cleaned = re.sub(r"[\s_]+", "_", cleaned)
    cleaned = cleaned.strip("_")

    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length]

    return cleaned if cleaned else "unnamed"
