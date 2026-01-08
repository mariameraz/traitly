# traitly/utils/convert_pdf.py

# ============================================================================
# STANDARD LIBRARY
# ============================================================================
import os
from typing import List, Optional

# ============================================================================
# THRID-PARTY LIBRARIES
# ============================================================================
import fitz  # PyMuPDF
import re

# ============================================================================
# LOCAL LIBRARIES
# ============================================================================
from traitly.utils.common_functions import detect_qr

try:
    import fitz  # pymupdf
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

if not HAS_PYMUPDF:
    raise RuntimeError(
        "pdf_to_img function requires PyMuPDF. To install, run:\n"
        "pip install traitly[pdf]"
    )

###################################################
# Converts PDF files to images using PyMuPDF (fitz) #
###################################################

def pdf_to_img(pdf_path: str, dpi: int = 300, output_dir: Optional[str] = None, 
               n_threads: Optional[int] = None, output_message: bool = True,
               qr_label: bool = False, output_format: str = 'jpg') -> List[str]:
    """
    Converts a PDF file (or all PDFs in a folder) to images (one per page).
    
    Args:
        pdf_path: Path to the input PDF file or folder containing PDF files.
        dpi: Conversion resolution (dots per inch).
        output_dir: Directory to save the images. If None, creates 'images_from_pdf' in the same folder as the PDF.
        n_threads: Number of threads for parallel processing. If None, uses 1 thread.
        output_message: Whether to print progress messages.
        qr_label: If True, detects QR codes and renames images with QR text. Falls back to default naming if no QR detected.
        output_format: Image format for output. Options: 'jpg', 'jpeg', 'png', 'tiff', 'tif', 'ppm', 'pnm', 'pgm', 'pbm', 'pam'. Default is 'jpg'.
    
    Returns:
        List of paths to the generated and renamed image files.
    
    Raises:
        ValueError: If the input file is not a valid PDF or folder contains no PDFs, or if output_format is invalid.
        RuntimeError: If the conversion process fails.
    """
    # Validate output format
    valid_formats = ['jpg', 'jpeg', 'png', 'tiff', 'tif', 'ppm', 'pnm', 'pgm', 'pbm', 'pam']
    output_format = output_format.lower()
    
    if output_format not in valid_formats:
        raise ValueError(f"Invalid output format '{output_format}'. Valid formats: {', '.join(valid_formats)}")
    
    # Normalize format extensions
    if output_format == 'jpeg':
        output_format = 'jpg'
    elif output_format == 'tif':
        output_format = 'tiff'
    
    # Check if path is a directory or file
    if os.path.isdir(pdf_path):
        # Find all PDF files in the directory
        pdf_files = [os.path.join(pdf_path, f) for f in os.listdir(pdf_path) 
                     if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(pdf_path, f))]
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in directory: {pdf_path}")
        
        if output_message:
            print(f"Extracting images may take a few minutes... ⋆✧｡٩(ˊᗜˋ )و✧*｡")
            print(f"Processing {len(pdf_files)} PDF files...")
        
        # Process all PDFs without individual messages
        all_saved_paths = []
        for pdf_file in pdf_files:
            paths = pdf_to_img(pdf_file, dpi=dpi, output_dir=output_dir, 
                             n_threads=n_threads, output_message=False, 
                             qr_label=qr_label, output_format=output_format)
            all_saved_paths.extend(paths)
        
        # Print final summary
        if output_message:
            final_output_dir = output_dir if output_dir else os.path.join(pdf_path, 'images_from_pdf')
            print(f"{len(all_saved_paths)} images saved in: {final_output_dir}")
        
        return all_saved_paths
    
    # Input validation for single file
    if not os.path.isfile(pdf_path):
        raise ValueError(f"File not found: {pdf_path}")
    
    if not pdf_path.lower().endswith('.pdf'):
        raise ValueError("Input file must be a PDF (.pdf extension)")
    
    # Set up output paths
    pdf_dir = os.path.dirname(pdf_path)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    if output_dir is None:
        output_dir = os.path.join(pdf_dir, 'images_from_pdf')
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        if output_message:
            print("Extracting images may take a few minutes... ⋆✧｡٩(ˊᗜˋ )و✧*｡")
        
        # Convert PDF to images using PyMuPDF
        doc = fitz.open(pdf_path)
        
        # Calculate zoom factor from DPI (72 is the base DPI in PDF)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        
        # Save images
        saved_paths = []
        used_names = {}  # Track used names to avoid duplicates
        
        for i in range(len(doc)):
            page = doc[i]
            pix = page.get_pixmap(matrix=mat)
            
            # Default image name with specified format
            img_name = f"{pdf_name}_page{i+1}.{output_format}"
            output_path = os.path.join(output_dir, img_name)
            
            # Save with default name first
            pix.save(output_path)
            
            # If qr_label is True, try to detect QR and rename
            if qr_label:
                qr_text, _ = detect_qr(img_path=output_path)
                
                # Debug print
                if output_message:
                    print(f"Page {i+1} - QR detected: {qr_text}")
                
                # If QR detected and has valid text
                if qr_text and qr_text != 'No QR code detected':
                    # Sanitize QR text for filename
                    sanitized_name = _sanitize_filename(qr_text)
                    
                    if output_message:
                        print(f"  Sanitized name: {sanitized_name}")
                    
                    # Handle duplicate names
                    if sanitized_name in used_names:
                        used_names[sanitized_name] += 1
                        final_name = f"{sanitized_name}_{used_names[sanitized_name]}.{output_format}"
                    else:
                        used_names[sanitized_name] = 0
                        final_name = f"{sanitized_name}.{output_format}"
                    
                    # Rename file
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


def _sanitize_filename(text: str) -> str:
    """
    Sanitizes text to be used as a filename by removing or replacing invalid characters.
    
    Args:
        text: The text to sanitize.
    
    Returns:
        A sanitized string safe for use as a filename.
    """
    # Remove or replace invalid filename characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', text)
    # Remove leading/trailing whitespace and dots
    sanitized = sanitized.strip('. ')
    # Limit length to avoid filesystem issues
    if len(sanitized) > 200:
        sanitized = sanitized[:200]
    return sanitized if sanitized else 'unnamed'



def _sanitize_filename(text: str, max_length: int = 100) -> str:
    """
    Sanitizes text to be a valid filename.
    
    Args:
        text: Text to sanitize
        max_length: Maximum length for the filename
    
    Returns:
        Sanitized filename (without extension)
    """
    # Remove or replace invalid characters for filenames
    # Keep alphanumeric, spaces, hyphens, and underscores
    sanitized = re.sub(r'[^\w\s-]', '_', text)
    
    # Replace multiple spaces/underscores with single underscore
    sanitized = re.sub(r'[\s_]+', '_', sanitized)
    
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    
    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    # If empty after sanitization, return default
    if not sanitized:
        sanitized = "unnamed"
    
    return sanitized
