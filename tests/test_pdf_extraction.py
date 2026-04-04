# tests/test_pdf_extraction.py

# ============================================================================
# STANDARD LIBRARY
# ============================================================================
from pathlib import Path

# ============================================================================
# THIRD-PARTY
# ============================================================================
import pytest

# ============================================================================
# INTERNAL IMPORTS
# ============================================================================
from traitly.utils.convert_pdf import pdf_to_img

pdf = Path(__file__).parent / "data/internal/cranberry_slices.pdf"


def test_pdf_extraction():
    temp = pdf_to_img(pdf_path=pdf, dpi=70, detect_qr=False, output_format="jpg")
    assert temp is not None
    assert len(temp) == 2
    assert all("_page" in Path(p).stem for p in temp)


def test_rename_with_qr():
    temp = pdf_to_img(pdf_path=pdf, dpi=150, detect_qr=True, output_format="jpg")
    assert temp is not None
    assert len(temp) == 2
    assert all("_page" not in Path(p).stem for p in temp)


valid_ext_msg = "Invalid output format: 'gif'. Supported formats are: \
jpg, jpeg, png, tiff, tif, ppm, pnm, pgm, pbm, pam"


def test_invalid_ext():
    with pytest.raises(ValueError, match=valid_ext_msg):
        temp = pdf_to_img(pdf_path=pdf, dpi=70, detect_qr=False, output_format="gif")


path_folder = Path(__file__).parent / "data/internal"


def test_folder_path():
    temp = pdf_to_img(
        pdf_path=path_folder, dpi=70, detect_qr=False, output_format="jpg"
    )
    assert temp is not None
    assert len(temp) == 2
