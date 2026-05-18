# tests/pdf/test_pdf_extraction.py

# ============================================================================
# STANDARD LIBRARY
# ============================================================================
from pathlib import Path

# ============================================================================
# THIRD-PARTY
# ============================================================================
import pytest

# ============================================================================
# INTERNAL
# ============================================================================
from traitly.pdf import pdf_to_img

path_folder = Path(__file__).parent.parent / "data" / "internal"

pdf = path_folder / "cranberry_slices.pdf"


def test_pdf_extraction(tmp_path):
    temp = pdf_to_img(pdf_path=pdf, dpi=70, detect_qr=False, output_format="jpg", output_path = tmp_path)
    assert temp is not None
    assert len(temp) == 1
    assert all("_page" in Path(p).stem for p in temp)


def test_rename_with_qr(tmp_path):
    temp = pdf_to_img(pdf_path=pdf, dpi=150, detect_qr=True, output_format="jpg", output_path = tmp_path)
    assert temp is not None
    assert len(temp) == 1
    assert all("_page" not in Path(p).stem for p in temp)


valid_ext_msg = "Invalid output format: 'gif'. Supported formats are: \
jpg, jpeg, png, tiff, tif, ppm, pnm, pgm, pbm, pam"


def test_invalid_ext(tmp_path):
    with pytest.raises(ValueError, match=valid_ext_msg):
        temp = pdf_to_img(pdf_path=pdf, dpi=70, detect_qr=False, output_format="gif", output_path = tmp_path)


def test_folder_path(tmp_path):
    temp = pdf_to_img(
        pdf_path=path_folder, dpi=70, detect_qr=False, output_format="jpg", output_path = tmp_path
    )
    assert temp is not None
    assert len(temp) == 1

# Unique image names
pdf_duplicates =  Path(__file__).parent.parent / "pdf" / "duplicated_image" / "cranberry_slices_duplicated.pdf"

def test_duplicate_qr(tmp_path):
    temp = pdf_to_img(
        pdf_path=pdf_duplicates, dpi=120, detect_qr=True, output_format="jpg", output_path = tmp_path
    )
    assert len(temp) == 3
    stems = [Path(p).stem for p in temp]
    assert len(set(stems)) == 3
    assert any(stem.endswith("_1") for stem in stems)
    assert any(stem.endswith("_2") for stem in stems)
