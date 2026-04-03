# tests/test_internal_analyzer.py

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
from traitly.fruit_phenotyping import FruitInternalAnalyzer

##########################################################################
# Valid cranberry image
##########################################################################

valid_img = Path(__file__).parent / "data/internal/img_test_1.jpg"


@pytest.fixture
def cranberry_valid():
    cranberry = FruitInternalAnalyzer(image_path=valid_img)
    cranberry.load_image(plot=False)
    cranberry.setup_measurements()
    cranberry.generate_fruit_mask(plot=False)
    cranberry.detect_fruits(plot=False)
    cranberry.analyze_morphology(plot=False, display_table=False)
    return cranberry


def test_load_image():
    cranberry = FruitInternalAnalyzer(image_path=valid_img)
    cranberry.load_image(plot=False)
    assert cranberry.img is not None


def test_morphology_columns(cranberry_valid):
    df = cranberry_valid.results.morphology_results
    assert "fruit_area_cm2" in df.columns
    assert "fruit_circularity" in df.columns
    assert len(df) == 25


def test_circularity_valid_ranges(cranberry_valid):
    df = cranberry_valid.results.morphology_results
    assert df["fruit_circularity"].between(0, 1).all()


def test_skip_yolo():
    cranberry = FruitInternalAnalyzer(image_path=valid_img)
    cranberry.load_image(plot=False)
    cranberry.setup_measurements(skip_yolo=True)
    cranberry.generate_fruit_mask(plot=False)
    cranberry.detect_fruits(plot=False)
    cranberry.analyze_morphology(plot=False, display_table=False)
    df = cranberry.results.morphology_results
    assert "fruit_area_px2" in df.columns
    assert "fruit_perimeter_px" in df.columns


##########################################################################
# Invalid image path
##########################################################################

invalid_path = Path(__file__).parent / "data/internal/img_test_5.jpg"


def test_invalid_path():
    with pytest.raises((FileNotFoundError, ValueError)):
        cranberry = FruitInternalAnalyzer(image_path=invalid_path)
        cranberry.load_image(plot=False)


##########################################################################
# Slices with not empty locules
##########################################################################

invalid_img = Path(__file__).parent / "data/internal/img_test_4.jpg"


def test_invalid_locule_segmentation():
    cranberry = FruitInternalAnalyzer(image_path=invalid_img)
    cranberry.load_image(plot=False)
    cranberry.setup_measurements(skip_yolo=True)
    cranberry.generate_fruit_mask(plot=False)
    cranberry.detect_fruits(plot=False)
    assert len(cranberry.fruit_locule_map.items()) == 0
