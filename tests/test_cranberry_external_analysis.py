# tests/test_external_analyzer.py

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
from traitly.fruit_phenotyping import FruitExternalAnalyzer

##########################################################################
# Valid cranberry image
##########################################################################

white_bg = Path(__file__).parent / "data/external/cranberry_white_bg.jpg"


@pytest.fixture
def cranberry_white():
    cranberry = FruitExternalAnalyzer(image_path=white_bg)
    cranberry.load_image(plot=False)
    cranberry.setup_measurements()
    return cranberry


def test_cranberry_white_bg(cranberry_white):
    cranberry_white.generate_fruit_mask(plot=False, background_color="white")
    cranberry_white.detect_fruits(plot=False)
    cranberry_white.analyze_morphology(plot=False, display_table=False)
    cranberry_white.analyze_color(plot=False, display_table=False)


def test_invalid_bg_color(cranberry_white):
    with pytest.raises(RuntimeError):
        cranberry_white.generate_fruit_mask(plot=False, background_color="red")


def test_skip_setup_measurements(cranberry_white):
    cranberry_white.generate_fruit_mask(plot=False, background_color="white")
    cranberry_white.detect_fruits(plot=False)
    cranberry_white.analyze_morphology(plot=False, display_table=False)
    cranberry_white.analyze_color(plot=False, display_table=False)


blue_bg = Path(__file__).parent / "data/external/cranberry_blue_bg.jpg"


@pytest.fixture
def cranberry_blue():
    cranberry = FruitExternalAnalyzer(image_path=blue_bg)
    cranberry.load_image(plot=False)
    return cranberry


def test_cranberry_blue_bg(cranberry_blue):
    cranberry_blue.setup_measurements()
    assert cranberry_blue.ref_roi
    cranberry_blue.generate_fruit_mask(plot=False, background_color="blue")
    cranberry_blue.detect_fruits(plot=False)
    cranberry_blue.analyze_morphology(plot=False, display_table=False)


def test_detect_label(cranberry_blue):
    cranberry_blue.setup_measurements(detect_label=True)
    assert cranberry_blue.label_roi
    assert cranberry_blue.label_text


def test_detect_color_checker(cranberry_blue):
    cranberry_blue.setup_measurements(detect_label=False, detect_color_checker=True)
    assert cranberry_blue.checker_coords


def test_skip_yolo_model(cranberry_blue):
    cranberry_blue.setup_measurements(skip_yolo=True)
    assert cranberry_blue.ref_roi is None
