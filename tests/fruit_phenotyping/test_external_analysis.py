# tests/fruit_phenotyping/test_external_analysis.py

# ============================================================================
# STANDARD LIBRARY
# ============================================================================
import copy
from pathlib import Path

# ============================================================================
# THIRD-PARTY
# ============================================================================
import pytest
import cv2

# ============================================================================
# INTERNAL
# ============================================================================
from traitly.fruit_phenotyping import FruitExternalAnalyzer

white_bg = Path(__file__).parent.parent / "data" / "external" / "white_bg" / "cranberry_white_bg.jpg"
blue_bg = Path(__file__).parent.parent / "data" / "external" / "blue_bg" / "cranberry_blue_bg.jpg"

@pytest.fixture(scope="module")
def _white_base():
    """Loaded + measured cranberry (white bg), default setup_measurements()."""
    cranberry = FruitExternalAnalyzer(path=white_bg)
    cranberry.load_image(plot=False)
    cranberry.setup_measurements()
    return cranberry

@pytest.fixture(scope="module")
def _blue_base():
    """Loaded + measured cranberry (blue bg), default setup_measurements()."""
    cranberry = FruitExternalAnalyzer(path=blue_bg)
    cranberry.load_image(plot=False)
    cranberry.setup_measurements()
    return cranberry

@pytest.fixture(scope="module")
def _blue_base_labeled():
    """Loaded + measured cranberry (blue bg) with label/QR detection on."""
    cranberry = FruitExternalAnalyzer(path=blue_bg)
    cranberry.load_image(plot=False)
    cranberry.setup_measurements(detect_label=True)
    return cranberry

@pytest.fixture(scope="module")
def _blue_base_no_label_no_yolo():
    """Loaded + measured cranberry (blue bg), label off, YOLO skipped entirely."""
    cranberry = FruitExternalAnalyzer(path=blue_bg)
    cranberry.load_image(plot=False)
    cranberry.setup_measurements(detect_label=False, skip_yolo=True)
    return cranberry

@pytest.fixture(scope="module")
def _blue_base_full_yolo():
    """Loaded + measured cranberry (blue bg) with YOLO label detection enabled.
    """
    cranberry = FruitExternalAnalyzer(path=blue_bg)
    cranberry.load_image(plot=False)
    cranberry.setup_measurements(detect_label=True, skip_yolo=False)
    return cranberry

# ============================================================================
# FUNCTION FIXTURES
# ============================================================================
@pytest.fixture
def cranberry_white(_white_base):
    return copy.deepcopy(_white_base)

@pytest.fixture
def cranberry_blue(_blue_base):
    return copy.deepcopy(_blue_base)


@pytest.fixture
def cranberry_blue_labeled(_blue_base_labeled):
    return copy.deepcopy(_blue_base_labeled)

@pytest.fixture
def cranberry_blue_no_label_no_yolo(_blue_base_no_label_no_yolo):
    return copy.deepcopy(_blue_base_no_label_no_yolo)

@pytest.fixture
def cranberry_blue_full_yolo(_blue_base_full_yolo):
    return copy.deepcopy(_blue_base_full_yolo)

# ============================================================================
# WHITE BACKGROUND TESTS
# ============================================================================
def test_circularity_valid_ranges(cranberry_white):
    cranberry_white.generate_fruit_mask(plot=False, background_color="white")
    cranberry_white.detect_fruits(plot=False)
    cranberry_white.analyze_morphology(plot=False, display_table=False)
    df = cranberry_white.results.morphology_results
    assert df["fruit_circularity"].between(0, 1).all()


def test_fruits_detected(cranberry_white):
    cranberry_white.generate_fruit_mask(plot=False, background_color="white")
    cranberry_white.detect_fruits(plot=False)
    assert len(cranberry_white.fruit_locule_map) > 0

def test_cranberry_white_bg_full_pipeline(cranberry_white):
    """End-to-end smoke test: mask -> detect -> morphology -> color -> save."""
    cranberry_white.generate_fruit_mask(plot=False, background_color="white")
    cranberry_white.detect_fruits(plot=False)
    cranberry_white.analyze_morphology(plot=False, display_table=False)
    cranberry_white.analyze_color(plot=False, display_table=False)
    cranberry_white.save_parameters()


def test_invalid_bg_color(cranberry_white):
    with pytest.raises(RuntimeError):
        cranberry_white.generate_fruit_mask(plot=False, background_color="red")

def test_pipeline_without_setup_measurements():
    """Pipeline should still run when setup_measurements() is never called."""
    cranberry = FruitExternalAnalyzer(path=white_bg)
    cranberry.load_image(plot=False)
    cranberry.generate_fruit_mask(plot=False, background_color="white")
    cranberry.detect_fruits(plot=False)
    cranberry.analyze_morphology(plot=False, display_table=False)
    cranberry.analyze_color(plot=False, display_table=False)

# ============================================================================
# BLUE BACKGROUND TESTS (default setup_measurements, no label detection)
# ============================================================================

def test_cranberry_blue_bg(cranberry_blue, tmp_path):
    cranberry_blue.setup_measurements()
    assert cranberry_blue._ref_roi
    cranberry_blue.generate_fruit_mask(plot=False, background_color="blue")
    cranberry_blue.detect_fruits(plot=False)
    cranberry_blue.analyze_color(display_table=False, get_color_histogram=True)
    cranberry_blue.results.save_csv(output_path=tmp_path, base_name="hist")
    cranberry_blue.analyze_morphology(plot=False, display_table=False)
    cranberry_blue.save_parameters()

    unwanted_file = tmp_path / "hist_morphology_results.csv"
    wanted_file = tmp_path / "hist_color_results.csv"
    assert not unwanted_file.exists(), f"File {unwanted_file} shouldn't be created"
    assert wanted_file.exists(), f"File {wanted_file} was not created"

def test_invalid_tissue(cranberry_blue):
    cranberry_blue.generate_fruit_mask(plot=False, background_color="blue")
    cranberry_blue.detect_fruits(plot=False)

    with pytest.raises(TypeError):
        cranberry_blue.analyze_color(
            plot=False, display_table=False, tissue="outer_pericarp"
        )

def test_analyze_color_channels(cranberry_blue):
    cranberry_blue.generate_fruit_mask(plot=False, background_color="blue")
    cranberry_blue.detect_fruits(plot=False)
    cranberry_blue.analyze_color(plot=False, display_table=False, color_space="lab")
    df = cranberry_blue.results.color_results
    assert df is not None
    assert len(df) > 0

# ============================================================================
# BLUE BACKGROUND TESTS (label detection enabled, QR/YOLO fallback allowed)
# ============================================================================
def test_detect_label(cranberry_blue_labeled):
    assert cranberry_blue_labeled._label_roi
    assert cranberry_blue_labeled.label_text
    assert cranberry_blue_labeled.img.any()
    assert cranberry_blue_labeled.img_name
    assert cranberry_blue_labeled._img_copy.any()
    assert cranberry_blue_labeled._img_rgb.any()
    assert cranberry_blue_labeled._img_hsv.any()
    assert cranberry_blue_labeled.l_transformed is None


def test_L_contrast(cranberry_blue_labeled):
    cranberry_blue_labeled.generate_fruit_mask(plot=False)
    cranberry_blue_labeled.enhance_locule_contrast(plot=False)
    assert cranberry_blue_labeled.l_transformed.any()

# ============================================================================
# BLUE BACKGROUND TESTS (label off, YOLO skipped)
# ============================================================================

def test_create_mask_attributes(cranberry_blue_no_label_no_yolo):
    cranberry_blue_no_label_no_yolo.generate_fruit_mask(plot=False)
    assert cranberry_blue_no_label_no_yolo.mask_fruit is not None
    assert cranberry_blue_no_label_no_yolo.mask_fruit.any()
    assert cranberry_blue_no_label_no_yolo.mask_locules is None
    assert cranberry_blue_no_label_no_yolo.contours is None
    assert cranberry_blue_no_label_no_yolo.fruit_locule_map is None

# ============================================================================
# YOLO label-detection path
# ============================================================================
@pytest.mark.slow
def test_setup_measurements_attributes(cranberry_blue_full_yolo):
    assert cranberry_blue_full_yolo.label_text
    assert cranberry_blue_full_yolo._label_roi
    assert cranberry_blue_full_yolo._ref_roi
    assert cranberry_blue_full_yolo.px_per_cm
