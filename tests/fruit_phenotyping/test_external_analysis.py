# tests/fruit_phenotyping/test_external_analysis.py

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
from traitly.fruit_phenotyping import FruitExternalAnalyzer


white_bg = Path(__file__).parent.parent / "data" / "external" / "white_bg" / "cranberry_white_bg.jpg"

@pytest.fixture
def cranberry_white():
    cranberry = FruitExternalAnalyzer(path=white_bg)
    cranberry.load_image(plot=False)
    cranberry.setup_measurements()
    return cranberry

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

def test_cranberry_white_bg(cranberry_white):
    cranberry_white.generate_fruit_mask(plot=False, background_color="white")
    cranberry_white.detect_fruits(plot=False)
    cranberry_white.analyze_morphology(plot=False, display_table=False)
    cranberry_white.analyze_color(plot=False, display_table=False)
    cranberry_white.save_parameters()


def test_invalid_bg_color(cranberry_white):
    with pytest.raises(RuntimeError):
        cranberry_white.generate_fruit_mask(plot=False, background_color="red")


def test_skip_setup_measurements(cranberry_white):
    cranberry_white.generate_fruit_mask(plot=False, background_color="white")
    cranberry_white.detect_fruits(plot=False)
    cranberry_white.analyze_morphology(plot=False, display_table=False)
    cranberry_white.analyze_color(plot=False, display_table=False)


blue_bg = Path(__file__).parent.parent / "data" / "external" / "blue_bg" / "cranberry_blue_bg.jpg"


@pytest.fixture
def cranberry_blue():
    cranberry = FruitExternalAnalyzer(path=blue_bg)
    cranberry.load_image(plot=False)
    return cranberry


def test_cranberry_blue_bg(cranberry_blue):
    cranberry_blue.setup_measurements()
    assert cranberry_blue.ref_roi
    cranberry_blue.generate_fruit_mask(plot=False, background_color="blue")
    cranberry_blue.detect_fruits(plot=False)
    cranberry_blue.analyze_color(display_table = False, get_color_histogram = True)
    cranberry_blue.results.save_csv(base_name = 'hist')
    cranberry_blue.analyze_morphology(plot=False, display_table=False)
    cranberry_blue.save_parameters()

    unwanted_file = Path("cranberry_blue") / "Results" / "hist_morphology.csv"
    wanted_file = Path("cranberry_blue") / "Results" / "hist_color.csv"
    assert not unwanted_file.exists(), (f"File {unwanted_file} shouldn't be created")
    assert not wanted_file.exists(), (f"File {wanted_file} created successfully")


def test_detect_label(cranberry_blue):
    cranberry_blue.setup_measurements(detect_label=True)
    assert cranberry_blue.label_roi
    assert cranberry_blue.label_text
    assert cranberry_blue.img.any()
    assert cranberry_blue.img_name
    assert cranberry_blue.img_copy.any()
    assert cranberry_blue.img_rgb.any()
    assert cranberry_blue.img_hsv.any()
    assert cranberry_blue.l_transformed is None

def test_L_contrast(cranberry_blue):
    cranberry_blue.setup_measurements(detect_label=True)
    cranberry_blue.generate_fruit_mask(plot = False)
    cranberry_blue.enhance_locule_contrast(plot = False)
    assert cranberry_blue.l_transformed.any()

def test_invalid_tissue(cranberry_blue):
    cranberry_blue.setup_measurements()
    cranberry_blue.generate_fruit_mask(plot=False, background_color="blue")
    cranberry_blue.detect_fruits(plot=False)

    with pytest.raises(TypeError):
        cranberry_blue.analyze_color(
            plot=False, display_table=False, tissue="outer_pericarp"
        )

def test_analyze_color_channels(cranberry_blue):
    cranberry_blue.setup_measurements()
    cranberry_blue.generate_fruit_mask(plot=False, background_color="blue")
    cranberry_blue.detect_fruits(plot=False)
    cranberry_blue.analyze_color(plot=False, display_table=False, color_space="lab")
    df = cranberry_blue.results.color_results
    assert df is not None
    assert len(df) > 0

def test_setup_measurements_attributes(cranberry_blue):
    cranberry_blue.setup_measurements(detect_label = True,
                                       detect_color_checker = True,
                                       skip_yolo = False)
    assert cranberry_blue.checker_coords
    assert cranberry_blue.label_text
    assert cranberry_blue.label_roi
    assert cranberry_blue.ref_roi
    assert cranberry_blue.px_per_cm


def test_create_mask_attributes(cranberry_blue):
    cranberry_blue.setup_measurements(detect_label = False,
                                       skip_yolo = True)
    cranberry_blue.generate_fruit_mask(plot = False)
    assert cranberry_blue.mask_fruit is not None
    assert cranberry_blue.mask_fruit.any()
    assert cranberry_blue.mask_locules is None
    assert cranberry_blue.contours is None
    assert cranberry_blue.fruit_locule_map is None
