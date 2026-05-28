# tests/fruit_phenotyping/test_internal_analysis.py

# ============================================================================
# STANDARD LIBRARY
# ============================================================================
from pathlib import Path
import os
# ============================================================================
# THIRD-PARTY
# ============================================================================
import pytest

# ============================================================================
# INTERNAL
# ============================================================================
from traitly.fruit_phenotyping import FruitInternalAnalyzer

##########################################################################
# Valid cranberry image
##########################################################################

valid_img = Path(__file__).parent.parent / "data" / "internal" / "img_test_1.jpg"

@pytest.fixture
def cranberry_valid():
    cranberry = FruitInternalAnalyzer(path=valid_img)
    cranberry.load_image(plot=False)
    cranberry.setup_measurements()
    cranberry.generate_fruit_mask(plot=False)
    cranberry.detect_fruits(plot=False)
    cranberry.analyze_morphology(plot=False, display_table=False)
    cranberry.analyze_color(plot=False, display_table=False, tissue="OUTER_PERICARP", color_space="rgb")
    cranberry.results.save_csv()
    cranberry.save_parameters()
    return cranberry


def test_load_image():
    cranberry = FruitInternalAnalyzer(path=valid_img)
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
    cranberry = FruitInternalAnalyzer(path=valid_img)
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

invalid_path = Path(__file__).parent.parent / "data" / "internal" / "img_test_5.jpg"


def test_invalid_path():
    with pytest.raises((FileNotFoundError, ValueError)):
        cranberry = FruitInternalAnalyzer(path=invalid_path)
        cranberry.load_image(plot=False)


##########################################################################
# Slices with not empty locules
##########################################################################

invalid_img = Path(__file__).parent.parent / "data" / "internal" / "img_test_4.jpg"

def test_invalid_locule_segmentation():
    cranberry = FruitInternalAnalyzer(path=invalid_img)
    cranberry.load_image(plot=False)
    cranberry.setup_measurements(skip_yolo=True)
    cranberry.generate_fruit_mask(plot=False)
    cranberry.detect_fruits(plot=False)
    assert len(cranberry.fruit_locule_map.items()) == 0

def test_fruits_detected(cranberry_valid):
    assert len(cranberry_valid.fruit_locule_map) > 0

def test_color_columns(cranberry_valid):
    df = cranberry_valid.results.color_results
    assert df is not None
    assert len(df) > 0


def test_axis_orientation(cranberry_valid):
    """Major axis/length should always be greater than minor axis/width"""
    df = cranberry_valid.results.morphology_results

    required_cols = [
        "fruit_major_axis_cm",
        "fruit_minor_axis_cm",
        "fruit_box_length_cm",
        "fruit_box_width_cm"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"

    assert (df["fruit_major_axis_cm"] >= df["fruit_minor_axis_cm"]).all(), \
        "Found rows where major_axis < minor_axis"

    assert (df["fruit_box_length_cm"] >= df["fruit_box_width_cm"]).all(), \
        "Found rows where box_length < box_width"


def test_no_negative_std(cranberry_valid):
    """Standard deviation columns must be postive"""
    df = cranberry_valid.results.morphology_results

    std_cols = [
        "outer_pericarp_std_thickness_cm",
        "locules_std_area_cm2",
        "locules_std_circularity",
    ]

    for col in std_cols:
        assert col in df.columns, f"Missing column: {col}"
        bad = df[df[col] < 0][col]
        assert bad.empty, (
            f"Column '{col}' has {len(bad)} negative std value(s):\n{bad.to_string()}"
        )

def test_no_negative_measurements(cranberry_valid):
    """All pixel and cm measurements must be positive"""
    df = cranberry_valid.results.morphology_results

    measurement_cols = [col for col in df.columns
                        if col.endswith("_cm")
                        or col.endswith("_cm2")
                        or col.endswith("_px")
                        or col.endswith("_px2")]

    assert measurement_cols, "No measurement columns found"

    for col in measurement_cols:
        bad = df[df[col] < 0][col]
        assert bad.empty, (
            f"Column '{col}' has {len(bad)} negative value(s):\n{bad.to_string()}"
        )
