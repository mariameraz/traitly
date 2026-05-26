# traitly/tests/color_correction/test_checker_detection.py

# ============================================================================
# STANDARD LIBRARY
# ============================================================================
from pathlib import Path
import warnings
from unittest.mock import patch

# ============================================================================
# THIRD-PARTY LIBRARIES
# ============================================================================
import pytest
import numpy as np
import cv2

# ============================================================================
# INTERNAL
# ============================================================================
from traitly.color_correction.color_checker import _detect_color_checker

### Paths
img_with_checker = Path(__file__).parent / "18-26.jpg"
img_without_checker = Path(__file__).parent.parent / "data" / "external" / "white_bg" / "cranberry_white_bg.jpg"

@pytest.fixture
def bgr_with_checker():
    assert img_with_checker.exists(), f"Test image not found: {img_with_checker}"
    img = cv2.imread(str(img_with_checker))
    assert img is not None, f"cv2.imread failed for: {img_with_checker}"
    return img

@pytest.fixture
def bgr_without_checker():
    assert img_without_checker.exists(), f"Test image not found: {img_without_checker}"
    img = cv2.imread(str(img_without_checker))
    assert img is not None, f"cv2.imread failed for: {img_without_checker}"
    return img


def test_none_image_raises():
    with pytest.raises(ValueError, match="Image cannot be None"):
        _detect_color_checker(None)

def test_non_array_raises():
    with pytest.raises(TypeError):
        _detect_color_checker("invalid_input")

def test_grayscale_raises(bgr_with_checker):
    img = cv2.cvtColor(bgr_with_checker, cv2.COLOR_BGR2GRAY)
    with pytest.raises(ValueError, match="3-channel"):
        _detect_color_checker(img)

def test_no_checker_returns_none(bgr_without_checker):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _detect_color_checker(bgr_without_checker, plot=False, verbose=False)
    assert result is None
    assert any("not detected" in str(warning.message) for warning in w)

def test_checker_detected_returns_tuple(bgr_with_checker):
    result = _detect_color_checker(bgr_with_checker, plot=False, verbose=False)
    assert result is not None
    assert isinstance(result, tuple)
    assert len(result) == 2 # checker_coords, charts

def test_checker_coords_format(bgr_with_checker):
    checker_coords, _ = _detect_color_checker(bgr_with_checker, plot=False, verbose=False)
    assert isinstance(checker_coords, dict)
    assert set(checker_coords.keys()) == {"x1", "y1", "x2", "y2"}
    assert checker_coords["x1"] >= 0
    assert checker_coords["y1"] >= 0
    assert checker_coords["x2"] > checker_coords["x1"]
    assert checker_coords["y2"] > checker_coords["y1"]

def test_checker_coords_within_image(bgr_with_checker):
    h, w = bgr_with_checker.shape[:2]
    checker_coords, _ = _detect_color_checker(bgr_with_checker, plot=False, verbose=False)
    assert checker_coords["x2"] <= w
    assert checker_coords["y2"] <= h

def test_charts_format(bgr_with_checker):
    _, charts = _detect_color_checker(bgr_with_checker, plot=False, verbose=False)
    assert charts is not None
    assert charts.shape == (72, 5)  # for the mcc24 with cv2.mcc: 24 patches x 3 channels, 5 stats

def test_mcc_not_available_returns_none(bgr_with_checker):
    with patch("traitly.color_correction.color_checker._MCC_AVAILABLE", False):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _detect_color_checker(bgr_with_checker, plot=False, verbose=False)
    assert result is None
    assert any("MCC detector not available" in str(warning.message) for warning in w)
