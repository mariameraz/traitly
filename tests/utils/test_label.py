#tests/utils/test_label.py

# ============================================================================
# STANDARD LIBRARY
# ============================================================================
import types
import sys
import textwrap
from unittest.mock import MagicMock, patch
from pathlib import Path

# ============================================================================
# THIRD-PARTY
# ============================================================================
import cv2
import numpy as np
import pytest

# ============================================================================
# INTERNAL
# ============================================================================
from traitly.utils.label import detect_qr
import traitly.utils.label as _mod

# Define input image
img_path = Path(__file__).parent.parent / "data" / "internal" / "img_test_1.jpg"
qr_text = "DP14-106"

@pytest.fixture(scope="session")
def qr_bgr():
    img = cv2.imread(str(img_path))
    assert img is not None, f"Could not load: {img_path}"
    return img

@pytest.fixture(scope="session")
def qr_gray(qr_bgr):
    return cv2.cvtColor(qr_bgr, cv2.COLOR_BGR2GRAY)

def test_raises_when_no_args():
    with pytest.raises(ValueError, match="Either img or img_path"):
        detect_qr()

def test_raises_when_invalid_path():
    with pytest.raises(ValueError, match="Could not load image"):
        detect_qr(img_path="/nonexistent/qr.png")

def test_decode_bgr_array(qr_bgr):
    assert detect_qr(img=qr_bgr) == qr_text

def test_decode_grayscale_array(qr_gray):
    assert detect_qr(img=qr_gray) == qr_text

def test_decode_from_file():
    assert detect_qr(img_path=str(img_path)) == qr_text

def test_no_qr_returns_falsy():
    blank = np.full((200, 200, 3), 255, dtype=np.uint8)
    assert not detect_qr(img=blank)

def test_wechat_branch_returns_first_result(qr_bgr):
    detector_mock = MagicMock()
    detector_mock.detectAndDecode.return_value = ([qr_text, "other"], [])
    with patch.object(_mod, "_WECHAT_AVAILABLE", True), \
         patch.object(_mod, "_detector", detector_mock):
        assert detect_qr(img=qr_bgr) == qr_text

def test_wechat_branch_empty_list_returns_none(qr_bgr):
    detector_mock = MagicMock()
    detector_mock.detectAndDecode.return_value = ([], [])
    with patch.object(_mod, "_WECHAT_AVAILABLE", True), \
         patch.object(_mod, "_detector", detector_mock):
        assert detect_qr(img=qr_bgr) is None

def test_fallback_uses_curved_when_first_fails(qr_bgr):
    """If QRCodeDetector falls, call to detectAndDecodeCurved."""
    det_mock = MagicMock()
    det_mock.detectAndDecode.return_value = ("", None, None)
    det_mock.detectAndDecodeCurved.return_value = (qr_text, None, None)
    with patch.object(_mod, "_WECHAT_AVAILABLE", False), \
         patch("cv2.QRCodeDetector", return_value=det_mock):
        assert detect_qr(img=qr_bgr) == qr_text
    det_mock.detectAndDecodeCurved.assert_called_once()

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
