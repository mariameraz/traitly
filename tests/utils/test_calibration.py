import numpy as np
import cv2
import pytest
from unittest.mock import patch, MagicMock

from traitly.utils.calibration import (
    _img_px_per_cm,
    _find_size_ref_circles,
    _get_yolo_model,
    _YOLO_MODEL_CACHE
)

#########################
# Testing _img_px_per_cm
#########################
def test_orientation_invariant_results():
    # height > width
    portrait = np.zeros((2710, 2190, 3), dtype=np.uint8)

    # width > height
    landscape = np.zeros((2190, 2710, 3), dtype=np.uint8)

    result_portrait = _img_px_per_cm(portrait, width_cm=21.9, length_cm=27.1)
    result_landscape = _img_px_per_cm(landscape, width_cm=21.9, length_cm=27.1)

    assert result_portrait == result_landscape, (
        f"Portrait {result_portrait} and landscape {result_landscape} should be equal"
    )

def test_predefined_size_orientation_invariant():
    # Same dimensions but using paper size predefined
    portrait = np.zeros((2790, 2160, 3), dtype=np.uint8)
    landscape = np.zeros((2160, 2790, 3), dtype=np.uint8)

    result_portrait = _img_px_per_cm(portrait, size="letter_ansi")
    result_landscape = _img_px_per_cm(landscape, size="letter_ansi")

    assert result_portrait == result_landscape, (
        f"Portrait {result_portrait} and landscape {result_landscape} should be equal"
    )

def test_length_greater_than_width_results():
    portrait = np.zeros((2710, 2190, 3), dtype=np.uint8)
    result = _img_px_per_cm(portrait, width_cm=21.9, length_cm=27.1)
    px_per_cm_w, px_per_cm_l, used_w_cm, used_l_cm = result

    assert used_l_cm > used_w_cm, (
        f"length_cm ({used_l_cm}) should be greater than width_cm ({used_w_cm})"
    )

def test_width_greater_than_length_raises_error():
    portrait = np.zeros((2710, 2190, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="width_cm cannot be greater than length_cm"):
        _img_px_per_cm(portrait, width_cm=27.1, length_cm=21.9)


#################################
# Testing _find_size_ref_circles
#################################

def make_white_roi(h=200, w=200):
    return np.ones((h, w), dtype=np.uint8) * 255


def test_detects_circles():
    roi = make_white_roi()
    cv2.circle(roi, (50, 50), 20, 0, -1)   # black circle
    cv2.circle(roi, (150, 150), 20, 0, -1)

    circles = _find_size_ref_circles(roi)

    assert len(circles) == 2
    for cx, cy, radius in circles:
        assert radius > 0


def test_no_circles():
    roi = make_white_roi()  # no circles
    circles = _find_size_ref_circles(roi)

    assert len(circles) == 0


def test_squares_not_detected():
    roi = make_white_roi()
    cv2.rectangle(roi, (30, 30), (90, 90), 0, -1)   # black square
    cv2.circle(roi, (150, 150), 20, 0, -1)

    circles = _find_size_ref_circles(roi, ref_circularity = 0.85)

    assert len(circles) == 1


def test_return_debug():
    roi = make_white_roi()
    cv2.circle(roi, (100, 100), 30, 0, -1)

    circles, debug = _find_size_ref_circles(roi, return_debug=True)

    assert isinstance(debug, dict)
    assert "binary" in debug
    assert "overlay" in debug
    assert "num_contours" in debug
    assert "num_circles" in debug
    assert debug["num_circles"] == len(circles)


def test_small_circles_filtered():
    roi = make_white_roi()
    cv2.circle(roi, (100, 100), 2, 0, -1)  # small dark circle

    circles = _find_size_ref_circles(roi)

    assert len(circles) == 0

#################################
# Testing YOLO model load
#################################

def setup_function():
    """
    Clean the cache before running the tests
    """
    _YOLO_MODEL_CACHE.clear()

def test_loads_model_first_time():
    mock_model = MagicMock()

    with patch("ultralytics.YOLO", return_value=mock_model) as mock_yolo:
        result = _get_yolo_model("fake/path/model.pt")

        mock_yolo.assert_called_once_with("fake/path/model.pt")
        assert result == mock_model

def test_uses_cache_second_call():
    mock_model = MagicMock()

    with patch("ultralytics.YOLO", return_value=mock_model) as mock_yolo:
        _get_yolo_model("fake/path/model.pt")
        _get_yolo_model("fake/path/model.pt")

        mock_yolo.assert_called_once()


def test_different_paths_load_separately():
    mock_model_1 = MagicMock()
    mock_model_2 = MagicMock()

    with patch("ultralytics.YOLO", side_effect=[mock_model_1, mock_model_2]):
        result_1 = _get_yolo_model("fake/path/model_1.pt")
        result_2 = _get_yolo_model("fake/path/model_2.pt")

        assert result_1 == mock_model_1
        assert result_2 == mock_model_2
        assert len(_YOLO_MODEL_CACHE) == 2
