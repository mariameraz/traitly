import numpy as np
import pytest

from traitly.utils.calibration import img_px_per_cm

def test_orientation_invariant_results():
    # height > width
    portrait = np.zeros((2710, 2190, 3), dtype=np.uint8)

    # width > height
    landscape = np.zeros((2190, 2710, 3), dtype=np.uint8)

    result_portrait = img_px_per_cm(portrait, width_cm=21.9, length_cm=27.1)
    result_landscape = img_px_per_cm(landscape, width_cm=21.9, length_cm=27.1)

    assert result_portrait == result_landscape, (
        f"Portrait {result_portrait} and landscape {result_landscape} should be equal"
    )

def test_predefined_size_orientation_invariant():
    # Same dimensions but using paper size predefined
    portrait = np.zeros((2790, 2160, 3), dtype=np.uint8)
    landscape = np.zeros((2160, 2790, 3), dtype=np.uint8)

    result_portrait = img_px_per_cm(portrait, size="letter_ansi")
    result_landscape = img_px_per_cm(landscape, size="letter_ansi")

    assert result_portrait == result_landscape, (
        f"Portrait {result_portrait} and landscape {result_landscape} should be equal"
    )

def test_length_greater_than_width_results():
    portrait = np.zeros((2710, 2190, 3), dtype=np.uint8)
    result = img_px_per_cm(portrait, width_cm=21.9, length_cm=27.1)
    px_per_cm_w, px_per_cm_l, used_w_cm, used_l_cm = result

    assert used_l_cm > used_w_cm, (
        f"length_cm ({used_l_cm}) should be greater than width_cm ({used_w_cm})"
    )

def test_width_greater_than_length_raises_error():
    portrait = np.zeros((2710, 2190, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="width_cm cannot be greater than length_cm"):
        img_px_per_cm(portrait, width_cm=27.1, length_cm=21.9)
