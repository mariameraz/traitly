# tests/fruit_phenotyping/test_mask.py
# ============================================================================
# STANDARD LIBRARY
# ============================================================================
import os
import sys

# ============================================================================
# THIRD-PARTY
# ============================================================================
import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib
matplotlib.use("Agg")  # headless backend, no windows popping up during tests

# ============================================================================
# INTERNAL
# ============================================================================
from traitly.fruit_phenotyping import mask

# Helpers / fixtures

def make_hsv_image(h=100, w=100, bg_hsv=(0, 0, 0), fg_hsv=(60, 200, 200),
                    fg_radius=25):
    """Build a synthetic HSV image: dark background + a bright circle (fruit)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = bg_hsv
    cv2.circle(img, (w // 2, h // 2), fg_radius, fg_hsv, -1)
    return img

def make_mask_with_hole(size=200, outer_r=80, hole_r=20):
    """Binary mask: filled circle with a smaller black circle (hole) inside."""
    m = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(m, (size // 2, size // 2), outer_r, 255, -1)
    cv2.circle(m, (size // 2, size // 2), hole_r, 0, -1)
    return m

def make_fruit_with_locules(size=300, fruit_r=100, n_locules=3, locule_r=15):
    """Binary mask of one fruit contour with `n_locules` small holes (locules) in it."""
    m = np.zeros((size, size), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    cv2.circle(m, (cx, cy), fruit_r, 255, -1)
    for i in range(n_locules):
        angle = 2 * np.pi * i / n_locules
        lx = int(cx + (fruit_r * 0.5) * np.cos(angle))
        ly = int(cy + (fruit_r * 0.5) * np.sin(angle))
        cv2.circle(m, (lx, ly), locule_r, 0, -1)
    return m

# ===========================================================================
# create_mask
# ===========================================================================

class TestCreateMask:
    def test_basic_segmentation(self):
        """A bright foreground circle on a dark background should be
        segmented as foreground (255)."""
        img = make_hsv_image()
        result = mask.create_mask(img, plot=False)
        assert result.dtype == np.uint8
        assert result.shape == img.shape[:2]
        center_val = result[50, 50]
        assert center_val == 255  # circle center -> foreground

    def test_background_color_preset(self):
        """background_color overrides lower_hsv/upper_hsv."""
        img = make_hsv_image()
        result = mask.create_mask(img, background_color='black', plot=False)
        assert result.shape == img.shape[:2]

    def test_invalid_background_color_raises(self):
        """valid colors: 'white', 'blue' and 'black' """
        img = make_hsv_image()
        with pytest.raises(RuntimeError):
            mask.create_mask(img, background_color='purple', plot=False)

    def test_non_ndarray_input_raises(self):
        with pytest.raises(RuntimeError):
            mask.create_mask([[1, 2, 3]], plot=False)

    def test_wrong_channel_count_raises(self):
        img = np.zeros((50, 50), dtype=np.uint8)  # 2D, not 3-channel
        with pytest.raises(RuntimeError):
            mask.create_mask(img, plot=False)

    def test_wrong_dtype_raises(self):
        img = np.zeros((50, 50, 3), dtype=np.float32)
        with pytest.raises(RuntimeError):
            mask.create_mask(img, plot=False)

    def test_even_kernel_open_raises(self):
        img = make_hsv_image()
        with pytest.raises(RuntimeError):
            mask.create_mask(img, kernel_open=4, plot=False)

    def test_canny_only_one_bound_raises(self):
        img = make_hsv_image()
        with pytest.raises(RuntimeError):
            mask.create_mask(img, canny_min=50, plot=False)

    def test_canny_min_greater_than_max_raises(self):
        img = make_hsv_image()
        with pytest.raises(RuntimeError):
            mask.create_mask(img, canny_min=200, canny_max=100, plot=False)

    def test_lower_greater_than_upper_raises(self):
        img = make_hsv_image()
        with pytest.raises(RuntimeError):
            mask.create_mask(img, lower_hsv=(100, 100, 100),
                              upper_hsv=(10, 10, 10), plot=False)

    def test_fill_holes_option(self):
        """fill_holes=True should remove internal black holes."""
        img = make_hsv_image(fg_radius=30)
        result_no_fill = mask.create_mask(img, fill_holes=False, plot=False)
        result_fill = mask.create_mask(img, fill_holes=True, plot=False)
        # fill_holes should never decrease the amount of foreground pixels
        assert result_fill.sum() >= result_no_fill.sum()

    def test_convex_hull_option_runs(self):
        img = make_hsv_image()
        result = mask.create_mask(img, apply_convex_hull=True, plot=False)
        assert result.dtype == np.uint8
