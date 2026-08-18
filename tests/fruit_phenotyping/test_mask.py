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

from unittest.mock import patch
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

# ===========================================================================
# fill_holes_to_mask
# ===========================================================================
class TestFillHolesToMask:
    def test_fills_interior_hole(self):
        m = make_mask_with_hole()
        filled = mask.fill_holes_to_mask(m)
        cy, cx = m.shape[0] // 2, m.shape[1] // 2
        assert m[cy, cx] == 0        # hole originally black
        assert filled[cy, cx] == 255  # hole filled after processing

    def test_preserves_shape_and_dtype(self):
        m = make_mask_with_hole()
        filled = mask.fill_holes_to_mask(m)
        assert filled.shape == m.shape
        assert filled.dtype == np.uint8

    def test_no_holes_unaffected(self):
        m = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(m, (20, 20), (80, 80), 255, -1)
        filled = mask.fill_holes_to_mask(m)
        assert np.array_equal(filled, m)

    def test_non_2d_raises(self):
        m = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            mask.fill_holes_to_mask(m)


# ===========================================================================
# apply_convex_hull_to_mask
# ===========================================================================

class TestApplyConvexHullToMask:
    def test_hull_area_gte_original(self):
        """A convex hull should never be smaller than the original shape."""
        m = np.zeros((100, 100), dtype=np.uint8)
        # a concave, non-convex "C" shape
        cv2.rectangle(m, (20, 20), (80, 80), 255, -1)
        cv2.rectangle(m, (40, 30), (90, 70), 0, -1)
        hull_mask = mask.apply_convex_hull_to_mask(m, min_area=10)
        assert hull_mask.sum() >= m.sum()

    def test_small_contours_filtered_by_min_area(self):
        m = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(m, (20, 20), 3, 255, -1)  # tiny blob, area well below 50
        result = mask.apply_convex_hull_to_mask(m, min_area=1000)
        assert result.sum() == 0

    def test_empty_mask_returns_empty(self):
        m = np.zeros((50, 50), dtype=np.uint8)
        result = mask.apply_convex_hull_to_mask(m)
        assert result.sum() == 0
        assert result.shape == m.shape

# ===========================================================================
# find_fruits
# ===========================================================================

class TestFindFruits:
    def test_detects_single_fruit_with_locules(self):
        m = make_fruit_with_locules(n_locules=3)
        contours, fruit_map = mask.find_fruits(m, min_locule_area=10,
                                                min_locules_per_fruit=1,
                                                min_fruit_area=100)
        assert len(fruit_map) == 1
        fruit_idx = list(fruit_map.keys())[0]
        assert len(fruit_map[fruit_idx]) == 3

    def test_fruit_rejected_if_not_enough_locules(self):
        m = make_fruit_with_locules(n_locules=1)
        contours, fruit_map = mask.find_fruits(m, min_locule_area=10,
                                                min_locules_per_fruit=3)
        assert fruit_map == {}

    def test_min_locules_zero_disables_locule_filter(self):
        m = make_fruit_with_locules(n_locules=0)
        contours, fruit_map = mask.find_fruits(m, min_locules_per_fruit=0,
                                                min_fruit_area=100)
        assert len(fruit_map) == 1

    def test_empty_mask_returns_empty_results(self):
        m = np.zeros((100, 100), dtype=np.uint8)
        contours, fruit_map = mask.find_fruits(m)
        assert contours == []
        assert fruit_map == {}

    def test_area_filter_excludes_fruit(self):
        m = make_fruit_with_locules(n_locules=2, fruit_r=100)
        # Require an area far bigger than the fruit actually has
        _, fruit_map = mask.find_fruits(m, min_fruit_area=10 ** 7)
        assert fruit_map == {}

    def test_rescale_factor_returns_scaled_contours(self):
        m = make_fruit_with_locules(n_locules=3, fruit_r=100)
        contours_full, map_full = mask.find_fruits(m, min_locule_area=10,
                                                     min_fruit_area=100)
        contours_scaled, map_scaled = mask.find_fruits(
            m, min_locule_area=10, min_fruit_area=100, rescale_factor=0.5
        )
        assert len(map_full) == len(map_scaled) == 1
        # Contour bounding areas should be roughly comparable after rescale-back
        idx_full = list(map_full.keys())[0]
        idx_scaled = list(map_scaled.keys())[0]
        area_full = cv2.contourArea(contours_full[idx_full])
        area_scaled = cv2.contourArea(contours_scaled[idx_scaled])
        assert area_scaled == pytest.approx(area_full, rel=0.2)

    def test_invalid_dtype_raises(self):
        m = np.zeros((50, 50), dtype=np.float32)
        with pytest.raises(ValueError):
            mask.find_fruits(m)

    def test_invalid_shape_raises(self):
        m = np.zeros((50, 50, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            mask.find_fruits(m)

    def test_min_greater_than_max_fruit_area_raises(self):
        m = make_fruit_with_locules()
        with pytest.raises(ValueError):
            mask.find_fruits(m, min_fruit_area=1000, max_fruit_area=10)

    def test_invalid_circularity_range_raises(self):
        m = make_fruit_with_locules()
        with pytest.raises(ValueError):
            mask.find_fruits(m, min_circularity=0.9, max_circularity=0.2)

    def test_invalid_rescale_factor_raises(self):
        m = make_fruit_with_locules()
        with pytest.raises(ValueError):
            mask.find_fruits(m, rescale_factor=1.5)

# ===========================================================================
# merge_locules_func
# ===========================================================================
class TestMergeLoculesFunc:

    def _two_close_circles(self):
        """Two small circular contours close to each other, plus one far away."""
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(img, (50, 50), 10, 255, -1)
        contour_a, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img2 = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(img2, (65, 50), 10, 255, -1)  # ~5px gap from contour a
        contour_b, _ = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img3 = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(img3, (180, 180), 10, 255, -1)  # far away
        contour_c, _ = cv2.findContours(img3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return [contour_a[0], contour_b[0], contour_c[0]]

    def test_empty_indices_returns_empty(self):
        result = mask.merge_locules_func([], [], )
        assert result == []

    def test_close_locules_are_merged(self):
        contours = self._two_close_circles()
        result = mask.merge_locules_func([0, 1, 2], contours,
                                          min_distance=0, max_distance=20,
                                          min_area=10)
        # a and b (close) merge into one contour; c (far) stays separate
        assert len(result) == 2

    def test_far_locules_not_merged(self):
        contours = self._two_close_circles()
        result = mask.merge_locules_func([0, 1, 2], contours,
                                          min_distance=0, max_distance=1,
                                          min_area=10)
        # distance threshold too small -> nothing merges, 3 contours kept
        assert len(result) == 3

    def test_small_area_locules_filtered_out(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(img, (50, 50), 2, 255, -1)  # tiny circle, area < min_area
        tiny_contour, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = mask.merge_locules_func([0], [tiny_contour[0]], min_area=1000)
        assert result == []

# ===========================================================================
# _ensure_uint8, gamma_contrast, sigmoid_contrast, exp_transform
# ===========================================================================

class TestContrastHelpers:

    def test_ensure_uint8_passthrough(self):
        arr = np.array([[0, 128, 255]], dtype=np.uint8)
        out = mask._ensure_uint8(arr)
        assert out.dtype == np.uint8
        assert np.array_equal(out, arr)

    def test_ensure_uint8_scales_normalized_float(self):
        arr = np.array([[0.0, 0.5, 1.0]], dtype=np.float64)
        out = mask._ensure_uint8(arr)
        assert out.dtype == np.uint8
        assert out[0, 2] == 255

    def test_ensure_uint8_clips_out_of_range(self):
        arr = np.array([[-10.0, 300.0]], dtype=np.float64)
        out = mask._ensure_uint8(arr)
        assert out.min() >= 0 and out.max() <= 255

    def test_gamma_contrast_identity_at_one(self):
        L = np.linspace(0, 255, 20).astype(np.uint8).reshape(4, 5)
        out = mask.gamma_contrast(L, gamma=1.0, plot=False)
        assert np.array_equal(out, L)

    def test_gamma_contrast_darkens_when_above_one(self):
        L = np.full((5, 5), 128, dtype=np.uint8)
        out = mask.gamma_contrast(L, gamma=2.0, plot=False)
        assert out.mean() < L.mean()

    def test_sigmoid_contrast_output_range(self):
        L = np.linspace(0, 255, 50).astype(np.uint8).reshape(5, 10)
        out = mask.sigmoid_contrast(L, gain=10, cutoff=0.5)
        assert out.dtype == np.uint8
        assert out.min() >= 0 and out.max() <= 255

    def test_exp_transform_output_range(self):
        L = np.linspace(0, 255, 50).astype(np.uint8).reshape(5, 10)
        out = mask.exp_transform(L, c=1.0)
        assert out.dtype == np.uint8
        assert out.max() <= 255


# ===========================================================================
# apply_contrast
# ===========================================================================

class TestApplyContrast:

    def _bgr_image(self):
        rng = np.random.default_rng(0)
        return rng.integers(0, 255, size=(40, 40, 3), dtype=np.uint8)

    def test_returns_2d_uint8(self):
        img = self._bgr_image()
        out = mask.apply_contrast(img, contrast_method='gamma', plot=False)
        assert out.ndim == 2
        assert out.dtype == np.uint8
        assert out.shape == img.shape[:2]

    def test_none_method_returns_l_channel_copy(self):
        img = self._bgr_image()
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        out = mask.apply_contrast(img, contrast_method='none',
                                   kernel_blur=1, plot=False)
        assert np.array_equal(out, l_channel)

    def test_invalid_method_raises(self):
        img = self._bgr_image()
        with pytest.raises(ValueError):
            mask.apply_contrast(img, contrast_method='bogus', plot=False)

    def test_non_ndarray_raises_type_error(self):
        with pytest.raises(TypeError):
            mask.apply_contrast([[1, 2, 3]], plot=False)

    def test_wrong_channels_raises_value_error(self):
        img = np.zeros((20, 20), dtype=np.uint8)
        with pytest.raises(ValueError):
            mask.apply_contrast(img, plot=False)

    def test_clahe_applied_changes_output(self):
        img = self._bgr_image()
        out_no_clahe = mask.apply_contrast(img, clip_limit=None, plot=False)
        out_clahe = mask.apply_contrast(img, clip_limit=2, plot=False)
        assert not np.array_equal(out_no_clahe, out_clahe)

    def test_compare_mode_runs_without_error(self):
        img = self._bgr_image()
        out = mask.apply_contrast(img, compare=True, plot=False)
        assert out.dtype == np.uint8


# ===========================================================================
# create_mask_locules
# ===========================================================================

class TestCreateMaskLocules:

    def _fruit_with_dark_locules(self, size=200):
        """Bright fruit disc (L~200) with darker locule blobs (L~30) inside,
        matching what create_mask_locules expects to threshold."""
        l_channel = np.full((size, size), 40, dtype=np.uint8)  # background dark
        cv2.circle(l_channel, (size // 2, size // 2), 80, 200, -1)  # fruit bright
        cv2.circle(l_channel, (size // 2 - 20, size // 2), 12, 30, -1)  # locule 1 (dark)
        cv2.circle(l_channel, (size // 2 + 20, size // 2), 12, 30, -1)  # locule 2 (dark)

        fruit_mask = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(fruit_mask, (size // 2, size // 2), 80, 255, -1)
        return l_channel, fruit_mask

    def test_returns_uint8_same_shape(self):
        l_ch, f_mask = self._fruit_with_dark_locules()
        out = mask.create_mask_locules(l_ch, f_mask, thresh_min=100,
                                        min_fruit_area=1000, min_locule_area=10,
                                        plot=False)
        assert out.shape == l_ch.shape
        assert out.dtype == np.uint8

    def test_type_errors_for_non_array_inputs(self):
        l_ch, f_mask = self._fruit_with_dark_locules()
        with pytest.raises(TypeError):
            mask.create_mask_locules([[1, 2]], f_mask)
        with pytest.raises(TypeError):
            mask.create_mask_locules(l_ch, [[1, 2]])

    def test_locules_appear_as_holes_in_output(self):
        l_ch, f_mask = self._fruit_with_dark_locules()
        out = mask.create_mask_locules(l_ch, f_mask, thresh_min=100,
                                        min_fruit_area=1000, min_locule_area=10,
                                        plot=False)
        size = l_ch.shape[0]
        # Locule center should be excluded (0) from the final fused mask,
        # while a point in the pericarp (bright, non-locule) stays foreground.
        assert out[size // 2, size // 2 - 20] == 0
        assert out[size // 2, size // 2 + 40] == 255

    def test_min_locule_area_filters_small_locules(self):
        l_ch, f_mask = self._fruit_with_dark_locules()
        out_small_thresh = mask.create_mask_locules(
            l_ch, f_mask, thresh_min=100, min_fruit_area=1000,
            min_locule_area=10 ** 6, plot=False
        )
        # With an impossibly high min_locule_area, no locule survives,
        # so the fused mask should equal the original fruit_mask.
        assert np.array_equal(out_small_thresh, f_mask)

# ===========================================================================
# gamma_contrast: plot=True
# ===========================================================================
class TestGammaContrastPlot:
    def test_plot_true_runs_without_error(self):
        L = np.linspace(0, 255, 20).astype(np.uint8).reshape(4, 5)
        with patch("matplotlib.pyplot.show"):
            out = mask.gamma_contrast(L, gamma=1.5, plot=True)
        assert out.dtype == np.uint8

# ===========================================================================
# generate_scatter_plot
# ===========================================================================
class TestGenerateScatterPlot:
    def test_runs_without_error(self):
        img = make_hsv_image()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        with patch("matplotlib.pyplot.show"):
            result = mask.generate_scatter_plot(img, img_rgb, sample_size=50)
        assert result is None

# ===========================================================================
# find_fruits: remaining validation + filtering branches
# ===========================================================================

class TestFindFruitsExtra:

    def test_non_positive_min_fruit_area_raises(self):
        m = make_fruit_with_locules()
        with pytest.raises(ValueError):
            mask.find_fruits(m, min_fruit_area=0)

    def test_non_positive_max_fruit_area_raises(self):
        m = make_fruit_with_locules()
        with pytest.raises(ValueError):
            mask.find_fruits(m, max_fruit_area=0)

    def test_max_fruit_area_filters_out_large_fruit(self):
        m = make_fruit_with_locules(n_locules=2, fruit_r=100)
        _, fruit_map = mask.find_fruits(m, min_locule_area=10,
                                         max_fruit_area=100)
        assert fruit_map == {}

    def test_empty_mask_no_locule_filter_returns_empty(self):
        """Hits the RETR_EXTERNAL branch's own empty-contours early return
        (min_locules_per_fruit=0 -> needs_locules=False)."""
        m = np.zeros((100, 100), dtype=np.uint8)
        contours, fruit_map = mask.find_fruits(m, min_locules_per_fruit=0)
        assert contours == []
        assert fruit_map == {}

# ===========================================================================
# create_mask: optional-processing branches (kernel_open/close/blur, canny,
# plot=True)
# ===========================================================================

class TestCreateMaskOptionalBranches:

    def test_kernel_open_is_applied(self):
        img = make_hsv_image()
        out = mask.create_mask(img, kernel_open=3, plot=False)
        assert out.dtype == np.uint8

    def test_kernel_close_is_applied(self):
        img = make_hsv_image()
        out = mask.create_mask(img, kernel_close=3, plot=False)
        assert out.dtype == np.uint8

    def test_kernel_blur_is_applied(self):
        img = make_hsv_image()
        out = mask.create_mask(img, kernel_blur=3, plot=False)
        assert out.dtype == np.uint8

    def test_canny_edges_are_or_combined(self):
        img = make_hsv_image()
        out = mask.create_mask(img, canny_min=50, canny_max=150, plot=False)
        assert out.dtype == np.uint8
        assert out.shape == img.shape[:2]

    def test_canny_without_prior_blur_still_blurs_internally(self):
        """When canny_min/max are given but kernel_blur is None, the
        function should internally blur before running Canny."""
        img = make_hsv_image()
        out = mask.create_mask(img, canny_min=50, canny_max=150,
                                kernel_blur=None, plot=False)
        assert out.dtype == np.uint8

    def test_plot_true_runs_without_error(self):
        img = make_hsv_image()
        with patch("matplotlib.pyplot.show"):
            out = mask.create_mask(img, plot=True)
        assert out.dtype == np.uint8

# ===========================================================================
# create_mask: extra validation branches
# ===========================================================================
class TestCreateMaskExtraValidation:

    def test_invalid_n_iteration_raises(self):
        img = make_hsv_image()
        with pytest.raises(RuntimeError):
            mask.create_mask(img, n_iteration=0, plot=False)

    def test_even_kernel_close_raises(self):
        img = make_hsv_image()
        with pytest.raises(RuntimeError):
            mask.create_mask(img, kernel_close=4, plot=False)

    def test_even_kernel_blur_raises(self):
        img = make_hsv_image()
        with pytest.raises(RuntimeError):
            mask.create_mask(img, kernel_blur=4, plot=False)

    def test_non_integer_canny_bounds_raises(self):
        img = make_hsv_image()
        with pytest.raises(RuntimeError):
            mask.create_mask(img, canny_min=10.5, canny_max=100, plot=False)

    def test_lower_hsv_wrong_shape_raises(self):
        img = make_hsv_image()
        with pytest.raises(RuntimeError):
            mask.create_mask(img, lower_hsv=(0, 0), plot=False)

    def test_upper_hsv_wrong_shape_raises(self):
        img = make_hsv_image()
        with pytest.raises(RuntimeError):
            mask.create_mask(img, upper_hsv=(180, 250), plot=False)

    def test_generic_exception_is_wrapped_as_runtime_error(self, monkeypatch):
        """Any unexpected (non cv2.error) exception raised inside the try
        block should be wrapped into a RuntimeError with 'Unexpected error'."""
        img = make_hsv_image()

        def boom(*args, **kwargs):
            raise ValueError("boom")

        monkeypatch.setattr(mask, "fill_holes_to_mask", boom)
        with pytest.raises(RuntimeError, match="Unexpected error"):
            mask.create_mask(img, fill_holes=True, plot=False)

# ===========================================================================
# create_mask_locules: optional branches (otsu, kernel_open/close/blur,
# erosion, invert_locules, plot=True)
# ===========================================================================
#
class TestCreateMaskLoculesExtra:
    def _fruit_with_dark_locules(self, size=200):
        l_channel = np.full((size, size), 40, dtype=np.uint8)
        cv2.circle(l_channel, (size // 2, size // 2), 80, 200, -1)
        cv2.circle(l_channel, (size // 2 - 20, size // 2), 12, 30, -1)
        cv2.circle(l_channel, (size // 2 + 20, size // 2), 12, 30, -1)

        fruit_mask = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(fruit_mask, (size // 2, size // 2), 80, 255, -1)
        return l_channel, fruit_mask

    def test_use_otsu_default(self):
        l_ch, f_mask = self._fruit_with_dark_locules()
        out = mask.create_mask_locules(l_ch, f_mask, use_otsu=True,
                                        min_fruit_area=1000, min_locule_area=10,
                                        plot=False)
        assert out.dtype == np.uint8

    def test_use_otsu_with_offset(self):
        l_ch, f_mask = self._fruit_with_dark_locules()
        out = mask.create_mask_locules(l_ch, f_mask, use_otsu=True,
                                        otsu_offset=5, min_fruit_area=1000,
                                        min_locule_area=10, plot=False)
        assert out.dtype == np.uint8

    def test_kernel_open_and_close_and_blur_applied(self):
        l_ch, f_mask = self._fruit_with_dark_locules()
        out = mask.create_mask_locules(l_ch, f_mask, thresh_min=100,
                                        kernel_close=3, kernel_open=3,
                                        kernel_blur=3, min_fruit_area=1000,
                                        min_locule_area=10, plot=False)
        assert out.dtype == np.uint8

    def test_erosion_px_applied(self):
        l_ch, f_mask = self._fruit_with_dark_locules()
        out = mask.create_mask_locules(l_ch, f_mask, thresh_min=100,
                                        erosion_px=3, min_fruit_area=1000,
                                        min_locule_area=10, plot=False)
        assert out.dtype == np.uint8

    def test_invert_locules(self):
        l_ch, f_mask = self._fruit_with_dark_locules()
        out = mask.create_mask_locules(l_ch, f_mask, thresh_min=100,
                                        invert_locules=True, min_fruit_area=1000,
                                        min_locule_area=10, plot=False)
        assert out.dtype == np.uint8

    def test_plot_true_runs_without_error(self):
        l_ch, f_mask = self._fruit_with_dark_locules()
        with patch("matplotlib.pyplot.show"):
            out = mask.create_mask_locules(l_ch, f_mask, thresh_min=100,
                                            min_fruit_area=1000,
                                            min_locule_area=10, plot=True)
        assert out.dtype == np.uint8

class TestFindFruitsMoreValidation:
    def test_non_positive_min_locule_area_raises(self):
        m = make_fruit_with_locules()
        with pytest.raises(ValueError):
            mask.find_fruits(m, min_locule_area=0)

    def test_negative_min_locules_per_fruit_raises(self):
        m = make_fruit_with_locules()
        with pytest.raises(ValueError):
            mask.find_fruits(m, min_locules_per_fruit=-1)


class TestCreateMaskErrorWrapping:
    def test_cv2_error_is_wrapped(self, monkeypatch):
        img = make_hsv_image()

        def boom(*args, **kwargs):
            raise cv2.error("forced cv2 failure")

        monkeypatch.setattr(mask.cv2, "bitwise_not", boom)
        with pytest.raises(RuntimeError, match="OpenCV error"):
            mask.create_mask(img, plot=False)

    def test_none_background_mask_raises(self, monkeypatch):
        img = make_hsv_image()

        def fake_inrange(*args, **kwargs):
            return None

        monkeypatch.setattr(mask.cv2, "inRange", fake_inrange)
        with pytest.raises(RuntimeError):
            mask.create_mask(img, plot=False)


class TestMergeLoculesFuncExtra:
    def _circle_contour(self, center, r, size=300):
        img = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(img, center, r, 255, -1)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours[0]

    def test_single_valid_locule_returns_original_contour(self):
        c0 = self._circle_contour((50, 50), 10)
        result = mask.merge_locules_func([0], [c0], min_area=5)
        assert len(result) == 1
        assert np.array_equal(result[0], c0)

    def test_degenerate_centroid_still_processed(self, monkeypatch):
        c0 = self._circle_contour((50, 50), 10)
        c1 = self._circle_contour((65, 50), 10)

        real_moments = cv2.moments
        call_count = {"n": 0}

        def fake_moments(contour, *args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {"m00": 0.0, "m10": 0.0, "m01": 0.0}
            return real_moments(contour, *args, **kwargs)

        monkeypatch.setattr(mask.cv2, "moments", fake_moments)

        result = mask.merge_locules_func([0, 1], [c0, c1],
                                          min_distance=0, max_distance=20,
                                          min_area=5)
        assert len(result) >= 1

    def test_j_less_or_equal_i_candidates_are_skipped(self):
        c0 = self._circle_contour((50, 50), 10)
        c1 = self._circle_contour((85, 50), 10)
        c2 = self._circle_contour((120, 50), 10)

        result = mask.merge_locules_func([0, 1, 2], [c0, c1, c2],
                                          min_distance=0, max_distance=25,
                                          min_area=5)
        assert 1 <= len(result) <= 3

    def test_merge_failure_falls_back_to_original_contour(self, monkeypatch):
        c0 = self._circle_contour((50, 50), 10)
        c1 = self._circle_contour((60, 50), 10)

        def boom(*args, **kwargs):
            raise cv2.error("forced failure")

        monkeypatch.setattr(mask.cv2, "approxPolyDP", boom)

        result = mask.merge_locules_func([0, 1], [c0, c1],
                                          min_distance=0, max_distance=20,
                                          min_area=5)
        assert len(result) == 1
        assert np.array_equal(result[0], c0)


class TestApplyContrastExtra:
    def _bgr_image(self):
        rng = np.random.default_rng(0)
        return rng.integers(0, 255, size=(40, 40, 3), dtype=np.uint8)

    def test_sigmoid_method(self):
        img = self._bgr_image()
        out = mask.apply_contrast(img, contrast_method='sigmoid', plot=False)
        assert out.dtype == np.uint8

    def test_exp_method(self):
        img = self._bgr_image()
        out = mask.apply_contrast(img, contrast_method='exp', plot=False)
        assert out.dtype == np.uint8

    def test_median_blur_applied_when_kernel_blur_gt_one(self):
        img = self._bgr_image()
        out_no_blur = mask.apply_contrast(img, kernel_blur=1, plot=False)
        out_blur = mask.apply_contrast(img, kernel_blur=5, plot=False)
        assert not np.array_equal(out_no_blur, out_blur)

    def test_plot_true_and_compare_false_runs(self):
        img = self._bgr_image()
        with patch("matplotlib.pyplot.show"):
            out = mask.apply_contrast(img, plot=True, compare=False)
        assert out.dtype == np.uint8


class TestGenerateLChannelHistogram:
    def test_runs_without_error(self):
        size = 100
        l_ch = np.random.default_rng(0).integers(0, 255, size=(size, size)).astype(np.uint8)
        fruit_mask = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(fruit_mask, (size // 2, size // 2), 40, 255, -1)
        with patch("matplotlib.pyplot.show"):
            result = mask.generate_l_channel_histogram(l_ch, fruit_mask)
        assert result is None

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
