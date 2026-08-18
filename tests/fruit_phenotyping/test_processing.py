# tests/fruit_phenotyping/test_processing.py

# ============================================================================
# THIRD-PARTY
# ============================================================================
import numpy as np
import cv2
import pytest

# ============================================================================
# INTERNAL
# ============================================================================
from traitly.fruit_phenotyping.processing import (
    calculate_fruit_centroids,
    precalculate_locules_data,
    get_fruit_contour,
    calculate_pericarp_thickness_radial,
    get_internal_pericarp_contour,
    get_internal_pericarp_area,
    annotate_all_fruits,
)

## Helpers (create 'fake' contours for the tests)
def _rect_contour(x, y, w, h):
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return cnts[0]

def _circle_contour(cx, cy, r):
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return cnts[0]

########

def test_calculate_fruit_centroids_basic():
    cnt = _rect_contour(50, 50, 20, 20)
    centroids = calculate_fruit_centroids([cnt])
    assert centroids[0] is not None
    assert 55 <= centroids[0][0] <= 65

def test_calculate_fruit_centroids_zero_area():
    degenerate = np.array([[[0, 0]], [[0, 0]], [[0, 0]]], dtype=np.int32)
    centroids = calculate_fruit_centroids([degenerate])
    assert centroids[0] is None

def test_precalculate_locules_data_valid():
    fruit = _rect_contour(0, 0, 100, 100)
    locule = _rect_contour(40, 40, 10, 10)
    data = precalculate_locules_data([fruit, locule], [1], (50, 50))
    assert len(data) == 1
    assert data[0]["contour_id"] == 1
    assert 0 <= data[0]["circularity"] <= 1

def test_precalculate_locules_data_skips_zero_area():
    degenerate = np.array([[[0, 0]], [[0, 0]], [[0, 0]]], dtype=np.int32)
    data = precalculate_locules_data([degenerate], [0], (0, 0))
    assert data == []

def test_get_fruit_contour_raw():
    cnt = _rect_contour(10, 10, 30, 30)
    result = get_fruit_contour([cnt], 0, contour_mode="raw")
    assert np.array_equal(result, cnt)

def test_get_fruit_contour_hull():
    cnt = _rect_contour(10, 10, 30, 30)
    result = get_fruit_contour([cnt], 0, contour_mode="hull")
    assert result.shape[1:] == (1, 2)

def test_get_fruit_contour_invalid_mode():
    cnt = _rect_contour(10, 10, 30, 30)
    with pytest.raises(ValueError):
        get_fruit_contour([cnt], 0, contour_mode="bad_mode")

def test_get_fruit_contour_index_out_of_range():
    cnt = _rect_contour(10, 10, 30, 30)
    with pytest.raises(IndexError):
        get_fruit_contour([cnt], 5)

def test_get_fruit_contour_circle_mode():
    cnt = _rect_contour(10, 10, 30, 30)
    result = get_fruit_contour([cnt], 0, contour_mode="circle")
    assert len(result) == 36

def test_calculate_pericarp_thickness_radial_valid():
    outer = _circle_contour(100, 100, 50)
    inner = _circle_contour(100, 100, 20)
    res = calculate_pericarp_thickness_radial(
        outer, inner, (100, 100), (200, 200), num_rays=36
    )
    assert res["outer_pericarp_mean_thickness_px"] > 0
    assert not np.isnan(res["outer_pericarp_mean_thickness_px"])

def test_calculate_pericarp_thickness_radial_with_px_per_cm():
    outer = _circle_contour(100, 100, 50)
    inner = _circle_contour(100, 100, 20)
    res = calculate_pericarp_thickness_radial(
        outer, inner, (100, 100), (200, 200), num_rays=36, px_per_cm=10
    )
    assert "outer_pericarp_mean_thickness_cm" in res

def test_get_internal_pericarp_contour_empty_locules():
    cnt = _rect_contour(10, 10, 30, 30)
    result = get_internal_pericarp_contour([], [cnt])
    assert result.size == 0

def test_get_internal_pericarp_contour_convex_hull():
    c1 = _rect_contour(10, 10, 20, 20)
    c2 = _rect_contour(50, 50, 20, 20)
    result = get_internal_pericarp_contour([0, 1], [c1, c2])
    assert len(result) > 0

def test_get_internal_pericarp_area_no_locules():
    cnt = _rect_contour(10, 10, 30, 30)
    area_cm, area_px, hull = get_internal_pericarp_area([], [cnt])
    assert np.isnan(area_cm)
    assert np.isnan(area_px)

def test_get_internal_pericarp_area_requires_img_when_draw():
    cnt = _rect_contour(10, 10, 30, 30)
    with pytest.raises(ValueError):
        get_internal_pericarp_area([0], [cnt], draw_inner_pericarp=True, img=None)


def test_annotate_all_fruits_invalid_label_position():
    cnt = _rect_contour(10, 10, 30, 30)
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        annotate_all_fruits({0: []}, [cnt], img, label_position="bad_pos")

def test_annotate_all_fruits_runs_without_error():
    fruit = _rect_contour(20, 20, 60, 60)
    locule = _rect_contour(40, 40, 10, 10)
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    annotate_all_fruits({0: [1]}, [fruit, locule], img, verbose=False)
    assert img.sum() > 0
