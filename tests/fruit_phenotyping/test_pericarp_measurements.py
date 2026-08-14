# tests/fruit_phenotyping/test_fruit_symmetry_synthetic.py
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
matplotlib.use("Agg")

# ============================================================================
# INTERNAL
# ============================================================================
from traitly.fruit_phenotyping.fruit_config import analyze_fruits_morphology


# ============================================================================
# Helpers
# ============================================================================

def _contour_from_circle(center, radius, size):
    """Rasterize a filled circle and return its outer contour."""
    m = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(m, center, radius, 255, -1)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea)

def _contour_from_star(center, outer_radius, inner_radius, n_points, size):
    """Rasterize a filled n-pointed star and return its outer contour.
    Used only to contrast lobedness behaviour against a perfect circle."""
    cx, cy = center
    pts = []
    for i in range(n_points * 2):
        r = outer_radius if i % 2 == 0 else inner_radius
        angle = np.pi * i / n_points
        pts.append((int(cx + r * np.cos(angle)), int(cy + r * np.sin(angle))))
    m = np.zeros((size, size), dtype=np.uint8)
    cv2.fillPoly(m, [np.array(pts, dtype=np.int32)], 255)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea)


def _run_pipeline(outer_contour, inner_contour, size=400, **kwargs):
    """Run the real production pipeline on a synthetic 1-fruit, 1-locule
    setup and return the flat per-fruit result dict."""
    contours = [outer_contour, inner_contour]
    fruit_locule_map = {0: [1]}
    img = np.zeros((size, size, 3), dtype=np.uint8)

    defaults = dict(
        px_per_cm=None,
        img_name="synthetic",
        label_text="synthetic",
        plot=False,
        min_locule_area=10,
        dilation_factor=None,
    )
    defaults.update(kwargs)

    results = analyze_fruits_morphology(
        img=img,
        contours=contours,
        fruit_locule_map=fruit_locule_map,
        **defaults,
    )
    assert len(results.morphology_results) == 1, "Expected exactly one fruit result"
    return results.morphology_results[0]

# ============================================================================
# Fixtures
# ============================================================================

SIZE = 400
CENTER = (SIZE // 2, SIZE // 2)
OUTER_R = 150
INNER_RATIO = 0.8
INNER_R = int(OUTER_R * INNER_RATIO)

@pytest.fixture(scope="module")
def concentric_circles_row():
    outer_contour = _contour_from_circle(CENTER, OUTER_R, SIZE)
    inner_contour = _contour_from_circle(CENTER, INNER_R, SIZE)
    row = _run_pipeline(outer_contour, inner_contour, size=SIZE)
    return row

# ============================================================================
# Tests
# ============================================================================

class TestConcentricCirclesGeometry:
    def test_both_contours_are_near_perfect_circles(self, concentric_circles_row):
        """fruit_circularity ~= 1 for a round fruit (4*pi*area/perimeter**2)."""
        row = concentric_circles_row
        assert row["fruit_circularity"] == pytest.approx(1.0, abs=0.15)

    def test_outer_pericarp_thickness_matches_known_gap(self, concentric_circles_row):
        """outer - inner radius = 20% of the outer radius here."""
        row = concentric_circles_row
        expected_thickness = OUTER_R - INNER_R  # = 0.2 * OUTER_R

        assert row["outer_pericarp_mean_thickness_px"] == pytest.approx(
            expected_thickness, abs=2
        )
        # Expressed as a fraction of the known outer radius:
        thickness_pct_of_radius = row["outer_pericarp_mean_thickness_px"] / OUTER_R
        assert thickness_pct_of_radius == pytest.approx(1 - INNER_RATIO, abs=0.02)

    def test_thickness_is_constant_around_a_round_fruit(self, concentric_circles_row):
        """Because both contours are round and concentric (circularity ~= 1),
        the pericarp thickness should barely vary with angle, so its std
        and coefficient of variation should be close to 0."""
        row = concentric_circles_row
        assert row["outer_pericarp_std_thickness_px"] == pytest.approx(0, abs=3)
        assert row["outer_pericarp_cv_thickness"] == pytest.approx(0, abs=5)

    def test_lobedness_is_near_zero_for_a_round_fruit(self, concentric_circles_row):
        """fruit_lobedness = std of the outer contour's radius sampled at
        many angles. For a circle centered on the fruit centroid this is
        close to 0 -> NOT equal to the outer radius itself."""
        row = concentric_circles_row
        assert row["fruit_lobedness_px"] == pytest.approx(0, abs=3)
        # Check if it is nowhere near the radius (i.e. we are really
        # testing the std-of-radius definition, not the mean radius).
        assert row["fruit_lobedness_px"] < 0.1 * OUTER_R

    def test_total_locules_area_matches_inner_circle_area(self, concentric_circles_row):
        row = concentric_circles_row
        expected_inner_area = np.pi * INNER_R ** 2
        assert row["total_locules_area_px2"] == pytest.approx(
            expected_inner_area, rel=0.03
        )

    def test_total_outer_pericarp_area_matches_ring_area(self, concentric_circles_row):
        """total_outer_pericarp_area = fruit_area - internal_fruit_area, i.e.
        the area of the ring between the outer and inner circle."""
        row = concentric_circles_row
        expected_outer_area = np.pi * OUTER_R ** 2
        expected_inner_area = np.pi * INNER_R ** 2
        expected_ring_area = expected_outer_area - expected_inner_area

        assert row["total_outer_pericarp_area_px2"] == pytest.approx(
            expected_ring_area, rel=0.03
        )

    def test_total_internal_pericarp_area_is_near_zero(self, concentric_circles_row):
        """total_internal_pericarp_area = hull(locules)_area - locules_area.
        Since our single locule (the inner circle) is already convex, its
        convex hull has (almost) the same area as the locule itself, so
        this should collapse to ~0."""
        row = concentric_circles_row
        expected_inner_area = np.pi * INNER_R ** 2
        assert row["total_internal_pericarp_area_px2"] == pytest.approx(
            0, abs=0.02 * expected_inner_area
        )


class TestLobednessContrastWithNonRoundFruit:
    """Companion test showing lobedness is NOT the radius, but does grow
    when the outer contour becomes irregular (a star), unlike the
    perfectly round case above."""

    def test_star_shaped_fruit_has_larger_lobedness_than_circle(self):
        circle_contour = _contour_from_circle(CENTER, OUTER_R, SIZE)
        star_contour = _contour_from_star(
            CENTER, outer_radius=OUTER_R, inner_radius=int(OUTER_R * 0.6),
            n_points=5, size=SIZE
        )
        # Use a tiny concentric inner circle as the "locule" in both cases
        # so we only isolate the outer-contour shape effect.
        tiny_inner = _contour_from_circle(CENTER, 20, SIZE)

        circle_row = _run_pipeline(circle_contour, tiny_inner, size=SIZE)
        star_row = _run_pipeline(star_contour, tiny_inner, size=SIZE)

        assert star_row["fruit_lobedness_px"] > circle_row["fruit_lobedness_px"]
        # The round fruit's circularity should also be clearly higher.
        assert circle_row["fruit_circularity"] > star_row["fruit_circularity"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
