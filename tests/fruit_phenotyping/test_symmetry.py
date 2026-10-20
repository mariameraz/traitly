# tests/fruit_phenotyping/test_symmetry.py
#
# ============================================================================
# THIRD-PARTY
# ============================================================================
import numpy as np
import pytest

# ============================================================================
# INTERNAL
# ============================================================================
from traitly.fruit_phenotyping.symmetry import angular_symmetry, precompute_ideal_angles

def make_fruit(angles_deg, radius=1.0):
    """ List of artificial fruits with polar_coord = (angle_rad, radius)."""
    return [{'polar_coord': (np.radians(a), radius)} for a in angles_deg]

def ideal_deg(n, rotation=0.0):
    return [(i * 360 / n + rotation) for i in range(n)]

def test_fewer_than_two_locules_returns_nan():
    assert np.isnan(angular_symmetry([]))
    assert np.isnan(angular_symmetry(make_fruit([30])))

## Perfect symmetry
@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 8, 12])
@pytest.mark.parametrize("rotation", [0, 3, 47.5, 123, 359])
def test_perfect_symmetry_rotated_is_zero(n, rotation):
    fruit = make_fruit(ideal_deg(n, rotation))
    assert angular_symmetry(fruit) == pytest.approx(0.0, abs=1e-9)

def test_perfect_symmetry_input_order_does_not_matter():
    angles = ideal_deg(5, 20)
    rng = np.random.default_rng(0)
    rng.shuffle(angles)
    assert angular_symmetry(make_fruit(angles)) == pytest.approx(0.0, abs=1e-9)

def test_perfect_symmetry_across_zero_boundary():
    # 355 and 5 sit on both sides of 0 / 360
    assert angular_symmetry(make_fruit([355, 85, 175, 265])) == pytest.approx(0.0, abs=1e-9)


# angles deviated like + dev, - dev, + dev, - dev
@pytest.mark.parametrize("n", [4, 6, 8, 12])
@pytest.mark.parametrize("dev", [2.0, 5.0])
def test_alternating_deviation(n, dev):
    ideal = ideal_deg(n)
    angles = [a + (dev if i % 2 == 0 else -dev) for i, a in enumerate(ideal)]
    expected = dev / (360 / n)
    assert angular_symmetry(make_fruit(angles)) == pytest.approx(expected, rel=1e-9)

def test_alternating_deviation_is_rotation_invariant():
    ideal = ideal_deg(6)
    base = [a + (6 if i % 2 == 0 else -6) for i, a in enumerate(ideal)]
    r0 = angular_symmetry(make_fruit(base))
    r1 = angular_symmetry(make_fruit([a + 77 for a in base]))
    assert r0 == pytest.approx(r1, abs=1e-9)
    assert r0 == pytest.approx(0.1, rel=1e-9)   # 6 deg / 60 deg

def test_same_normalized_error_comparable_across_n():
    # Same fraction of the step (10%) for n=4 and n=12 -> same score
    def fruit(n):
        step = 360 / n
        return make_fruit([a + (0.1*step if i % 2 == 0 else -0.1*step)
                           for i, a in enumerate(ideal_deg(n))])
    assert angular_symmetry(fruit(4)) == pytest.approx(angular_symmetry(fruit(12)), rel=1e-9)
    assert angular_symmetry(fruit(4)) == pytest.approx(0.1, rel=1e-9)


# Assymetric fruits
def test_all_locules_clumped_is_one_for_n4():
    assert angular_symmetry(make_fruit([0, 0, 0, 0])) == pytest.approx(1.0)

def test_clustered_locules_known_value():
    # 0, 20, 40, 60 -> best rotation gives 70 deg mean error -> 70/90
    assert angular_symmetry(make_fruit([0, 20, 40, 60])) == pytest.approx(70 / 90)

def test_one_locule_displaced_known_value():
    # Ideal marks: 0,60,120,180,240,300. Last locule at 350 -> +50 deg off.
    # Best rotation leaves the 5 aligned locules at 0 error -> mean 50/6 deg.
    displaced = angular_symmetry(make_fruit([0, 60, 120, 180, 240, 350]))
    assert displaced == pytest.approx((50 / 6) / 60)
    assert displaced > angular_symmetry(make_fruit(ideal_deg(6)))

def test_error_grows_with_noise():
    rng = np.random.default_rng(42)
    base = np.array(ideal_deg(8))
    noise = rng.normal(size=(200, 8))
    means = []
    for sd in [1, 5, 10, 20]:
        vals = [angular_symmetry(make_fruit(base + sd * z)) for z in noise]
        means.append(np.mean(vals))
    assert means == sorted(means)

def test_result_is_never_negative_and_finite():
    rng = np.random.default_rng(1)
    for _ in range(200):
        n = int(rng.integers(2, 13))
        v = angular_symmetry(make_fruit(rng.uniform(0, 360, n)))
        assert np.isfinite(v) and v >= 0


# Check precomputation of ideal angles by locule count
def test_precompute_returns_vectors_of_shape_n():
    pre = precompute_ideal_angles(np.array([1, 2, 3, 4, 12]))
    assert set(pre.keys()) == {2, 3, 4, 12}          # n < 2 skipped
    for n, v in pre.items():
        assert isinstance(n, int)
        assert v.shape == (n,)
        np.testing.assert_allclose(v, np.arange(n) * 2*np.pi / n)

@pytest.mark.parametrize("n", [3, 4, 6, 9])
def test_precomputed_matches_on_the_fly(n):
    rng = np.random.default_rng(n)
    fruit = make_fruit(rng.uniform(0, 360, n))
    pre = precompute_ideal_angles(np.array([n]))
    assert angular_symmetry(fruit, precomputed_ideals=pre) == pytest.approx(
        angular_symmetry(fruit), abs=1e-12)

def test_precomputed_missing_n_falls_back():
    fruit = make_fruit(ideal_deg(5, 10))
    pre = precompute_ideal_angles(np.array([3, 4]))   # no 5
    assert angular_symmetry(fruit, precomputed_ideals=pre) == pytest.approx(0.0, abs=1e-9)

def test_precomputed_old_matrix_format_gives_wrong_result():
    """Documents the bug: old (angle_shifts, n) matrices silently give garbage."""
    n = 4
    fruit = make_fruit(ideal_deg(n, 15))              # perfect -> should be 0
    shifts = np.linspace(0, 2*np.pi, 500, endpoint=False)
    old = {n: (np.arange(n)*2*np.pi/n)[None, :] + shifts[:, None]}
    assert old[n].shape == (500, 4)
    assert angular_symmetry(fruit, precomputed_ideals=old) != pytest.approx(0.0, abs=1e-6)
