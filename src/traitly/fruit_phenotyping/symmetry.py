

# traitly/fruit_phenotyping/symmetry.py
"""
Locule symmetry metrics for fruit phenotyping pipelines.

Provides functions to quantify the angular and radial arrangement of
locules within a fruit. Designed to be called from
:func:`~traitly.fruit_phenotyping.analysis._calculate_symmetry_metrics`.

Precomputation utilities (:func:`get_unique_locule_counts` and
:func:`precompute_ideal_angles`) are intended to be called once per
image before the per-fruit loop to avoid redundant computation.
"""

# ============================================================================
# STANDARD LIBRARIES
# ============================================================================
from typing import Optional

# ============================================================================
# THIRD-PARTY LIBRARIES
# ============================================================================
import numpy as np
from scipy.stats import circmean
from scipy.optimize import linear_sum_assignment


#################################################################################################
# Precomputation utilities
#################################################################################################

def get_unique_locule_counts(
    fruit_locules_map: dict,
) -> np.ndarray:
    """
    Extract the sorted unique locule counts across all fruits.

    Used by :func:`precompute_ideal_angles` to determine which locule
    counts need precomputed ideal angle matrices before the per-fruit
    analysis loop.

    Parameters
    ----------
    fruit_locules_map : dict of {int : list of int}
        Mapping from fruit contour index to list of locule contour
        indices, as returned by
        :func:`~traitly.fruit_phenotyping.mask.find_fruits`.

    Returns
    -------
    np.ndarray
        Sorted 1-D array of unique locule counts across all fruits.
    """
    locule_counts = [len(locule_ids) for locule_ids in fruit_locules_map.values()]
    return np.unique(locule_counts)

##################################################################################################
# Precompute ideal angles for locule counts
##################################################################################################

def precompute_ideal_angles(unique_locule_counts: np.ndarray):
    "Precompute shifted ideal angle matrices for each unique locule count"
    precomputed = {}
    for n in unique_locule_counts:
        n = int(n)
        if n < 2:
            continue
        precomputed[n] = np.arange(n) * (2*np.pi / n)
    return precomputed

#################################################################################################
# Angular locule symmetry
#################################################################################################

def angular_symmetry(
    locules_data: list,
    precomputed_ideals: Optional[dict] = None,
) -> float:
    """
    Calculate angular symmetry of the locule arrangement around the fruit centroid.

    Parameters
    ----------
    locules_data : list of dict
        Per-locule data dicts as returned by
        :func:`~traitly.fruit_phenotyping.processing.precalculate_locules_data`.
        Each dict must contain ``'polar_coord'``, where
        ``polar_coord[0]`` is the locule angle in radians relative to
        the fruit centroid.
    precomputed_ideals : dict of {int : np.ndarray} or None, optional
        Precomputed ideal angle vectors of shape ``(n,)`` from
        :func:`precompute_ideal_angles`, keyed by locule count. If
        ``None`` or the current locule count is absent, the ideal angles
        are computed internally. Default is ``None``.

    Returns
    -------
    float
        Mean circular angular error at the best-matching rotation,
        as a fraction of the ideal spacing ``2π / n`` for dimensionless.
        A value of 0 indicates perfect angular symmetry and larger values indicate less symmetry.
        Returns ``NaN`` if fewer than 2 locules are present.

    """

    # Only process fruits with >= 2 locules
    if len(locules_data) < 2:
        return np.nan

    # Get all the locule angles and sort them
    angles = np.sort(np.array([d['polar_coord'][0] for d in locules_data]) % (2*np.pi))
    n = len(angles)
    step = 2*np.pi / n # get the ideal angle distance between n locules in radians [0, 2pi)

    # Get the ideal angles array for a fruit of reuse a pre computed array
    # it's included to avoid recalculate ideals for same n locules
    if precomputed_ideals is not None and n in precomputed_ideals:
        ideal = precomputed_ideals[n]
    else:
        ideal = np.arange(n) * step # in case the function is run in a different pipeline

    d = (angles - ideal) % (2*np.pi) # compare real vs ideal angles
    diff = np.abs(d[:, None] - d[None, :])  # rows are candidate rotations and cols are locule angles
    diff = np.minimum(diff, 2*np.pi - diff)  # get circular distance (shortest way around the circle)

    # keep the min distance and normalize the results by steps to make it comparable
    # between fruits with different n locules
    diff_norm = diff.mean(axis=1).min() / step

    return diff_norm

#################################################################################################
# Radial locules symmetry
#################################################################################################

def radial_symmetry(locules_data: list) -> float:
    """
    Calculate radial symmetry of locule arrangement using the CV of distances.

    Extracts the radial distance of each locule from the fruit centroid
    from ``locules_data`` and returns the coefficient of variation
    (CV = std / mean × 100). Lower values indicate more uniform radial
    spacing and thus better radial symmetry.

    Parameters
    ----------
    locules_data : list of dict
        Per-locule data dicts as returned by
        :func:`~traitly.fruit_phenotyping.processing.precalculate_locules_data`.
        Each dict must contain ``'polar_coord'``, where
        ``polar_coord[1]`` is the radial distance in pixels from the
        fruit centroid.

    Returns
    -------
    float
        Coefficient of variation of radial distances as a percentage,
        or ``NaN`` if fewer than 2 locules are present or the mean
        radius is zero.
    """
    if not locules_data or len(locules_data) < 2: # If there is fewer than 2 locules, symettry is undefined (no symmetry)
        return np.nan

    # Extract precalculated radii for each locule's data
    radii = [data['polar_coord'][1] for data in locules_data]

    # Calculate the coefficient of variation (CV = standard deviation / mean)
    radii_cv = (np.std(radii) / np.mean(radii) * 100) if np.mean(radii) > 0 else np.nan
    return radii_cv
