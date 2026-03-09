

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

def precompute_ideal_angles(
    unique_locule_counts: np.ndarray,
    angle_shifts: int = 500,
) -> dict:
    """
    Precompute shifted ideal angle matrices for each unique locule count.

    For each count ``n`` in ``unique_locule_counts``, generates ``n``
    evenly spaced ideal angles in [0, 2π) and shifts them by
    ``angle_shifts`` equally spaced offsets. The resulting matrix is
    used directly by :func:`angular_symmetry` to avoid redundant
    computation across fruits with the same locule count.

    Counts below 2 are skipped because symmetry is undefined for fewer
    than two locules.

    Parameters
    ----------
    unique_locule_counts : np.ndarray
        Sorted array of unique locule counts as returned by
        :func:`get_unique_locule_counts`.
    angle_shifts : int, optional
        Number of angular offsets to test. Higher values improve
        alignment accuracy at the cost of speed. Default is 500.

    Returns
    -------
    dict of {int : np.ndarray}
        Mapping from locule count ``n`` to a shifted ideal angle matrix
        of shape ``(angle_shifts, n)``, where each row is a rotated
        version of the ideal equally-spaced arrangement.
    """
    shifts = np.linspace(0, 2*np.pi, angle_shifts, endpoint=False)
    precomputed = {}
    
    for n in unique_locule_counts:
        if n < 2:
            continue

        ideal_angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        shifted_ideals = (ideal_angles[None, :] + shifts[:, None]) % (2 * np.pi)
        precomputed[n] = shifted_ideals
    
    return precomputed

#################################################################################################
# Angular locule symmetry
#################################################################################################

def angular_symmetry(
    locules_data: list,
    angle_shifts: int = 500,
    precomputed_ideals: Optional[dict] = None,
) -> float:
    """
    Calculate angular symmetry of locule arrangement around the fruit centroid.

    Extracts the polar angle of each locule from ``locules_data``,
    centers them by subtracting the circular mean, and finds the rotation
    of an ideal equally-spaced arrangement that best matches the observed
    angles. Optimal assignment at each rotation is solved with the
    Hungarian algorithm via :func:`scipy.optimize.linear_sum_assignment`.
    The best (minimum) mean angular error across all rotations is
    returned.

    Lower values indicate better angular symmetry. The result is in
    radians and can be normalized externally if needed.

    Parameters
    ----------
    locules_data : list of dict
        Per-locule data dicts as returned by
        :func:`~traitly.fruit_phenotyping.processing.precalculate_locules_data`.
        Each dict must contain ``'polar_coord'``, where
        ``polar_coord[0]`` is the locule angle in radians relative to
        the fruit centroid.
    angle_shifts : int, optional
        Number of angular offsets to test when ``precomputed_ideals``
        is ``None`` or does not contain the required locule count.
        Default is 500.
    precomputed_ideals : dict of {int : np.ndarray} or None, optional
        Precomputed shifted ideal angle matrices from
        :func:`precompute_ideal_angles`, keyed by locule count. If
        ``None`` or the current locule count is absent, ideal angles
        are computed on the fly. Default is ``None``.

    Returns
    -------
    float
        Mean angular error in radians at the best-matching rotation,
        or ``NaN`` if fewer than 2 locules are present.
    """
    if len(locules_data) < 2:
        return np.nan
    
    angles = np.array([d['polar_coord'][0] for d in locules_data]) % (2 * np.pi)
    n = len(angles)
    
    mean_angle = circmean(angles)
    angles_centered = (angles - mean_angle) % (2 * np.pi)
    
    # Use precomputed ideals if available
    if precomputed_ideals is not None and n in precomputed_ideals:
        shifted_ideals = precomputed_ideals[n]
    else:
        ideal_angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        shifts = np.linspace(0, 2*np.pi, angle_shifts, endpoint=False)
        shifted_ideals = (ideal_angles[None, :] + shifts[:, None]) % (2 * np.pi)
    
    # Compute cost matrices
    diff = np.abs(angles_centered[None, :, None] - shifted_ideals[:, None, :])
    cost_matrices = np.minimum(diff, 2*np.pi - diff)
    
    # Find best alignment
    best_error = np.inf
    for i in range(len(shifted_ideals)):
        row_ind, col_ind = linear_sum_assignment(cost_matrices[i])
        error = cost_matrices[i, row_ind, col_ind].mean()
        if error < best_error:
            best_error = error
    
    return best_error

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

