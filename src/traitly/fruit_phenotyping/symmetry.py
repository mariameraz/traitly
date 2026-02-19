

# traitly/internal_structure/symmetry.py

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

def get_unique_locule_counts(fruit_locules_map: dict):
    """
    Extract unique locule counts from fruit_locules_map.
    
    Args:
        fruit_locules_map (Dict[int, List[int]]): Dictionary mapping fruit_id -> list of locule_ids
        
    Returns:
        np.ndarray: Sorted array of unique locule counts
    """
    locule_counts = [len(locule_ids) for locule_ids in fruit_locules_map.values()]
    return np.unique(locule_counts)

##################################################################################################
# Precompute ideal angles for locule counts
##################################################################################################

def precompute_ideal_angles(unique_locule_counts: np.ndarray, 
                            angle_shifts: int =1000):
    """
    Precompute ideal angles and shifts for each unique locule count.
    
    Args:
        unique_locule_counts (array-like): Array of unique locule counts in dataset
        angle_shifts (int): Number of angular shifts to test
        
    Returns:
        dict: Dictionary mapping locule_count -> shifted_ideal_angles matrix
              Each matrix has shape (angle_shifts, locule_count)
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

def angular_symmetry(locules_data: list, 
                     angle_shifts: int = 500, 
                     precomputed_ideals: Optional[dict] = None):
    """
    Calculate angular symmetry by comparing actual locule angles with the most symmetrical arrangement.
    
    Args:
        REQUIRED:
            - locules_data (List[Dict]): List of dictionaries, each containing at least the 'polar_coord'
              of a locule, where 'polar_coord'[0] is the angle in radians from the reference centroid.
        OPTIONAL:
            - angle_shifts (int): Number of angular shifts to test (default = 100).
            - precomputed_ideals (dict): Precomputed shifted ideal angles. If None, computes on-the-fly.
              
    Returns:
        float: Normalized angular error in range [0, 1]
    """
    if not locules_data or len(locules_data) < 2:
        return np.nan
    
    angles = np.array([d['polar_coord'][0] for d in locules_data]) % (2 * np.pi)
    n = len(angles)
    
    mean_angle = circmean(angles)
    angles_centered = (angles - mean_angle) % (2 * np.pi)
    
    # Use precomputed ideals if available, otherwise compute on-the-fly
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

def radial_symmetry(locules_data: list):
    """
    Calculate radial symmetry using coefficient of variation (CV) of distances.
    Args:
        REQUIRED:
            - locules_data (List[Dict]): List of dictionaries, where each dictionary contains the centroid coordinates (x,y) of a locule and precalculated 'polar_coordinates'.


    Returns:
        radii (List[float]): List of radial distances for each locule.
        float: CV of distances (0 = perfect symmetry, nan = undefined).
    """
    if not locules_data or len(locules_data) < 2: # If there is fewer than 2 locules, symettry is undefined (no symmetry) 
        return np.nan

    # Extract precalculated radii for each locule's data
    radii = [data['polar_coord'][1] for data in locules_data] 
    
    # Calculate the coefficient of variation (CV = standard deviation / mean)
    radii_cv = (np.std(radii) / np.mean(radii) * 100) if np.mean(radii) > 0 else np.nan 
    return radii_cv

