

# traitly/internal_structure/symmetry.py

# ============================================================================
# THIRD-PARTY LIBRARIES
# ============================================================================
import numpy as np
from scipy.stats import circmean
from scipy.optimize import linear_sum_assignment


#################################################################################################
# Angular locule symmetry
#################################################################################################

def angular_symmetry(locules_data, num_shifts=500):
    """
    Calculate angular symmetry by comparing actual locule angles with the most symmetrical arrangement.

    Args:
        REQUIRED:
            - locules_data (List[Dict]): List of dictionaries, each containing at least the 'polar_coord'
              of a locule, where 'polar_coord'[0] is the angle in radians from the reference centroid.
        OPTIONAL:
            - num_shifts (int): Number of angular shifts to test when trying to align the ideal angles
              to the observed angles (default = 1000).

    Returns:
        float: Normalized angular error in range [0, 1]:
               - 0.0  → perfect angular symmetry.
               - 1.0  → maximum possible angular deviation for given number of locules.
               - nan  → undefined if fewer than 2 locules.
    """
    if len(locules_data) < 2:  # If fewer than 2 locules, angular symmetry is undefined
        return np.nan

    angles = np.array([d['polar_coord'][0] for d in locules_data]) % (2 * np.pi) # Extract angles (in radians) for each locule, normalized to [0, 2π)
    n = len(angles)  # Total number of locules

    
    mean_angle = circmean(angles) # Center angles around their circular mean 
    angles_centered = (angles - mean_angle) % (2 * np.pi)

    ideal_angles = np.linspace(0, 2*np.pi, n, endpoint=False) # Define the ideal angles for a perfectly symmetric arrangement

    # Initialize best alignment search
    best_error = np.inf
    best_shift = None

    
    for shift in np.linspace(0, 2*np.pi, num_shifts, endpoint=False): # Test multiple rotational shifts to find best alignment with minimal angular deviation
        shifted_ideal = (ideal_angles + shift) % (2*np.pi)

        diff = np.abs(angles_centered[:, None] - shifted_ideal[None, :])  # Compute angular differences, considering wrap-around at 2π
        cost_matrix = np.minimum(diff, 2*np.pi - diff)
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix) # Find optimal assignment of observed to ideal angles using Hungarian algorithm
        angle_error = cost_matrix[row_ind, col_ind].mean()
        
        if angle_error < best_error: # Keep the best shift (smallest mean angular error)
            best_error = angle_error
            #best_shift = shift

    # Maximum possible mean angular error for given number of locules
    #max_angle_error = np.pi / n

    # Debug prints (can be commented out in production)
    #print(f"Best angle error (rad): {best_error}")
    #print(f"Max angle error (rad): {max_angle_error}")
    #print(f"Best shift (rad): {best_shift}")

    # Normalize error to range [0, 1]
    #angle_error_norm = min(best_error / max_angle_error, 1.0)
    return best_error



#################################################################################################
# Radial locules symmetry
#################################################################################################

def radial_symmetry(locules_data):
    """
    Calculate radial symmetry using coefficient of variation (CV) of distances.
    Args:
        REQUIRED:
            - locules_data (List[Dict]): List of dictionaries, where each dictionary contains the centroid coordinates (x,y) of a locule and precalculated 'polar_coordinates'.


    Returns:
        float: CV of distances (0 = perfect symmetry, nan = undefined).
    """
    if len(locules_data) < 2: # If there is fewer than 2 locules, symettry is undefined (no symmetry) 
        return np.nan

    radii = [data['polar_coord'][1] for data in locules_data] # Extract precalculated radii for each locule's data
    
    return np.std(radii) / np.mean(radii) if np.mean(radii) > 0 else 0.0 # Compute coefficient of variation (CV = standard deviation / mean)


#################################################################################################
# Rotational symmetry 
#################################################################################################

def rotational_symmetry(locules_data, angle_error=None, angle_weight=0.5, radius_weight=0.5, min_radius_threshold=0.1):
    """
    Calculates rotational symmetry for a fruit using both angular and radial asymmetry.
    0 = perfect symmetry, 1 = maximum asymmetry.
    Optionally accepts a precomputed angular error to avoid recalculation.

    Args:
        REQUIRED:
            - locules_data (List[Dict]): Each dict contains 'polar_coord' = (angle, radius)
        OPTIONAL:
            - angle_error (float, optional): Precomputed angular error (0-1). If None, it is calculated internally.
            - angle_weight (float): Weight of angular error in combined metric (default=0.5)
            - radius_weight (float): Weight of radial error in combined metric (default=0.5)
            - min_radius_threshold (float): Ignore locules with radius < fraction of mean (default=0.1)

    Returns:
        float: Combined rotational symmetry metric in [0,1], or np.nan if undefined.
    """

    if len(locules_data) < 2: # Check for minimum number of locules
        return np.nan  # Cannot define symmetry with fewer than 2 locules

    # Extract and normalize radial distances
    radii = np.array([d['polar_coord'][1] for d in locules_data]) # Extract radius for each locule
    radii_normalized = radii / np.mean(radii) # Normalize by mean radius for comparability
    valid_mask = radii_normalized >= min_radius_threshold # Ignore very small locules (likely noise)
    radii_normalized = radii_normalized[valid_mask] # Keep only valid radii

    if len(radii_normalized) < 2: # Check if enough locules remain after filtering    
        return np.nan  # Symmetry undefined if too few valid locules remain

    # Calculate radial error using Median Absolute Deviation (MAD)
    median_abs_dev = np.median(np.abs(radii_normalized - 1.0)) # Typical deviation from mean radius
    radius_error_norm = np.tanh(median_abs_dev / 0.6745) # Normalize radial error to ~[0,1), robust to outliers
    
    if angle_error is None:
        angle_error = angular_symmetry(locules_data) # Compute angular asymmetry

    total_weight = angle_weight + radius_weight 
    combined_error = (angle_weight * angle_error + radius_weight * radius_error_norm) / total_weight # Combine angular and radial errors (weighted average)

    return np.clip(combined_error, 0.0, 1.0) # Ensure combined error is within [0,1]

