# ============================================================================
# THIRD-PARTY LIBRARIES
# ============================================================================
import numpy as np

# Reference Lab D65 values for the X-Rite ColorChecker Classic 24 patches.
# Obtained from the X-Rite D50 data (after November 2014) by converting
# Lab D50 -> XYZ (D50) -> Bradford chromatic adaptation to D65 -> Lab D65
# with (colour-science 0.4.7). Cols order: L, a, b:
#
#    cs = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']
#    xyz = colour.Lab_to_XYZ(CHECKER_LAB_D50, cs['D50'])
#    xyz = colour.adaptation.chromatic_adaptation_VonKries(
#    xyz, colour.xy_to_XYZ(cs['D50']), colour.xy_to_XYZ(cs['D65']), transform='Bradford')
#    CHECKER_LAB_D65 = colour.XYZ_to_Lab(xyz, cs['D65'])

CHECKER_LAB_D65 = np.array([
 [ 37.31, 13.37, 14.58], # A1: dark skin
 [ 64.37, 18.03, 17.05], # B1: light skin
 [ 49.62, -1.18, -22.17], # C1: blue sky
 [ 43.35, -14.64, 22.86], # D1: foliage
 [ 55.18, 12.14, -24.57], # E1: blue flower
 [ 70.67, -31.94, 0.08], # F1: bluish green
 [ 62.11, 33.38, 55.76], # A2: orange
 [ 40.06, 16.25, -44.37], # B2: purplish blue
 [ 50.06, 48.10, 15.60], # C2: moderate red
 [ 30.21, 24.39, -20.88], # D2: purple
 [ 71.52, -28.42, 58.85], # E2: yellow green
 [ 70.96, 14.75, 67.25], # F2: orange yellow
 [ 29.15, 21.66, -48.74], # A3: blue
 [ 54.35, -42.68, 32.87], # B3: green
 [ 41.82, 50.33, 27.36], # C3: red
 [ 81.30, -1.87, 80.91], # D3: yellow
 [ 50.40, 52.49, -14.82], # E3: magenta
 [ 50.10, -24.99, -27.52], # F3: cyan
 [ 95.17, -1.30, 2.92], # A4: white
 [ 81.29, -0.61, 0.44], # B4: neutral 80
 [ 66.90, -0.74, -0.05], # C4: neutral 65
 [ 50.76, -0.14, 0.14], # D4: neutral 50
 [ 35.64, -0.41, -0.47], # E4: neutral 35
 [ 20.64, 0.11, -0.46], # F4: black
], dtype=np.float32)

CHECKER_LAB_D50 = np.array([
 [ 37.54, 14.37, 14.92],
 [ 64.66, 19.27, 17.50],
 [ 49.32, -3.82, -22.54],
 [ 43.46, -12.74, 22.72],
 [ 54.94, 9.61, -24.79],
 [ 70.48, -32.26, -0.37],
 [ 62.73, 35.83, 56.50],
 [ 39.43, 10.75, -45.17],
 [ 50.57, 48.64, 16.67],
 [ 30.10, 22.54, -20.87],
 [ 71.77, -24.13, 58.19],
 [ 71.51, 18.24, 67.37],
 [ 28.37, 15.42, -49.80],
 [ 54.38, -39.72, 32.27],
 [ 42.43, 51.05, 28.62],
 [ 81.80, 2.67, 80.41],
 [ 50.63, 51.28, -14.12],
 [ 49.57, -29.71, -28.32],
 [ 95.19, -1.03, 2.93],
 [ 81.29, -0.57, 0.44],
 [ 66.89, -0.75, -0.06],
 [ 50.76, -0.13, 0.14],
 [ 35.63, -0.46, -0.48],
 [ 20.64, 0.07, -0.46],
], dtype=np.float32)

CHECKER_PATCH_NAMES = [
    "A1: dark skin", "B1: light skin", "C1: blue sky", "D1: foliage", "E1: blue flower", "F1: bluish green",
    "A2: orange", "B2: purplish blue", "C2: moderate red", "D2: purple", "E2: yellow green", "F2: orange yellow",
    "A3: blue", "B3: green", "C3: red", "D3: yellow", "E3: magenta", "F3: cyan",
    "A4: white", "B4: neutral 80", "C4: neutral 65", "D4: neutral 50", "E4: neutral 35", "F4: black"
]
