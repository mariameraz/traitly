# ============================================================================
# THIRD-PARTY LIBRARIES
# ============================================================================
import numpy as np

# Reference Lab D65 and D50 values for the X-Rite ColorChecker Classic 24 patches.
# Original X-Rite D50 data (after November 2014):
# https://www.xrite.com/service-support/new_color_specifications_for_colorchecker_sg_and_classic_charts
# D65 obtained converting from Lab D50 -> XYZ -> Lab D65 (Brandford adaptation using `colours-science` Python library)

# Cols order: L, a, b
CHECKER_LAB_D65 = np.array([
 [ 37.54, 15.5541, 21.7589], # A1: dark skin
 [ 64.66, 21.0425, 28.2429], # B1: light skin
 [ 49.32, -2.4785, -10.5961], # C1: blue sky
 [ 43.46, -11.5636, 29.7716], # D1: foliage
 [ 54.94, 11.1334, -11.7910], # E1: blue flower
 [ 70.48, -30.6153, 12.8387], # F1: bluish green
 [ 62.73, 37.6423, 63.5024], # A2: orange
 [ 39.43, 11.9560, -32.7331], # B2: purplish blue
 [ 50.57, 50.2610, 25.3395], # C2: moderate red
 [ 30.10, 23.6087, -12.0020], # D2: purple
 [ 71.77, -22.4191, 66.4204], # E2: yellow green
 [ 71.51, 20.1501, 74.7496], # F2: orange yellow
 [ 28.37, 16.4183, -38.6391], # A3: blue
 [ 54.38, -38.4465, 40.1414], # B3: green
 [ 42.43, 52.5131, 34.9933], # C3: red
 [ 81.80, 4.7191, 88.2050], # D3: yellow
 [ 50.63, 52.9150, -2.7205], # E3: magenta
 [ 49.57, -28.4883, -15.8272], # F3: cyan
 [ 95.19, 1.2801, 19.6119], # A4: white
 [ 81.29, 1.4529, 15.2242], # B4: neutral 80
 [ 66.89, 0.9722, 12.5744], # C4: neutral 65
 [ 50.76, 1.2593, 10.2991], # D4: neutral 50
 [ 35.63, 0.6127, 7.4287], # E4: neutral 35
 [ 20.64, 0.8332, 5.1631], # F4: black
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
