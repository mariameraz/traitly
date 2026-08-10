#tests/utils/test_basic_functions.py

# ============================================================================
# STANDARD LIBRARY
# ============================================================================
from pathlib import Path
from unittest.mock import patch

# ============================================================================
# THIRD-PARTY
# ============================================================================
import pytest
import numpy as np

# ============================================================================
# INTERNAL
# ============================================================================
from traitly.utils.basic_functions import load_img

##########################
# Image load and process #
# ########################

## 1. Invalid paths
invalid_paths = ["/img.txt", "/img.pdf", "/img.gif", "/img"]

@pytest.mark.parametrize("bad_paths", invalid_paths)
def test_load_img_invalid_extension_raises(bad_paths):
    with pytest.raises(ValueError, match="Unsupported image format"):
        load_img(bad_paths)

@pytest.mark.parametrize("bad_paths", invalid_paths)
def test_load_img_invalid_extension_raises(bad_paths):
    with pytest.raises(ValueError, match="Unsupported image format"):
        load_img(bad_paths)

## 2. Empty file
def test_load_img_cannot_load_img():
    with pytest.raises(ValueError, match="Cannot load image"):
        load_img("nonexistent.jpg")

## 3. Import color image and cut ROI

img_path = Path(__file__).parent.parent / "data" / "external" / "blue_bg" / "cranberry_blue_bg.jpg"


def test_load_img_shape():
    img = load_img(img_path)
    assert img is not None
    assert img.shape == (3456, 5184, 3) # loaded as BGR

@pytest.mark.parametrize("x, y, w, h, expected_shape", [
    (0, 0, 500, 300, (300, 500, 3)),
    (100, 200, 800, 400, (400, 800, 3)),
    (0, 0, 5184, 3456, (3456, 5184, 3)),
])

def test_load_img_crop_shape(x, y, w, h, expected_shape):
    img = load_img(img_path, x=x, y=y, w=w, h=h)
    assert img is not None
    assert img.shape == expected_shape

def test_load_img_crop_still_color():
    img = load_img(img_path, x=0, y=0, w=500, h=300)
    assert img is not None
    assert img.shape[2] == 3 #still bgr

# 4. Check how the cache is working

img_path_2 = Path(__file__).parent.parent / "data" / "external" / "white_bg" / "cranberry_white_bg.jpg"

# hits = how many times it founds the image in the cache (no in the disk)
# misses = how many paths are being loaded and going to the disk
# maxsize= entry max size (128 in this case)
# currsize = how many entries do we have saved in the cache
