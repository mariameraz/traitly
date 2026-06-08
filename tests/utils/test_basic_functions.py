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
from traitly.utils.basic_functions import load_img, _load_img_cached

##########################
# Image load and process #
# ########################

## 1. Invalid paths
invalid_paths = ["/img.txt", "/img.pdf", "/img.gif", "/img"]

@pytest.mark.parametrize("bad_paths", invalid_paths)
def test_extension_invalid(bad_paths):
    assert load_img(bad_paths) is None
    with pytest.raises(ValueError, match="Unsupported image format"):
            _load_img_cached(bad_paths)

@pytest.mark.parametrize("bad_paths", invalid_paths)
def test_load_img_cached_invalid_extension_raises(bad_paths):
    _load_img_cached.cache_clear() # clean lru after every test
    with pytest.raises(ValueError, match="Unsupported image format"):
        _load_img_cached(bad_paths)

## 2. Empty file
def test_load_img_cached_cannot_load_img():
    with pytest.raises(ValueError, match="Cannot load image"):
        _load_img_cached("nonexistent.jpg")

## 3. Import color image and cut ROI

img_path = Path(__file__).parent.parent / "data" / "external" / "blue_bg" / "cranberry_blue_bg.jpg"

@pytest.fixture(autouse=True)
def clear_cache():
    _load_img_cached.cache_clear()
    yield
    _load_img_cached.cache_clear()

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

def test_lru_cache_hits():
    _load_img_cached.cache_clear()

    _load_img_cached(str(img_path))
    info = _load_img_cached.cache_info()
    assert info.hits == 0 # First call, so cache is empty
    assert info.misses == 1

    _load_img_cached(str(img_path))
    info = _load_img_cached.cache_info()
    assert info.hits == 1 # second call, find the previous read image
    assert info.misses == 1

    _load_img_cached(str(img_path))
    info = _load_img_cached.cache_info()
    assert info.hits == 2 # third call, cache hit it again
    assert info.misses == 1

def test_lru_cache_different_paths():
    _load_img_cached.cache_clear()

    # two different paths
    _load_img_cached(str(img_path))
    _load_img_cached(str(img_path_2))
    info = _load_img_cached.cache_info()
    assert info.misses == 2
    assert info.hits == 0

def test_lru_cache_clear():
    _load_img_cached(str(img_path))
    _load_img_cached.cache_clear()

    info = _load_img_cached.cache_info()
    assert info.currsize == 0   # empty cache
