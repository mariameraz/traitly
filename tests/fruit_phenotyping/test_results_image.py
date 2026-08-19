# tests/fruit_phenotyping/test_results_image.py

# ============================================================================
# STANDARD
# ============================================================================
#
import os

# ============================================================================
# THIRD-PARTY
# ============================================================================
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

# ============================================================================
# INTERNAL
# ============================================================================
from traitly.fruit_phenotyping.results_image import ResultsImage


@pytest.fixture
def sample_img():
    return np.zeros((10, 10, 3), dtype=np.uint8)

def test_init_defaults(sample_img):
    ri = ResultsImage(sample_img)
    assert ri.morphology_results == []
    assert ri.color_results == []
    assert ri.path is None

def test_init_copies_image_for_color(sample_img):
    ri = ResultsImage(sample_img)
    ri.color_image[0, 0, 0] = 255
    assert ri.morphology_image[0, 0, 0] == 0

def test_init_raises_on_none_image():
    with pytest.raises(ValueError):
        ResultsImage(None)

def test_resolve_img_to_save_explicit_morphology(sample_img):
    ri = ResultsImage(sample_img)
    ri._resolve_img_to_save(image_type="morphology")
    assert ri._img_to_save is ri.morphology_image

def test_resolve_img_to_save_explicit_color(sample_img):
    ri = ResultsImage(sample_img)
    ri._resolve_img_to_save(image_type="color")
    assert ri._img_to_save is ri.color_image


def test_resolve_img_to_save_invalid_type(sample_img):
    ri = ResultsImage(sample_img)
    with pytest.raises(ValueError):
        ri._resolve_img_to_save(image_type="bad")


def test_resolve_img_to_save_auto_prefers_morph(sample_img):
    ri = ResultsImage(sample_img, morphology_results=[{"a": 1}])
    ri._resolve_img_to_save(image_type="auto")
    assert ri._img_to_save is ri.morphology_image

def test_resolve_img_to_save_auto_falls_back_to_color(sample_img):
    ri = ResultsImage(sample_img, color_results=[{"a": 1}])
    ri._resolve_img_to_save(image_type="auto")
    assert ri._img_to_save is ri.color_image

@patch("traitly.fruit_phenotyping.results_image._save_img")
@patch("traitly.fruit_phenotyping.results_image._format_output_path")
def test_save_img_calls_save(mock_format, mock_save, sample_img):
    mock_format.return_value = ("/tmp", "out.jpg")
    ri = ResultsImage(sample_img, path="/tmp/original.jpg")
    ri.save_img()
    mock_save.assert_called_once()

@patch("traitly.fruit_phenotyping.results_image._save_df")
@patch("traitly.fruit_phenotyping.results_image._save_img")
@patch("traitly.fruit_phenotyping.results_image._format_output_path")
def test_save_all_calls_all_savers(mock_format, mock_save_img, mock_save_df, sample_img):
    mock_format.return_value = ("/tmp", "out.csv")
    ri = ResultsImage(sample_img, morphology_results=[{"a": 1}], path="/tmp/original.jpg")
    ri.save_all()
    assert mock_save_img.called
    assert mock_save_df.call_count == 2

@patch("traitly.fruit_phenotyping.results_image._format_output_path", side_effect=Exception("boom"))
def test_save_all_wraps_exceptions(mock_format, sample_img):
    ri = ResultsImage(sample_img, path="/tmp/original.jpg")
    with pytest.raises(RuntimeError):
        ri.save_all()

def test_get_base_path_no_path_no_original_raises(sample_img):
    ri = ResultsImage(sample_img)
    with pytest.raises(ValueError):
        ri._get_base_path(None, "results")

def test_get_base_path_uses_original_path(sample_img):
    ri = ResultsImage(sample_img, path="/tmp/imgs/original.jpg")
    base = ri._get_base_path(None, "results")
    assert base == os.path.join("/tmp/imgs", "results")

def test_get_base_path_invalid_extension(tmp_path, sample_img):
    ri = ResultsImage(sample_img)
    bad_file = tmp_path / "out.txt"
    with pytest.raises(ValueError):
        ri._get_base_path(str(bad_file), "results")

@patch("traitly.fruit_phenotyping.results_image._save_results")
def test_save_csv_invalid_mode_raises(mock_save, sample_img, tmp_path):
    ri = ResultsImage(sample_img, path=str(tmp_path / "img.jpg"))
    with pytest.raises(ValueError):
        ri.save_csv(data="not_a_mode")

@patch("traitly.fruit_phenotyping.results_image._save_results")
def test_save_csv_morphology_mode(mock_save, sample_img, tmp_path):
    ri = ResultsImage(sample_img, morphology_results=[{"a": 1}], path=str(tmp_path / "img.jpg"))
    ri.save_csv(data="morphology")
    mock_save.assert_called_once()
    args, kwargs = mock_save.call_args
    assert kwargs.get("require_morph") is True
