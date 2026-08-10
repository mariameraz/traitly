# tests/fruit_phenotyping/test_color_plot.py
#
# ============================================================================
# STANDARD LIBRARY
# ============================================================================
from pathlib import Path

# ============================================================================
# THIRD-PARTY
# ============================================================================
import pytest
from unittest.mock import patch
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
# ============================================================================
# INTERNAL
# ============================================================================
from traitly.fruit_phenotyping import plot_color_histogram

@pytest.fixture
def df_hist():
    path = Path(__file__).parent.parent / "data" / "external" / "blue_bg" / "hist_color_results.csv"
    return pd.read_csv(path)

@pytest.fixture
def df_means():
    """Generate color_results.csv if it doesn't exist."""
    from traitly.fruit_phenotyping import FruitExternalAnalyzer

    data_dir = Path(__file__).parent.parent / "data"
    external_folder_blue = data_dir / "external" / "blue_bg"
    results_file = external_folder_blue / "Results" / "color_results.csv"

    if not results_file.exists():
        analyzer = FruitExternalAnalyzer(path=external_folder_blue)
        analyzer.analyze_folder()

    return pd.read_csv(results_file)

@pytest.fixture(autouse=True)
def no_plots():
    with patch("matplotlib.pyplot.show"):
        yield

def test_valid_df_histogram(df_hist):
    result = plot_color_histogram(
        df=df_hist,
        color_space='rgb',
        overlay=True,
        alpha=1
    )
    assert result is None

def test_invalid_df_histogram(df_means):
    with pytest.raises(ValueError, match="does not contain histogram columns"):
        plot_color_histogram(
            df=df_means,
            color_space='rgb',
            overlay=True,
            alpha=1
        )

def test_histogram_returns_none(df_hist):
    result = plot_color_histogram(df=df_hist, color_space='rgb')
    assert result is None

@pytest.mark.parametrize("color_space", ["rgb", "lab", "hsv", "gray"])
def test_histogram_valid_color_spaces(df_hist, color_space):
    """All valid color spaces (including grey alias) should work."""
    result = plot_color_histogram(df=df_hist, color_space=color_space)
    assert result is None

@pytest.mark.parametrize("position", ["top-right", "top-left", "bottom-right", "bottom-left", "none"])
def test_histogram_valid_legend_positions(df_hist, position):
    """All valid legend positions should work."""
    result = plot_color_histogram(df=df_hist, color_space='rgb', legend_position=position)
    assert result is None

def test_histogram_invalid_color_space(df_hist):
    """Unknown color space should raise ValueError."""
    with pytest.raises(ValueError, match="Invalid color_space"):
        plot_color_histogram(df=df_hist, color_space='ycbcr')

def test_histogram_invalid_fruit_id(df_hist):
    """A fruit_id not present in the DataFrame should raise ValueError."""
    with pytest.raises(ValueError, match="fruit_id"):
        plot_color_histogram(df=df_hist, fruit_id=100)
