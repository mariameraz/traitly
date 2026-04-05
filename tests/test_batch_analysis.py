# tests/test_batch.py

# ============================================================================
# STANDARD LIBRARY
# ============================================================================
import os
from pathlib import Path

# ============================================================================
# THIRD-PARTY
# ============================================================================
import pytest
import pandas as pd
# ============================================================================
# INTERNAL IMPORTS
# ============================================================================
from traitly.fruit_phenotyping import FruitExternalAnalyzer, FruitInternalAnalyzer

external_folder_blue = Path(__file__).parent / "data/external/blue_bg"
external_folder_white = Path(__file__).parent / "data/external/white_bg"
internal_folder = Path(__file__).parent / "data/internal"


def test_internal_analyzer():
    test = FruitInternalAnalyzer(path = internal_folder)
    test.analyze_folder()

    error_report = Path(internal_folder) / "Results" / "error_report.txt"
    session_report = Path(internal_folder) / "Results" / "session_report.txt"
    color_res = Path(internal_folder) / "Results" / "color_results.csv"
    morpho_res = Path(internal_folder) / "Results" / "morphology_results.csv"

    assert os.path.exists(error_report), (f"error_report.txt not found")
    assert os.path.exists(session_report), (f"session_report.txt not found")
    assert os.path.exists(color_res), f"color_results.csv not found"
    assert os.path.exists(morpho_res), (f"morphology_results.csv not found")

def test_int_json():
    json = Path(internal_folder) / "img_test_1_parameters.json"
    test = FruitInternalAnalyzer(path = internal_folder)
    test.analyze_folder(json_path = json)

def test_external_analyzer():
    test = FruitExternalAnalyzer(path = external_folder_blue)
    test.analyze_folder()

def test_pass_params():
    test = FruitExternalAnalyzer(path=external_folder_white)
    test.analyze_folder(background_color='white')

    color_res = Path(external_folder_white) / "Results" / "color_results.csv"
    assert color_res.exists(), f"color_results.csv not found"

    df = pd.read_csv(color_res)
    assert "fruit_id" in df.columns, "color_results.csv doesn't include 'fruit_id' column"
    assert len(df) == 45, f"Expectin 45 fruits detected, but obtain: {len(df)}"

def test_json():
    json_blue = Path(external_folder_blue) / "cranberry_blue_bg.json"
    test_blue = FruitExternalAnalyzer(path = external_folder_blue)
    test_blue.analyze_folder(json_path = json_blue, analyze_color = False)

    json_white = Path(external_folder_white) / "cranberry_white_bg.json"
    test_white = FruitExternalAnalyzer(path = external_folder_white)
    test_white.analyze_folder(json_path = json_white, analyze_morphology = False)
