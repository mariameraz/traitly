# tests/fruit_phenotyping/test_batch_analysis.py
#
# ============================================================================
# STANDARD LIBRARY
# ============================================================================
import os
from pathlib import Path
import shutil
# ============================================================================
# THIRD-PARTY
# ============================================================================
import pytest
import pandas as pd

# ============================================================================
# INTERNAL
# ============================================================================

# Paths ############################################################################
from traitly.fruit_phenotyping import FruitExternalAnalyzer, FruitInternalAnalyzer
data_dir = Path(__file__).parent.parent / "data"

external_folder_blue = data_dir / "external" / "blue_bg"
external_folder_white =  data_dir / "external" / "white_bg"
internal_folder = data_dir / "internal"
####################################################################################

def get_input_images(folder: Path) -> list:
    images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
    return [p for p in images if "_processed" not in p.stem]

@pytest.fixture(autouse=True)
def cleanup_results(request):
    """Remove Results folders after each test to avoid output contamination."""
    yield
    for folder in [external_folder_blue, external_folder_white, internal_folder]:
        results = folder / "Results"
        if results.exists():
            shutil.rmtree(results)

def test_internal_analyzer():
    test = FruitInternalAnalyzer(path = internal_folder)
    test.analyze_folder()

    error_report = data_dir / "internal" / "Results" / "error_report.txt"
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
    assert len(df) == 57, f"Expectin 57 fruits detected, but obtain: {len(df)}"

def test_json():
    json_blue = Path(external_folder_blue) / "cranberry_blue_bg.json"
    test_blue = FruitExternalAnalyzer(path = external_folder_blue)
    test_blue.analyze_folder(json_path = json_blue, analyze_color = False)

    json_white = Path(external_folder_white) / "cranberry_white_bg.json"
    test_white = FruitExternalAnalyzer(path = external_folder_white)
    test_white.analyze_folder(json_path = json_white, analyze_morphology = False)

def test_processed_images_created():
    test = FruitExternalAnalyzer(path=external_folder_blue)
    test.analyze_folder()
    for img_path in get_input_images(external_folder_blue):
        expected = external_folder_blue / "Results" / f"{img_path.stem}_processed.jpg"
        assert expected.exists(), f"Processed image not found for: {img_path.name}"
