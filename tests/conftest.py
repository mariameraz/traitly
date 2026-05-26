# tests/conftest.py

from pathlib import Path
import pytest
from traitly.fruit_phenotyping.internal import FruitInternalAnalyzer

valid_img = Path(__file__).parent / "data" / "internal" / "img_test_1.jpg"

@pytest.fixture
def cranberry_valid():
    cranberry = FruitInternalAnalyzer(path=valid_img)
    cranberry.load_image(plot=False)
    cranberry.setup_measurements()
    cranberry.generate_fruit_mask(plot=False)
    cranberry.detect_fruits(plot=False)
    cranberry.analyze_morphology(plot=False, display_table=False)
    cranberry.analyze_color(plot=False, display_table=False, tissue="OUTER_PERICARP", color_space="rgb")
    cranberry.results.save_csv()
    cranberry.save_parameters()
    return cranberry
