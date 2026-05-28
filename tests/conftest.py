# tests/conftest.py

from pathlib import Path
import warnings

import pytest
from traitly.fruit_phenotyping import FruitInternalAnalyzer


# Ignore PyTorch MPS warning on apple silicon
@pytest.fixture(autouse=True)
def ignore_pin_memory_warning():
    warnings.filterwarnings(
        "ignore",
        message="'pin_memory' argument is set as true but not supported on MPS",
        category=UserWarning
    )

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
