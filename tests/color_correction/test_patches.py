import numpy as np

from traitly.color_correction.color_charts import CHECKER_LAB_D50, CHECKER_PATCH_NAMES
from traitly.color_correction.color_analysis import _get_lab_patches


def test_get_lab_patches_output_shape():
    fake_chart = np.random.rand(72, 5).astype(np.float32)
    result = _get_lab_patches(fake_chart)
    assert result.shape == (24, 3)

def test_patch_names_extraction():
    names = [name.split(": ", 1)[1] for name in CHECKER_PATCH_NAMES]
    assert names[0] == "dark skin"
    assert names[-1] == "black"
    assert len(names) == 24

def test_patch_names_extraction():
    names = [name.split(": ", 1)[1] for name in CHECKER_PATCH_NAMES]
    assert names[0] == "dark skin"
    assert names[-1] == "black"
    assert len(names) == 24
