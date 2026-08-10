from traitly.color_correction.color_charts import CHECKER_LAB_D50, CHECKER_PATCH_NAMES
from traitly.color_correction.color_analysis import _delta_e

def test_delta_e_shape():
    result = _delta_e(CHECKER_LAB_D50)
    assert result.shape == (24,)
