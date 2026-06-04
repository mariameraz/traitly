import pytest
import multiprocessing as mp
from traitly.utils.validation import _validate_num_cores

################################################################
### Validate num_cores for multiprocessing in batch analysis
################################################################

def test_validate_num_cores_zero():
    cores, msg = _validate_num_cores(0)
    assert cores == 1
    assert "must be at least 1" in msg

def test_validate_num_cores_exceeds_max():
    max_cores = mp.cpu_count()
    cores, msg = _validate_num_cores(max_cores + 1)
    assert cores == max_cores
    assert "exceeds system cores" in msg

def test_validate_num_cores_valid():
    cores, msg = _validate_num_cores(2)
    assert cores == 2
    assert msg is None
