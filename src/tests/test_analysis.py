import numpy as np
import pytest

from qat.purr.compiler.analysis import extract_iter_bounds
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()

count = 100
cases = [([1, 2, 4, 10], None), ([-0.1, 0, 0.1, 0.2, 0.3], (-0.1, 0.1, 0.3, 5))] + [
    (np.linspace(b[0], b[1], count), (b[0], 1, b[1], count))
    for b in [(1, 100), (-50, 49)]
]


@pytest.mark.parametrize("value, bounds", cases)
def test_extract_bounds(value, bounds):
    if bounds is None:
        with pytest.raises(ValueError):
            extract_iter_bounds(value)
    else:
        assert extract_iter_bounds(value) == bounds
