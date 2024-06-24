from dataclasses import dataclass
from typing import Any, List, Union

import numpy as np

from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


@dataclass
class Attribute:
    name: str = None
    value: Any = None


def extract_iter_bounds(value: Union[List, np.ndarray]):
    """
    Extracts bounds from given value if it's linearly and evenly
    spaced or fails otherwise.
    """

    if value is None:
        raise ValueError(f"Cannot process value {value}")

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if not value:
        raise ValueError(f"Cannot process value {value}")

    start = value[0]
    step = 0
    end = value[-1]
    count = len(value)

    if count >= 2:
        step = value[1] - value[0]

    if not np.isclose(step, (end - start) / (count - 1)):
        raise ValueError(f"Not a regularly partitioned space {value}")

    return start, step, end, count
