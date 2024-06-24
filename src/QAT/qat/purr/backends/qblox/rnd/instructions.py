from typing import Union, List

import numpy as np

from qat.purr.compiler.instructions import SweepValue, Sweep


class ReducedSweepValue(SweepValue):
    def __init__(self, name, value):
        super().__init__(name, value)
        self.start, self.step, self.count = self._extract_triple()

    def _extract_triple(self):
        if self.value is None:
            return None, None, None

        value = self.value
        if isinstance(self.value, np.ndarray):
            value = self.value.tolist()

        start = None
        step = None
        count = len(value)

        if value:
            start = value[0]

        if len(value) >= 2:
            step = value[1] - value[0]

        for i in range(len(value) - 1):
            if np.isclose(step - (value[i + 1] - value[i]), 0):
                raise ValueError(f"Not a regularly partitioned space {value}")

        return start, step, count


class ReducedSweep(Sweep):
    def __init__(self, operations: Union[ReducedSweepValue, List[ReducedSweepValue]] = None):
        super().__init__(operations)

        if operations is None:
            operations = []
        elif not isinstance(operations, List):
            operations = [operations]

        self.operations = operations
