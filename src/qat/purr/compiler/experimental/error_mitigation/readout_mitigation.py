from typing import Dict

import numpy as np
from pydantic import BaseModel, ConfigDict


class ReadoutMitigation(BaseModel):
    """
    Attributes
    ----------
    linear: Dict[str, Dict[str, float]]
        Linear maps each individual qubit to its <0/1> given <0/1> probability according to
        {
            "<qubit_number>": {
                "0|0": p(0|0),
                "1|0": p(1|0),
                "0|1": p(0|1),
                "1|1": p(1|1),
            }
        }.
        Note that linear assumes no cross-talk effects and considers each qubit independent.
    matrix: np.array
        The entire 2**n x 2**n process matrix of p(<bitstring_1>|<bitstring_2>).
    m3: bool
        Runtime mitigation strategy that builds up the calibration it needs at runtime, hence a bool
        of available or not. For more info https://github.com/Qiskit-Partners/mthree.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    linear: Dict[str, Dict[str, float]] = None
    matrix: np.array = None
    m3: bool = False
