# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, field_validator

from qat.utils.pydantic import (
    CalibratableUnitInterval2x2Array,
    FrozenDict,
    NoExtraFieldsModel,
    QubitId,
)


class ReadoutMitigation(NoExtraFieldsModel):
    """
    :param linear: Maps each individual qubit to its <0/1> state given the <0/1> probabilistic
        error that happens on the device. Note that linear assumes no cross-talk effects and
        considers each qubit independent.
        linear = {
            <qubit_number>: NDArray(2, 2)
        }
        The matrix element (i, j) of the NDArray maps the probability p(i|j).
    :param m3_available: Flag to enable a runtime mitigation strategy that builds up the calibration
                        it needs at runtime. For more info https://github.com/Qiskit-Partners/mthree.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # TODO: COMPILER-706 linear readout mitigation currently a 2x2 matrix,
    #  we may want to change this to be a dictionary like in the old hardware model.
    linear: FrozenDict[QubitId, CalibratableUnitInterval2x2Array]
    matrix: NDArray | None = None
    m3_available: bool = False

    @field_validator("linear")
    @classmethod
    def validate_linear(cls, linear):
        for qubit_idx, qubit_map in linear.items():
            if not np.allclose(np.sum(qubit_map, axis=0), np.ones(qubit_map.shape[0])):
                raise ValueError(
                    f"Please provide a linear probability map for qubit {qubit_idx} where all probabilities p(i|{qubit_idx} sum to 1."
                )

        return linear

    @property
    def qubits(self):
        return list(self.linear.keys())


class ErrorMitigation(NoExtraFieldsModel):
    """
    A collection of error mitigation strategies. Currently, this holds a single mitigation strategy,
    but this can be expanded in the future as we add more error correction schemes.

    :param readout_mitigation: Linear readout mitigation.
    """

    readout_mitigation: ReadoutMitigation | None = None

    @property
    def is_enabled(self):
        return (
            True
            if self.readout_mitigation and len(self.readout_mitigation.linear)
            else False
        )

    @property
    def qubits(self):
        return self.readout_mitigation.qubits if self.readout_mitigation else []
