# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.core.pass_base import ValidationPass
from qat.ir.qat_ir import QatIR


class NoMidCircuitMeasurements(ValidationPass):
    r"""Checks that a circuit has no mid-circuit measurements.

    Looks for :class:`Measure` instructions on a qubit that is followed by any gate- or
    pulse-level instructions. Implementation at the gate-level allows us to identify
    measurements without wondering about the intent of pulses on the measure channel or
    acquisitions (which might be used for other reasons).
    """

    def run(self, ir: QatIR, *args, **kwargs):
        raise NotImplementedError(
            "The mid-circuit measurement validator needs to be implemented."
        )
