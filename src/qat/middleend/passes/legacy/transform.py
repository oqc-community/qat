# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
from qat.core.pass_base import TransformPass
from qat.purr.backends.qiskit_simulator import QiskitBuilder, QiskitBuilderWrapper
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.instructions import Acquire, AcquireMode


class IntegratorAcquireSanitisation(TransformPass):
    """Changes `AcquireMode.INTEGRATOR` acquisitions to `AcquireMode.RAW`.

    The legacy echo/RTCS engines expect the acquisition mode to be either `RAW` or `SCOPE`.
    While the actual execution can process `INTEGRATOR` by treating it as `RAW`, they are
    typically santitised the runtime using :meth:`EchoEngine.optimize()`. If not done in the
    new pipelines, it will conflict with :class:`PostProcessingSantisiation`, and return the
    wrong results. The new echo engine supports all acquisition modes, so this is not a
    problem here.
    """

    def run(
        self,
        ir: InstructionBuilder,
        *args,
        **kwargs,
    ):
        """:param ir: The list of instructions stored in an :class:`InstructionBuilder`."""
        for inst in [instr for instr in ir.instructions if isinstance(instr, Acquire)]:
            if inst.mode == AcquireMode.INTEGRATOR:
                inst.mode = AcquireMode.RAW
        return ir


class QiskitInstructionsWrapper(TransformPass):
    """Wraps the Qiskit builder in a wrapper to match the pipelines API.

    A really silly pass needed to wrap the :class:`QiskitBuilder` in an object that allows
    `QiskitBuilderWrapper.instructions` to be called, allowing the builder to be used in the
    the :class:`LegacyRuntime`. This is needed because the qiskit engine has a different API
    to other `purr` engines, requiring the whole builder to be passed (as opposed to
    `builder.instructions`).
    """

    def run(self, ir: QiskitBuilder, *args, **kwargs) -> QiskitBuilderWrapper:
        """:param ir: The Qiskit instructions"""
        return QiskitBuilderWrapper(ir)
