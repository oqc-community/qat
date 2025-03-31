# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from functools import singledispatchmethod

from qat.ir.gates.base import GateBase
from qat.ir.gates.native import X_pi_2, Z_phase, ZX_pi_4
from qat.ir.gates.operation import Barrier, Measure, Reset
from qat.ir.instructions import PhaseShift, QuantumInstruction, Synchronize
from qat.ir.measure import Acquire
from qat.ir.waveforms import Pulse, SampledWaveform, Waveform
from qat.middleend.decompositions.base import DecompositionBase
from qat.middleend.decompositions.gates import (
    DefaultGateDecompositions,
    GateDecompositionBase,
)
from qat.model.hardware_model import PhysicalHardwareModel


class PulseDecompositionBase(DecompositionBase):
    """Base object for implementing decompositions of :class:`NativeGate` and other qubit
    operations such as :class:`Measure`, :class:`Reset` and :class:`Barrier`.

    This handles decompositions of gates / operations to pulse level instructions. Details
    of gate-to-gate decompositions should not be specified here, but can be injected as
    :attr:`gate_decompositions` to allow for decomposition of higher-level gates into
    pulses.
    """

    end_nodes = (QuantumInstruction,)

    def __init__(
        self,
        gate_decompositions: GateDecompositionBase = None,
    ):
        """
        :param gate_decompositions: The DAG for decomposing higher-level, non-native gates
            can be provided as an optional argument. Uses the
            :class:`DefaultGateDecompositions` by default.
        """

        if not gate_decompositions:
            gate_decompositions = DefaultGateDecompositions()
        self.gate_decompositions = gate_decompositions

    @singledispatchmethod
    def decompose_op(self, gate: GateBase, model: PhysicalHardwareModel):
        """Implements the definition of a decomposition of a gate.

        The definition does not have to be in terms of native gates, but decompositions
        must form a DAG. This will call :attr:`gate_decompositions` for higher-level gates.
        Decompositon details of native gates and other qubit operations should be
        implemented here.
        """
        return self.gate_decompositions.decompose(gate)


class DefaultPulseDecompositions(PulseDecompositionBase):
    """Implements the standard gate-to-pulse decompositions used in OQC hardware.

    Provides decomposition rules for the native gate set {:class:`Z_phase`,
    :class:`X_pi_2`, :class:`ZX_pi_4`}. Gates which are not part of
    the native set are first decomposed into native gates using
    :class:`DefaultGateDecompositions`, and then decomposed into pulses using the rules
    provided here.
    """

    end_nodes = (QuantumInstruction,)

    def __init__(self, gate_decompositions: DecompositionBase = None):
        """
        :param gate_decompositions: The DAG for decomposing higher-level, non-native gates
            can be provided as an optional argument. Uses the:class:`DefaultGateDecompositions`
            by default.
        """

        if not gate_decompositions:
            gate_decompositions = DefaultGateDecompositions()
        self.gate_decompositions = gate_decompositions

    @singledispatchmethod
    def decompose_op(self, gate: GateBase, *args):
        """Implements the definition of a decomposition of a gate.

        The definition does not have to be in terms of native gates, but decompositions
        must form a DAG. This will call :attr:`gate_decompositions` for higher-level gates.
        Decompositon details of native gates and other qubit operations should be
        implemented here.
        """
        return self.gate_decompositions.decompose(gate)

    @decompose_op.register(X_pi_2)
    def _(self, gate: X_pi_2, model: PhysicalHardwareModel):
        """Decomposes the :class:`X_pi_2` intruction into a drive pulse."""

        qubit = model.qubit_with_index(gate.qubit)
        pulse_channel = qubit.drive_pulse_channel
        pulse_info = pulse_channel.pulse.model_dump()
        pulse_waveform = pulse_channel.pulse.waveform_type(**pulse_info)
        return [Pulse(targets=pulse_channel.uuid, waveform=pulse_waveform)]

    @decompose_op.register(Z_phase)
    def _(self, gate: Z_phase, model: PhysicalHardwareModel):
        """Decomposes the :class:`Z_phase` intruction into a :class:`PhaseShift` on all
        appropiate channels."""

        qubit = model.qubit_with_index(gate.qubit)
        pulse_channel = qubit.drive_pulse_channel
        instructions = [PhaseShift(targets=pulse_channel.uuid, phase=gate.theta)]
        for (
            qid,
            crc_pulse_channel,
        ) in qubit.cross_resonance_cancellation_pulse_channels.items():
            coupled_qubit = model.qubit_with_index(qid)
            cr_pulse_channel = coupled_qubit.cross_resonance_pulse_channels[gate.qubit]

            instructions.append(PhaseShift(targets=cr_pulse_channel.uuid, phase=gate.theta))
            instructions.append(
                PhaseShift(targets=crc_pulse_channel.uuid, phase=gate.theta)
            )
        return instructions

    @decompose_op.register(ZX_pi_4)
    def _(self, gate: ZX_pi_4, model: PhysicalHardwareModel):
        r"""Decomposes a :math:`ZX(\pi/4)` into cross-resonance pulses."""

        qubit1 = model.qubits[gate.qubit1]
        qubit2 = model.qubits[gate.qubit2]
        target1_pulse_channel = qubit1.cross_resonance_pulse_channels[gate.qubit2]
        target2_pulse_channel = qubit2.cross_resonance_cancellation_pulse_channels[
            gate.qubit1
        ]

        if target1_pulse_channel is None or target2_pulse_channel is None:
            raise ValueError(
                f"Tried to perform cross resonance on {str(qubit2)} that isn't linked to "
                f"{str(qubit1)}."
            )

        pulse_info = target1_pulse_channel.zx_pi_4_pulse.model_dump()
        if pulse_info is None:
            raise ValueError(
                f"No `zx_pi_4_pulse` available on {qubit1} with index {gate.qubit1}."
            )

        return [
            Synchronize(targets=[target1_pulse_channel.uuid, target2_pulse_channel.uuid]),
            Pulse(
                targets=target1_pulse_channel.uuid,
                waveform=target1_pulse_channel.zx_pi_4_pulse.waveform_type(**pulse_info),
            ),
            Pulse(
                targets=target2_pulse_channel.uuid,
                waveform=target1_pulse_channel.zx_pi_4_pulse.waveform_type(**pulse_info),
            ),
        ]

    @decompose_op.register(Barrier)
    def _(self, gate: Barrier, model: PhysicalHardwareModel):
        r"""Decomposes a :class:`Barrier` into a :class:`Synchronize` across pulse channels."""

        pulse_channel_ids = set()
        for qid in gate.qubits:
            target = model.qubit_with_index(qid)
            pulse_channel_ids.add(target.acquire_pulse_channel.uuid)
            pulse_channel_ids.add(target.measure_pulse_channel.uuid)
            qubit_pulse_channel_ids = [
                pulse_channel.uuid for pulse_channel in target.all_pulse_channels
            ]
            pulse_channel_ids.add(*qubit_pulse_channel_ids)
        return [Synchronize(targets=pulse_channel_ids)]

    @decompose_op.register(Measure)
    def _(self, gate: Measure, model: PhysicalHardwareModel):
        r"""Decomposes a :class:`Measure` object into a measure pulse and an acquisition."""
        # TODO: deal with complexities of the output_variable: classical registers, random
        # generation of names, etc... (COMPILER-285)
        # TODO: post processing responsibilities? should we have a "return mode" in the
        # measure instruction. The post-processing path is decided by acquire mode + return
        # mode. (COMPILER-287)

        qubit = model.qubit_with_index(gate.qubit)

        # Measure-related info.
        measure_channel = qubit.measure_pulse_channel
        measure_instruction = Pulse(
            targets=measure_channel.uuid,
            waveform=Waveform(**measure_channel.pulse.model_dump()),
        )

        # Acquire-related info.
        acquire_channel = qubit.acquire_pulse_channel
        acquire_duration = (
            measure_channel.pulse.width
            if acquire_channel.acquire.sync
            else acquire_channel.acquire.width
        ) - acquire_channel.acquire.delay
        if acquire_channel.acquire.use_weights is False:
            filter = Pulse(
                waveform=SampledWaveform(samples=acquire_channel.acquire.weights),
                duration=acquire_duration,
            )
        else:
            filter = None

        acquire_instruction = Acquire(
            targets=acquire_channel.uuid,
            duration=acquire_duration,
            mode=gate.mode,
            delay=acquire_channel.acquire.delay,
            filter=filter,
            output_variable=None,
        )

        return [
            Barrier(qubits=(gate.qubit)),
            measure_instruction,
            acquire_instruction,
        ]

    @decompose_op.register(Reset)
    def _(self, gate: Reset, model: PhysicalHardwareModel):
        r"""Decomposes a :class:`Reset` into a "passive reset" (i.e. just wait for the state
        to decohere).

        Not implemented for default pulse decompositions.
        """

        return NotImplementedError("The Reset operation is not yet implemented.")
