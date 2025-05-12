# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd


from qat.ir.gates.gates_2q import ECR
from qat.ir.gates.native import X_pi_2, Z_phase
from qat.ir.instructions import PhaseShift, Synchronize
from qat.ir.waveforms import Pulse
from qat.middleend.decompositions.pulses import DefaultPulseDecompositions
from qat.utils.hardware_model import generate_hw_model


class TestDefaultPulseDecompositions:
    def test_X_pi_2_decomposition(self):
        model = generate_hw_model(4, seed=42)
        decomps = DefaultPulseDecompositions()
        instrs = decomps.decompose(X_pi_2(qubit=0), model)
        assert len(instrs) == 1
        assert isinstance(instrs[0], Pulse)
        assert instrs[0].pulse_channel == model.qubit_with_index(0).drive_pulse_channel.uuid

    def test_Z_phase_decomposition(self):
        model = generate_hw_model(4, seed=42)
        decomps = DefaultPulseDecompositions()
        instrs = decomps.decompose(Z_phase(qubit=0, theta=0.521), model)
        num_instructions = 1 + 2 * len(
            model.qubit_with_index(0).cross_resonance_cancellation_pulse_channels
        )
        assert len(instrs) == num_instructions
        assert all([isinstance(instr, PhaseShift) for instr in instrs])

    def test_ECR_decomposition(self):
        """Tests that the ECR decomposes into native pulse instructions, and ZX_pi_4 gives
        the correct too."""

        model = generate_hw_model(4, seed=42)
        decomps = DefaultPulseDecompositions()
        qubit2 = next(iter(model.physical_connectivity[0]))
        instrs = decomps.decompose(ECR(qubit1=0, qubit2=qubit2), model)
        assert len([instr for instr in instrs if isinstance(instr, Pulse)]) == 6
        assert len([instr for instr in instrs if isinstance(instr, Synchronize)]) == 2

        # two Z gates on each qubit
        num_instructions_Z1 = len(decomps.decompose(Z_phase(qubit=0, theta=0.254), model))
        num_instructions_Z2 = len(
            decomps.decompose(Z_phase(qubit=qubit2, theta=0.254), model)
        )

        # a total of five z_phase gates on q1 and two on q2
        num_instructions = 5 * num_instructions_Z1 + 2 * num_instructions_Z2
        assert (
            len([instr for instr in instrs if isinstance(instr, PhaseShift)])
            == num_instructions
        )
